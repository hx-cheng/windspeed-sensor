from picamera2 import Picamera2
import cv2
import time
import numpy as np
import json
import os

import serial
import threading
from threading import Event, Lock

import numpy as np
from collections import deque

class WindSpeedFilter:
    def __init__(self,
                 alpha=0.15,          # EMA 系数：越小越平滑
                 window_size=5,       # median filter 窗口
                 outlier_abs=1.0,     # 绝对跳变阈值 (m/s)
                 outlier_rel=0.4):    # 相对跳变阈值（40%）
        self.alpha = alpha
        self.window_size = window_size
        self.outlier_abs = outlier_abs
        self.outlier_rel = outlier_rel

        self.history = deque(maxlen=window_size)  # 原始 wind speed 历史
        self.filtered = None                      # EMA 状态

    def update(self, v_meas):
        """
        v_meas: 当前帧测到的 wind speed (m/s)，可以为 0 或噪声。
        返回：滤波后的 wind speed
        """

        # 1) 处理明显错误值（<=0 当成缺失值）
        if (v_meas is None) or (v_meas <= 0):
            # detection fail 的时候，用上一帧 filtered 值“顶住”
            if self.filtered is not None:
                return self.filtered
            else:
                # 如果一开始就坏点，只能先用 0 或者原值
                v_meas = 0.0

        # 2) 取上一帧历史
        if self.history:
            v_prev_raw = self.history[-1]
        else:
            v_prev_raw = v_meas

        # 3) outlier 检测：同时看绝对差和相对差
        diff_abs = abs(v_meas - v_prev_raw)
        diff_rel = diff_abs / max(v_prev_raw, 1e-6)

        is_outlier = (diff_abs > self.outlier_abs) and (diff_rel > self.outlier_rel)

        # 4) 更新 history，并做 median 处理
        self.history.append(v_meas)

        if is_outlier and len(self.history) >= 3:
            # outlier 时，用当前窗口的 median 修正
            v_med = float(np.median(list(self.history)))
        else:
            # 正常点直接用当前值（也可以用 median，更平滑）
            v_med = v_meas

        # 5) EMA
        if self.filtered is None:
            self.filtered = v_med
        else:
            #self.filtered = self.alpha * v_med + (1.0 - self.alpha) * self.filtered
            self.filtered = v_med

        return self.filtered

slope_cap = 15          # Low-pass Upper bound

shared_slope_clean = 0.0
slope_lock = Lock()
stop_event = Event()

wind_filter = WindSpeedFilter(alpha=0.15,
                              window_size=5,
                              outlier_abs=1.0,
                              outlier_rel=0.4)

# 0. Configure UART
ser = serial.Serial(
    port="/dev/ttyAMA0",
    baudrate=115200,
    timeout=1,
    write_timeout=2
)
print("Sending over UART. Press Ctrl+C to stop.\n")

# 1. Open Pi Camera（Picamera2）
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(0.3)   # short warm-up for AE/AWB

print("Camera started. Press 'q' to quit.")

# 2. Laod config（HSV + ROI）
CONFIG_FILE = 'config.json'
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    roi = config["roi"]  # [x, y, w, h]
    lower = np.array(config["lower_bound"], dtype=np.uint8)
    upper = np.array(config["upper_bound"], dtype=np.uint8)
    print(f"Loaded config: ROI={roi}, HSV={lower} to {upper}")

    x, y, w, h = roi
else:
    print("config.json not found! Using defaults.")

    lower = np.array([60, 50, 50], dtype=np.uint8)
    upper = np.array([85, 255, 255], dtype=np.uint8)

    tmp = picam2.capture_array()
    H, W = tmp.shape[:2]
    x, y, w, h = 0, 0, W, H

def uart_thread():
    global shared_slope_clean

    target_hz = 200.0
    dt = 1.0 / target_hz
    next_t = time.perf_counter()

    while not stop_event.is_set():

        # Read latest slope_clean
        with slope_lock:
            slope_to_send = shared_slope_clean
        
        msg = f"{slope_to_send:.5f}\r\n"
        ser.write(msg.encode("utf-8"))
        ser.flush()
        print("Sent:", msg.strip())

        # 200 Hz timer
        next_t += dt
        sleep_time = next_t - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_t = time.perf_counter()

    print("UART thread stopped.")

# Start uart thread
t_uart = threading.Thread(target=uart_thread, daemon=True)
t_uart.start()

# 3. main loop
while True:
    frame = picam2.capture_array()     # RGB888
    frame_bgr = frame

    # Crop to ROI
    roi_frame = frame_bgr[y:y+h, x:x+w]
    if roi_frame.size == 0:
        print("ROI invalid. Please check config.json.")
        break

    # Convert to HSV and create mask
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find contours 
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy for drawing
    result = frame_bgr.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)

    if contours:
        # Find largest contour
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 50:
            # Fit line
            [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)

            # Extract scalar values
            vx, vy = float(vx), float(vy)
            x0, y0 = float(x0), float(y0)

            # Slpoe calculation
            if abs(vx) < 1e-6:
                slope = float('inf')     # Vertical
            else:
                slope = vy / vx

            # Low-pass filtering
            if slope == float("inf") or slope > slope_cap:
                slope_clean = slope_cap
            elif slope < -slope_cap:
                slope_clean = -slope_cap
            else:
                slope_clean = slope

            # slope_clean = (
            #     0.028 * (slope_clean ** 4)
            #     - 0.51 * (slope_clean ** 3)
            #     + 3.36 * (slope_clean ** 2)
            #     - 10.04 * slope_clean
            #     + 15.80
            # )

            slope_clean = (
                -0.053 * (slope_clean ** 4)
                + 0.745 * (slope_clean ** 3)
                - 3.447 * (slope_clean ** 2)
                + 5.651 * slope_clean
                + 5.758
            )

            wind_speed_filtered = wind_filter.update(slope_clean)
                              
            if slope_clean == float("inf") or slope_clean > 10:
                slope_clean = 0

            # Output filtered slope data through UART
            with slope_lock:
                shared_slope_clean = slope_clean

            if abs(vx) < 1e-6:
                # Draw vertical line
                X = int(x + x0)  # x0 是 ROI 内坐标
                pt1 = (X, y)
                pt2 = (X, y + roi_frame.shape[0] - 1)
            else:
                # Calculate points to draw the line (relative to ROI)
                cols = roi_frame.shape[1]
                lefty  = int((-x0 * vy / vx) + y0)
                righty = int(((cols - x0) * vy / vx) + y0)
                
                # Adjust to full frame coordinates
                pt1 = (x + 0, y + lefty)
                pt2 = (x + cols - 1, y + righty)

            # Draw line
            cv2.line(result, pt1, pt2, (0, 255, 0), 2)

            # Draw contour for reference (adjust offset)
            c_shifted = c + np.array([x, y])
            cv2.drawContours(result, [c_shifted], -1, (0, 0, 255), 1)

    # Display side-by-side
    show = np.hstack((frame_bgr, result))
    cv2.imshow("Frame | Line Fitted (Picamera2)", show)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        stop_event.set()
        break

cv2.destroyAllWindows()
picam2.stop()

stop_event.set()
time.sleep(0.1)
t_uart.join(timeout=0.1)

ser.close()
print("UART closed.")