from picamera2 import Picamera2
import cv2
import time
import numpy as np
import json
import os

# ---------------------------------------------------------
# 1. Open Pi Camera（Picamera2）
# ---------------------------------------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (1280, 720)})
picam2.configure(config)
picam2.start()
time.sleep(0.3)   # short warm-up for AE/AWB

print("Camera started. Press 'q' to quit.")

# ---------------------------------------------------------
# 2. Laod config 加载配置文件（HSV + ROI）
# ---------------------------------------------------------
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

    # 先读取一帧确认尺寸
    tmp = picam2.capture_array()
    H, W = tmp.shape[:2]
    x, y, w, h = 0, 0, W, H

# ====== FILTER INITIALISATION ======
slope_history = []     # 用于 median filter（窗口最多 3）
slope_filtered = None  # EMA 的状态量
alpha = 0.2            # EMA 系数，可调
outlier_threshold = 0.5  # slope 跳变阈值，可按实际调节

# ---------------------------------------------------------
# 3. 主循环：实时处理
# ---------------------------------------------------------

while True:
    # 拍摄一帧
    frame = picam2.capture_array()     # RGB888
    frame_bgr = frame #cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Crop to ROI ROI 裁剪
    roi_frame = frame_bgr[y:y+h, x:x+w]
    if roi_frame.size == 0:
        print("ROI invalid. Please check config.json.")
        break

    # Convert to HSV and create mask HSV 分割
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up mask去噪
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # # Find contours 查找轮廓
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

            # === 计算斜率 ===
            if abs(vx) < 1e-6:
                slope = float('inf')     # 垂直线
            else:
                slope = vy / vx

            # ====== SCHEME 1: Adaptive Median + EMA ======

            # 初始化历史 slope（用于 outlier 检测）
            if slope_history:
                slope_prev = slope_history[-1]
            else:
                slope_prev = slope

            # 判断是否是 outlier（可根据实际情况调节 outlier_threshold）
            if abs(slope - slope_prev) > outlier_threshold:
                # 使用 adaptive median（只对 outlier 修正）
                slope_history.append(slope)
                if len(slope_history) > 3:
                    slope_history.pop(0)
                slope_med = float(np.median(slope_history))
            else:
                # 非 outlier 直接使用原 slope
                slope_history.append(slope)
                if len(slope_history) > 3:
                    slope_history.pop(0)
                slope_med = slope

            # EMA 滤波
            if slope_filtered is None:
                slope_filtered = slope_med
            else:
                slope_filtered = alpha * slope_med + (1 - alpha) * slope_filtered

            # 使用 slope_filtered 作为最终输出
            print("Slope (raw):", slope, " | Filtered (scheme1):", slope_filtered)

            # print("Slope:", slope)
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

                #lefty  = np.clip(lefty, 0, roi_frame.shape[0]-1)
                #righty = np.clip(righty, 0, roi_frame.shape[0]-1)
                
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
    

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
