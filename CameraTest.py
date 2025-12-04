#!/usr/bin/env python3
import serial
import time
from datetime import datetime

# Configure UART
ser = serial.Serial(
    port="/dev/ttyAMA0",
    baudrate=115200,
    timeout=1,
    write_timeout=2
)

print("Sending current time over UART. Press Ctrl+C to stop.\n")
count = 0
try:
    while True:
        # Prepare message
        count += 1
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"10000000{count}\r\n"

        # Write and flush
        ser.write(message.encode("utf-8"))
        ser.flush()  # Ensure all bytes are transmitted

        print(f"Sent: {message.strip()}")

        # Optional: small delay to prevent overwhelming slow receivers
        time.sleep(0.005)

except KeyboardInterrupt:
    print("\nStopped by user.")

except serial.SerialTimeoutException:
    print("⚠️ Serial write timeout — the UART buffer may be full.")

finally:
    ser.close()
