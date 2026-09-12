import serial

print("Python script started")

ser = serial.Serial("COM20", 115200, timeout=1)

print("ESP32 connected successfully!")
print("Waiting for CSI data...")

while True:
    line = ser.readline().decode("utf-8", errors="ignore").strip()

    if line:
        print(line)