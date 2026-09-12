import serial
import csv
import time

PORT = 'COM3'
BAUD = 921600
LABEL = int(input("Enter label (0 = No Human, 1 = Standing, 2 = Walking): "))
FILENAME = 'esp32_hardware_data.csv'

print(f"Logging data for label {LABEL}. Press Ctrl+C to stop.")

ser = serial.Serial(PORT, BAUD, timeout=0.1)

with open(FILENAME, mode='a', newline='') as f:
    writer = csv.writer(f)
    # Write header if file is empty
    if f.tell() == 0:
        writer.writerow(['mean_diff', 'max_diff', 'var_diff', 'label'])
        
    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line.startswith('LOG,'):
                parts = line.split(',')
                mean_d, max_d, var_d = float(parts[1]), float(parts[2]), float(parts[3])
                writer.writerow([mean_d, max_d, var_d, LABEL])
                print(f"Saved -> Mean: {mean_d:.2f}, Max: {max_d:.2f}, Var: {var_d:.2f}")
    except KeyboardInterrupt:
        print("\nLogging stopped. Data saved successfully!")