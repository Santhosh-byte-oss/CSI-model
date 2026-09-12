import serial
import time
import numpy as np
import pandas as pd
from collections import deque
import os

# --- CONFIGURATION ---
ESP32_PORT = 'COM3'
BAUD_RATE = 921600
CALIBRATION_PACKETS = 30
CSV_FILE = 'csi_training_features.csv'

def collect_data():
    print(f"Opening {ESP32_PORT}...")
    try:
        ser = serial.Serial(ESP32_PORT, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f"Failed to connect to ESP32: {e}")
        return

    # 1. Calibration Phase
    print("\n--- STEP 1: CALIBRATION ---")
    print("Please step OUTSIDE the room. Calibrating in 3 seconds...")
    time.sleep(3)
    
    calibration_data = []
    expected_length = None
    
    while len(calibration_data) < CALIBRATION_PACKETS:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith('CSI'):
            parts = line.split(',')
            if len(parts) > 10:
                payload = np.array(parts[11:], dtype=np.float32)
                amplitudes = np.sqrt(payload[0::2]**2 + payload[1::2]**2)
                
                if expected_length is None and len(amplitudes) > 10:
                    expected_length = len(amplitudes)
                if expected_length is not None and len(amplitudes) == expected_length:
                    calibration_data.append(amplitudes)
                    print(f"Calibrating: {len(calibration_data)}/{CALIBRATION_PACKETS}", end='\r')
                    
    calib_array = np.array(calibration_data)
    means = np.mean(calib_array, axis=0)
    variances = np.var(calib_array, axis=0)
    valid_indices = np.where((means > 0) & (variances < 50))[0]
    if len(valid_indices) == 0:
        valid_indices = np.arange(len(means))
    live_baseline = means[valid_indices]
    print("\nCalibration complete!")

    # 2. Data Collection Phase
    print("\n--- STEP 2: DATA COLLECTION ---")
    print("0: Empty Room | 1: Standing (Corners) | 2: Walking (Corners/Edges)")
    
    try:
        label = int(input("Enter label to record (0, 1, or 2): "))
    except ValueError:
        print("Invalid label.")
        return
        
    duration = int(input("How many seconds to record? (e.g., 60): "))
    
    print(f"\nRecording label {label} for {duration} seconds... GO!")
    
    packet_buffer = deque(maxlen=3)
    features_list = []
    
    start_time = time.time()
    while time.time() - start_time < duration:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line.startswith('CSI'):
            parts = line.split(',')
            if len(parts) > 10:
                payload = np.array(parts[11:], dtype=np.float32)
                amplitudes = np.sqrt(payload[0::2]**2 + payload[1::2]**2)
                
                if len(amplitudes) == expected_length:
                    clean_amplitudes = amplitudes[valid_indices]
                    packet_buffer.append(clean_amplitudes)
                    
                    if len(packet_buffer) == 3:
                        smoothed_data = np.mean(packet_buffer, axis=0)
                        diff = np.abs(smoothed_data - live_baseline)
                        
                        features_list.append({
                            'label': label,
                            'mean_diff': np.mean(diff),
                            'max_diff': np.max(diff),
                            'var_diff': np.var(diff)
                        })
    
    print(f"\nFinished recording! Captured {len(features_list)} samples.")
    
    # Save to CSV
    df = pd.DataFrame(features_list)
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(CSV_FILE, index=False)
        
    print(f"Data saved to {CSV_FILE}.")

if __name__ == '__main__':
    collect_data()