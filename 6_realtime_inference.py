import serial
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from collections import deque

# --- CONFIGURATION ---
COM_PORT = 'COM3'  # <-- Change this to match your ESP32 COM port
BAUD_RATE = 115200 
MODEL_FILE = 'csi_human_detection_model.pkl'
CALIBRATION_PACKETS = 30  # Number of packets to read for baseline calibration

print("Loading ML model...")
try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

labels = {0: 'No human', 1: 'Human present (Standing)', 2: 'Movement detected (Walking)'}
packet_buffer = deque(maxlen=3)
calibration_data = []
valid_indices = None

print(f"Connecting to ESP32 on {COM_PORT}...")
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print("Connected!")
except Exception as e:
    print(f"Serial connection failed: {e}")
    exit()

print("\n" + "="*40)
print("   AUTO-CALIBRATION: PLEASE STAND BACK")
print("   Ensure the room is empty/still.")
print("="*40 + "\n")

while True:
    try:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        
        if line.startswith('CSI'):
            parts = line.split(',')
            
            if len(parts) == 613:
                # 1. Parse payload to amplitude
                payload = [int(x) for x in parts[11:]]
                amplitudes = []
                for i in range(0, len(payload)-1, 2):
                    real = payload[i]
                    imag = payload[i+1]
                    amplitudes.append(np.sqrt(real**2 + imag**2))
                
                # 2. Calibration Phase
                if valid_indices is None:
                    calibration_data.append(amplitudes)
                    print(f"Calibrating... {len(calibration_data)}/{CALIBRATION_PACKETS} packets", end='\r')
                    
                    if len(calibration_data) >= CALIBRATION_PACKETS:
                        calib_array = np.array(calibration_data)
                        means = np.mean(calib_array, axis=0)
                        variances = np.var(calib_array, axis=0)
                        
                        # Keep only subcarriers with signal and without extreme noise spikes
                        valid_indices = np.where((means > 0) & (variances < 50))[0]
                        print(f"\nCalibration complete! Locked onto {len(valid_indices)} clean subcarriers.")
                        print("-" * 40)
                        print("Starting real-time AI inference...\n")
                    continue
                
                # 3. Live Inference Phase (Only using valid subcarriers)
                clean_amplitudes = np.array(amplitudes)[valid_indices]
                packet_buffer.append(clean_amplitudes)
                
                if len(packet_buffer) == 3:
                    # Apply moving average across the time axis
                    smoothed_data = np.mean(packet_buffer, axis=0)
                    
                    # Extract spatial features using ONLY the clean data
                    features = pd.DataFrame([{
                        'mean_amp': np.mean(smoothed_data),
                        'std_amp': np.std(smoothed_data),
                        'variance_amp': np.var(smoothed_data),
                        'max_amp': np.max(smoothed_data),
                        'min_amp': np.min(smoothed_data),
                        'range_amp': np.max(smoothed_data) - np.min(smoothed_data),
                        'rms_amp': np.sqrt(np.mean(smoothed_data**2))
                    }])
                    
                    # Run inference
                    prediction = model.predict(features)[0]
                    status = labels[prediction]
                    
                    # Output real-time result
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"{timestamp}  {status}")
                    
    except KeyboardInterrupt:
        print("\nStopping real-time inference.")
        ser.close()
        break
    except Exception as e:
        pass