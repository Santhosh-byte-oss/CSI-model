import serial
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from collections import deque

# --- CONFIGURATION ---
COM_PORT = 'COM3'  # <-- Verify this is correct
BAUD_RATE = 115200 
MODEL_FILE = 'robust_csi_model.pkl'
CALIBRATION_PACKETS = 30 

print("Loading robust ML model...")
try:
    model = joblib.load(MODEL_FILE)
except Exception as e:
    print(f"Error: {e}")
    exit()

labels = {0: 'No human', 1: 'Human present (Standing)', 2: 'Movement detected (Walking)'}
packet_buffer = deque(maxlen=3)
calibration_data = []
valid_indices = None
live_baseline = None

try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to ESP32 on {COM_PORT}!")
except Exception as e:
    print(f"Connection failed: {e}")
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
                payload = [int(x) for x in parts[11:]]
                amplitudes = []
                for i in range(0, len(payload)-1, 2):
                    real = payload[i]
                    imag = payload[i+1]
                    amplitudes.append(np.sqrt(real**2 + imag**2))
                
                # --- CALIBRATION PHASE ---
                if live_baseline is None:
                    calibration_data.append(amplitudes)
                    print(f"Calibrating baseline... {len(calibration_data)}/{CALIBRATION_PACKETS}", end='\r')
                    
                    if len(calibration_data) >= CALIBRATION_PACKETS:
                        calib_array = np.array(calibration_data)
                        means = np.mean(calib_array, axis=0)
                        variances = np.var(calib_array, axis=0)
                        
                        valid_indices = np.where((means > 0) & (variances < 50))[0]
                        live_baseline = means[valid_indices] # Lock in the empty room signature
                        
                        print(f"\nCalibration complete! Locked {len(valid_indices)} clean subcarriers.")
                        print("-" * 40)
                    continue
                
                # --- LIVE INFERENCE PHASE ---
                clean_amplitudes = np.array(amplitudes)[valid_indices]
                packet_buffer.append(clean_amplitudes)
                
                if len(packet_buffer) == 3:
                    smoothed_data = np.mean(packet_buffer, axis=0)
                    
                    # Calculate difference from the live baseline
                    diff = np.abs(smoothed_data - live_baseline)
                    
                    features = pd.DataFrame([{
                        'mean_diff': np.mean(diff),
                        'max_diff': np.max(diff),
                        'var_diff': np.var(diff)
                    }])
                    
                    prediction = model.predict(features)[0]
                    status = labels[prediction]
                    
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"{timestamp}  {status}")
                    
    except KeyboardInterrupt:
        print("\nStopping real-time inference.")
        ser.close()
        break
    except Exception:
        pass