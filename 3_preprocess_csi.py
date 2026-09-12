import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_calculate_amplitude(filepath):
    amplitudes_list = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for parts in reader:
                # Strictly filter for valid CSI packets
                if parts and parts[0] == 'CSI' and len(parts) == 613:
                    payload = [int(x) for x in parts[11:]]
                    
                    # Convert to amplitude: sqrt(real^2 + imag^2)
                    amplitudes = []
                    for i in range(0, len(payload)-1, 2):
                        real = payload[i]
                        imag = payload[i+1]
                        amplitudes.append(np.sqrt(real**2 + imag**2))
                    amplitudes_list.append(amplitudes)
        return np.array(amplitudes_list)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def find_valid_subcarriers(empty_data):
    # Calculate mean and variance of the empty room (baseline)
    means = np.mean(empty_data, axis=0)
    variances = np.var(empty_data, axis=0)
    
    # Valid subcarriers must have a signal (mean > 0) 
    # and must NOT be an artificial pilot spike (variance < 50 in an empty room)
    valid_indices = np.where((means > 0) & (variances < 50))[0]
    
    print(f"Kept {len(valid_indices)} valid subcarriers out of {empty_data.shape[1]}")
    return valid_indices

def apply_moving_average(data, window_size=3):
    # Apply a rolling mean across the time axis (rows) for each subcarrier (columns)
    df = pd.DataFrame(data)
    smoothed_data = df.rolling(window=window_size, min_periods=1).mean().values
    return smoothed_data

def save_cleaned_data(filename, data):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")

# --- Main Execution ---
print("1. Loading raw data...")
raw_empty = load_and_calculate_amplitude('empty_room.csv')
raw_standing = load_and_calculate_amplitude('standing.csv')
raw_walking = load_and_calculate_amplitude('walking.csv')

print("\n2. Sanitizing subcarriers based on Empty Room baseline...")
# Use the empty room as the baseline to find which subcarriers are physically valid
valid_indices = find_valid_subcarriers(raw_empty)

# Filter all datasets using the exact same valid subcarrier indices
clean_empty = raw_empty[:, valid_indices]
clean_standing = raw_standing[:, valid_indices]
clean_walking = raw_walking[:, valid_indices]

print("\n3. Applying Temporal Smoothing (Moving Average)...")
# A window of 3 smooths out rapid radio jitter while preserving walking/standing features
smooth_empty = apply_moving_average(clean_empty, window_size=3)
smooth_standing = apply_moving_average(clean_standing, window_size=3)
smooth_walking = apply_moving_average(clean_walking, window_size=3)

print("\n4. Saving preprocessed datasets...")
save_cleaned_data('clean_empty.csv', smooth_empty)
save_cleaned_data('clean_standing.csv', smooth_standing)
save_cleaned_data('clean_walking.csv', smooth_walking)

# --- Visualization ---
plt.figure(figsize=(12, 5))
plt.plot(np.var(smooth_empty, axis=0), label='Empty Room', color='blue')
plt.plot(np.var(smooth_standing, axis=0), label='Standing', color='orange')
plt.plot(np.var(smooth_walking, axis=0), label='Walking', color='green')

plt.title('Cleaned CSI Amplitude Variance (Pilots & Nulls Removed)')
plt.xlabel('Valid Subcarrier Index')
plt.ylabel('Variance over time')
plt.legend()
plt.tight_layout()
plt.savefig('csi_cleaned_variance.png')

print("\nPreprocessing complete. Check 'csi_cleaned_variance.png' for the final signal quality.")