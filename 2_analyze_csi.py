import csv
import numpy as np
import matplotlib.pyplot as plt

def process_csi(filepath):
    amplitudes_list = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for parts in reader:
                # We only process rows that have exactly 613 fields to avoid malformed packets
                if parts and parts[0] == 'CSI' and len(parts) == 613:
                    # Payload starts at index 11
                    payload = [int(x) for x in parts[11:]]
                    
                    # Convert consecutive pairs into amplitude
                    amplitudes = []
                    for i in range(0, len(payload)-1, 2):
                        real = payload[i]
                        imag = payload[i+1]
                        amp = np.sqrt(real**2 + imag**2)
                        amplitudes.append(amp)
                    amplitudes_list.append(amplitudes)
        
        if not amplitudes_list:
            print(f"Warning: No valid CSI packets found in {filepath}")
            return None
        return np.array(amplitudes_list)
        
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
        return None

print("Parsing CSV files...")
empty_amp = process_csi('empty_room.csv')
standing_amp = process_csi('standing.csv')
walking_amp = process_csi('walking.csv')

# Plot 1: Amplitude Variance over Time
plt.figure(figsize=(12, 6))

if empty_amp is not None:
    plt.plot(np.var(empty_amp, axis=0), label='Empty Room')
if standing_amp is not None:
    plt.plot(np.var(standing_amp, axis=0), label='Standing')
if walking_amp is not None:
    plt.plot(np.var(walking_amp, axis=0), label='Walking')

plt.title('CSI Amplitude Variance per Subcarrier (Motion Indicator)')
plt.xlabel('Subcarrier Index')
plt.ylabel('Variance over time')
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('csi_variance_plot.png')
print("Analysis complete. Check 'csi_variance_plot.png' in your folder.")