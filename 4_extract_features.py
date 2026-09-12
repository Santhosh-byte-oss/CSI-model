import pandas as pd
import numpy as np

def extract_features(file_path, label_name, label_id):
    try:
        # Load the cleaned dataset
        df = pd.read_csv(file_path)
        
        # Create an empty dataframe to hold our new features
        features = pd.DataFrame()
        
        # Extract Spatial Features (calculating across subcarriers for each time packet)
        features['mean_amp'] = df.mean(axis=1)
        features['std_amp'] = df.std(axis=1)
        features['variance_amp'] = df.var(axis=1)
        features['max_amp'] = df.max(axis=1)
        features['min_amp'] = df.min(axis=1)
        features['range_amp'] = features['max_amp'] - features['min_amp']
        features['rms_amp'] = np.sqrt((df ** 2).mean(axis=1))
        
        # Add the classification labels for ML training
        features['label_name'] = label_name
        features['label_id'] = label_id
        
        return features
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame()

print("1. Extracting features from cleaned data...")
empty_features = extract_features('clean_empty.csv', 'Empty', 0)
standing_features = extract_features('clean_standing.csv', 'Standing', 1)
walking_features = extract_features('clean_walking.csv', 'Walking', 2)

print("2. Combining into a single ML dataset...")
# Combine all three conditions into one master dataset
all_features = pd.concat([empty_features, standing_features, walking_features], ignore_index=True)

# Drop any rows that might contain NaN values from the smoothing process
all_features = all_features.dropna()

# Save the final dataset
output_file = 'ml_features_dataset.csv'
all_features.to_csv(output_file, index=False)

print(f"\nFeature extraction complete! Saved as '{output_file}'.")
print("-" * 40)
print(f"Total packets ready for ML: {all_features.shape[0]}")
print(f"Total features per packet: {all_features.shape[1] - 2}") # Subtracting the 2 label columns
print("\nPacket breakdown:")
print(all_features['label_name'].value_counts().to_string())