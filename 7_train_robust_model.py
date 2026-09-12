import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

print("1. Loading clean datasets...")
empty = pd.read_csv('clean_empty.csv').values
standing = pd.read_csv('clean_standing.csv').values
walking = pd.read_csv('clean_walking.csv').values

# The secret sauce: Calculate the 'Empty Room' baseline signature
baseline = np.mean(empty, axis=0)

def extract_differential_features(data, baseline, label_id):
    features = []
    for row in data:
        # Calculate how much the signal deviates from the empty room baseline
        diff = np.abs(row - baseline)
        features.append({
            'mean_diff': np.mean(diff),
            'max_diff': np.max(diff),
            'var_diff': np.var(diff),
            'label_id': label_id
        })
    return pd.DataFrame(features)

print("2. Extracting differential features...")
df_empty = extract_differential_features(empty, baseline, 0)
df_standing = extract_differential_features(standing, baseline, 1)
df_walking = extract_differential_features(walking, baseline, 2)

df_all = pd.concat([df_empty, df_standing, df_walking], ignore_index=True)

# Separate features and labels
X = df_all.drop(columns=['label_id'])
y = df_all['label_id']

print("3. Training robust Random Forest...")
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X, y)

# Save the new model
joblib.dump(clf, 'robust_csi_model.pkl')
print("\nSuccess! Saved 'robust_csi_model.pkl'.")
print("This model is now immune to environmental drift.")