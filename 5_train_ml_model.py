import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 1. Load the dataset
print("Loading feature dataset...")
df = pd.read_csv('ml_features_dataset.csv')

# Separate features (X) from the target labels (y)
X = df.drop(columns=['label_name', 'label_id'])
y = df['label_id']  # 0: Empty, 1: Standing, 2: Walking
target_names = ['Empty', 'Standing', 'Walking']

# 2. Split into Training and Testing sets (80% train, 20% test)
# stratify=y ensures the 80/20 split maintains the ratio of our 3 classes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training on {len(X_train)} packets, Testing on {len(X_test)} packets.\n")

# 3. Initialize and Train the Model
print("Training Random Forest Classifier...")
# n_estimators=100 means we are building 100 decision trees
clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate the Model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("-" * 40)
print(f"MODEL ACCURACY: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion Matrix:")
matrix = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(matrix, index=[f"Actual {n}" for n in target_names], 
                     columns=[f"Predicted {n}" for n in target_names])
print(df_cm)
print("-" * 40)

# 5. Feature Importance (What is the AI actually looking at?)
print("\nFeature Importance (Top 3 most critical metrics for detection):")
importances = pd.Series(clf.feature_importances_, index=X.columns)
print(importances.sort_values(ascending=False).head(3).to_string())

# 6. Save the trained model for real-time deployment
model_filename = 'csi_human_detection_model.pkl'
joblib.dump(clf, model_filename)
print(f"\nModel successfully saved as '{model_filename}'. Ready for real-time edge deployment.")