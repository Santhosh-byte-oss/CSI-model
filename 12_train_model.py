import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

CSV_FILE = 'csi_training_features.csv'
MODEL_FILE = 'robust_csi_model.pkl'

def train_model():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please run data collector first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(CSV_FILE)
    
    print(f"Total samples: {len(df)}")
    print("Samples per class:")
    print(df['label'].value_counts())
    
    if len(df['label'].unique()) < 3:
        print("\nWARNING: You don't have all 3 classes (0, 1, 2) in your dataset yet!")
        print("The model needs data for Empty Room, Standing, and Walking to work properly.")
    
    X = df[['mean_diff', 'max_diff', 'var_diff']]
    y = df['label']
    
    print("\nSplitting data and training Random Forest model...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # We use a slightly deeper Random Forest to capture the subtle edge-case patterns
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nModel Accuracy on test set: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, MODEL_FILE)
    print(f"\nSUCCESS: Model saved as {MODEL_FILE}")
    print("You can now run your 10_web_dashboard.py script to test it in real-time!")

if __name__ == '__main__':
    train_model()
