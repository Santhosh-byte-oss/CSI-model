import joblib
from micromlgen import port
import os

MODEL_FILE = 'robust_csi_model.pkl'
HEADER_FILE = 'CSI_Model.h'

def export_model_to_cpp():
    if not os.path.exists(MODEL_FILE):
        print(f"Error: {MODEL_FILE} not found. Train your model first.")
        return

    print("Loading Random Forest model...")
    model = joblib.load(MODEL_FILE)
    
    print("Compiling model to C++ for ESP32...")
    # The port function translates the decision trees into native C++ if/else logic
    c_code = port(model)
    
    with open(HEADER_FILE, 'w') as f:
        f.write(c_code)
        
    print(f"\nSUCCESS: Edge AI model saved as {HEADER_FILE}")
    print("Move this file into your Arduino sketch folder!")

if __name__ == '__main__':
    export_model_to_cpp()