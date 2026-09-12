import joblib
from micromlgen import port

model_file = 'robust_csi_model.pkl'

try:
    print("Loading robust model...")
    clf = joblib.load(model_file)
    
    print("Translating Random Forest to C++...")
    # This generates the raw C code for the decision trees
    c_code = port(clf)
    
    # Save it as a header file for the Arduino IDE
    header_filename = 'CSI_Classifier.h'
    with open(header_filename, 'w') as f:
        f.write(c_code)
        
    print(f"\nSuccess! Your AI model has been exported to '{header_filename}'.")
    print("You can now include this in your ESP32 firmware.")
    
except Exception as e:
    print(f"Error during export: {e}")