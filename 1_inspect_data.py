import pandas as pd
import os

# Define the files to check
files = ['empty_room.csv', 'standing.csv', 'walking.csv']

def inspect_csi_data(filename):
    print(f"\n{'='*40}")
    print(f"Inspecting: {filename}")
    print(f"{'='*40}")
    
    if not os.path.exists(filename):
        print(f"Error: {filename} not found in the current directory.")
        return

    try:
        # Read the first few rows to understand structure
        df = pd.read_csv(filename)
        
        print(f"Total Rows: {df.shape[0]}")
        print(f"Total Columns: {df.shape[1]}")
        print("\nColumns detected:")
        print(list(df.columns))
        
        print("\nFirst 3 rows of data:")
        print(df.head(3))
        
        # Check for missing values
        missing_data = df.isnull().sum().sum()
        print(f"\nTotal missing values in file: {missing_data}")
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

# Run inspection on all three datasets
for f in files:
    inspect_csi_data(f)