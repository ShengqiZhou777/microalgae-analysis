
import os
import pandas as pd

def check_paths():
    print("Checking image paths in dataset_test.csv...")
    if not os.path.exists("data/dataset_test.csv"):
        print("dataset_test.csv missing.")
        return
        
    df = pd.read_csv("data/dataset_test.csv")
    
    # Check column name
    path_col = 'Source_Path' if 'Source_Path' in df.columns else 'file'
    if path_col not in df.columns:
        print(f"Cannot find path column. Columns: {df.columns.tolist()}")
        return
    
    # Check first valid path
    sample_path = df[path_col].iloc[0]
    print(f"Sample Path Raw: {sample_path}")
    
    # Logic to fix path if needed
    # If path starts with /home/shockley/myproject/DATASET4, it is absolute on host.
    # In docker, we are at /workspace/ (mapped to above).
    # So we might need to strip prefix.
    
    if os.path.exists(sample_path):
        print("✅ Path exists exactly as is.")
    else:
        print("❌ Path does not exist. Trying relative adjustment...")
        # Check if basename exists in data/images? Or assuming clear structure.
        # Let's try prepending current dir if it's relative
        if os.path.exists(os.path.abspath(sample_path)):
             print(f"✅ Absolute path works: {os.path.abspath(sample_path)}")
        else:
             print("❌ Still not found. We might need path mapping logic.")

if __name__ == "__main__":
    check_paths()
