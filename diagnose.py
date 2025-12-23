import os
import json
import pandas as pd

TARGETS = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]

print("--- Diagnostic Report ---")
print(f"Current Directory: {os.getcwd()}")
print(f"Weights Dir exists: {os.path.exists('weights')}")

for target in TARGETS:
    prefix = f"weights/{target}_stochastic"
    meta_path = f"{prefix}_metadata.json"
    exists = os.path.exists(meta_path)
    print(f"Target: {target}")
    print(f"  Expected Metadata: {meta_path}")
    print(f"  Exists: {exists}")
    
    if exists:
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print(f"  Metadata loaded! Target in meta: {meta.get('target_name')}")
            print(f"  Features count: {len(meta.get('feature_cols', []))}")
        except Exception as e:
            print(f"  Error loading metadata: {e}")
    else:
        # Check for similar files
        print("  Looking for similar files in weights/...")
        all_files = os.listdir('weights')
        similar = [f for f in all_files if target in f]
        print(f"  Similar files found: {similar}")
print("--- End ---")
