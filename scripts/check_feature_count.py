
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from algae_fusion.config import NON_FEATURE_COLS, WINDOW_SIZE

def check_feature_count():
    print(">>> Checking Feature Count Logic...")
    
    # 1. Load a sample of the dataset (e.g. test set)
    df = pd.read_csv("data/dataset_test.csv")
    
    # 2. Identify Static Features
    all_cols = df.columns.tolist()
    feature_cols = [c for c in all_cols if c not in NON_FEATURE_COLS]
    
    print(f"\n[1] Static Features (Original): {len(feature_cols)}")
    # print(f"    (First 5): {feature_cols[:5]}")
    
    # 3. Identify Morphology Columns (subset of features used for history)
    # Logic from pipeline.py:
    morph_cols = [c for c in feature_cols if "intensity" in c or "area" in c or "perimeter" in c or "eccentricity" in c]
    print(f"\n[2] Morphological Columns (for History): {len(morph_cols)}")
    
    # 4. Calculate Expected Dynamic Count
    # Formula: 
    #   Total = Static 
    #         + (WINDOW_SIZE) * Morph_Cols  (Prev1, Prev2, Prev3...)
    #         + (WINDOW_SIZE) * Morph_Cols  (Velocity - if enabled) [CURRENTLY ENABLED]
    #         + (WINDOW_SIZE-1) * Morph_Cols (Accel - if enabled) [CURRENTLY ENABLED]
    #         + Extra Metadata (Prev_Time, dt, etc. if not clean)
    
    N = len(morph_cols)
    static = len(feature_cols)
    
    # Current Code in sliding_window_stochastic.py:
    # It adds Prev{k} for k in 1..WINDOW_SIZE (default 3)
    step1 = static + (3 * N)
    print(f"    -> If minimal history (Prev1/2/3): {static} + 3*{N} = {step1}")
    
    # It adds Velocity for k=1 (1 set)
    step2 = step1 + N 
    print(f"    -> If +Velocity (Vel1): {step1} + {N} = {step2}")
    
    # It adds Acceleration for k=2 (1 set)
    step3 = step2 + N
    print(f"    -> If +Acceleration (Accel1): {step2} + {N} = {step3}")

    print(f"\n[3] Your Observation: 381 Features")
    
    if 381 == step3:
        print(f"   MATCH! 381 = {static} (Static) + {N}*3 (Prev) + {N} (Vel) + {N} (Accel)")
    elif 381 == step2:
         print(f"   MATCH! 381 = {static} (Static) + {N}*3 (Prev) + {N} (Vel)")
    elif 381 == step1:
         print(f"   MATCH! 381 = {static} (Static) + {N}*3 (Prev)")
    else:
        diff = 381 - step1
        print(f"   Mismatch. Residual = {diff}. Maybe metadata columns (Prev_Time)?")

if __name__ == "__main__":
    check_feature_count()
