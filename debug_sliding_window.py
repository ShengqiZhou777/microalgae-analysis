import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic

def run_debug():
    print("=== DEBUG: Real Data Sliding Window Logic Verification ===")
    
    csv_path = "data/dataset_test.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        return

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter for Light (Simulating the pipeline environment)
    # Note: Logic usually handles Initial -> Light mapping internally or assumes balanced
    if 'Initial' in df['condition'].unique():
        df = df[df['condition'] == 'Light'].reset_index(drop=True)
    else:
        df = df[df['condition'] == 'Light'].reset_index(drop=True)
        
    print(f"Loaded {len(df)} samples for Condition='Light'.")
    
    # Identify Morph Cols
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    candidates = [c for c in all_numeric_cols if 'area' in c.lower()]
    morph_col = candidates[0] if candidates else all_numeric_cols[0]
    print(f"Using feature '{morph_col}' for verification.")
    
    # --- REPLICATE MATCHING LOGIC MANUALLY ---
    print("\n---> Replicating Pipeline Sorting & Indexing...")
    # The pipeline sorts by [time, condition, file]
    df_sorted = df.sort_values(['time', 'condition', 'file']).reset_index(drop=True)
    # Then assigns group_idx
    df_sorted['group_idx'] = df_sorted.groupby(['time', 'condition']).cumcount()
    
    print("\n---> Running compute_sliding_window_features_stochastic (window=2)...")
    df_out = compute_sliding_window_features_stochastic(df, window_size=2, morph_cols=[morph_col])
    
    prev1_col = f"Prev1_{morph_col}"
    prev2_col = f"Prev2_{morph_col}"
    
    # 3. Explicit Verification
    print("\n[VERIFICATION] Tracing a single sample's Pseudo-History...")
    
    # 3.1 Pick a sample at t=24
    if 24 not in df_out['time'].values:
        t_target = df_out['time'].max()
    else:
        t_target = 24
        
    # Pick the first one for simplicity (Limit 1)
    sample_out = df_out[df_out['time'] == t_target].iloc[0]
    
    # Identify this sample in our manually sorted DF to get its group_idx
    # We match by file ID to be sure
    file_id = sample_out['file']
    manual_row = df_sorted[df_sorted['file'] == file_id].iloc[0]
    group_idx = manual_row['group_idx']
    
    print(f"Selected Sample: '{file_id}' (Time={t_target}h)")
    print(f"  -> Assigned Group Index (Rank): {group_idx}")
    print(f"  -> Current Value: {sample_out[morph_col]:.4f}")
    print(f"  -> Pipeline Generated Prev1: {sample_out[prev1_col]:.4f}")
    
    # 3.2 Determine Expected Ancestor Index
    # Logic from sliding_window_stochastic.py:
    # all_times = sorted unique times.
    all_times = sorted(df_sorted['time'].unique())
    curr_t_idx = all_times.index(t_target)
    
    # Calculate Prev1 Target Index (k=1)
    # If standard: prev_idx = curr_t_idx - 1
    # If custom logic for 1h? (Not applicable for 24h)
    prev_t_idx = curr_t_idx - 1
    prev_time = all_times[prev_t_idx]
    
    print(f"\n--> Manual History Lookup:")
    print(f"  -> Current Time List Index: {curr_t_idx} ({t_target}h)")
    print(f"  -> Expected Prev1 Time Index: {prev_t_idx} ({prev_time}h)")
    print(f"  -> Looking for sample at Time={prev_time}h with Group Index={group_idx}...")
    
    # 3.3 Find the match
    ancestor = df_sorted[(df_sorted['time'] == prev_time) & (df_sorted['group_idx'] == group_idx)]
    
    if not ancestor.empty:
        true_val = ancestor.iloc[0][morph_col]
        ancestor_file = ancestor.iloc[0]['file']
        print(f"  [FOUND] Ancestor File: '{ancestor_file}'")
        print(f"  [FOUND] Ancestor Value: {true_val:.4f}")
        
        if np.isclose(true_val, sample_out[prev1_col]):
            print(f"  [SUCCESS] Match Confirmed! (Diff: {abs(true_val - sample_out[prev1_col]):.6f})")
        else:
            print(f"  [FAIL] Mismatch! Pipeline: {sample_out[prev1_col]}, Manual: {true_val}")
    else:
        print("  [WARN] Ancestor row not found (Index out of bounds due to dropout?).")

if __name__ == "__main__":
    run_debug()
