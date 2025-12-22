
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic

def naive_baseline_check(target="Dry_Weight"):
    print(f"\n--- Checking Naive Baseline (Persistence) for {target} ---")
    
    # 1. Load Data
    df = pd.read_csv("data/dataset_train.csv")
    
    # 2. Get Previous Value (Simulate Dynamic Feature Engineering)
    # We use the same function to ensure fairness
    morph_cols = [target] # Trick: Ask for target as a "morph feature" to fetch its history
    
    print("Computing Previous Values...")
    df_aug = compute_sliding_window_features_stochastic(df, window_size=1, morph_cols=morph_cols)
    
    # Drop rows where history is missing (e.g., Time 0 or missing link)
    prev_col = f"Prev1_{target}"
    df_clean = df_aug.dropna(subset=[target, prev_col])
    
    y_true = df_clean[target]
    y_pred_naive = df_clean[prev_col] # The "Naivest" Prediction: Tomorrow = Today
    
    # 3. Calculate Scores
    r2 = r2_score(y_true, y_pred_naive)
    
    print(f"Data Points: {len(df_clean)}")
    print(f"Naive Baseline R2: {r2:.4f}")
    
    if r2 > 0.90:
        print("=> CONCLUSION: High R2 is expected due to strong autocorrelation.")
    else:
        print("=> CONCLUSION: Autocorrelation is weak. High model R2 implies genuine learning (or leakage).")

if __name__ == "__main__":
    for t in ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]:
        naive_baseline_check(t)
