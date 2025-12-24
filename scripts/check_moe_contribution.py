
import os
import sys

# Add project root to path immediately
# Script is in scripts/, so root is one level up
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

print(f"[DEBUG] Project Root: {project_root}")
print(f"[DEBUG] Sys Path: {sys.path}")

import torch
import json
import joblib
import pandas as pd
import numpy as np

# Now imports should work
from algae_fusion.models.moe import GatingNetwork
from algae_fusion.config import NON_FEATURE_COLS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_contribution(target, condition, variance="stochastic"):
    model_prefix = f"weights/{target}_{condition}_{variance}"
    meta_path = f"{model_prefix}_metadata.json"
    gating_path = f"{model_prefix}_gating.pth"
    scaler_path = f"{model_prefix}_gating_scaler.joblib"
    
    if not os.path.exists(meta_path):
        print(f"Skipping {target} ({condition}): No metadata found.")
        return None

    # Load Metadata
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    gating_cols = meta['gating_cols']
    
    # Load Data
    train_path = os.path.join(project_root, "data/dataset_train.csv")
    test_path = os.path.join(project_root, "data/dataset_test.csv")
    
    if not os.path.exists(train_path):
         # Try relative if running from root
         train_path = "data/dataset_train.csv"
         test_path = "data/dataset_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Filter Condition
    if condition != "All":
        df = df[df['condition'] == condition]
    
    if df.empty:
        return None

    # Prepare Features
    missing = [c for c in gating_cols if c not in df.columns]
    if missing:
        if any(m.startswith("Prev") for m in missing):
            print(f"  [Warning] Missing history features for {target} {condition}. Skipping...")
            return None
        for c in missing:
            df[c] = 0
            
    X_gating = df[gating_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    
    # Load Scaler
    scaler = joblib.load(scaler_path)
    try:
        X_gating = scaler.transform(X_gating)
    except ValueError:
        print(f"  [Error] Scaler feature mismatch for {target} {condition}.")
        return None

    # Load Model
    state = torch.load(gating_path, map_location=DEVICE)
    if 'net.6.weight' in state:
        num_experts = state['net.6.weight'].shape[0]
    else:
        print(f"  [Error] Could not infer num_experts from state dict keys: {state.keys()}")
        return None

    g_net = GatingNetwork(input_dim=X_gating.shape[1], num_experts=num_experts).to(DEVICE)
    g_net.load_state_dict(state)
    g_net.eval()
    
    X_node = torch.tensor(X_gating, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        weights = g_net(X_node).cpu().numpy()
        
    avg_weights = weights.mean(axis=0)
    return avg_weights, num_experts

def main():
    targets = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
    conditions = ["Light", "Dark"]
    
    print(f"{'Target':<15} {'Cond':<10} {'XGB':<10} {'LGB':<10} {'LSTM':<10} {'CNN':<10}")
    print("-" * 70)
    
    for t in targets:
        for c in conditions:
            res = check_contribution(t, c)
            if res:
                avg, n = res
                xgb = f"{avg[0]*100:.1f}%"
                lgb = f"{avg[1]*100:.1f}%"
                
                if n == 3:
                     lstm = "N/A"
                     cnn = f"{avg[2]*100:.1f}%"
                elif n == 4:
                     lstm = f"{avg[2]*100:.1f}%"
                     cnn = f"{avg[3]*100:.1f}%"
                else:
                     lstm = "?"
                     cnn = "?"
                
                print(f"{t:<15} {c:<10} {xgb:<10} {lgb:<10} {lstm:<10} {cnn:<10}")

if __name__ == "__main__":
    main()
