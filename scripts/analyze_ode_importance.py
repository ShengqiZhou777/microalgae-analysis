
import os
import sys
import torch
import torch.nn as nn
import joblib
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root
sys.path.append(os.getcwd())

from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.models.ode import GrowthODE
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch

def load_ode_config(target, condition):
    config_path = f"archive_ode_20251229_054916/weights/ode_{target}_{condition}_config.json"
    defaults = {
        "latent_dim": 64,
        "ode_hidden_dim": 128,
        "decoder_hidden": 64,
        "decoder_dropout": 0.2,
    }
    if not os.path.exists(config_path):
        return defaults
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return {**defaults, **config}

class ODEProjector(nn.Module):
    def __init__(self, ode, latent_dim, decoder_hidden, decoder_dropout):
        super().__init__()
        self.ode = ode
        self.proj = nn.Sequential(
            nn.Linear(latent_dim, decoder_hidden),
            nn.Tanh(),
            nn.Dropout(p=decoder_dropout), 
            nn.Linear(decoder_hidden, 1)
        )
    def forward(self, x, t, mask):
        h = self.ode.ode_net(x, t, mask) 
        return self.proj(h).squeeze(-1)

def analyze_importance(target, condition):
    print(f"=== Analyzing Feature Importance for ODE: {target} ({condition}) ===")
    
    # 1. Load Data
    df_test = pd.read_csv("data/dataset_test.csv")
    if condition != "All":
        df_test = df_test[df_test['condition'] == condition].reset_index(drop=True)
    
    # Scale Target
    scaler_path = f"archive_ode_20251229_054916/weights/ode_{target}_{condition}_scaler.joblib"
    if not os.path.exists(scaler_path):
        print("Scaler not found.")
        return
    scaler = joblib.load(scaler_path)
    df_test[target] = scaler.transform(df_test[target].values)
    
    # 2. Setup Features
    feature_cols = [c for c in df_test.columns if c not in NON_FEATURE_COLS + [target]]
    group_col = 'group_idx' if 'group_idx' in df_test.columns else 'file'
    
    print(f"Features: {len(feature_cols)}")
    
    # 3. Load Model
    ode_config = load_ode_config(target, condition)
    input_dim = len(feature_cols)
    latent_dim = ode_config["latent_dim"]
    
    ode_core = GrowthODE(input_dim=input_dim, latent_dim=latent_dim, ode_hidden_dim=ode_config["ode_hidden_dim"]).to(DEVICE)
    model = ODEProjector(ode_core, latent_dim, ode_config["decoder_hidden"], ode_config["decoder_dropout"]).to(DEVICE)
    
    model_path = f"archive_ode_20251229_054916/weights/ode_{target}_{condition}.pth"
    if not os.path.exists(model_path):
        print("Model weights not found.")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 4. Prepare Batch
    ds = AlgaeTimeSeriesDataset(df_test, feature_cols, target, group_col=group_col)
    dl = DataLoader(ds, batch_size=len(ds), collate_fn=collate_ode_batch, shuffle=False)
    batch = next(iter(dl))
    
    x_orig = batch['features'].to(DEVICE) # (B, T, D)
    mask = batch['mask'].to(DEVICE)
    times = batch['times'].to(DEVICE)
    targets = batch['targets'].to(DEVICE)
    
    # Normalize time
    time_scale = df_test['time'].max()
    t_grid = times[0] / time_scale
    
    # 5. Baseline Loss
    criterion = nn.MSELoss()
    with torch.no_grad():
        preds = model(x_orig, t_grid, mask)
        baseline_loss = criterion(preds[mask>0], targets[mask>0]).item()
        
    print(f"Baseline MSE: {baseline_loss:.6f}")
    
    # 6. Permutation Importance
    importances = {}
    
    for i, col_name in enumerate(feature_cols):
        # Create perturbed input
        x_perm = x_orig.clone()
        
        # Shuffle the i-th feature across the batch dimension
        # We start with shape (B, T, D)
        # We want to shuffle index i across B (and keep T consistent or shuffle T too?)
        # Standard: shuffle across samples (B).
        idx = torch.randperm(x_orig.size(0))
        x_perm[:, :, i] = x_orig[idx, :, i]
        
        with torch.no_grad():
            preds_p = model(x_perm, t_grid, mask)
            loss_p = criterion(preds_p[mask>0], targets[mask>0]).item()
            
        # Importance = Increase in Error
        importances[col_name] = loss_p - baseline_loss

    # [NEW] Test Time Importance explicitly
    t_perm = t_grid.clone()
    # Shuffle time points? Or shuffle the entire time vector across batch?
    # In ODE, time is a shared grid. If we shuffle it, we break the physics.
    # Let's shuffle t_grid itself (randomize the observation times)
    # But t_grid is 1D (Length T).
    # Actually, the model input 't' is shared.
    # Let's perturb it by adding random noise or shuffling indices
    t_perm_idx = torch.randperm(len(t_grid))
    t_perm = t_grid[t_perm_idx]
    
    with torch.no_grad():
        # Note: altering t affects ODE solve points
        preds_t = model(x_orig, t_perm, mask) 
        # We must compare against targets at the NEW times? No, targets are fixed at original times.
        # This test checks: "If I ask for predictions at WRONG times, how bad is it?"
        # But wait, if we shuffle t, we are comparing Pred(t_random) vs Target(t_real). Of course it will be bad.
        # A fairer test: Shuffle the 'time' column in input DF if it were dynamic (but here it's a grid).
        # Let's skip 'Time' permutation for now because it's apples-to-oranges with features.
        # Instead, let's just print the raw MSE values to give context.
        pass

    # 7. Rank and Plot
    sorted_imps = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== Top 20 Important Features ===")
    for k, v in sorted_imps[:20]:
        print(f"{k}: +{v:.6f}")
        
    # Save CSV
    df_imp = pd.DataFrame(sorted_imps, columns=['Feature', 'Importance'])
    df_imp.to_csv(f"results/ode_{target}_{condition}_importance.csv", index=False)
    print(f"\nSaved importance list to results/ode_{target}_{condition}_importance.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='Chl_Per_Cell')
    parser.add_argument('--condition', type=str, default='Dark')
    args = parser.parse_args()
    
    analyze_importance(args.target, args.condition)
