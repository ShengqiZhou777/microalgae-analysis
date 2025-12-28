
import os
import sys
import torch
import torch.nn as nn
import joblib
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import argparse
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.getcwd())

from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.models.ode import GrowthODE
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch

def load_ode_config(target, condition):
    config_path = f"weights/ode_{target}_{condition}_config.json"
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

def test_ode_forecast(target, condition, cutoff_time=6):
    print(f"=== Testing Neural ODE Forecasting: {target} ({condition}) ===")
    print(f"Goal: Predict values after {cutoff_time}h using ONLY data <= {cutoff_time}h")
    
    # 1. Load Data
    df_test = pd.read_csv("data/dataset_test.csv")
    if condition != "All":
        df_test = df_test[df_test['condition'] == condition].reset_index(drop=True)
        
    # 2. Load Scaler
    scaler_path = f"weights/ode_{target}_{condition}_scaler.joblib"
    if not os.path.exists(scaler_path):
        print("Scaler not found.")
        return
    scaler = joblib.load(scaler_path)
    
    # Scale Target
    df_test[target] = scaler.transform(df_test[target].values)
    
    # 3. Create Dataset (Normal for Loading)
    feature_cols = [c for c in df_test.columns if c not in NON_FEATURE_COLS + [target]]
    group_col = 'group_idx' if 'group_idx' in df_test.columns else 'file'
    
    from algae_fusion.models.ode import GrowthODE
    ode_config = load_ode_config(target, condition)
    
    # ODEProjector Definition (Copied from pipeline)
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

    # 4. Load Model
    input_dim = len(feature_cols)
    latent_dim = ode_config["latent_dim"]
    ode_hidden_dim = ode_config["ode_hidden_dim"]
    ode_core = GrowthODE(input_dim=input_dim, latent_dim=latent_dim, ode_hidden_dim=ode_hidden_dim).to(DEVICE)
    model = ODEProjector(
        ode_core,
        latent_dim,
        ode_config["decoder_hidden"],
        ode_config["decoder_dropout"],
    ).to(DEVICE)
    
    model_path = f"weights/ode_{target}_{condition}.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # === Forecasting Logic ===
    ds = AlgaeTimeSeriesDataset(df_test, feature_cols, target, group_col=group_col)
    
    # We create a batch with ALL sequences
    dl = DataLoader(ds, batch_size=len(ds), collate_fn=collate_ode_batch, shuffle=False)
    
    batch = next(iter(dl))
    x = batch['features'].to(DEVICE)
    mask = batch['mask'].to(DEVICE)          # [B, T] Original mask (where data exists)
    targets = batch['targets'].to(DEVICE)
    times = batch['times'].to(DEVICE)        # [B, T] Actual time values
    
    # === Apply Forecasting Mask ===
    # We want to set mask=0 for any time > cutoff_time
    # This forces the model to ignore any features from the future and depend on ODE solving.
    forecast_mask = mask.clone()
    
    print(f"Original Mask Count (Total Points): {mask.sum().item()}")
    
    # Set future points to 0 in mask
    forecast_mask[times > cutoff_time] = 0
    
    print(f"Forecast Mask Count (Visible Points <= {cutoff_time}h): {forecast_mask.sum().item()}")
    
    # Run Model with Forecast Mask
    # Note: We still pass 'x', but since forecast_mask is 0, 'x' should be ignored by ODERNN logic locally.
    # We verified ODERNN: h = m * GRU(x, h) + (1-m) * h
    # So if m=0, GRU(x, h) is calculated but multiplied by 0. 
    # The state h is carried over (via ODE integration from previous step).
    
    time_scale = pd.to_numeric(df_test['time'], errors='coerce').max()
    time_scale = float(time_scale) if pd.notna(time_scale) and time_scale > 0 else 1.0
    t_grid = batch['times'][0].to(DEVICE) / time_scale
    
    with torch.no_grad():
        # Predict using partial history
        pred_y_forecast = model(x, t_grid, forecast_mask) # [B, T, 1]
    
    # Evaluate on the HIDDEN part (Future)
    # We want to check accuracy on times > cutoff_time
    # Original mask tells us where valid ground truth is.
    # Future mask is: (times > cutoff_time) & (original_mask == 1)
    
    future_indices = (times > cutoff_time) & (mask > 0)
    
    # Extract
    pred_future = pred_y_forecast.squeeze(-1)[future_indices].cpu().numpy()
    true_future = targets[future_indices].cpu().numpy()
    
    # Inverse Scale
    pred_inv = scaler.inverse_transform(pred_future.reshape(-1, 1)).flatten()
    true_inv = scaler.inverse_transform(true_future.reshape(-1, 1)).flatten()
    
    # Metrics
    if len(true_inv) > 0:
        r2 = r2_score(true_inv, pred_inv)
        mse = np.mean((true_inv - pred_inv)**2)
        print("\n=== Forecasting Results (Future Points Only) ===")
        print(f"Time Range: > {cutoff_time}h (e.g., 12h, 24h, 48h, 72h)")
        print(f"Number of Future Points Predicted: {len(true_inv)}")
        print(f"R2 Score: {r2:.4f}")
        print(f"MSE:      {mse:.8f}")
        
        # Plotting (Optional - first 5 sequences)
        # Choose a sample with long history
        sample_idx = 0
        t_seq = times[sample_idx].cpu().numpy()
        y_seq_true = targets[sample_idx].cpu().numpy()
        y_seq_pred = pred_y_forecast.squeeze(-1)[sample_idx].cpu().numpy()
        
        y_vis_true = scaler.inverse_transform(y_seq_true.reshape(-1,1)).flatten()
        y_vis_pred = scaler.inverse_transform(y_seq_pred.reshape(-1,1)).flatten()
        
        # Cutoff Line
        print("\n[Sample Trajectory Check]")
        print("Time | True      | Pred (Forecast)")
        for t_val, yt, yp in zip(t_seq, y_vis_true, y_vis_pred):
            marker = "*" if t_val > cutoff_time else " "
            print(f"{t_val:3.0f}h | {yt:.6f} | {yp:.6f} {marker}")
            
    else:
        print("No future points found to evaluate!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='Dry_Weight')
    parser.add_argument('--condition', type=str, default='Light')
    parser.add_argument('--cutoff', type=float, default=6.0)
    args = parser.parse_args()
    
    test_ode_forecast(args.target, args.condition, args.cutoff)
