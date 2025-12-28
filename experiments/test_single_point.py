import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from algae_fusion.models.latent_ode import LatentODE
from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch
from torch.utils.data import DataLoader

def main():
    print("=== Testing Single Point Inference ===")
    
    # 1. Config
    BATCH_SIZE = 1 # Single sample inference
    LATENT_DIM = 128
    TARGET_COL = 'Dry_Weight'
    MODEL_PATH = "experiments/latent_ode.pth"
    SCALER_PATH = "experiments/weights/latent_ode_scaler.joblib"
    
    # 2. Model
    # Load Scaler first to know input dim
    scaler = joblib.load(SCALER_PATH)
    feature_cols = list(scaler.feature_names_in_)
    input_dim = len(feature_cols)
    print(f"Loaded Scaler. Feature Count: {input_dim}")
    
    model = LatentODE(input_dim, LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Loaded Model.")
    
    # 3. Data (Validation Set)
    df_val = pd.read_csv("data/dataset_val.csv")
    
    # Create Meta Group
    df_val['meta_group'] = df_val['group_idx'].astype(str) + "_" + df_val['condition']
    
    # Scale Data
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])
    
    ds = AlgaeTimeSeriesDataset(df_val, feature_cols, TARGET_COL, group_col='meta_group')
    dl = DataLoader(ds, batch_size=1, collate_fn=collate_ode_batch, shuffle=True) # Shuffle to get random sample
    
    # 4. Inference Logic
    # We want to pick ONE sample, keep only ONE time point (e.g. t=24), and predict the REST.
    
    with torch.no_grad():
        # Get one sample details
        batch = next(iter(dl))
        x_full = batch['features'].to(DEVICE) # [1, T, D]
        t_full = batch['times'].to(DEVICE)    # [1, T] (0-72)
        
        # Select target observation time (e.g. closest to 24h)
        # Assuming sorted times. 24h is usually index 6 or so depending on resolution.
        # Let's pick a random index != 0 to make it hard.
        idx_obs = 3 # t=3 roughly. Or we can find where t=24.
        
        # Construct Sparse Input
        # We need to feed (x_obs, t_obs) to Encoder.
        x_obs = x_full[:, idx_obs:idx_obs+1, :] # [1, 1, D]
        t_obs = t_full[:, idx_obs:idx_obs+1]    # [1, 1]
        
        t_val = t_obs.item()
        print(f"Observing ONLY t = {t_val:.1f} hours.")
        
        # Scale Time for Model [0, 1]
        t_obs_scaled = t_obs / 72.0
        t_eval_scaled = torch.linspace(0, 1, 100).to(DEVICE) # Dense evaluation for smooth curve
        t_eval_real = t_eval_scaled * 72.0
        
        # Forward Pass
        # Encoder sees: 1 point.
        mu, logvar = model.encoder(x_obs, t_obs_scaled)
        z0 = mu # Deterministic inference
        
        # Decode
        # Integrate from z0 -> z(t)
        # Note: z0 is state at t=0. But we encoded from t=24.
        # Our ODERNN encoder (if trained with random starts) learns p(z0 | x_t).
        # Let's hope it generalized well!
        z_traj = model.ode_func(t_eval_scaled, z0) # wait, ode_func is func.
        
        from torchdiffeq import odeint
        z_traj = odeint(model.ode_func, z0, t_eval_scaled, rtol=1e-3, atol=1e-3)
        
        T, B, L = z_traj.shape
        x_pred = model.decoder(z_traj.view(T*B, L)).view(T, B, -1)
        
        # Inverse Scale for Visualization
        # Just first feature (usually a biomass proxy or similar)
        feat_idx = 0 
        feat_name = feature_cols[0]
        
        x_true_np = x_full[0, :, feat_idx].cpu().numpy()
        t_true_np = t_full[0].cpu().numpy()
        
        x_pred_np = x_pred[:, 0, feat_idx].cpu().numpy()
        t_pred_np = t_eval_real.cpu().numpy()
        
        # Inverse Transform specific value (approximate for just one col)
        # Actually easier to just plot scaled values to see fit shape.
        # Or inverse whole matrix.
        
        # Plot
        os.makedirs("results/ode_plots", exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(t_true_np, x_true_np, 'ko', label='True Data points')
        plt.plot(t_pred_np, x_pred_np, 'r-', label=f'Inferred from t={t_val}h')
        
        # Highlight the observation used
        x_obs_val = x_obs[0, 0, feat_idx].cpu().item()
        plt.plot(t_val, x_obs_val, 'bo', markersize=12, label='Observation')
        
        plt.title(f"Single Point Inference (Observed t={t_val:.1f}h) -> Full Trajectory")
        plt.xlabel("Time (h)")
        plt.ylabel(f"Feature: {feat_name} (Scaled)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = "results/ode_plots/single_point_inference.png"
        plt.savefig(out_path)
        print(f"Saved inference plot to {out_path}")

if __name__ == "__main__":
    main()
