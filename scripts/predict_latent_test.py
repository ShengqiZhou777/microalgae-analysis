import argparse
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import os
import sys

# Ensure project root is in path
sys.path.append(os.getcwd())

from algae_fusion.models.latent_ode import LatentODE
from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Dry_Weight")
    args = parser.parse_args()

    print("=== Testing Optimzed Latent ODE ===")
    
    # 1. Config
    LATENT_DIM = 128
    MODEL_PATH = "experiments/latent_ode.pth"
    SCALER_PATH = "experiments/weights/latent_ode_scaler.joblib"
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    # 2. Model & Data
    scaler = joblib.load(SCALER_PATH)
    feature_cols = list(scaler.feature_names_in_)
    input_dim = len(feature_cols)
    print(f"Loaded Scaler. Input Dim: {input_dim}")
    
    model = LatentODE(input_dim, LATENT_DIM).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Loaded Optimized Latent ODE Model.")
    
    # Load Test Data
    df_test = pd.read_csv("data/dataset_test.csv")
    
    # Filter for Target Condition if needed (Current latent_ode trained on ALL, but we can visualize split)
    # The user was asking about Light/Dark, let's keep all and split in existing
    pass 
    
    # Create Meta Group for robust grouping
    if "group_idx" not in df_test.columns:
        df_test['group_idx'] = df_test['file'] # Fallback
        
    df_test['meta_group'] = df_test['group_idx'].astype(str) + "_" + df_test['condition']
    
    # Scale Data
    df_scaled = df_test.copy()
    df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])
    
    ds = AlgaeTimeSeriesDataset(df_scaled, feature_cols, args.target, group_col='meta_group')
    dl = DataLoader(ds, batch_size=32, collate_fn=collate_ode_batch, shuffle=False)
    
    # 3. Inference
    all_preds = []
    all_trues = []
    all_times = []
    all_groups = []
    all_conds = []
    
    print("Running Inference...")
    with torch.no_grad():
        for batch in dl:
            x_seq = batch['features'].to(DEVICE) # [B, T, D]
            t_seq = batch['times'].to(DEVICE)    # [B, T]
            
            # Important: Apply the same Time Scaling as Training!
            t_seq_scaled = t_seq / 72.0
            
            # Forward Pass: Encode -> z0 -> Decode Trajectory
            # Note: We use the full sequence to infer z0 (Hindsight Inference)
            # This tests "Reconstruction" capability on unseen data.
            t_max_batch = t_seq_scaled.max(dim=1)[0] # Per sample max time
            
            # Since lengths vary, we need to be careful.
            # But LatentODE handles t_seq.
            # However, LatentODE.forward returns (x_pred, mu, logvar)
            # And it expects t_eval (unique times)
            
            # Simplified approach: Pass (x, t) to encoder, get z0.
            # Then decode at exact t sequence.
            
            mu, logvar = model.encoder(x_seq, t_seq_scaled)
            z0 = mu # Deterministic z for testing
            
            # Decode
            # We must decode individually per sample if times differ, 
            # Or use the unique time trick if they align.
            # Here, let's just loop for safety/correctness
            
            B = x_seq.size(0)
            for i in range(B):
                t_sample = t_seq_scaled[i]
                # Remove padding (t < 0 or duplicate large padding?)
                # Actually pipeline pads with 0, but dataset defines sequence length.
                # Let's rely on Valid Length logic if we had it.
                # For now, just decode all timepoints.
                
                # Decode
                z_traj = model.ode_func(t_sample, z0[i]) # This is linear approx? No, ode_func is derivative
                # Use solver
                from torchdiffeq import odeint
                # Sort times for solver (required)
                t_sorted, inv_idx = torch.sort(t_sample)
                z_traj_sorted = odeint(model.ode_func, z0[i], t_sorted, rtol=1e-3, atol=1e-3)
                
                # Unsort
                z_traj = torch.zeros_like(z_traj_sorted)
                z_traj[inv_idx] = z_traj_sorted
                
                # Project
                x_rec = model.decoder(z_traj) # [T, D]
                
                # Collect Result (Feature 0 is usually target proxy, but herein we reconstructed features)
                # Wait, LatentODE reconstructs FEATURES (91 dim).
                # We need to map back to original space.
                
                x_rec_np = x_rec.cpu().numpy()
                x_true_np = x_seq[i].cpu().numpy()
                
                all_preds.append(x_rec_np)
                all_trues.append(x_true_np)
                all_times.append(t_seq[i].cpu().numpy())
                
                # Meta
                # Need to map batch index back to dataframe
                # This is tricky with DataLoader batching.
                # Let's use simple list logic if batch size matches.
                pass

    # Reassemble seems hard with batching.
    # Let's iterate dataset directly without DataLoader for 1-to-1 mapping
    pass

    # --- SIMPLIFIED INFERENCE LOOP ---
    # Re-do purely iterating groups to keep metadata synced
    
    unique_groups = df_test['meta_group'].unique()
    results = []
    
    print(f"Processing {len(unique_groups)} groups...")
    
    with torch.no_grad():
        for gid in unique_groups:
            sub = df_scaled[df_scaled['meta_group'] == gid].sort_values('time')
            
            x_in = torch.tensor(sub[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(DEVICE) # [1, T, D]
            t_in = torch.tensor(sub['time'].values, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            
            t_in_scaled = t_in / 72.0
            
            # Encode
            mu, _ = model.encoder(x_in, t_in_scaled)
            z0 = mu[0]
            
            # Decode
            t_eval = t_in_scaled[0]
            from torchdiffeq import odeint
            # Solver needs sorted time
            t_sorted, inv_idx = torch.sort(t_eval)
            z_traj_sorted = odeint(model.ode_func, z0, t_sorted, rtol=1e-3, atol=1e-3)
            
            # Project
            x_rec_sorted = model.decoder(z_traj_sorted)
            
            # Reorder
            x_rec = torch.zeros_like(x_rec_sorted)
            x_rec[inv_idx] = x_rec_sorted
            
            x_rec_np = x_rec.cpu().numpy()
            
            # INVERSE TRANSFORM
            # We predict 91 features. We need to save them.
            # But user cares about Target (Dry_Weight).
            # Dry_Weight IS one of the features? 
            # Check feature list. Usually Target is part of features in Autoencoder if we want to predict it?
            # Or is it separate? 
            # Based on dataset.py: feature_cols = [c for c in df.columns if ...]
            # If target is in df, it might be in feature_cols.
            
            if args.target in feature_cols:
                feat_idx = feature_cols.index(args.target)
                target_pred_scaled = x_rec_np[:, feat_idx]
                
                # Inverse Scale ONLY this column is hard.
                # Construct full matrix dummy
                dummy = np.zeros((len(t_eval), len(feature_cols)))
                dummy[:, feat_idx] = target_pred_scaled
                
                # This is risky if scaler centers data (it does).
                # Correct way: Inverse transform the WHOLE prediction matrix.
                x_rec_inv = scaler.inverse_transform(x_rec_np)
                target_pred = x_rec_inv[:, feat_idx]
                
            else:
                # If target is NOT in features (e.g. external label), LatentODE (Reconstruction) cannot predict it directly!
                # It only reconstructs INPUT features.
                # BUT, Dry_Weight IS usually a feature in our dataset, unless explicitly excluded.
                # Let's assume it IS in features.
                # If not, we have a problem: LatentODE is unsupervised.
                pass
                
                # Fallback check
                if args.target not in feature_cols:
                    # Try to map latent z to target? (Not trained)
                    # We assume target is in input for reconstruction task.
                    print(f"Warning: {args.target} not in feature columns. Cannot predict.")
                    continue
                else:
                     feat_idx = feature_cols.index(args.target)
                     x_rec_inv = scaler.inverse_transform(x_rec_np)
                     target_pred = x_rec_inv[:, feat_idx]

            # Store
            for i in range(len(sub)):
                results.append({
                    'meta_group': gid,
                    'time': sub.iloc[i]['time'],
                    'condition': sub.iloc[i]['condition'],
                    'True_Val': sub.iloc[i][args.target], # Original Unscaled
                    'Pred_Val': target_pred[i]
                })

    res_df = pd.DataFrame(results)
    
    # --- SCATTER PLOT ---
    print("Generating Plots...")
    os.makedirs("results/ode_plots_latent", exist_ok=True)
    
    r2 = r2_score(res_df['True_Val'], res_df['Pred_Val'])
    rmse = np.sqrt(mean_squared_error(res_df['True_Val'], res_df['Pred_Val']))
    
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=res_df, x='True_Val', y='Pred_Val', hue='condition', alpha=0.6)
    
    # Diagonal
    min_val = min(res_df['True_Val'].min(), res_df['Pred_Val'].min())
    max_val = max(res_df['True_Val'].max(), res_df['Pred_Val'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel(f"True {args.target}")
    plt.ylabel(f"Predicted {args.target} (Reconstructed)")
    plt.title(f"Optimized Latent ODE Reconstruction\nR2={r2:.3f}, RMSE={rmse:.3f}")
    
    out_path = f"results/ode_plots_latent/scatter_{args.target}.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved scatter plot to {out_path}")
    
    # --- TRAJECTORIES (Sample) ---
    plt.figure(figsize=(10, 6))
    conds = res_df['condition'].unique()
    colors = {'Light': 'orange', 'Dark': 'blue'}
    
    # Plot top 5 groups for each condition
    for cond in conds:
        groups = res_df[res_df['condition'] == cond]['meta_group'].unique()[:5]
        for gid in groups:
            sub = res_df[res_df['meta_group'] == gid].sort_values('time')
            plt.plot(sub['time'], sub['True_Val'], 'o', color=colors.get(cond, 'gray'), alpha=0.3)
            plt.plot(sub['time'], sub['Pred_Val'], '-', color=colors.get(cond, 'gray'), alpha=0.7)
            
    plt.title("Sample Trajectories (Lines=Pred, Dots=True)")
    plt.xlabel("Time (h)")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"results/ode_plots_latent/traj_{args.target}.png", dpi=150)

if __name__ == "__main__":
    main()
