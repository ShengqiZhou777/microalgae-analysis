
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root
sys.path.append(os.getcwd())

from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch
from algae_fusion.models.latent_ode import LatentODE

def train_latent_ode():
    print("=== Training Autonomous Latent ODE (VAE) ===")
    
    # 1. Config
    TARGET_COL = 'Dry_Weight' # Just for dataset compatibility, we reconstruct FEATURES
    BATCH_SIZE = 32 # 400 reduces updates too much. 32 strikes balance.
    LATENT_DIM = 128 # 64 was too small for 91 features. 128 gives more "memory".
    EPOCHS = 500 # More epochs needed if batch size is small, but also if learning is hard
    LR = 3e-3 # Slightly more aggressive
    
    # 2. Data
    df_train = pd.read_csv("data/dataset_train.csv") # Using test for demo/dev as it's cleaner
    # Create unique group ID to separate Light/Dark duplicates
    df_train['meta_group'] = df_train['group_idx'].astype(str) + "_" + df_train['condition']
    
    feature_cols = [c for c in df_train.columns if c not in NON_FEATURE_COLS + [TARGET_COL, 'meta_group']]
    input_dim = len(feature_cols)
    print(f"Features: {input_dim}, Latent: {LATENT_DIM}")
    
    # === Normalization (CRITICAL for VAE/ODE) ===
    from sklearn.preprocessing import StandardScaler
    import joblib
    
    scaler = StandardScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    
    # Save scaler for inference
    os.makedirs("experiments/weights", exist_ok=True)
    joblib.dump(scaler, "experiments/weights/latent_ode_scaler.joblib")
    print("Scaler saved to experiments/weights/latent_ode_scaler.joblib")
    # ============================================

    print("\n[Data] Loading Training Set...")
    ds = AlgaeTimeSeriesDataset(df_train, feature_cols, TARGET_COL, group_col='meta_group')
    dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_ode_batch, shuffle=True, num_workers=4, pin_memory=True)
    
    # Validation Data
    print("\n[Data] Loading Validation Set...")
    df_test = pd.read_csv("data/dataset_val.csv")
    df_test['meta_group'] = df_test['group_idx'].astype(str) + "_" + df_test['condition']
    # Use SAME scaler
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])
    ds_val = AlgaeTimeSeriesDataset(df_test, feature_cols, TARGET_COL, group_col='meta_group')
    dl_val = DataLoader(ds_val, batch_size=BATCH_SIZE, collate_fn=collate_ode_batch, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. Model
    model = LatentODE(input_dim, LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # 4. Training Loop
    os.makedirs("experiments/plots", exist_ok=True)
    
    # Pre-fetch fixed sample for visualization (so we track SAME cell)
    viz_dl = DataLoader(ds, batch_size=1, collate_fn=collate_ode_batch, shuffle=False)
    viz_batch = next(iter(viz_dl))
    viz_x = viz_batch['features'].to(DEVICE)
    viz_t = viz_batch['times'].to(DEVICE)
    
    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        total_mse = 0
        total_kl = 0
        
        for batch in dl:
            x_seq = batch['features'].to(DEVICE) # [B, T, D]
            t_seq = batch['times'].to(DEVICE) / 72.0 # Scale Time to [0, 1] for stable ODE
            
            optimizer.zero_grad()
            
            # Handling t_eval for odeint (Must be strictly increasing/unique)
            t_unique, inverse_indices = torch.unique(t_seq[0], sorted=True, return_inverse=True)
            t_eval = t_unique
             
            # Forward
            x_pred_unique, mu, logvar = model(x_seq, t_seq, t_eval)
            x_pred = x_pred_unique[inverse_indices] # [T_full, B, D]
            
            # Loss: ELBO
            # Loss: ELBO
            x_pred = x_pred.permute(1, 0, 2) # [B, T, D]
            
            recon_loss = torch.mean((x_pred - x_seq)**2)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl_loss / (BATCH_SIZE * input_dim) 
            
            # KL Annealing
            beta = 0.0 if epoch < 100 else 1e-5
            loss = recon_loss + beta * kl_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += recon_loss.item()
            total_kl += kl_loss.item()
            
        train_loss = total_loss / len(dl)
        
        # --- VALIDATION ---
        model.eval()
        val_loss = 0
        
        # Split Metrics
        mse_light = 0.0
        mse_dark = 0.0
        count_light = 0
        count_dark = 0
        
        with torch.no_grad():
            for batch in dl_val:
                x_seq = batch['features'].to(DEVICE)
                t_seq = batch['times'].to(DEVICE) / 72.0 # Scale Time also in Val!
                conds = batch['conditions'].to(DEVICE) # 1=Light, 0=Dark
                
                t_unique, inverse_indices = torch.unique(t_seq[0], sorted=True, return_inverse=True)
                x_pred_unique, mu, logvar = model(x_seq, t_seq, t_unique)
                x_pred = x_pred_unique[inverse_indices].permute(1, 0, 2)
                
                # mse_batch: [B]
                mse_batch = torch.mean((x_pred - x_seq)**2, dim=(1, 2))
                
                # Accumulate
                is_light = (conds == 1.0)
                is_dark = (conds == 0.0)
                
                if is_light.any():
                    mse_light += mse_batch[is_light].sum().item()
                    count_light += is_light.sum().item()
                
                if is_dark.any():
                    mse_dark += mse_batch[is_dark].sum().item()
                    count_dark += is_dark.sum().item()
                    
                val_loss += mse_batch.mean().item() # Approx mean
                
        # Final calculations
        val_mse_light = mse_light / max(count_light, 1)
        val_mse_dark = mse_dark / max(count_dark, 1)
        val_mse_avg = (val_mse_light + val_mse_dark) / 2 # Balanced avg
        
        # Step Scheduler
        scheduler.step(val_mse_avg)
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val L: {val_mse_light:.4f} | Val D: {val_mse_dark:.4f} | KL Beta: {beta:.1e} | LR: {current_lr:.1e}")
            
            # Visualize FIXED sample
            with torch.no_grad():
                # Scale viz time too
                viz_t_scaled = viz_t.clone() / 72.0
                
                # Get unique times for viz batch
                t_unique_viz, inverse_viz = torch.unique(viz_t_scaled[0], sorted=True, return_inverse=True)
                
                # Forward with fixed sample (use scaled time)
                x_pred_viz, _, _ = model(viz_x, viz_t_scaled, t_unique_viz)
                
                # Map back
                x_pred_viz = x_pred_viz[inverse_viz].permute(1, 0, 2)
                
                t_plot = viz_t[0].cpu().numpy() # Plot with original time (0-72) for readability
                x_true = viz_x[0, :, 0].cpu().numpy() # First feature
                x_est = x_pred_viz[0, :, 0].cpu().numpy()
                
                plt.figure(figsize=(6, 4))
                plt.plot(t_plot, x_true, 'k-', label='True')
                plt.plot(t_plot, x_est, 'r--', label='Latent ODE')
                plt.title(f"Val L: {val_mse_light:.4f} | D: {val_mse_dark:.4f}")
                plt.legend()
                plt.savefig(f"experiments/plots/epoch_{epoch:03d}.png")
                plt.close()

    # 5. Save
    torch.save(model.state_dict(), "experiments/latent_ode.pth")
    print("Model saved to experiments/latent_ode.pth")

if __name__ == "__main__":
    train_latent_ode()
