
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from algae_fusion.config import DEVICE, BATCH_SIZE
from algae_fusion.models.boost import XGBoostExpert, LightGBMExpert
from algae_fusion.models.ode import GrowthODE, ODEProjector
from algae_fusion.data.dataset import ODETimeSeriesDataset, collate_ode_batch
from algae_fusion.data.processing import get_all_features

def run_boost_training(X_tab_train, X_tab_val, y_train_scaled, target_scaler, mode="full"):
    """
    Runs the boost model training (Layer 1 Stacking + Layer 2 Prediction).
    
    Args:
        X_tab_train (pd.DataFrame): Tabular features for training.
        X_tab_val (pd.DataFrame): Tabular features for validation.
        y_train_scaled (np.array): Scaled targets for training.
        target_scaler (object): Scaler to inverse transform predictions.
        mode (str): Training mode (full, boost_only, etc).
        
    Returns:
        results (dict): Dictionary containing:
            - 'val_preds_xgb': Predictions from XGBoost on validation set (unscaled).
            - 'val_preds_lgb': Predictions from LightGBM on validation set (unscaled).
            - 'models': Dictionary of trained models {'xgb1': ..., 'xgb2': ..., 'lgb2': ...}
            - 'X_train_aug': Augmented training features (with Layer 1 meta-features).
            - 'X_val_aug': Augmented validation features.
    """
    models = {}
    val_preds_xgb = np.zeros(len(X_tab_val))
    val_preds_lgb = np.zeros(len(X_tab_val))
    
    # --- XGBoost ---
    if mode in ["full", "boost_only", "xgb_only"]:
        # Standard XGBoost
        xgb = XGBoostExpert(n_estimators=800, learning_rate=0.05, max_depth=6, tree_method="hist")
        xgb.fit(X_tab_train, y_train_scaled)
        models['xgb'] = xgb
        
        p_scaled = xgb.predict(X_tab_val)
        val_preds_xgb = target_scaler.inverse_transform(p_scaled)
        
    # --- LightGBM ---
    if mode in ["full", "boost_only", "lgb_only"]:
        lgb = LightGBMExpert(n_estimators=800, learning_rate=0.05, num_leaves=31)
        lgb.fit(X_tab_train, y_train_scaled)
        models['lgb'] = lgb
        
        p_scaled = lgb.predict(X_tab_val)
        val_preds_lgb = target_scaler.inverse_transform(p_scaled)
        
    return {
        'val_preds_xgb': val_preds_xgb,
        'val_preds_lgb': val_preds_lgb,
        'models': models
    }
        
    return {
        'val_preds_xgb': val_preds_xgb,
        'val_preds_lgb': val_preds_lgb,
        'models': models,
        'X_train_aug': X_train_aug,
        'X_val_aug': X_val_aug
    }

def run_ode_training(df_train, df_val, df_test, target_name, condition, target_scaler, 
                     ode_window_size=None, use_future_masking=True):
    """
    Encapsulates the entire ODE training, evaluation, and visualization workflow.
    """
    
    # Feature Discovery
    feature_cols = get_all_features(df_train)

    
    y_train_scaled = target_scaler.transform(df_train[target_name].values)
    y_val_scaled = target_scaler.transform(df_val[target_name].values)
    y_test_scaled = target_scaler.transform(df_test[target_name].values)

    # Dataset Creation
    group_col = 'group_idx' if 'group_idx' in df_train.columns else 'file'
    
    # Pass 'scaled target array' directly
    if ode_window_size:
        print(f"  [ODE Engine] Using Sliding Window Mode (Size={ode_window_size})")
        ds_mode = 'window'
    else:
        print(f"  [ODE Engine] Using Full Trajectory Mode")
        ds_mode = 'trajectory'

    train_ds = ODETimeSeriesDataset(df_train, feature_cols, y_train_scaled, group_col=group_col, mode=ds_mode, window_size=ode_window_size)
    val_ds = ODETimeSeriesDataset(df_val, feature_cols, y_val_scaled, group_col=group_col, mode=ds_mode, window_size=ode_window_size)
    
    # Data Stats
    # Note: In ODETimeSeriesDataset, len(ds) is the number of samples.
    # In 'trajectory' mode, 1 sample = 1 full sequence.
    num_sequences = len(train_ds)
    # Estimate average length from the first few samples to avoid iterating all if big
    if num_sequences > 0:
        sample_lens = [len(train_ds[i]['times']) for i in range(min(num_sequences, 50))]
        avg_seq_len = float(np.mean(sample_lens))
    else:
        avg_seq_len = 0.0
    
    # Heuristic: Use small model if very few samples (e.g. < 50).
    # Ignore sequence length check for sliding window mode (where len < 10 is normal)
    if ode_window_size:
        small_data = (num_sequences < 50)
    else:
        small_data = (num_sequences <= 2 or avg_seq_len <= 10)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_ode_batch)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_ode_batch)
    
    print(f"  [ODE Engine] Training on {len(train_ds)} sequences, Validating on {len(val_ds)} sequences")
    
    # 2. Model Initialization
    input_dim = len(feature_cols)
    if small_data:
        print("  [ODE Warning] Very few time points detected; using smaller params.")
        latent_dim, ode_hidden, dec_hidden = 8, 32, 16
        epochs, patience, lr = 150, 30, 5e-4
        use_future_masking = False 
    else:
        latent_dim, ode_hidden, dec_hidden = 64, 128, 64
        epochs, patience, lr = 300, 100, 1e-3
        # use_future_masking stays as passed argument (True by default)

    ode_core = GrowthODE(input_dim=input_dim, latent_dim=latent_dim, ode_hidden_dim=ode_hidden).to(DEVICE)
    model = ODEProjector(ode_core, latent_dim, dec_hidden).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    criterion = nn.MSELoss(reduction='none') 
    
    # 3. Training Loop
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    time_scale = float(pd.to_numeric(df_train['time'], errors='coerce').max())
    
    print(f"  [ODE Engine] Starting Training for {epochs} epochs...")
    
    for ep in range(epochs):
        model.train()
        ep_loss = 0
        count = 0
        for batch in train_loader:
            x = batch['features'].to(DEVICE)
            y = batch['targets'].to(DEVICE)
            t_grid = batch['times'][0].to(DEVICE) / time_scale
            loss_mask = batch['mask'].to(DEVICE)
            input_mask = loss_mask.clone()
            
            # Scheduled Sampling
            if use_future_masking and np.random.rand() > 0.4 and x.shape[1] > 2:
                cut_idx = np.random.randint(1, x.shape[1] - 1)
                input_mask[:, cut_idx:] = 0
            
            optimizer.zero_grad()
            pred = model(x, t_grid, input_mask)
            loss = (criterion(pred, y) * loss_mask).sum() / loss_mask.sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
            count += 1
            
        # Validation
        model.eval()
        val_loss, val_count = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['features'].to(DEVICE)
                y = batch['targets'].to(DEVICE)
                t_grid = batch['times'][0].to(DEVICE) / time_scale
                mask = batch['mask'].to(DEVICE)
                pred = model(x, t_grid, mask)
                loss = (criterion(pred, y) * mask).sum() / mask.sum()
                val_loss += loss.item()
                val_count += 1
        
        avg_val = val_loss / max(1, val_count)
        scheduler.step(avg_val)
        
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  [Epoch {ep}] Train: {ep_loss/max(1,count):.4f} | Val: {avg_val:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
            
        if avg_val < best_loss:
            best_loss = avg_val
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  [ODE Engine] Early stopping at epoch {ep}")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        
    # 4. Save Logic
    os.makedirs("weights", exist_ok=True)
    model_prefix = f"weights/ode_{target_name}_{condition}"
    torch.save(model.state_dict(), f"{model_prefix}.pth")
    if target_scaler:
         joblib.dump(target_scaler, f"{model_prefix}_scaler.joblib")
         
    # 5. Visualization (Evaluation)
    _visualize_ode_results(model, df_test, feature_cols, y_test_scaled, group_col, time_scale, condition, target_scaler, original_target_name=target_name)

    print("  [ODE Engine] Run Complete.")


def _visualize_ode_results(model, df_test, feature_cols, target_input, group_col, time_scale, condition, target_scaler, original_target_name=None):
    """
    Internal helper to plot Growth Curve and Scatter.
    """
    test_ds = ODETimeSeriesDataset(df_test, feature_cols, target_input, group_col=group_col, mode='trajectory')
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False, collate_fn=collate_ode_batch)
    
    model.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        x = batch['features'].to(DEVICE)
        y_true_all = batch['targets'].cpu().numpy()
        t_grid = batch['times'][0].to(DEVICE) / time_scale
        mask = batch['mask'].to(DEVICE)
        
        # Predict
        y_pred_all = model(x, t_grid, mask).cpu().numpy()
        times = batch['times'][0].numpy()
        mask_np = mask.cpu().numpy()
        
        # --- Plot 1: Population Trajectory ---
        # Mean across samples
        y_true_pop = np.nanmean(np.where(mask_np, y_true_all, np.nan), axis=0)
        y_pred_mean = np.nanmean(np.where(mask_np, y_pred_all, np.nan), axis=0)
        y_pred_std = np.nanstd(np.where(mask_np, y_pred_all, np.nan), axis=0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, y_true_pop, 'o-', color='blue', label='Ground Truth (Mean)', linewidth=3)
        plt.plot(times, y_pred_mean, 's--', color='red', label='Prediction (Mean)', linewidth=2)
        plt.fill_between(times, y_pred_mean - y_pred_std, y_pred_mean + y_pred_std, color='red', alpha=0.2)
        
        t_name = original_target_name if original_target_name else "Target"
        plt.title(f'Neural ODE Trajectory: {condition} ({t_name})')
        plt.xlabel('Time (h)')
        plt.ylabel('Scaled Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs('results/ode_plots', exist_ok=True)
        plt.savefig(f'results/ode_plots/{t_name}_{condition}_trajectory.png', dpi=150)
        plt.close()
        
        # --- Plot 2: Scatter Test ---
        mask_flat = mask_np.flatten()
        y_true_flat = y_true_all.flatten()[mask_flat > 0]
        y_pred_flat = y_pred_all.flatten()[mask_flat > 0]
        
        if len(y_true_flat) > 0:
            if target_scaler:
                 y_true_flat = target_scaler.inverse_transform(y_true_flat)
                 y_pred_flat = target_scaler.inverse_transform(y_pred_flat)
                 
            plt.figure(figsize=(6, 6))
            plt.scatter(y_true_flat, y_pred_flat, alpha=0.5)
            
            # Diagonal
            mn, mx = min(y_true_flat.min(), y_pred_flat.min()), max(y_true_flat.max(), y_pred_flat.max())
            plt.plot([mn, mx], [mn, mx], 'r--')
            
            score = r2_score(y_true_flat, y_pred_flat)
            plt.title(f'Test Scatter: R2={score:.4f}')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.grid(True, alpha=0.3)
            plt.savefig(f'results/ode_plots/{t_name}_{condition}_scatter.png', dpi=150)
            plt.close()
