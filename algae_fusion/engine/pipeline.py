import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GroupKFold, LeaveOneOut, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from torch.utils.data import DataLoader
from torchvision import transforms
import joblib
import json
from time import time

from algae_fusion.config import *
from algae_fusion.data.dataset import MaskedImageDataset
from algae_fusion.models.backbones import ResNetRegressor
from algae_fusion.models.moe import GatingNetwork
from algae_fusion.features.kinetic import compute_kinetic_features
from algae_fusion.features.sliding_window import compute_sliding_window_features
from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
from algae_fusion.engine.trainer import train_epoch, eval_epoch
from algae_fusion.models.lstm import MorphLSTM

from algae_fusion import omics_loader

class Log1pScaler:
    def __init__(self, s): self.s = s
    def inverse_transform(self, yp): return np.expm1(self.s.inverse_transform(yp.reshape(-1, 1)).flatten())

class StandardWrapper:
    def __init__(self, s): self.s = s
    def inverse_transform(self, yp): return self.s.inverse_transform(yp.reshape(-1, 1)).flatten()

def prepare_lstm_tensor(df, morph_cols, scaler=None):
    """
    stacks [Prev2, Prev1, Curr] into (N, 3, F) tensor
    If scaler is provided, applies transform to each time step.
    """
    # Verify cols exist
    p1 = [f"Prev1_{c}" for c in morph_cols]
    p2 = [f"Prev2_{c}" for c in morph_cols]
    
    # Check if Prev cols exist (Dynamic Mode)
    if not all(c in df.columns for c in p1):
        return None
        
    # Stack: T-2, T-1, T
    # Fill NaN with 0 or handle before
    data_t2 = df[p2].fillna(0).values
    data_t1 = df[p1].fillna(0).values
    data_t0 = df[morph_cols].fillna(0).values
    
    if scaler:
        # Transform each step using the same scaler (trained on T0)
        # Note: if fillna(0) was used, scaling 0 might shift it.
        # But 0 usually implies mean if centered? No, 0 implies missing. 
        # Ideally we impute means. But here we assume data is clean after sliding window.
        data_t2 = scaler.transform(data_t2)
        data_t1 = scaler.transform(data_t1)
        data_t0 = scaler.transform(data_t0)
    
    # Stack along new axis 1 -> (N, 3, F)
    x_seq = np.stack([data_t2, data_t1, data_t0], axis=1)
    return torch.FloatTensor(x_seq)
    
    # Stack along new axis 1 -> (N, 3, F)
    x_seq = np.stack([data_t2, data_t1, data_t0], axis=1)
    return torch.FloatTensor(x_seq)


def run_pipeline(target_name="Dry_Weight", mode="full", hidden_times=None, stochastic_window=False, condition=None, window_size=None):
    """
    Orchestrate the full training pipeline.
    
    Args:
        target_name: The regression target column.
        mode: "full" (all features), "cnn_only", or "boost_only".
    Args:
        target_name: The regression target column.
        mode: "full" (all features), "cnn_only", or "boost_only".
        hidden_times: List of timepoints to exclude from training (LOO CV).
        stochastic_window: If True, uses randomized history matching.
        condition: If None, train on both. If "Light" or "Dark", train specific.
        window_size: Sliding window size (history length). Defaults to config value.
    """
    # [CONFIG] Window Size for History
    if window_size is None:
        window_size = WINDOW_SIZE
    
    # Window=3 means we look back at t-3, t-2, t-1 (plus current t)
    # Total Channels = (3+1) * 3 = 12
    # in_channels calculation uses window_size
    
    in_channels = (window_size + 1) * 3 if stochastic_window else 3
    print(f"\n\n{'='*40}")
    status = "DYNAMIC (History)" if stochastic_window else "STATIC (No History)"
    print(f"STARTING PIPELINE: Target={target_name}, Mode={mode} | {status} (Win={window_size}, Ch={in_channels})")
    print(f"{'='*40}\n")

    # --- [DETERMINISTIC DATA POOLING] ---
    train_df = pd.read_csv("data/dataset_train.csv")
    test_df = pd.read_csv("data/dataset_test.csv") if os.path.exists("data/dataset_test.csv") else pd.DataFrame()
    
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # --- [FILTER CONDITION] ---
    if condition is not None and condition != "All":
        print(f"  [Filter] Selecting only '{condition}' samples (plus Initial 0h if available)...")
        df = df[df['condition'] == condition].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No samples found for condition '{condition}'!")
    
    # 1. Sort by condition/time/file for stable group_idx
    df = df.sort_values(by=['condition', 'time', 'file']).reset_index(drop=True)
    df['group_idx'] = df.groupby(['condition', 'time']).cumcount()
    
    # 2. Assign deterministic split based on threshold (80/20)
    # We apply the split per (condition, time) group
    def assign_split(group):
        N = len(group)
        test_thresh = int(N * 0.8)
        val_thresh = int(test_thresh * 0.8)
        
        splits = ['TRAIN'] * N
        for i in range(val_thresh, test_thresh):
            splits[i] = 'VAL'
        for i in range(test_thresh, N):
            splits[i] = 'TEST'
        group['split_set'] = splits
        return group

    df = df.groupby(['condition', 'time'], group_keys=False).apply(assign_split)
    
    df_hidden = pd.DataFrame()
    if hidden_times:
        mask = df['time'].isin(hidden_times)
        df_hidden = df[mask].copy().reset_index(drop=True)
        df = df[~mask].copy().reset_index(drop=True)

    df['Source_Path'] = df['Source_Path'].astype(str)
    
    # --- [DYNAMIC FEATURE SELECTION] ---
    # Automatically identify all morphological columns (starting with cell_)
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    morph_cols = [c for c in all_numeric_cols if c.startswith('cell_') and c not in NON_FEATURE_COLS]
    print(f"  [Info] Identified {len(morph_cols)} morphological features.")

    # --- [TRANSFORMS] ---
    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # --- SINGLE PASS TRAINING SETUP ---
    train_pool_mask = df['split_set'].isin(['TRAIN', 'VAL'])
    df_pool = df[train_pool_mask].reset_index(drop=True)
    
    tr_indices = df_pool.index[df_pool['split_set'] == 'TRAIN'].tolist()
    va_indices = df_pool.index[df_pool['split_set'] == 'VAL'].tolist()
    
    print(f"  [Split] Train: {len(tr_indices)}, Val: {len(va_indices)}")

    df_train_fold, df_val_fold = df_pool.iloc[tr_indices].copy(), df_pool.iloc[va_indices].copy()

    # [NEW Logic] Isolate Feature Engineering
    if stochastic_window:
        print(f"  [Fold] Computing History (Win={window_size}) for TRAIN (Dense)...")
        df_train_fold = compute_sliding_window_features_stochastic(df_train_fold, window_size=window_size, morph_cols=morph_cols)
        
        print(f"  [Fold] Computing History (Win={window_size}) for VAL (Sparse - Simulating Test)...")
        df_val_fold = compute_sliding_window_features_stochastic(df_val_fold, window_size=window_size, morph_cols=morph_cols)
    else:
        print("  [Static Mode] No History computed.")
    
    df_test = df[df['split_set'] == 'TEST'].copy()
    df_train = df_pool.copy()
    
    # Filter Condition if needed
    if condition:
        print(f"  [Filter] Using only {condition} samples.")
        df_train = df_train[df_train['condition'] == condition]
        df_test = df_test[df_test['condition'] == condition]

    # 3. K-Fold CV on Training Set
    # We use GroupKFold on 'group_id' to prevent leakage
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # Ideally should be GroupKFold but for simplicity KFold on mixed groups
    
    # Placeholder for results
    fold_metrics = []
    
    # Feature Columns (exclude non-feature ones)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and "Prev" not in c]
    # 'Prev' features are generated dynamically if stochastic
    
    # Identify Morphology Columns for History Gen
    morph_cols = [c for c in feature_cols if "intensity" in c or "area" in c or "perimeter" in c or "eccentricity" in c]

    for fold, (train_idx, val_idx) in enumerate(kf.split(df_train)):
        # print(f"\n[Fold {fold+1}/5]")
        
        df_train_fold = df_train.iloc[train_idx].copy()
        df_val_fold   = df_train.iloc[val_idx].copy()
        
        # 4. Stochastic Sliding Window (Dynamic Features)
        if stochastic_window:
            # print(f"  [Fold] Computing History (Win={window_size}) for TRAIN...")
            df_train_fold = compute_sliding_window_features_stochastic(df_train_fold, window_size=window_size, morph_cols=morph_cols)
            # print(f"  [Fold] Computing History (Win={window_size}) for VAL...")
            df_val_fold   = compute_sliding_window_features_stochastic(df_val_fold, window_size=window_size, morph_cols=morph_cols)
    
    tab_cols = [c for c in df_train_fold.columns if c not in NON_FEATURE_COLS]
    X_tab_train = df_train_fold[tab_cols].select_dtypes(exclude=['object'])
    X_tab_val = df_val_fold[tab_cols].select_dtypes(exclude=['object'])
    common_cols = X_tab_train.columns.intersection(X_tab_val.columns)
    X_tab_train, X_tab_val = X_tab_train[common_cols], X_tab_val[common_cols]
    
    y_train_orig, y_val_orig = df_train_fold[target_name].values, df_val_fold[target_name].values
    
    # Target Transform (Simplified logic for modular version)
    y_train_min, y_train_max = y_train_orig.min(), y_train_orig.max()
    ratio = y_train_max / (y_train_min + 1e-9) if y_train_min >= 0 else 0
    
    pt_fold = None
    if y_train_min >= 0 and ratio > 10:
        y_train_transformed = np.log1p(y_train_orig)
        y_val_transformed = np.log1p(y_val_orig)
        scaler_target = StandardScaler()
        y_train_scaled = scaler_target.fit_transform(y_train_transformed.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_target.transform(y_val_transformed.reshape(-1, 1)).flatten()
        pt_fold = Log1pScaler(scaler_target)
    else:
        scaler_target = StandardScaler()
        y_train_scaled = scaler_target.fit_transform(y_train_orig.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_target.transform(y_val_orig.reshape(-1, 1)).flatten()
        pt_fold = StandardWrapper(scaler_target)

    # Layer 1: XGB1
    if mode in ["full", "boost_only"]:
        xgb1 = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, tree_method="hist")
        xgb1.fit(X_tab_train, y_train_scaled)
        X_train_aug = X_tab_train.copy()
        X_val_aug = X_tab_val.copy()
        X_train_aug["XGB1_Feature"] = xgb1.predict(X_tab_train)
        X_val_aug["XGB1_Feature"] = xgb1.predict(X_tab_val) # Corrected from LGB1_Feature
    else:
        # In cnn_only/xgb_only, we still need Aug features for gating if we merge
        X_train_aug, X_val_aug = X_tab_train, X_tab_val # Initialize if not done by XGB1
        pass

    # Layer 1.5: LSTM (Optional Dynamic Expert)
    # [REVERTED] User found LSTM degrades performance compared to pure Boosting.
    # We skip LSTM training.
    val_preds_lstm = None
    if False and stochastic_window and mode in ["full", "boost_only"]:
        print("\n  [Training] LSTM Expert (Morphological Sequence)...")

        
        # Save Scaler
        cond_str = condition if condition else "All"
        os.makedirs("weights", exist_ok=True)
        joblib.dump(lstm_scaler, os.path.join("weights", f"{target_name}_{cond_str}_lstm_scaler.joblib"))
        
        # Prepare Tensors with Scaling
        X_seq_train = prepare_lstm_tensor(df_train_fold, morph_cols, scaler=lstm_scaler) 
        X_seq_val = prepare_lstm_tensor(df_val_fold, morph_cols, scaler=lstm_scaler)
        
        if X_seq_train is not None and X_seq_val is not None:
            # Simple Dataset/Loader
            from torch.utils.data import TensorDataset
            lstm_ds_train = TensorDataset(X_seq_train, torch.FloatTensor(y_train_scaled).squeeze())
            lstm_ds_val = TensorDataset(X_seq_val, torch.FloatTensor(y_val_scaled).squeeze())
            
            lstm_loader_tr = DataLoader(lstm_ds_train, batch_size=BATCH_SIZE, shuffle=True)
            lstm_loader_val = DataLoader(lstm_ds_val, batch_size=BATCH_SIZE, shuffle=False)
            
            lstm_model = MorphLSTM(input_dim=len(morph_cols)).to(DEVICE)
            lstm_opt = optim.Adam(lstm_model.parameters(), lr=1e-3)
            lstm_crit = nn.MSELoss()
            
            # Fast Train Loop (20 Epochs)
            best_lstm_loss = float('inf')
            best_lstm_state = None
            for ep in range(20):
                lstm_model.train()
                ep_loss = 0
                for xb, yb in lstm_loader_tr:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    lstm_opt.zero_grad()
                    pred = lstm_model(xb)
                    loss = lstm_crit(pred, yb)
                    loss.backward()
                    lstm_opt.step()
                    ep_loss += loss.item()
                
                # Val
                lstm_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for xb, yb in lstm_loader_val:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        p = lstm_model(xb)
                        val_loss += lstm_crit(p, yb).item()
                
                if val_loss < best_lstm_loss:
                    best_lstm_loss = val_loss
                    best_lstm_state = lstm_model.state_dict()
            
            # Load Best & Predict
            if best_lstm_state:
                lstm_model.load_state_dict(best_lstm_state)
            lstm_model.eval()
            
            # Save Model
            cond_str = condition if condition else "All"
            os.makedirs("weights", exist_ok=True)
            torch.save(lstm_model.state_dict(), os.path.join("weights", f"{target_name}_{cond_str}_lstm.pth"))
            
            # Generate Val Preds for Gating
            preds = []
            with torch.no_grad():
                for xb, _ in lstm_loader_val:
                     xb = xb.to(DEVICE)
                     preds.append(lstm_model(xb).cpu().numpy())
            p_scaled = np.concatenate(preds)
            val_preds_lstm = pt_fold.inverse_transform(p_scaled)
            
            print(f"  [LSTM] Best Val Loss: {best_lstm_loss:.4f}")
            
            # [PARALLEL MODE] We do NOT add LSTM features to X_train_aug.
            # Experts (XGB/LGB) operate independently of LSTM.
            # Gating Network will fuse them.

    # Layer 2: XGB2/LGB2
    val_preds_xgb = np.zeros(len(va_indices))
    if mode in ["full", "xgb_only", "boost_only"]:
        xgb2 = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, tree_method="hist")
        xgb2.fit(X_train_aug, y_train_scaled)
        # Prediction needs inverse transform
        p_scaled = xgb2.predict(X_val_aug)
        val_preds_xgb = pt_fold.inverse_transform(p_scaled)
    
    val_preds_lgb = np.zeros(len(va_indices))
    if mode in ["full", "lgb_only", "boost_only"]:
        lgb2 = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31, verbose=-1)
        lgb2.fit(X_train_aug, y_train_scaled)
        p_scaled = lgb2.predict(X_val_aug)
        val_preds_lgb = pt_fold.inverse_transform(p_scaled)

    # Layer 2: CNN
    val_preds_cnn = np.zeros(len(va_indices))
    if mode in ["full", "cnn_only"]:
        in_channels = (window_size + 1) * 3 if stochastic_window else 3
        train_ds = MaskedImageDataset(df_train_fold, target_name, IMG_SIZE, train_transform, labels=y_train_scaled, in_channels=in_channels)
        val_ds = MaskedImageDataset(df_val_fold, target_name, IMG_SIZE, val_transform, labels=y_val_scaled, in_channels=in_channels)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
        
        cnn = ResNetRegressor(BACKBONE, in_channels=in_channels).to(DEVICE)
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-2)
        
        best_loss = float('inf')
        best_state = None
        patience, no_improve = 5, 0
        for ep in range(EPOCHS):
            tr_loss = train_epoch(cnn, train_loader, criterion, optimizer)
            val_loss, _ = eval_epoch(cnn, val_loader, criterion, max_batches=MAX_VAL_BATCHES)
            print(f"  [Epoch {ep+1}/{EPOCHS}] Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss, best_state, no_improve = val_loss, cnn.state_dict(), 0
            else:
                no_improve += 1
                if no_improve >= patience: break
        
        if best_state: cnn.load_state_dict(best_state)
        _, p_scaled = eval_epoch(cnn, val_loader, criterion)
        val_preds_cnn = pt_fold.inverse_transform(p_scaled)

    # Hidden Set Prediction (Restored)
    if not df_hidden.empty:
        folds_run += 1
        X_tab_hidden = df_hidden[tab_cols].select_dtypes(exclude=['object'])
        for c in common_cols:
            if c not in X_tab_hidden.columns: X_tab_hidden[c] = 0
        X_tab_hidden = X_tab_hidden[common_cols]
        
        if mode in ["full", "boost_only"]:
            X_hidden_aug = X_tab_hidden.copy()
            X_hidden_aug["XGB1_Feature"] = xgb1.predict(X_tab_hidden)
        else:
            X_hidden_aug = X_tab_hidden
            
        if mode in ["full", "xgb_only", "boost_only"]:
            p = xgb2.predict(X_hidden_aug)
            hidden_accum_xgb += pt_fold.inverse_transform(p)
        if mode in ["full", "lgb_only", "boost_only"]:
            p = lgb2.predict(X_hidden_aug)
            hidden_accum_lgb += pt_fold.inverse_transform(p)
        if mode in ["full", "cnn_only"]:
            h_ds = MaskedImageDataset(df_hidden, target_name, IMG_SIZE, val_transform)
            h_loader = DataLoader(h_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
            _, ps = eval_epoch(cnn, h_loader, criterion)
            hidden_accum_cnn += pt_fold.inverse_transform(ps)

    # --- MoE Gating (Train on Validation Set) ---
    # We use the predictions on the validation set as inputs to the Gater
    
    gating_cols = [c for c in df_pool.columns if c not in NON_FEATURE_COLS + [target_name]]
    # Note: df_pool has raw features, but we need the ones from df_val_fold (with history if dynamic)
    # Check if gating cols are available in df_val_fold
    # df_val_fold has history! so use it.
    
    # We need to ensure we pick numerical columns available in val fold
    X_gating_val = df_val_fold[gating_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    
    gating_scaler = StandardScaler()
    X_gating_val = gating_scaler.fit_transform(X_gating_val)
    
    # Stack Available Experts
    # Order: XGB, LGB, [LSTM], CNN
    expert_preds_list = [val_preds_xgb, val_preds_lgb]
    if val_preds_lstm is not None:
        expert_preds_list.append(val_preds_lstm)
    expert_preds_list.append(val_preds_cnn)
    
    num_experts = len(expert_preds_list)
    print(f"  [Gating] Training Gating Network with {num_experts} experts...")
    
    E_preds_val = torch.tensor(np.vstack(expert_preds_list).T, dtype=torch.float32).to(DEVICE)
    y_g_val = torch.tensor(y_val_orig, dtype=torch.float32).to(DEVICE).view(-1, 1)
    X_g_val = torch.tensor(X_gating_val, dtype=torch.float32).to(DEVICE)
    
    g_net = GatingNetwork(input_dim=X_gating_val.shape[1], num_experts=num_experts).to(DEVICE)
    opt_g = optim.Adam(g_net.parameters(), lr=0.001)
    crit_g = nn.MSELoss()
    y_g_val = torch.tensor(y_val_orig, dtype=torch.float32).to(DEVICE).view(-1, 1)
    X_g_val = torch.tensor(X_gating_val, dtype=torch.float32).to(DEVICE)
    
    g_net.train()
    for _ in range(50):
        weights = g_net(X_g_val)
        y_p = torch.sum(weights * E_preds_val, dim=1).view(-1, 1)
        loss = crit_g(y_p, y_g_val)
        opt_g.zero_grad(); loss.backward(); opt_g.step()
    
    g_net.eval()
    with torch.no_grad():
        final_weights = g_net(X_g_val).cpu().numpy()
        final_valid_pred = np.sum(final_weights * E_preds_val.cpu().numpy(), axis=1)

    # Save results (Using Validation IDs)
    # Re-assemble validation dataframe with predictions

    # Save results
    # Save results structured
    history_str = "Dynamic" if stochastic_window else "Static"
    
    # [FIX] Include condition in output path to prevent overwrites
    cond_str = condition if condition else "All"
    output_dir = os.path.join("output", target_name, mode, history_str, cond_str)
    os.makedirs(output_dir, exist_ok=True)
    
    # [FIX] Use df_val_fold for saving results as it's the only one we predicted on
    # We want to save predictions for the Validation Set (as our "OOF" replacement)
    df_val_for_save = df_val_fold.copy()
    
    df_val_for_save[f"Predicted_{target_name}"] = final_valid_pred
    
    # [OPTIMIZATION] Save individual experts for One-Shot Ablation Study
    if mode == "full":
        df_val_for_save[f"Pred_{target_name}_XGB"] = val_preds_xgb
        df_val_for_save[f"Pred_{target_name}_LGB"] = val_preds_lgb
        if val_preds_lstm is not None:
             df_val_for_save[f"Pred_{target_name}_LSTM"] = val_preds_lstm
        df_val_for_save[f"Pred_{target_name}_CNN"] = val_preds_cnn

    result_csv = os.path.join(output_dir, "predictions_oof.csv")
    df_val_for_save.to_csv(result_csv, index=False)
    
    score = r2_score(y_val_orig, final_valid_pred)
    print(f"R2 Score: {score:.4f}")

    # Append to Summary CSV (Keep global summary too)
    os.makedirs("results", exist_ok=True)
    summary_file = os.path.join("results", "training_summary.csv")
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write("Target,Mode,History,R2_Score,Path\n")
            
    with open(summary_file, "a") as f:
        f.write(f"{target_name},{mode},{history_str},{score:.4f},{result_csv}\n")
    
    # Generate Visualization
    visualize_results(result_csv, target_name, output_dir)
    
    # --- MODEL SAVING ---
    os.makedirs("weights", exist_ok=True)
    # Suffix logic: 'stochastic' for dynamic history, 'mean' for static (no history)
    suffix = "stochastic" if stochastic_window else "mean"
    cond_str = condition if condition else "All"
    model_prefix = f"weights/{target_name}_{cond_str}_{suffix}"
    
    # 1. Save Tabular Models
    if mode in ["full", "boost_only"]:
        xgb1.save_model(f"{model_prefix}_xgb1.json")
        xgb2.save_model(f"{model_prefix}_xgb2.json")
    if mode in ["full", "lgb_only", "boost_only"]:
        joblib.dump(lgb2, f"{model_prefix}_lgb.joblib")
    
    if mode in ["full", "cnn_only"]:
        torch.save(cnn.state_dict(), f"{model_prefix}_cnn.pth")
    
    # 2. Save Gating Network & Scaler (Using Val Gating which simulates test)
    torch.save(g_net.state_dict(), f"{model_prefix}_gating.pth")
    joblib.dump(gating_scaler, f"{model_prefix}_gating_scaler.joblib")
    
    # 3. Save Target Scaler
    if pt_fold:
        joblib.dump(pt_fold, f"{model_prefix}_target_scaler.joblib")
    
    # 4. Save Metadata (Features used)
    # Use columns from train fold (should be same as val)
    metadata = {
        'target_name': target_name,
        'feature_cols': [c for c in X_tab_train.columns.tolist() if c not in ['split_set']],
        'gating_cols': [c for c in gating_cols if c not in ['split_set']],
        'stochastic': stochastic_window,
        'window_size': window_size,
        'mode': mode
    }
    with open(f"{model_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    print(f"  [SUCCESS] All models saved to: {model_prefix}_*")

def visualize_results(csv_path, target_name, output_dir="."):
    df = pd.read_csv(csv_path)
    
    # Increase height to accommodate split plots
    plt.figure(figsize=(18, 12)) 
    
    # Subplot 1: All Samples Scatter Plot (Left Half)
    plt.subplot(1, 2, 1)
    y_true = df[target_name]
    y_pred = df[f"Predicted_{target_name}"]
    
    sns.scatterplot(x=y_true, y=y_pred, hue=df['condition'], alpha=0.5)
    
    # Diagonal line
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    
    plt.title(f"Sample-level Accuracy (R2: {r2_score(y_true, y_pred):.4f})")
    plt.xlabel(f"True {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    
    # Helper for trajectory plotting
    def plot_trajectory(ax, condition, color):
        subset = df[df['condition'] == condition]
        if subset.empty: return
        
        # Group by time and calculate mean/std
        true_stats = subset.groupby('time')[target_name].agg(['mean', 'std'])
        pred_stats = subset.groupby('time')[f"Predicted_{target_name}"].agg(['mean', 'std'])
        
        # Plot True
        ax.plot(true_stats.index, true_stats['mean'], '--o', color=color, alpha=0.6, label=f'True {condition}')
        # Plot Pred
        ax.plot(pred_stats.index, pred_stats['mean'], '-s', color=color, label=f'Pred {condition}', linewidth=2)
        
        # Fill Error
        ax.fill_between(true_stats.index, true_stats['mean'] - true_stats['std'], 
                         true_stats['mean'] + true_stats['std'], color=color, alpha=0.1)
        
        ax.set_title(f"{target_name} Population Trajectory ({condition})")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(target_name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    # Subplot 2: Light Trajectory (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    plot_trajectory(ax2, 'Light', 'red')
    
    # Subplot 3: Dark Trajectory (Bottom Right)
    ax3 = plt.subplot(2, 2, 4)
    plot_trajectory(ax3, 'Dark', 'blue')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f"validation_plot.png")
    plt.savefig(viz_path, dpi=300)
    print(f"  [SUCCESS] Saved result visualization to: {viz_path}")

def run_loo_experiment(target_name="Dry_Weight", stochastic_window=False, window_size=None):
    all_times = [1, 2, 3, 6, 12, 24, 48, 72]
    for t in all_times:
        run_pipeline(target_name=target_name, mode="full", hidden_times=[t], stochastic_window=stochastic_window, window_size=window_size)
