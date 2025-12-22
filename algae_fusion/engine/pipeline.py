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
from algae_fusion import omics_loader

class Log1pScaler:
    def __init__(self, s): self.s = s
    def inverse_transform(self, yp): return np.expm1(self.s.inverse_transform(yp.reshape(-1, 1)).flatten())

def run_pipeline(target_name="Dry_Weight", mode="full", cv_method="group", max_folds=None, hidden_times=None, stochastic_window=False):
    """
    target_name: "Dry_Weight" or "Fv_Fm"
    mode: "full", "xgb_only", "lgb_only", "cnn_only"
    cv_method: "random" (KFold) or "group" (GroupKFold)
    """
    print(f"\n\n{'='*40}")
    status = "DYNAMIC (History)" if stochastic_window else "STATIC (No History)"
    print(f"STARTING PIPELINE: Target={target_name}, Mode={mode}, CV={cv_method} ({status})")
    print(f"{'='*40}\n")

    df = pd.read_csv("data/dataset_train.csv")

    
    df.loc[df['condition'] == 'Initial', 'condition'] = 'Light'
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df['file'] = df['file'].astype(str)
    
    # if 'cell_mean_area' in df.columns:
    #     df['cell_mean_area_sq'] = df['cell_mean_area'] ** 2
    #     df['cell_mean_area_cub'] = df['cell_mean_area'] ** 3

    df_hidden = pd.DataFrame()
    if hidden_times:
        mask = df['time'].isin(hidden_times)
        df_hidden = df[mask].copy().reset_index(drop=True)
        df = df[~mask].copy().reset_index(drop=True)

    df['Source_Path'] = df['Source_Path'].astype(str)
    
    df['Source_Path'] = df['Source_Path'].astype(str)
    
    # Define morphological columns 
    morph_cols = [
        'cell_mean_area', 
        'cell_mean_mean_intensity', 
        'cell_mean_eccentricity',
        'cell_mean_solidity'
    ]

    # [HISTORY RESTORED] User wants sliding window history (Prev_ features) but NO calculated kinetic rates.
    # We use stochastic window to get these history raw features.
    
    if stochastic_window:
        print("  [Feature Engineering] Computing Sliding Window History (Prev_ features)...")
        # We use stochastic window (better for individual trajectories in this dataset context)
        df = compute_sliding_window_features_stochastic(df, window_size=3, morph_cols=morph_cols)
    else:
        print("  [Static Mode] No History/Sliding Window features computed.")
    
    # [Note] We do NOT call compute_kinetic_features, so Rate_ and Rel_ are NOT computed in either case.


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

    final_pred = np.zeros(len(df), dtype=np.float32)
    oof_preds_xgb = np.zeros(len(df))
    oof_preds_lgb = np.zeros(len(df))
    oof_preds_cnn = np.zeros(len(df))
    oof_targets   = np.zeros(len(df))
    
    if not df_hidden.empty:
        hidden_accum_xgb = np.zeros(len(df_hidden))
        hidden_accum_lgb = np.zeros(len(df_hidden))
        hidden_accum_cnn = np.zeros(len(df_hidden))
        folds_run = 0
    
    groups = df['file']
    if cv_method == "loocv":
        splitter = LeaveOneOut().split(df)
    elif cv_method == "group":
        if max_folds == 1:
            splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(df, groups=groups)
        # else: # Fallback to standard GroupKFold logic which works better with groups passed to split
            # splitter = GroupKFold(n_splits=N_SPLITS).split(df, groups=groups)
        else:
             # Standard GroupKFold
             splitter = GroupKFold(n_splits=N_SPLITS).split(df, groups=groups)
    else:
        splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42).split(df)

    processed_indices = []
    feature_names = None

    for fold, (tr, va) in enumerate(splitter):
        if max_folds is not None and fold >= max_folds:
            break
        
        processed_indices.extend(va)
        # print(f"\n--- FOLD {fold+1} ---")
        
        df_train_fold, df_val_fold = df.iloc[tr].copy(), df.iloc[va].copy()
        
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
            pt_fold = scaler_target

        # Layer 1: XGB1
        if mode in ["full", "boost_only"]:
            xgb1 = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, tree_method="hist")
            xgb1.fit(X_tab_train, y_train_scaled)
            X_train_aug = X_tab_train.copy()
            X_val_aug = X_tab_val.copy()
            X_train_aug["XGB1_Feature"] = xgb1.predict(X_tab_train)
            X_val_aug["XGB1_Feature"] = xgb1.predict(X_tab_val)
        else:
            X_train_aug, X_val_aug = X_tab_train, X_tab_val

        # Layer 2: XGB2/LGB2
        pred_xgb_scaled = np.zeros(len(va))
        if mode in ["full", "xgb_only", "boost_only"]:
            xgb2 = XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=6, tree_method="hist")
            xgb2.fit(X_train_aug, y_train_scaled)
            pred_xgb_scaled = xgb2.predict(X_val_aug)

        pred_lgb_scaled = np.zeros(len(va))
        if mode in ["full", "lgb_only", "boost_only"]:
            lgb2 = LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=31)
            lgb2.fit(X_train_aug, y_train_scaled)
            pred_lgb_scaled = lgb2.predict(X_val_aug)

        # Layer 2: CNN
        pred_cnn_scaled = np.zeros(len(va))
        if mode in ["full", "cnn_only"]:
            train_ds = MaskedImageDataset(df_train_fold, target_name, IMG_SIZE, train_transform, labels=y_train_scaled)
            val_ds = MaskedImageDataset(df_val_fold, target_name, IMG_SIZE, val_transform, labels=y_val_scaled)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
            
            cnn = ResNetRegressor(BACKBONE).to(DEVICE)
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
            _, pred_cnn_scaled = eval_epoch(cnn, val_loader, criterion)

        # Hidden Set Prediction
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
                hidden_accum_xgb += pt_fold.inverse_transform(p) if hasattr(pt_fold, 'inverse_transform') else p
            if mode in ["full", "lgb_only", "boost_only"]:
                p = lgb2.predict(X_hidden_aug)
                hidden_accum_lgb += pt_fold.inverse_transform(p) if hasattr(pt_fold, 'inverse_transform') else p
            if mode in ["full", "cnn_only"]:
                h_ds = MaskedImageDataset(df_hidden, target_name, IMG_SIZE, val_transform)
                h_loader = DataLoader(h_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
                _, ps = eval_epoch(cnn, h_loader, criterion)
                hidden_accum_cnn += pt_fold.inverse_transform(ps) if hasattr(pt_fold, 'inverse_transform') else ps

        # OOF Storage
        inv = lambda p: pt_fold.inverse_transform(p.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else p
        oof_preds_xgb[va] = inv(pred_xgb_scaled)
        oof_preds_lgb[va] = inv(pred_lgb_scaled)
        oof_preds_cnn[va] = inv(pred_cnn_scaled)
        oof_targets[va] = y_val_orig

    # MoE Gating
    if max_folds is not None and max_folds < N_SPLITS:
        pi = np.array(processed_indices)
        oof_preds_xgb, oof_preds_lgb, oof_preds_cnn = oof_preds_xgb[pi], oof_preds_lgb[pi], oof_preds_cnn[pi]
        oof_targets = oof_targets[pi]
        df = df.iloc[pi].reset_index(drop=True)

    gating_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + [target_name]]
    X_gating = df[gating_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    X_gating = StandardScaler().fit_transform(X_gating)
    
    g_net = GatingNetwork(input_dim=X_gating.shape[1]).to(DEVICE)
    opt_g = optim.Adam(g_net.parameters(), lr=0.001)
    crit_g = nn.MSELoss()
    
    E_preds = torch.tensor(np.vstack([oof_preds_xgb, oof_preds_lgb, oof_preds_cnn]).T, dtype=torch.float32).to(DEVICE)
    y_g = torch.tensor(oof_targets, dtype=torch.float32).to(DEVICE).view(-1, 1)
    X_g = torch.tensor(X_gating, dtype=torch.float32).to(DEVICE)
    
    g_net.train()
    for _ in range(50):
        weights = g_net(X_g)
        y_p = torch.sum(weights * E_preds, dim=1).view(-1, 1)
        loss = crit_g(y_p, y_g)
        opt_g.zero_grad(); loss.backward(); opt_g.step()
    
    g_net.eval()
    with torch.no_grad():
        final_weights = g_net(X_g).cpu().numpy()
        final_pred = np.sum(final_weights * E_preds.cpu().numpy(), axis=1)

    # Save results
    # Save results structured
    history_str = "Dynamic" if stochastic_window else "Static"
    output_dir = os.path.join("output", target_name, mode, history_str)
    os.makedirs(output_dir, exist_ok=True)
    
    df[f"Predicted_{target_name}"] = final_pred
    
    # [OPTIMIZATION] Save individual experts for One-Shot Ablation Study
    if mode == "full":
        df[f"Pred_{target_name}_XGB"] = oof_preds_xgb
        df[f"Pred_{target_name}_LGB"] = oof_preds_lgb
        df[f"Pred_{target_name}_CNN"] = oof_preds_cnn

    result_csv = os.path.join(output_dir, "predictions_oof.csv")
    df.to_csv(result_csv, index=False)
    
    score = r2_score(oof_targets, final_pred)
    print(f"R2 Score: {score:.4f}")

    # Append to Summary CSV (Keep global summary too)
    summary_file = "training_summary.csv"
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
    model_prefix = f"weights/{target_name}_{suffix}"
    
    # 1. Save Tabular Models
    if mode in ["full", "boost_only"]:
        xgb1.save_model(f"{model_prefix}_xgb1.json")
        xgb2.save_model(f"{model_prefix}_xgb2.json")
    if mode in ["full", "lgb_only", "boost_only"]:
        joblib.dump(lgb2, f"{model_prefix}_lgb.joblib")
    
    if mode in ["full", "cnn_only"]:
        torch.save(cnn.state_dict(), f"{model_prefix}_cnn.pth")
    
    # 2. Save Gating Network & Scaler
    torch.save(g_net.state_dict(), f"{model_prefix}_gating.pth")
    gating_scaler = StandardScaler().fit(X_gating)
    joblib.dump(gating_scaler, f"{model_prefix}_gating_scaler.joblib")
    
    # 3. Save Target Scaler
    joblib.dump(pt_fold, f"{model_prefix}_target_scaler.joblib")
    
    # 4. Save Metadata (Features used)
    metadata = {
        'target_name': target_name,
        'feature_cols': tab_cols,
        'gating_cols': gating_cols,
        'stochastic': stochastic_window,
        'mode': mode
    }
    with open(f"{model_prefix}_metadata.json", "w") as f:
        json.dump(metadata, f)
    
    print(f"  [SUCCESS] All models saved to: {model_prefix}_*")

def visualize_results(csv_path, target_name, output_dir="."):
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(15, 6))
    
    # Subplot 1: All Samples Scatter Plot
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
    
    # Subplot 2: Population Trajectory (Mean over Time)
    plt.subplot(1, 2, 2)
    
    for cond, color in [('Light', 'red'), ('Dark', 'blue')]:
        subset = df[df['condition'] == cond]
        if subset.empty: continue
        
        # Group by time and calculate mean/std for True and Pred
        true_stats = subset.groupby('time')[target_name].agg(['mean', 'std'])
        pred_stats = subset.groupby('time')[f"Predicted_{target_name}"].agg(['mean', 'std'])
        
        # Plot True values (Dashed lines with markers)
        plt.plot(true_stats.index, true_stats['mean'], '--o', color=color, alpha=0.5, label=f'True {cond}')
        # Plot Predicted values (Solid lines with markers)
        plt.plot(pred_stats.index, pred_stats['mean'], '-s', color=color, label=f'Pred {cond}')
        
        # Fill error bands (std)
        plt.fill_between(true_stats.index, true_stats['mean'] - true_stats['std'], 
                         true_stats['mean'] + true_stats['std'], color=color, alpha=0.1)

    plt.title(f"{target_name} Population Trajectory")
    plt.xlabel("Time (h)")
    plt.ylabel(target_name)
    plt.legend()
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f"validation_plot.png")
    plt.savefig(viz_path, dpi=300)
    print(f"  [SUCCESS] Saved result visualization to: {viz_path}")

def run_loo_experiment(target_name="Dry_Weight", stochastic_window=False):
    all_times = [1, 2, 3, 6, 12, 24, 48, 72]
    for t in all_times:
        run_pipeline(target_name=target_name, mode="full", hidden_times=[t], stochastic_window=stochastic_window)
