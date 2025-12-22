import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from torch.utils.data import DataLoader
from torchvision import transforms

from algae_fusion.config import IMG_SIZE, DEVICE, BACKBONE
from algae_fusion.models.backbones import ResNetRegressor
from algae_fusion.models.moe import GatingNetwork
from algae_fusion.data.dataset import MaskedImageDataset

# Define targets and their model file suffixes
TARGETS = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]

def load_moe_ensemble(target, model_prefix):
    """
    Loads XGB1, XGB2, LGB2, CNN, and Gating Network for a specific target.
    Returns a dictionary of loaded artifacts.
    """
    artifacts = {}
    
    # Check metadata
    meta_path = f"{model_prefix}_metadata.json"
    if not os.path.exists(meta_path):
        print(f"  [WARN] Metadata not found for {target}: {meta_path}")
        return None
        
    with open(meta_path, "r") as f:
        artifacts['meta'] = json.load(f)
        
    feat_cols = artifacts['meta']['feature_cols']
    gating_cols = artifacts['meta'].get('gating_cols', [])
        
    # Load Scalers
    scaler_path = f"{model_prefix}_target_scaler.joblib"
    if os.path.exists(scaler_path):
        artifacts['scaler'] = joblib.load(scaler_path)
        
    gating_scaler_path = f"{model_prefix}_gating_scaler.joblib"
    if os.path.exists(gating_scaler_path):
        artifacts['gating_scaler'] = joblib.load(gating_scaler_path)

    # 1. Load Tabular Experts
    if os.path.exists(f"{model_prefix}_xgb1.json"):
        xgb1 = XGBRegressor()
        xgb1.load_model(f"{model_prefix}_xgb1.json")
        artifacts['xgb1'] = xgb1
    
    if os.path.exists(f"{model_prefix}_xgb2.json"):
        xgb2 = XGBRegressor()
        xgb2.load_model(f"{model_prefix}_xgb2.json")
        artifacts['xgb2'] = xgb2
    
    lgb_path = f"{model_prefix}_lgb.joblib"
    if os.path.exists(lgb_path):
        artifacts['lgb2'] = joblib.load(lgb_path)

    # 2. Load Visual Expert (CNN)
    cnn_path = f"{model_prefix}_cnn.pth"
    if os.path.exists(cnn_path):
        cnn = ResNetRegressor(BACKBONE).to(DEVICE)
        state_dict = torch.load(cnn_path, map_location=DEVICE, weights_only=True)
        cnn.load_state_dict(state_dict)
        cnn.eval()
        artifacts['cnn'] = cnn
    else:
        print(f"  [WARN] CNN weights not found: {cnn_path}. Expert will be missing.")

    # 3. Load Gating Network
    gating_path = f"{model_prefix}_gating.pth"
    if os.path.exists(gating_path):
        # We need input_dim for GatingNetwork. It should be len(gating_cols)
        # If we can't determine it easily, we might fail. 
        # But gating_cols is in metadata.
        if gating_cols:
            g_net = GatingNetwork(input_dim=len(gating_cols)).to(DEVICE)
            g_net.load_state_dict(torch.load(gating_path, map_location=DEVICE, weights_only=True))
            g_net.eval()
            artifacts['g_net'] = g_net
        else:
            print("  [WARN] Gating columns missing in metadata. Cannot load Gating Network.")
            
    return artifacts

def predict_single_target(df, target, artifacts):
    """
    Runs the MoE inference for a single target.
    Returns: (final_predictions, expert_weights)
    """
    feature_cols = artifacts['meta']['feature_cols']
    gating_cols = artifacts['meta'].get('gating_cols', [])
    
    # --- Prepare Tabular Inputs ---
    X = df[feature_cols].copy()
    X = X.select_dtypes(exclude=['object']) # Ensure numeric
    
    # Layer 1: XGB1
    xgb1 = artifacts.get('xgb1')
    if xgb1:
        l1_feat = xgb1.predict(X)
    else:
        l1_feat = np.zeros(len(df))
    
    # Layer 2: Augment
    X_aug = X.copy()
    X_aug["XGB1_Feature"] = l1_feat
    
    # --- Run Experts ---
    preds_map = {} # 'xgb', 'lgb', 'cnn'
    
    # 1. XGB2 (Expert 1)
    xgb2 = artifacts.get('xgb2')
    if xgb2:
        preds_map['xgb'] = xgb2.predict(X_aug)
    else:
        preds_map['xgb'] = np.zeros(len(df))
        
    # 2. LGB2 (Expert 2)
    lgb2 = artifacts.get('lgb2')
    if lgb2:
        preds_map['lgb'] = lgb2.predict(X_aug)
    else:
        preds_map['lgb'] = np.zeros(len(df))
        
    # 3. CNN (Expert 3)
    cnn = artifacts.get('cnn')
    if cnn:
        # Prepare Images
        val_transform = transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        # We assume df has 'Source_Path'. If not, we fail gracefully or warn?
        # Dataset expects labels but for inference we can pass zeros
        # Actually MaskedImageDataset requires labels arg if not provided? 
        # Let's check constructor signature: def __init__(self, dataframe, target_col, img_size, transform=None, labels=None):
        # If labels is None, it might fail in __getitem__ if it tries to return label.
        # Let's verify dataset code logic. Assuming it handles None or we pass dummy.
        # We'll pass zeros.
        dummy_labels = np.zeros(len(df))
        ds = MaskedImageDataset(df, target, IMG_SIZE, transform=val_transform, labels=dummy_labels)
        
        # In predict script, maybe we don't have images for all rows?
        # We assume the user provided valid input with images.
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
        
        raw_cnn_preds = []
        with torch.no_grad():
            for imgs, _ in dl: # __getitem__ returns image, label
                imgs = imgs.to(DEVICE)
                out = cnn(imgs)
                raw_cnn_preds.append(out.cpu().numpy().flatten())
        
        preds_map['cnn'] = np.concatenate(raw_cnn_preds)
        
        # Inverse transform CNN preds here?
        # NO. The trained pipeline trains CNN on SCALED targets.
        # XGB/LGB were also trained on SCALED targets.
        # So all experts output scaled values. This is consistent.
    else:
        preds_map['cnn'] = np.zeros(len(df))

    # --- MoE Gating ---
    g_net = artifacts.get('g_net')
    gating_scaler = artifacts.get('gating_scaler')
    
    if g_net and gating_cols:
        # Prepare Gating Input
        X_gate = df[gating_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
        if gating_scaler:
            X_gate = gating_scaler.transform(X_gate)
        
        X_gate_t = torch.tensor(X_gate, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            weights = g_net(X_gate_t) # [N, 3]
            weights = weights.cpu().numpy()
            
    else:
        # Fallback to simple average [0.33, 0.33, 0.33]
        print("  [WARN] Gating Network missing. Using simple average.")
        weights = np.ones((len(df), 3)) / 3.0
        
    # --- Aggregate ---
    # Stack experts: [N, 3] -> order MUST match training: XGB, LGB, CNN
    E_preds = np.vstack([preds_map['xgb'], preds_map['lgb'], preds_map['cnn']]).T # [N, 3]
    
    # Weighted Sum
    # Element-wise multiply and sum across dim 1
    # weights: [N, 3], E_preds: [N, 3]
    raw_final_pred = np.sum(weights * E_preds, axis=1)
    
    # --- Inverse Transform ---
    scaler = artifacts.get('scaler')
    if scaler:
        if hasattr(scaler, 'inverse_transform'):
            final_pred = scaler.inverse_transform(raw_final_pred.reshape(-1, 1)).flatten()
        else:
            final_pred = raw_final_pred
    else:
        final_pred = raw_final_pred
        
    return final_pred, weights

def main():
    parser = argparse.ArgumentParser(description="Multi-Target MoE Inference")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, default="Final_MoE_Predictions.csv")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Loading input data from: {args.input}")
    df = pd.read_csv(args.input)
    
    # --- 1. Dynamic Features Context (Identical to original) ---
    df_dynamic = df.copy()
    TRAIN_DB_PATH = "data/dataset_train.csv"
    if os.path.exists(TRAIN_DB_PATH):
        print("   [Info] Loading Training DB for History Context...")
        df_train_db = pd.read_csv(TRAIN_DB_PATH)
        df_dynamic['is_test'] = True
        df_train_db['is_test'] = False
        
        common_cols = df_dynamic.columns.intersection(df_train_db.columns)
        df_combined = pd.concat([df_dynamic[common_cols], df_train_db[common_cols]], ignore_index=True)
        
        from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
        
        morph_cols = ['cell_mean_area', 'cell_mean_mean_intensity', 'cell_mean_eccentricity', 'cell_mean_solidity']
        print("   [Info] Computing Stochastic Sliding Window...")
        df_combined_aug = compute_sliding_window_features_stochastic(df_combined, window_size=3, morph_cols=morph_cols)
        
        df_dynamic_aug = df_combined_aug[df_combined_aug['is_test'] == True].copy()
        # Handle potential duplicates from self-testing merge (many-to-many)
        df_dynamic_aug = df_dynamic_aug.drop_duplicates(subset=['file'])
        df_dynamic_aug = df_dynamic_aug.set_index('file').reindex(df['file']).reset_index()
        df_dynamic = df_dynamic_aug
    else:
        print("   [WARN] Training DB not found. Dynamic models will likely fail.")

    results = df.copy()
    
    print("\nStarting Multi-Target MoE Inference...")
    print("======================================")
    
    for target in TARGETS:
        print(f"-> Processing Target: {target}")
        
        # 1. Static Prediction
        print(f"   [Static] Predicting...")
        try:
            artifacts_static = load_moe_ensemble(target, f"weights/{target}_mean")
            if artifacts_static:
                pred_static, w_static = predict_single_target(df, target, artifacts_static)
                results[f"Pred_{target}_Static"] = pred_static
                # Save weights for analysis
                results[f"W_XGB_{target}_Static"] = w_static[:, 0]
                results[f"W_LGB_{target}_Static"] = w_static[:, 1]
                results[f"W_CNN_{target}_Static"] = w_static[:, 2]
            else:
                print(f"   [WARN] Static model not found for {target}")
        except Exception as e:
            print(f"   [ERROR] Static inference failed: {e}")
            import traceback
            traceback.print_exc()

        # 2. Dynamic Prediction
        print(f"   [Dynamic] Predicting...")
        try:
            artifacts_dynamic = load_moe_ensemble(target, f"weights/{target}_stochastic")
            if artifacts_dynamic:
                pred_dynamic, w_dynamic = predict_single_target(df_dynamic, target, artifacts_dynamic)
                results[f"Pred_{target}_Dynamic"] = pred_dynamic
                results[f"W_XGB_{target}_Dynamic"] = w_dynamic[:, 0]
                results[f"W_LGB_{target}_Dynamic"] = w_dynamic[:, 1]
                results[f"W_CNN_{target}_Dynamic"] = w_dynamic[:, 2]
            else:
                print(f"   [WARN] Dynamic model not found for {target}")
        except Exception as e:
            print(f"   [ERROR] Dynamic inference failed: {e}")

    # Save
    out_dir = os.path.dirname(args.output)
    if out_dir: os.makedirs(out_dir, exist_ok=True)
    results.to_csv(args.output, index=False)
    print("\n======================================")
    print(f"Done! MoE results saved to: {args.output}")
    
    # Visualize
    visualize_moe_results(results, TARGETS)

def visualize_moe_results(df, targets):
    for target in targets:
        pred_static_col = f"Pred_{target}_Static"
        
        if pred_static_col not in df.columns:
            continue
            
        plt.figure(figsize=(20, 6))
        
        # 1. Scatter Plot
        plt.subplot(1, 3, 1)
        if pred_static_col in df.columns:
            sns.scatterplot(x=df[target], y=df[pred_static_col], alpha=0.5, label='Static')
            
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        plt.title(f"{target}: True vs Pred")
        
        # 2. Expert Weights Distribution (Boxplot)
        plt.subplot(1, 3, 2)
        # Gather weights
        w_cols = [f"W_XGB_{target}_Static", f"W_LGB_{target}_Static", f"W_CNN_{target}_Static"]
        w_df = df[w_cols].copy()
        w_df.columns = ["XGB", "LGB", "CNN"]
        w_df_melted = w_df.melt(var_name="Expert", value_name="Weight")
        
        sns.boxplot(x="Expert", y="Weight", data=w_df_melted, hue="Expert", legend=False, palette="Set2")
        plt.title(f"Expert Contribution (MoE Gating)")
        plt.ylim(0, 1.1)
        
        # 3. Trajectory
        plt.subplot(1, 3, 3)
        df_sorted = df.sort_values('time')
        for cond, style in [('Light', '-'), ('Dark', '--')]:
            subset = df_sorted[df_sorted['condition'] == cond]
            if subset.empty: continue
            
            true_means = subset.groupby('time')[target].mean()
            plt.plot(true_means.index, true_means.values, style + 'o', label=f'True {cond}', color='black')
            
            if pred_static_col in df.columns:
                stat_means = subset.groupby('time')[pred_static_col].mean()
                plt.plot(stat_means.index, stat_means.values, style + 'x', label=f'Static {cond}', color='blue')
                
        plt.title(f"{target} Dynamics")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"MoE_Result_{target}.png", dpi=300)
        print(f"  -> Saved plot: MoE_Result_{target}.png")

if __name__ == "__main__":
    main()
