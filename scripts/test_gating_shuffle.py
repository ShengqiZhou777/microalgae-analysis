
import pandas as pd
import numpy as np
import argparse
import os
import sys
from sklearn.metrics import r2_score

# Add project root to sys.path BEFORE importing local modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

import joblib
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from algae_fusion.config import IMG_SIZE, DEVICE, BACKBONE, NON_FEATURE_COLS, WINDOW_SIZE
from algae_fusion.models.cnn import ResNetRegressor
from algae_fusion.models.tabular import XGBoostExpert, LightGBMExpert
from algae_fusion.models.moe import GatingNetwork
from algae_fusion.data.dataset import MaskedImageDataset

# === Reusing Logic from predict.py ===
def load_moe_ensemble(target, model_prefix):
    artifacts = {}
    
    # Metadata
    meta_path = f"{model_prefix}_metadata.json"
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f: artifacts['meta'] = json.load(f)
        
    # Scalers
    if os.path.exists(f"{model_prefix}_target_scaler.joblib"):
        artifacts['scaler'] = joblib.load(f"{model_prefix}_target_scaler.joblib")
    if os.path.exists(f"{model_prefix}_gating_scaler.joblib"):
        artifacts['gating_scaler'] = joblib.load(f"{model_prefix}_gating_scaler.joblib")

    # Experts
    if os.path.exists(f"{model_prefix}_xgb1.json"):
        xgb1 = XGBoostExpert(); xgb1.load(f"{model_prefix}_xgb1.json"); artifacts['xgb1'] = xgb1
    if os.path.exists(f"{model_prefix}_xgb2.json"):
        xgb2 = XGBoostExpert(); xgb2.load(f"{model_prefix}_xgb2.json"); artifacts['xgb2'] = xgb2
    if os.path.exists(f"{model_prefix}_lgb.joblib"):
        lgb2 = LightGBMExpert(); lgb2.load(f"{model_prefix}_lgb.joblib"); artifacts['lgb2'] = lgb2

    # CNN (Optional/Missing)
    cnn_path = f"{model_prefix}_cnn.pth"
    if os.path.exists(cnn_path):
        state_dict = torch.load(cnn_path, map_location=DEVICE, weights_only=True)
        in_ch = state_dict['backbone.conv1.weight'].shape[1]
        cnn = ResNetRegressor(BACKBONE, in_channels=in_ch).to(DEVICE)
        cnn.load_state_dict(state_dict)
        cnn.eval()
        artifacts['cnn'] = cnn
        artifacts['in_channels'] = in_ch

    # Gating
    gating_path = f"{model_prefix}_gating.pth"
    if os.path.exists(gating_path):
        state_dict_g = torch.load(gating_path, map_location=DEVICE, weights_only=True)
        in_dim = state_dict_g['net.0.weight'].shape[1]
        out_dim = state_dict_g['net.6.weight'].shape[0] if 'net.6.weight' in state_dict_g else 3
        g_net = GatingNetwork(input_dim=in_dim, num_experts=out_dim).to(DEVICE)
        g_net.load_state_dict(state_dict_g)
        g_net.eval()
        artifacts['g_net'] = g_net
            
    return artifacts

def predict_single_target_custom(df, target, artifacts):
    # Modified to be simpler and just return pred
    feature_cols = [c for c in artifacts['meta']['feature_cols'] if c != 'split_set']
    gating_cols = [c for c in artifacts['meta'].get('gating_cols', []) if c != 'split_set']
    
    # Check features exist (handle missing by fill 0)
    for c in feature_cols:
        if c not in df.columns: df[c] = 0
            
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    
    # 1. XGB1
    xgb1 = artifacts.get('xgb1')
    l1 = xgb1.predict(X) if xgb1 else np.zeros(len(df))
    
    # 2. Augment
    X_aug = X.copy()
    X_aug["XGB1_Feature"] = l1
    
    # 3. XGB2 & LGB
    xgb2 = artifacts.get('xgb2')
    pred_xgb = xgb2.predict(X_aug) if xgb2 else np.zeros(len(df))
    
    lgb2 = artifacts.get('lgb2')
    pred_lgb = lgb2.predict(X_aug) if lgb2 else np.zeros(len(df))
    
    # 4. CNN (Placeholder 0)
    pred_cnn = np.zeros(len(df)) 
    
    # 5. Gating
    g_net = artifacts.get('g_net')
    gating_scaler = artifacts.get('gating_scaler')
    
    if g_net:
        # Gating Input: Use gating_cols (Base features usually)
        # Ensure cols exist
        for c in gating_cols: 
            if c not in df.columns: df[c] = 0
            
        X_gate = df[gating_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
        if gating_scaler: 
            try: X_gate = gating_scaler.transform(X_gate)
            except: pass
            
        xt = torch.tensor(X_gate, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            w = g_net(xt).cpu().numpy() # [N, 3]
    else:
        # Fallback average
        w = np.ones((len(df), 3)) / 3.0
        
    # Fuse: XGB, LGB, CNN
    # Assume 3 experts order
    # Note: If weights shape is [N,2], we adjust.
    if w.shape[1] == 2:
        final = w[:,0]*pred_xgb + w[:,1]*pred_lgb
    else:
        final = w[:,0]*pred_xgb + w[:,1]*pred_lgb + w[:,2]*pred_cnn
        
    # Inverse Transform
    scaler = artifacts.get('scaler')
    if scaler:
        try: final = scaler.inverse_transform(final.reshape(-1,1)).flatten()
        except: pass
        
    return final

def main():
    target = "Dry_Weight"
    condition = "Light"
    
    # 1. Load Data
    input_file = "data/dataset_test.csv"
    print(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    df = df[df['condition'] == condition].reset_index(drop=True)
    
    # 2. History Features
    print("Generating History Features...")
    df = df.sort_values(['group_idx', 'time'])
    from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
    
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    morph_cols = [c for c in all_numeric_cols if c.startswith('cell_') and c not in NON_FEATURE_COLS]
    df = compute_sliding_window_features_stochastic(df, window_size=3, morph_cols=morph_cols)
    
    # 3. Load Model
    model_path = f"archive_run_20251225_153705/weights/{target}_{condition}_stochastic" # or mean
    print(f"Loading Ensemble from {model_path}...")
    artifacts = load_moe_ensemble(target, model_path)
    
    if not artifacts:
        print("Model not found! Check paths.")
        return

    # 4. Predict Original
    print("Predicting Original...")
    pred_orig = predict_single_target_custom(df.copy(), target, artifacts)
    y_true = df[target].values
    r2_orig = r2_score(y_true, pred_orig)
    print(f"Original R2: {r2_orig:.4f}")
    
    # 5. Predict Shuffled
    print("Predicting Shuffled (Counterfactual)...")
    df_shuf = df.copy()
    
    # Identify feature columns to shuffle
    # Use meta features + gating features
    feat_cols = artifacts['meta']['feature_cols']
    gate_cols = artifacts['meta'].get('gating_cols', [])
    cols_to_shuffle = list(set(feat_cols + gate_cols))
    
    # Shuffle
    for c in cols_to_shuffle:
        if c in df_shuf.columns:
            df_shuf[c] = np.random.permutation(df_shuf[c].values)
            
    pred_shuf = predict_single_target_custom(df_shuf, target, artifacts)
    r2_shuf = r2_score(y_true, pred_shuf)
    print(f"Shuffled R2: {r2_shuf:.4f}")
    
    if r2_shuf < 0:
        print("\nCONCLUSION: Model Collapsed (Good!). Strict reliance on features confirmed.")
    else:
        print(f"\nCONCLUSION: Model Resilient? (Delta={r2_orig - r2_shuf:.2f})")

if __name__ == "__main__":
    main()
