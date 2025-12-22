import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
import argparse
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic as compute_sw_stochastic
from algae_fusion.engine.pipeline import GatingNetwork, NON_FEATURE_COLS

def predict(input_csv, model_prefix, output_path=None):
    # 1. Load Metadata
    with open(f"{model_prefix}_metadata.json", "r") as f:
        meta = json.load(f)
    
    target_name = meta['target_name']
    tab_cols = meta['feature_cols']
    gating_cols = meta['gating_cols']
    mode = meta['mode']
    
    # 2. Load Models
    print(f"Loading models from {model_prefix}...")
    
    xgb1, xgb2, lgb2 = None, None, None
    if mode in ["full", "boost_only"]:
        xgb1 = XGBRegressor()
        xgb1.load_model(f"{model_prefix}_xgb1.json")
        xgb2 = XGBRegressor()
        xgb2.load_model(f"{model_prefix}_xgb2.json")
    
    if mode in ["full", "lgb_only", "boost_only"]:
        lgb2 = joblib.load(f"{model_prefix}_lgb.joblib")
    
    g_net = GatingNetwork(input_dim=len(gating_cols)).to("cpu")
    g_net.load_state_dict(torch.load(f"{model_prefix}_gating.pth", map_location="cpu"))
    g_net.eval()
    
    g_scaler = joblib.load(f"{model_prefix}_gating_scaler.joblib")
    t_scaler = joblib.load(f"{model_prefix}_target_scaler.joblib")
    
    # 3. Prepare Data
    df_input = pd.read_csv(input_csv)
    df_input.loc[df_input['condition'] == 'Initial', 'condition'] = 'Light'
    
    # [Conditional History Logic]
    # We check if the trained model expects sliding window features (Prev_...)
    # We look for a representative feature, e.g., 'Prev1_cell_mean_area'
    
    use_history = any(c.startswith('Prev') for c in tab_cols)
    
    if use_history:
        print("  [Inference] Model expects history. Computing Sliding Window...")
        # 3a. Sliding Window (DB Load required)
        # CRITICAL: Use only TRAINING data for history to avoid leakage
        TRAIN_CSV = "data/dataset_train.csv"
        if not os.path.exists(TRAIN_CSV):
             # Fallback if DB missing: assume pure static inference on just input? 
             # But model will crash if missing cols. So we must error.
             raise FileNotFoundError(f"Training database {TRAIN_CSV} not found. Required for history context.")
        
        df_db = pd.read_csv(TRAIN_CSV)
        df_db.loc[df_db['condition'] == 'Initial', 'condition'] = 'Light'
        
        # To prevent merge explosion if input files are already in DB, tag them
        df_input['_orig_file'] = df_input['file']
        df_input['file'] = df_input['file'].astype(str) + "_INFER"
        
        df_full = pd.concat([df_db, df_input], axis=0).reset_index(drop=True)
        morph_cols = ['cell_mean_area', 'cell_mean_mean_intensity', 'cell_mean_eccentricity', 'cell_mean_solidity']
        
        # Compute History Raw Features (Prev_)
        df_processed = compute_sw_stochastic(df_full, window_size=3, morph_cols=morph_cols)
        
        # Separate back: Filter by the marked filename 
        df_final = df_processed[df_processed['file'].str.contains("_INFER", na=False)].copy()
        
        # Restore original filename immediately
        if '_orig_file' in df_final.columns:
            df_final['file'] = df_final['_orig_file']
            df_final.drop(columns=['_orig_file'], inplace=True)
            
    else:
        print("  [Inference] Model is Static (No History). Skipping Sliding Window.")
        df_final = df_input.copy()
    
    # Polynomial features for area
    if 'cell_mean_area' in df_final.columns:
        df_final['cell_mean_area_sq'] = df_final['cell_mean_area'] ** 2
        df_final['cell_mean_area_cub'] = df_final['cell_mean_area'] ** 3
        
    X_tab = df_final[tab_cols].select_dtypes(exclude=['object']).fillna(0)
    
    # 4. Inference
    print("Running inference...")
    
    # Layer 1
    if xgb1:
        X_aug = X_tab.copy()
        X_aug["XGB1_Feature"] = xgb1.predict(X_tab)
    else:
        X_aug = X_tab
        
    # Layer 2 Predictions (Scaled)
    pred_xgb = xgb2.predict(X_aug) if xgb2 else np.zeros(len(df_final))
    pred_lgb = lgb2.predict(X_aug) if lgb2 else np.zeros(len(df_final))
    pred_cnn = np.zeros(len(df_final)) # CNN skipped in boost_only
    
    # MoE Gating
    X_g = df_final[gating_cols].select_dtypes(include=[np.number]).fillna(0).values
    X_g_scaled = g_scaler.transform(X_g)
    X_g_tensor = torch.tensor(X_g_scaled, dtype=torch.float32)
    
    E_preds = torch.tensor(np.vstack([pred_xgb, pred_lgb, pred_cnn]).T, dtype=torch.float32)
    
    with torch.no_grad():
        weights = g_net(X_g_tensor).numpy()
        final_pred_scaled = np.sum(weights * E_preds.numpy(), axis=1)
    
    # Inverse Scale
    inv = lambda p: t_scaler.inverse_transform(p.reshape(-1, 1)).flatten() if hasattr(t_scaler, 'inverse_transform') else p
    final_pred = inv(final_pred_scaled)
    
    df_final[f"Predicted_{target_name}"] = final_pred
    
    # Restore original filename logic removed (not needed as we didn't rename)
    
    if output_path:
        # Simplify output columns per user request
        cols_to_keep = ['file', 'time', 'condition', target_name, f"Predicted_{target_name}"]
        # Only keep columns that actually exist (target might not be in input if it's real inference)
        out_cols = [c for c in cols_to_keep if c in df_final.columns]
        
        df_final[out_cols].to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return df_final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model_prefix", required=True)
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    
    predict(args.input, args.model_prefix, args.output)
