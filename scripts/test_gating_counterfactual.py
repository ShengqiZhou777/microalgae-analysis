
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import argparse
import glob

# Add path
sys.path.append(os.getcwd())
from algae_fusion.config import NON_FEATURE_COLS

# === Helper Classes ===
class Log1pScaler:
    def __init__(self, s): self.s = s
    def transform(self, y): return self.s.transform(np.log1p(y).reshape(-1, 1)).flatten()
    def inverse_transform(self, yp): return np.expm1(self.s.inverse_transform(yp.reshape(-1, 1)).flatten())

class StandardWrapper:
    def __init__(self, s): self.s = s
    def transform(self, y): return self.s.transform(y.reshape(-1, 1)).flatten()
    def inverse_transform(self, yp): return self.s.inverse_transform(yp.reshape(-1, 1)).flatten()

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.net(x)

def test_gating_counterfactual(target, condition):
    print(f"=== Testing Full Gating Ensemble: {target} ({condition}) ===")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Data
    df_test = pd.read_csv("data/dataset_test.csv")
    if condition:
        df_test = df_test[df_test['condition'] == condition].reset_index(drop=True)
        
    df_test = df_test.sort_values(['group_idx', 'time'])
    feature_cols_base = [c for c in df_test.columns if c not in NON_FEATURE_COLS and "Prev" not in c]
    
    print(f"Generating History Features (Window=3)...")
    for lag in range(1, 4):
        shifted = df_test.groupby('group_idx')[feature_cols_base].shift(lag)
        shifted.columns = [f"Prev{lag}_{c}" for c in shifted.columns]
        df_test = pd.concat([df_test, shifted], axis=1)
    df_test = df_test.fillna(0)
    
    print(f"Test Set Size: {len(df_test)}")
    feature_cols = [c for c in df_test.columns if c in feature_cols_base or "Prev" in c]
    
    # 2. Load Components
    archive_dir = "archive_run_20251225_153705/weights"
    
    # Helper to find file
    def find_file(suffix):
        pat = f"{archive_dir}/{target}_{condition}_stochastic_{suffix}"
        cands = glob.glob(pat)
        if not cands: 
            # Try looser match
            pat = f"{archive_dir}/{target}_{condition}_*_{suffix}"
            cands = glob.glob(pat)
        return cands[0] if cands else None

    path_xgb1 = find_file("xgb1.json")
    path_xgb2 = find_file("xgb2.json")
    path_lgb = find_file("lgb.joblib")
    path_gating = find_file("gating.pth")
    path_scaler = find_file("target_scaler.joblib")
    
    if not all([path_xgb1, path_xgb2, path_lgb, path_gating, path_scaler]):
        print("Missing model components!")
        print(f"XGB1: {path_xgb1}")
        print(f"XGB2: {path_xgb2}")
        print(f"LGB: {path_lgb}")
        print(f"Gating: {path_gating}")
        return

    print("Loading Models...")
    xgb1 = xgb.XGBRegressor(); xgb1.load_model(path_xgb1)
    xgb2 = xgb.XGBRegressor(); xgb2.load_model(path_xgb2)
    lgb_model = joblib.load(path_lgb)
    scaler = joblib.load(path_scaler)
    
    # Gating uses BASE features (91), Experts use FULL features (364)
    gating_net = GatingNetwork(input_dim=len(feature_cols_base), num_experts=3).to(DEVICE)
    gating_net.load_state_dict(torch.load(path_gating, map_location=DEVICE))
    gating_net.eval()
    
    y_test = df_test[target].values
    
    # === Inference Function ===
    def run_inference(df_in):
        # 1. Prepare Features
        # X_raw now contains ALL features (including history)
        X_all = df_in[feature_cols]
        # X_base for Gating (exclude history)
        X_base = df_in[feature_cols_base]
        
        # 2. XGB1 (Layer 1)
        # Check XGB1 features
        feat_xgb1 = xgb1.get_booster().feature_names
        # Note: XGB1 uses features from training, which likely includes history if available
        pred_xgb1 = xgb1.predict(X_all[feat_xgb1] if feat_xgb1 else X_all)
        
        # 3. Stack Features
        X_aug = X_all.copy()
        X_aug["XGB1_Feature"] = pred_xgb1
        
        # 4. XGB2 (Layer 2)
        feat_xgb2 = xgb2.get_booster().feature_names
        pred_xgb2 = xgb2.predict(X_aug[feat_xgb2] if feat_xgb2 else X_aug)
        
        # 5. LGB (Layer 2)
        # 5. LGB (Layer 2)
        # Fix: Filter features to match training set (227 vs 365)
        if hasattr(lgb_model, 'feature_name_'):
             lgb_feats = lgb_model.feature_name_
             # Ensure all needed features are present
             missing = [f for f in lgb_feats if f not in X_aug.columns]
             if missing:
                 print(f"Server Warning: LGB model expects {len(lgb_feats)} features, {len(missing)} missing in input.")
             pred_lgb = lgb_model.predict(X_aug[lgb_feats])
        else:
             pred_lgb = lgb_model.predict(X_aug)
        
        # 6. CNN (Missing)
        pred_cnn = np.zeros_like(pred_xgb2)
        
        # 7. Gating
        # Input to gating is BASE Features only
        x_tensor = torch.tensor(X_base.values, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            weights = gating_net(x_tensor).cpu().numpy() # [N, 3]
            
        # 8. Fuse
        w_xgb = weights[:, 0]
        w_lgb = weights[:, 1]
        w_cnn = weights[:, 2]
        
        final_pred_scaled = (w_xgb * pred_xgb2) + (w_lgb * pred_lgb) + (w_cnn * pred_cnn)
        
        # 9. Inverse Transform
        try:
             final_pred = scaler.inverse_transform(final_pred_scaled)
        except:
             final_pred = scaler.inverse_transform(final_pred_scaled.reshape(-1,1)).flatten()
             
        return final_pred

    # === Test Original ===
    pred_orig = run_inference(df_test)
    r2_orig = r2_score(y_test, pred_orig)
    print(f"[Original Ensemble] R2: {r2_orig:.4f}")
    
    # === Test Shuffled ===
    print("Shuffling Features...")
    df_shuf = df_test.copy()
    # Shuffle only feature columns, keeping targets and others intact?
    # Or shuffle ALL columns for "feature destruction"?
    # The previous test shuffled X_test. Here we shuffle inside df_shuf.
    for c in feature_cols:
        df_shuf[c] = np.random.permutation(df_shuf[c].values)
        
    pred_shuf = run_inference(df_shuf)
    r2_shuf = r2_score(y_test, pred_shuf)
    print(f"[Shuffled Ensemble] R2: {r2_shuf:.4f}")

    print("-" * 30)
    print("CONCLUSION:")
    if r2_shuf < r2_orig - 0.5:
        print(">> Gating Ensemble COLLAPSED. Reliance on Features CONFIRMED.")
    else:
        print(">> Gating Ensemble Resilient (Unexpected).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='Dry_Weight')
    parser.add_argument('--condition', type=str, default='Light')
    args = parser.parse_args()
    
    test_gating_counterfactual(args.target, args.condition)
