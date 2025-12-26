
import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import argparse

# Add Valid Paths
sys.path.append(os.getcwd())
from algae_fusion.config import NON_FEATURE_COLS

# === Helper Classes for Scaler Loading ===
class Log1pScaler:
    def __init__(self, s): self.s = s
    def transform(self, y): return self.s.transform(np.log1p(y).reshape(-1, 1)).flatten()
    def inverse_transform(self, yp): return np.expm1(self.s.inverse_transform(yp.reshape(-1, 1)).flatten())

class StandardWrapper:
    def __init__(self, s): self.s = s
    def transform(self, y): return self.s.transform(y.reshape(-1, 1)).flatten()
    def inverse_transform(self, yp): return self.s.inverse_transform(yp.reshape(-1, 1)).flatten()

# Inject into sys.modules so joblib can find them if they were saved as algae_fusion.engine.pipeline.Log1pScaler
# Actually, joblib might look for them in the original module.
# Check where they are expected. Usually joblib saves the class definition ref.
# If they fail to load, we might need to mock the module structure.
# But let's try defining them here first.

def test_boost_counterfactual(target, condition):
    print(f"=== Testing Boost Model Feature Reliance: {target} ({condition}) ===")
    
    # ... (Data Loading is same) ...
    # 1. Load Data
    df_test = pd.read_csv("data/dataset_test.csv")
    if condition:
        df_test = df_test[df_test['condition'] == condition].reset_index(drop=True)
    
    # Generate History Features (Window=3)
    df_test = df_test.sort_values(['group_idx', 'time'])
    feature_cols_base = [c for c in df_test.columns if c not in NON_FEATURE_COLS and "Prev" not in c]
    
    print(f"Generating History Features (Window=3)...")
    for lag in range(1, 4):
        shifted = df_test.groupby('group_idx')[feature_cols_base].shift(lag)
        shifted.columns = [f"Prev{lag}_{c}" for c in shifted.columns]
        df_test = pd.concat([df_test, shifted], axis=1)
    df_test = df_test.fillna(0)
    
    # 2. Load Model & Scaler
    archive_dir = "archive_run_20251225_153705/weights"
    model_name = f"{target}_{condition}_stochastic_xgb1.json"
    model_path = os.path.join(archive_dir, model_name) # Assuming xgb1 was the one saved/used
    # Note: filename might vary (xgb1 vs xgb2). Using find logic.
    
    import glob
    candidates = glob.glob(f"{archive_dir}/{target}_{condition}_*xgb*.json")
    if not candidates:
        print(f"Model not found for {target} {condition}")
        return
    model_path = candidates[0] # Pick first one (likely xgb1 or mean_xgb1)
    
    # Find Scaler (Assumed Same dir, name similar but scaler.joblib)
    # The scaler name in pipeline.py: weights/xgb_full_..._scaler.joblib or similar
    # In archive: {Target}_{Condition}_stochastic_target_scaler.joblib
    scaler_pattern = f"{archive_dir}/{target}_{condition}_*_target_scaler.joblib"
    scaler_candidates = glob.glob(scaler_pattern)
    scaler = None
    if scaler_candidates:
        print(f"Loading Scaler: {scaler_candidates[0]}")
        try:
            scaler = joblib.load(scaler_candidates[0])
        except Exception as e:
            print(f"Scaler load failed: {e}")
            # Try mapping module if needed
            # Assuming defined in __main__ or pipeline.
            pass

    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # Get features
    booster = model.get_booster()
    model_features = booster.feature_names
    if model_features is None:
        model_features = [c for c in df_test.columns if c in feature_cols_base or "Prev" in c]

    X_test = df_test[model_features]
    y_test = df_test[target] # Raw Targets
    
    # 3. Predict Original
    pred_raw = model.predict(X_test)
    
    # Inverse Transform
    if scaler:
        try:
            pred_orig = scaler.inverse_transform(pred_raw)
        except:
             # Fallback if scaler is standard scaler directly
             pred_orig = scaler.inverse_transform(pred_raw.reshape(-1,1)).flatten()
    else:
        pred_orig = pred_raw # Assume no scaling
        
    r2_orig = r2_score(y_test, pred_orig)
    mse_orig = mean_squared_error(y_test, pred_orig)
    
    print(f"[Original] R2: {r2_orig:.4f}, MSE: {mse_orig:.4f}")
    
    # 4. Predict Shuffled
    X_shuffled = X_test.copy()
    for col in X_shuffled.columns:
        X_shuffled[col] = np.random.permutation(X_shuffled[col].values)
        
    pred_shuf_raw = model.predict(X_shuffled)
    
    if scaler:
        try:
             pred_shuf = scaler.inverse_transform(pred_shuf_raw)
        except:
             pred_shuf = scaler.inverse_transform(pred_shuf_raw.reshape(-1,1)).flatten()
    else:
        pred_shuf = pred_shuf_raw

    r2_shuf = r2_score(y_test, pred_shuf)
    mse_shuf = mean_squared_error(y_test, pred_shuf)
    
    print(f"[Shuffled] R2: {r2_shuf:.4f}, MSE: {mse_shuf:.4f}")
    
    print("-" * 30)
    print("CONCLUSION:")
    if r2_orig > 0 and r2_shuf < 0:
        print(">> Boost Model COLLAPSED. Reliance on Features CONFIRMED.")
    elif r2_shuf < r2_orig - 0.5:
        print(">> Boost Model Degradation Significant.")
        print(f"   Delta R2: {r2_shuf - r2_orig:.4f}")
    else:
        print(">> Boost Model Resilience? (Unexpected)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='Dry_Weight')
    parser.add_argument('--condition', type=str, default='Light')
    args = parser.parse_args()
    
    test_boost_counterfactual(args.target, args.condition)
