
import os
import sys
import torch
import torch.nn as nn
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

# Add project root
sys.path.append(os.getcwd())

from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.models.ode import GrowthODE
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch

def test_ode_counterfactual(target, condition):
    print(f"=== Testing Neural ODE Feature Reliance: {target} ({condition}) ===")
    
    # 1. Load Data
    df_test = pd.read_csv("data/dataset_test.csv")
    if condition != "All":
        df_test = df_test[df_test['condition'] == condition].reset_index(drop=True)
        
    # 2. Load Scaler
    scaler_path = f"weights/ode_{target}_{condition}_scaler.joblib"
    if not os.path.exists(scaler_path):
        print("Scaler not found.")
        return
    scaler = joblib.load(scaler_path)
    
    # Scale Target
    df_test[target] = scaler.transform(df_test[target].values)
    
    # 3. Create Dataset (Normal)
    feature_cols = [c for c in df_test.columns if c not in NON_FEATURE_COLS + [target]]
    group_col = 'group_idx' if 'group_idx' in df_test.columns else 'file'
    
    # 4. Load Model
    from algae_fusion.models.ode import GrowthODE
    
    # ODEProjector is defined locally in pipeline.py, so we must redefine it here matching that structure
    class ODEProjector(nn.Module):
        def __init__(self, ode, latent_dim):
            super().__init__()
            self.ode = ode
            # MLP Decoder instead of linear for better mapping
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.Tanh(),
                nn.Dropout(p=0.2), # Decoder Regularization
                nn.Linear(64, 1)
            )
        def forward(self, x, t, mask):
            # ODERNN returns [B, T, latent]
            h = self.ode.ode_net(x, t, mask) 
            return self.proj(h) # pipeline.py uses .squeeze(-1) but here we handle shape carefully

    input_dim = len(feature_cols)
    hidden_dim = 32 # From pipeline default
    ode_core = GrowthODE(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
    model = ODEProjector(ode_core, hidden_dim).to(DEVICE)
    
    model_path = f"weights/ode_{target}_{condition}.pth"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # === Helper Inference ===
    def run_inference(df_in):
        ds = AlgaeTimeSeriesDataset(df_in, feature_cols, target, group_col=group_col)
        dl = DataLoader(ds, batch_size=len(ds), collate_fn=collate_ode_batch, shuffle=False)
        
        all_preds = []
        all_y = []
        
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                t = batch['times'].to(DEVICE) # [B, T]
                
                # Note: ODERNN forward expects [T] usually if strictly numerical.
                # But our collate returns [B, T].
                # Let's check GrowthODE forward. It just passes to ODERNN.
                # ODERNN forward(x, t, mask).
                # t needs to be passed correctly. 
                # If t is [B, T], code needs to support it. 
                # Assuming visualization script works, it expects t as [T] usually?
                # visualize_ode_latent passed: t = torch.arange(x.shape[1])
                # Let's stick to that index-based time for now as trained.
                t_idx = torch.arange(x.shape[1], dtype=torch.float32).to(DEVICE)
                
                pred_y = model(x, t_idx, mask) # [B, T, 1]
                
                # Flatten valid predictions
                mask_bool = mask.bool()
                valid_preds = pred_y.squeeze(-1)[mask_bool]
                valid_y = targets[mask_bool]
                
                all_preds.append(valid_preds.cpu().numpy())
                all_y.append(valid_y.cpu().numpy())
                
        return np.concatenate(all_preds), np.concatenate(all_y)

    # === Original ===
    print("Predicting Original...")
    pred_orig, y_true = run_inference(df_test)
    
    # Inverse transform for R2
    pred_orig_inv = scaler.inverse_transform(pred_orig.reshape(-1, 1)).flatten()
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
    
    r2_orig = r2_score(y_true_inv, pred_orig_inv)
    print(f"[Original] R2: {r2_orig:.4f}")
    
    # === Shuffled ===
    print("Predicting Shuffled...")
    df_shuf = df_test.copy()
    # Shuffle feature cols
    for c in feature_cols:
        df_shuf[c] = np.random.permutation(df_shuf[c].values)
    
    pred_shuf, _ = run_inference(df_shuf)
    pred_shuf_inv = scaler.inverse_transform(pred_shuf.reshape(-1, 1)).flatten()
    
    r2_shuf = r2_score(y_true_inv, pred_shuf_inv)
    print(f"[Shuffled] R2: {r2_shuf:.4f}")
    
    if r2_shuf > r2_orig - 0.1:
        print("\nCONCLUSION: ODE Model is ROBUST. It relies on Time/Dynamics, not individual features.")
    else:
        print(f"\nCONCLUSION: ODE Model Collapsed (Delta={r2_orig - r2_shuf:.2f}). It uses features!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='Dry_Weight')
    parser.add_argument('--condition', type=str, default='Light')
    args = parser.parse_args()
    
    test_ode_counterfactual(args.target, args.condition)
