
import os
import sys
import torch
import torch.nn as nn
import joblib
import json
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

# Add project root
sys.path.append(os.getcwd())

from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.models.ode import GrowthODE
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch

def load_ode_config(target, condition):
    config_path = f"weights/ode_{target}_{condition}_config.json"
    defaults = {
        "latent_dim": 64,
        "ode_hidden_dim": 128,
        "decoder_hidden": 64,
        "decoder_dropout": 0.2,
    }
    if not os.path.exists(config_path):
        return defaults
    with open(config_path, "r") as config_file:
        config = json.load(config_file)
    return {**defaults, **config}

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
    ode_config = load_ode_config(target, condition)
    
    # ODEProjector is defined locally in pipeline.py, so we must redefine it here matching that structure
    class ODEProjector(nn.Module):
        def __init__(self, ode, latent_dim, decoder_hidden, decoder_dropout):
            super().__init__()
            self.ode = ode
            # MLP Decoder instead of linear for better mapping
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, decoder_hidden),
                nn.Tanh(),
                nn.Dropout(p=decoder_dropout), # Decoder Regularization
                nn.Linear(decoder_hidden, 1)
            )
        def forward(self, x, t, mask):
            # ODERNN returns [B, T, latent]
            h = self.ode.ode_net(x, t, mask) 
            return self.proj(h).squeeze(-1)

    input_dim = len(feature_cols)
    latent_dim = ode_config["latent_dim"]
    ode_hidden_dim = ode_config["ode_hidden_dim"]
    ode_core = GrowthODE(input_dim=input_dim, latent_dim=latent_dim, ode_hidden_dim=ode_hidden_dim).to(DEVICE)
    model = ODEProjector(
        ode_core,
        latent_dim,
        ode_config["decoder_hidden"],
        ode_config["decoder_dropout"],
    ).to(DEVICE)
    
    model_path = f"weights/ode_{target}_{condition}.pth"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # === Helper Inference ===
    def run_inference(df_in):
        time_scale = pd.to_numeric(df_in['time'], errors='coerce').max()
        time_scale = float(time_scale) if pd.notna(time_scale) and time_scale > 0 else 1.0
        ds = AlgaeTimeSeriesDataset(df_in, feature_cols, target, group_col=group_col)
        dl = DataLoader(ds, batch_size=len(ds), collate_fn=collate_ode_batch, shuffle=False)
        
        all_preds = []
        all_y = []
        
        with torch.no_grad():
            for batch in dl:
                x = batch['features'].to(DEVICE)
                mask = batch['mask'].to(DEVICE)
                targets = batch['targets'].to(DEVICE)
                t_grid = batch['times'][0].to(DEVICE) / time_scale
                
                pred_y = model(x, t_grid, mask) # [B, T, 1]
                
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
