
import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from torch.utils.data import DataLoader
from torchvision import transforms

# Add project root to path
sys.path.append(os.getcwd())

try:
    from algae_fusion.config import NON_FEATURE_COLS, DEVICE, PATH_PREFIX
    from algae_fusion.models.backbones import ResNetRegressor
    from algae_fusion.models.moe import GatingNetwork
    from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
    from algae_fusion.data.dataset import MaskedImageDataset
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def verify_pipeline():
    print(">>> Verifying Model Input/Output Shapes using REAL DATA (data/dataset_test.csv)")
    
    # 1. Load Data
    try:
        df = pd.read_csv("data/dataset_test.csv")
        print(f"    Loaded Dataset Rows: {len(df)}")
    except Exception as e:
        print(f"[ERROR] Could not load dataset: {e}")
        return

    # Prepare Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # helper to print shape
    def verify_expert_io(name, model, x, expected_out_shape, is_torch=False):
        try:
            if is_torch:
                out = model(x)
                out_shape = tuple(out.shape)
            else:
                out = model.predict(x)
                out_shape = out.shape
                
            status = "[SUCCESS]" if out_shape == expected_out_shape else "[WARN]"
            print(f"    {status} {name}: Input {x.shape} -> Output {out_shape}")
        except Exception as e:
            print(f"    [ERROR] {name} Failed: {e}")

    # ==========================
    # PART A: STATIC MODE CHECK
    # ==========================
    print("\n" + "="*40)
    print("      PART A: STATIC MODE VERIFICATION")
    print("="*40)
    
    # Static Features: Just raw columns - blacklist
    static_cols = [c for c in df.columns if c not in NON_FEATURE_COLS and "Prev" not in c]
    X_static = df[static_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    
    print(f"[Data] Static Table Shape: {X_static.shape}")
    print(f"[Data] Static Features Count: {X_static.shape[1]}")
    
    # 1. Static XGB/LGB
    y_dummy = np.random.rand(len(X_static))
    print("\n--- Static Tabular Experts ---")
    
    xgb_static = xgb.XGBRegressor(n_estimators=1, max_depth=1)
    xgb_static.fit(X_static, y_dummy)
    verify_expert_io("Static XGB", xgb_static, X_static[:10], (10,), is_torch=False)
    
    lgb_static = lgb.LGBMRegressor(n_estimators=1, max_depth=1, verbose=-1)
    lgb_static.fit(X_static, y_dummy)
    verify_expert_io("Static LGB", lgb_static, X_static[:10], (10,), is_torch=False)
    
    # 2. Static CNN
    print("\n--- Static CNN Expert ---")
    STATIC_CHANNELS = 3
    ds_static = MaskedImageDataset(df.head(10), target_col='Dry_Weight', transform=transform, in_channels=STATIC_CHANNELS)
    dl_static = DataLoader(ds_static, batch_size=4)
    imgs_static, _ = next(iter(dl_static))
    
    print(f"[Data] Loaded Static Batch Images: {imgs_static.shape}")
    
    cnn_static = ResNetRegressor(backbone="resnet34", in_channels=STATIC_CHANNELS)
    # Mock forward
    verify_expert_io("Static CNN", cnn_static, imgs_static, (4, 1), is_torch=True)


    # ==========================
    # PART B: DYNAMIC MODE CHECK
    # ==========================
    print("\n" + "="*40)
    print("      PART B: DYNAMIC MODE VERIFICATION")
    print("="*40)
    
    # Dynamic Features: Run Stochastic generator
    # Must use FULL DF to ensure history exists
    print("[Data] Generating Dynamic Features (Window=3)...")
    feature_cols_for_hist = [c for c in df.columns if c not in NON_FEATURE_COLS and "Prev" not in c]
    
    df_dyn = compute_sliding_window_features_stochastic(df, window_size=3, morph_cols=feature_cols_for_hist)
    
    # Filter valid columns
    dyn_cols = [c for c in df_dyn.columns if c not in NON_FEATURE_COLS]
    X_dyn = df_dyn[dyn_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    
    print(f"[Data] Dynamic Table Shape: {X_dyn.shape}")
    print(f"[Data] Dynamic Features Count: {X_dyn.shape[1]}")
    
    # 3. Dynamic XGB/LGB
    print("\n--- Dynamic Tabular Experts ---")
    xgb_dyn = xgb.XGBRegressor(n_estimators=1, max_depth=1)
    xgb_dyn.fit(X_dyn, y_dummy) # y_dummy length matches df/X_dyn
    verify_expert_io("Dynamic XGB", xgb_dyn, X_dyn[:10], (10,), is_torch=False)
    
    lgb_dyn = lgb.LGBMRegressor(n_estimators=1, max_depth=1, verbose=-1)
    lgb_dyn.fit(X_dyn, y_dummy)
    verify_expert_io("Dynamic LGB", lgb_dyn, X_dyn[:10], (10,), is_torch=False)
    
    # 4. Dynamic CNN
    print("\n--- Dynamic CNN Expert ---")
    DYN_CHANNELS = 12 # (3+1)*3
    ds_dyn = MaskedImageDataset(df_dyn.head(10), target_col='Dry_Weight', transform=transform, in_channels=DYN_CHANNELS)
    dl_dyn = DataLoader(ds_dyn, batch_size=4)
    imgs_dyn, _ = next(iter(dl_dyn))
    
    print(f"[Data] Loaded Dynamic Batch Images: {imgs_dyn.shape}")
    
    cnn_dyn = ResNetRegressor(backbone="resnet34", in_channels=DYN_CHANNELS)
    verify_expert_io("Dynamic CNN", cnn_dyn, imgs_dyn, (4, 1), is_torch=True)
    
    # 5. Dynamic Gating
    print("\n--- Dynamic Gating Network ---")
    NUM_EXPERTS = 3 # XGB, LGB, CNN
    print(f"[Info] Gating Input Dim: {X_dyn.shape[1]}, Experts: {NUM_EXPERTS}")
    
    gater_dyn = GatingNetwork(input_dim=X_dyn.shape[1], num_experts=NUM_EXPERTS)
    # Pass tensor
    X_dyn_tensor = torch.tensor(X_dyn[:4]) 
    verify_expert_io("Gating Net", gater_dyn, X_dyn_tensor, (4, NUM_EXPERTS), is_torch=True)

if __name__ == "__main__":
    verify_pipeline()
