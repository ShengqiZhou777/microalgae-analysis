import os
import random
import json
import joblib
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from algae_fusion.config import DEVICE, BATCH_SIZE, WINDOW_SIZE, RANDOM_SEED
# from torchvision.transforms import transforms # Removed as logic moved to transforms.py
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer

# from algae_fusion.data.dataset import MaskedImageDataset # Removed
# from algae_fusion.models.cnn import ResNetRegressor # Moved to image_engine
# from algae_fusion.models.tabular import XGBoostExpert, LightGBMExpert # Moved to tabular_engine
from algae_fusion.models.moe import GatingNetwork
from algae_fusion.models.ode import GrowthODE, NeuralODEParameterizer
from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
from algae_fusion.engine.trainer import train_epoch, eval_epoch # Still used? Check usages.

from algae_fusion.utils.model_utils import set_seed, Log1pScaler, StandardWrapper, save_pipeline_models
from algae_fusion.data.processing import (
    get_all_features, prepare_modeling_data, load_and_split_data, get_morph_features
)
from algae_fusion.data.transforms import get_transforms
from algae_fusion.engine.tabular_engine import run_boost_training, run_ode_training
from algae_fusion.engine.image_engine import run_image_training



def run_pipeline(target_name="Dry_Weight", mode="full", stochastic_window=False, condition=None, window_size=None, population_mean=False, ode_window_size=None):
    """
    Orchestrate the full training pipeline.
    Args:
        target_name: The regression target column.
        mode: "full" (all features), "cnn_only", or "boost_only".
        stochastic_window: If True, uses randomized history matching.
        condition: If None, train on both. If "Light" or "Dark", train specific.
        window_size: Sliding window size (history length) for CNN/Tabular. Defaults to config value.
        ode_window_size: Sliding window size for ODE mode (if set, cuts trajectories).
    """

    # [MODE SWITCH for ODE]
    # If mode is ODE, we delegate strictly to the ODE Engine and return.
    # We do duplication of data loading for now to decouple, OR we load data here and pass it.
    # To keep pipeline clean, let's load data here and pass it.
    # [CONFIG] Window Size for History
    if window_size is None:
        window_size = WINDOW_SIZE
    
    # Set seed for reproducibility
    set_seed(RANDOM_SEED)
    
    # Window=3 means we look back at t-3, t-2, t-1 (plus current t)
    # Total Channels = (3+1) * 3 = 12
    # in_channels calculation uses window_size
    
    in_channels = (window_size + 1) * 3 if stochastic_window else 3
    print(f"\n\n{'='*40}")
    status = "DYNAMIC" if stochastic_window else "STATIC"
    print(f"STARTING PIPELINE: Target={target_name}, Mode={mode} | {status} (Win={window_size}, Ch={in_channels})")
    print(f"{'='*40}\n")

    # --- [DATA LOADING] ---
    df_train, df_val, df_test = load_and_split_data(condition=condition)
    print(f"  [Split] Loaded Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}")
    
    # Identify morphological columns (needed for history generation)
    morph_cols = get_morph_features(df_train)
    
    # --- [TRANSFORMS] ---
    train_transform = get_transforms(split='train')
    val_transform = get_transforms(split='val')
    
    # helper for stochastic history
    def apply_history(df_in):
        if stochastic_window:
             return compute_sliding_window_features_stochastic(df_in, window_size=window_size, morph_cols=morph_cols)
        return df_in

    
    # 3. Apply History (if stochastic)
    df_train = apply_history(df_train)
    df_val = apply_history(df_val)
    df_test = apply_history(df_test)
    
    # --- [DATA PREP] ---
    # Prepare features and adaptive scaling for targets
    X_tab_train, X_tab_val, y_train_scaled, y_val_scaled, target_scaler, tab_cols = prepare_modeling_data(df_train, df_val, target_name)
    
    # [FIX] Define original validation targets for gating/metrics
    y_val_orig = df_val[target_name].values

    # --- [TABULAR ENGINE] ---
    # Runs Layer 1 (Stacking) and Layer 2 (Prediction) for XGB/LGB
    tab_results = run_boost_training(X_tab_train, X_tab_val, y_train_scaled, target_scaler, mode=mode)
    
    val_preds_xgb = tab_results['val_preds_xgb']
    val_preds_lgb = tab_results['val_preds_lgb']
    models = tab_results['models']

    img_results = run_image_training(
        df_train=df_train,
        df_val=df_val,
        target_name=target_name,
        train_transform=train_transform,
        val_transform=val_transform,
        y_train_scaled=y_train_scaled,
        y_val_scaled=y_val_scaled,
        target_scaler=target_scaler,
        mode=mode,
        window_size=window_size,
        stochastic_window=stochastic_window
    )
    
    val_preds_cnn = img_results['val_preds_cnn']
    cnn = img_results['model']


    if mode == "ode":
        # We need y_test_scaled as well for ODE
        # Lazy calc if not available? prepare_modeling_data gave us y_val_scaled.
        # But prepare_modeling_data doesn't return y_test_scaled.
        # We can quickly compute it here using target_scaler.
        run_ode_training(
            df_train=df_train,
            df_val=df_val,
            df_test=df_test,
            target_name=target_name,
            condition=condition if condition else "All", # Ensure string for file naming
            target_scaler=target_scaler,
            ode_window_size=ode_window_size
        )
        return


    # --- MoE Gating (Train on Validation Set) ---
    # We use the predictions on the validation set as inputs to the Gater
    from algae_fusion.engine.moe_engine import train_and_fuse_experts
    from algae_fusion.utils.visualization import visualize_results

    gating_cols = get_all_features(df_train)
    
    # Stack Available Experts Dynamically
    # This design allows for any number of experts (Tabular, CNN, ODE, Transformer, etc.)
    expert_preds_list = []
    
    # 1. Tabular Experts
    if 'val_preds_xgb' in locals() and np.any(val_preds_xgb): 
        expert_preds_list.append(val_preds_xgb)
        print("  [MoE] Added Tabular Expert: XGB")
    if 'val_preds_lgb' in locals() and np.any(val_preds_lgb): 
        expert_preds_list.append(val_preds_lgb)
        print("  [MoE] Added Tabular Expert: LGB")
        
    # 2. Image Experts
    if 'val_preds_cnn' in locals() and np.any(val_preds_cnn):
        expert_preds_list.append(val_preds_cnn)
        print("  [MoE] Added Image Expert: CNN")
    
    if not expert_preds_list:
        print("  [Warning] No experts available for MoE fusion. Utilizing mean of dummy zeros (Bad state).")
        expert_preds_list = [np.zeros(len(df_val))]
    
    # [MODEL SAVING PREFIX SETUP]
    os.makedirs("weights", exist_ok=True)
    suffix = "stochastic" if stochastic_window else "mean"
    cond_str = condition if condition else "All"
    model_prefix = f"weights/{target_name}_{cond_str}_{suffix}"
    
    final_valid_pred, g_net, gating_scaler = train_and_fuse_experts(
        df_val=df_val,
        target_name=target_name,
        expert_preds_list=expert_preds_list,
        gating_feature_cols=gating_cols,
        y_val_orig=y_val_orig,
        save_prefix=model_prefix # Integrates saving internally
    )

    # Save results (Using Validation IDs)
    history_str = "Dynamic" if stochastic_window else "Static"
    output_dir = os.path.join("output", target_name, mode, history_str, cond_str)
    os.makedirs(output_dir, exist_ok=True)
    
    df_val_for_save = df_val.copy()
    df_val_for_save[f"Predicted_{target_name}"] = final_valid_pred
    
    # [OPTIMIZATION] Save individual experts for One-Shot Ablation Study
    if mode == "full":
        df_val_for_save[f"Pred_{target_name}_XGB"] = val_preds_xgb
        df_val_for_save[f"Pred_{target_name}_LGB"] = val_preds_lgb
        df_val_for_save[f"Pred_{target_name}_CNN"] = val_preds_cnn

    result_csv = os.path.join(output_dir, "predictions_oof.csv")
    df_val_for_save.to_csv(result_csv, index=False)
    
    from sklearn.metrics import r2_score
    score = r2_score(y_val_orig, final_valid_pred)
    print(f"R2 Score: {score:.4f}")

    # Append to Summary CSV
    os.makedirs("results", exist_ok=True)
    summary_file = os.path.join("results", "training_summary.csv")
    if not os.path.exists(summary_file):
        with open(summary_file, "w") as f:
            f.write("Target,Mode,History,R2_Score,Path\n")
            
    with open(summary_file, "a") as f:
        f.write(f"{target_name},{mode},{history_str},{score:.4f},{result_csv}\n")
    
    # Generate Visualization (External Call)
    visualize_results(result_csv, target_name, output_dir)
    
    # --- MODEL SAVING (Experts) ---
    from algae_fusion.utils.model_utils import save_pipeline_models
    
    metadata = {
        'target_name': target_name,
        'feature_cols': get_all_features(df_train),
        'gating_cols': gating_cols,
        'stochastic': stochastic_window,
        'window_size': window_size,
        'mode': mode
    }
    
    save_pipeline_models(
        mode=mode,
        model_prefix=model_prefix,
        xgb=models.get('xgb'),
        lgb=models.get('lgb'),
        cnn=cnn if 'cnn' in locals() else None,
        target_scaler=target_scaler,
        metadata=metadata
    )
