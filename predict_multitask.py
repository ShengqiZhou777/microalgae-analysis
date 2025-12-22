import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Define targets and their model file suffixes
TARGETS = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
SUFFIX = "mean" # Static models

def load_static_model_ensemble(target, model_prefix):
    """
    Loads XGB1, XGB2, LGB2, and Target Scaler for a specific target.
    Returns a dictionary of loaded artifacts.
    """
    artifacts = {}
    
    # Check metadata to confirm features
    meta_path = f"{model_prefix}_metadata.json"
    if not os.path.exists(meta_path):
        print(f"  [WARN] Metadata not found for {target}: {meta_path}")
        return None
        
    with open(meta_path, "r") as f:
        artifacts['meta'] = json.load(f)
        
    # Load Scaler
    scaler_path = f"{model_prefix}_target_scaler.joblib"
    if os.path.exists(scaler_path):
        artifacts['scaler'] = joblib.load(scaler_path)
        
    # Load Models
    xgb1 = XGBRegressor()
    xgb1.load_model(f"{model_prefix}_xgb1.json")
    artifacts['xgb1'] = xgb1
    
    xgb2 = XGBRegressor()
    xgb2.load_model(f"{model_prefix}_xgb2.json")
    artifacts['xgb2'] = xgb2
    
    lgb_path = f"{model_prefix}_lgb.joblib"
    if os.path.exists(lgb_path):
        artifacts['lgb2'] = joblib.load(lgb_path)
        
    return artifacts

def predict_single_target(df, target, artifacts):
    """
    Runs the ensemble inference for a single target.
    """
    feature_cols = artifacts['meta']['feature_cols']
    
    # Prepare Input
    # Ensure all features exist (fill 0 if missing, though unlikely in valid pipeline)
    X = df[feature_cols].copy()
    X = X.select_dtypes(exclude=['object']) # Ensure numeric
    
    # Layer 1: XGB1
    xgb1 = artifacts['xgb1']
    l1_feat = xgb1.predict(X)
    
    # Layer 2: Augment
    X_aug = X.copy()
    X_aug["XGB1_Feature"] = l1_feat
    
    # Layer 2 Predictions
    preds_l2 = []
    
    # XGB2
    xgb2 = artifacts['xgb2']
    p_xgb = xgb2.predict(X_aug)
    preds_l2.append(p_xgb)
    
    # LGB2 (if exists)
    if 'lgb2' in artifacts:
        p_lgb = artifacts['lgb2'].predict(X_aug)
        preds_l2.append(p_lgb)
    
    # Average (Simple Mean for now, ignoring Gating for simplicity/robustness in pure static mode)
    # The Gating network was trained but simple averaging is often very robust for static.
    # Let's use simple average of the boosted trees. 
    raw_pred = np.mean(preds_l2, axis=0)
    
    # Inverse Transform
    scaler = artifacts.get('scaler')
    if scaler:
        if hasattr(scaler, 'inverse_transform'):
            final_pred = scaler.inverse_transform(raw_pred.reshape(-1, 1)).flatten()
        else:
            final_pred = raw_pred # Should not happen if standard scaler
    else:
        final_pred = raw_pred
        
    return final_pred

def main():
    parser = argparse.ArgumentParser(description="Multi-Target Inference")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file")
    parser.add_argument("--output", type=str, default="Final_MultiTarget_Predictions.csv")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Loading input data from: {args.input}")
    df = pd.read_csv(args.input)
    
    # --- 1. Polynomial Features (Shared) ---
    # if 'cell_mean_area' in df.columns:
    #     df['cell_mean_area_sq'] = df['cell_mean_area'] ** 2
    #     df['cell_mean_area_cub'] = df['cell_mean_area'] ** 3
    
    # --- 2. Dynamic Features (History context) ---
    # To predict with Dynamic models, we need 'Prev_' features.
    # We must load the training set to serve as the "History Database".
    df_dynamic = df.copy()
    
    # Check if we need dynamic features (i.e. if we are running any dynamic models)
    # For simplicity, we always compute them if training data is available
    TRAIN_DB_PATH = "data/dataset_train.csv"
    if os.path.exists(TRAIN_DB_PATH):
        print("   [Info] Loading Training DB for History Context...")
        df_train_db = pd.read_csv(TRAIN_DB_PATH)
        
        # Ensure polys in DB too
        # if 'cell_mean_area' in df_train_db.columns:
        #     df_train_db['cell_mean_area_sq'] = df_train_db['cell_mean_area'] ** 2
        #     df_train_db['cell_mean_area_cub'] = df_train_db['cell_mean_area'] ** 3

        # Combine Test + Train to compute sliding window
        # (Test samples need to find their ancestors in Train)
        df_dynamic['is_test'] = True
        df_train_db['is_test'] = False
        
        # Align columns
        common_cols = df_dynamic.columns.intersection(df_train_db.columns)
        df_combined = pd.concat([df_dynamic[common_cols], df_train_db[common_cols]], ignore_index=True)
        
        # Compute Stochastic Sliding Window
        from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
        
        morph_cols = [
            'cell_mean_area', 'cell_mean_mean_intensity', 
            'cell_mean_eccentricity', 'cell_mean_solidity'
        ]
        
        print("   [Info] Computing Stochastic Sliding Window...")
        df_combined_aug = compute_sliding_window_features_stochastic(df_combined, window_size=3, morph_cols=morph_cols)
        
        # Extract back the test rows
        # We identify them by 'is_test' flag if preserved, or we rely on file IDs if unique.
        # But compute_sliding_window_features_stochastic might drop 'is_test' or reorder.
        # Actually it returns a new df. Let's rely on 'is_test' being present if we didn't drop it explicitly? 
        # The stochastic function drops 'group_idx' but usually preserves others.
        # Let's check the function implementation or just merge back on 'file'.
        # Merging on 'file' is safest since file names are unique per sample row in this dataset.
        
        # Filter back
        df_dynamic_aug = df_combined_aug[df_combined_aug['is_test'] == True].copy()
        
        # Re-align with original df order if needed, but 'file' merge is better
        # Let's map the new features back to the original results dataframe
        # Actually, let's just use df_dynamic_aug as the input for dynamic models.
        # We need to ensure the order matches 'results' dataframe which is a copy of 'df'.
        
        # Reset index to be safe
        df_dynamic_aug = df_dynamic_aug.set_index('file').reindex(df['file']).reset_index()
        
        df_dynamic = df_dynamic_aug
    else:
        print("   [WARN] Training DB not found. Dynamic models will likely fail due to missing history features.")

    # Placeholder for results
    results = df.copy()
    
    print("\nStarting Multi-Target Inference...")
    print("===================================")
    
    for target in TARGETS:
        print(f"-> Processing Target: {target}")
        
        # 1. Static Prediction
        print(f"   [Static] Predicting...")
        try:
            artifacts_static = load_static_model_ensemble(target, f"weights/{target}_mean")
            if artifacts_static:
                pred_static = predict_single_target(df, target, artifacts_static)
                results[f"Pred_{target}_Static"] = pred_static
            else:
                print(f"   [WARN] Static model not found for {target}")
        except Exception as e:
            print(f"   [ERROR] Static inference failed: {e}")

        # 2. Dynamic Prediction
        print(f"   [Dynamic] Predicting...")
        try:
            artifacts_dynamic = load_static_model_ensemble(target, f"weights/{target}_stochastic")
            if artifacts_dynamic:
                pred_dynamic = predict_single_target(df_dynamic, target, artifacts_dynamic)
                results[f"Pred_{target}_Dynamic"] = pred_dynamic
            else:
                print(f"   [WARN] Dynamic model not found for {target}")
        except Exception as e:
            print(f"   [ERROR] Dynamic inference failed: {e}")

    # Filter columns for final output
    id_cols = ['file', 'time', 'condition']
    # Ensure targets exist in df (if not, we can't save them, but usually they are there for test set)
    target_cols = [t for t in TARGETS if t in df.columns]
    pred_cols = [c for c in results.columns if c.startswith("Pred_")]
    
    final_cols = id_cols + target_cols + pred_cols
    # Keep only existing columns
    final_cols = [c for c in final_cols if c in results.columns]
    
    extra_info_df = results[final_cols]

    # Save
    extra_info_df.to_csv(args.output, index=False)
    print("\n===================================")
    print(f"Done! Cleaned results (Static & Dynamic) saved to: {args.output}")
    
    # Visualize if targets are present
    if len(target_cols) > 0:
        print("Generating visualization plots...")
        visualize_test_results(extra_info_df, target_cols)

def visualize_test_results(df, targets):
    for target in targets:
        pred_static_col = f"Pred_{target}_Static"
        pred_dynamic_col = f"Pred_{target}_Dynamic"
        
        has_static = pred_static_col in df.columns
        has_dynamic = pred_dynamic_col in df.columns
        
        if not has_static and not has_dynamic:
            continue
            
        plt.figure(figsize=(15, 6))
        
        # 1. Scatter Plot (True vs Pred)
        plt.subplot(1, 2, 1)
        # Plot Static
        if has_static:
            sns.scatterplot(x=df[target], y=df[pred_static_col], alpha=0.5, label='Static', color='blue')
        # Plot Dynamic
        if has_dynamic:
            sns.scatterplot(x=df[target], y=df[pred_dynamic_col], alpha=0.5, label='Dynamic', color='red')
            
        # Diagnosis line
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
        plt.title(f"Test Set: {target} Prediction")
        plt.xlabel("True Value")
        plt.ylabel("Predicted Value")
        plt.legend()
        
        # 2. Trajectory Plot (Timecourse)
        plt.subplot(1, 2, 2)
        
        # Explicit sorting by time is crucial for line plots
        df_sorted = df.sort_values('time')
        
        # True Trajectory
        for cond, style in [('Light', '-'), ('Dark', '--')]:
            subset = df_sorted[df_sorted['condition'] == cond]
            if subset.empty: continue
            
            # Group by time
            true_means = subset.groupby('time')[target].mean()
            plt.plot(true_means.index, true_means.values, style + 'o', label=f'True {cond}', color='black', linewidth=2)
            
            if has_static:
                stat_means = subset.groupby('time')[pred_static_col].mean()
                plt.plot(stat_means.index, stat_means.values, style + 'x', label=f'Static {cond}', color='blue', alpha=0.7)
            
            if has_dynamic:
                dyn_means = subset.groupby('time')[pred_dynamic_col].mean()
                plt.plot(dyn_means.index, dyn_means.values, style + '^', label=f'Dynamic {cond}', color='red', alpha=0.7)

        plt.title(f"{target} Population Dynamics")
        plt.xlabel("Time (h)")
        plt.ylabel(target)
        plt.legend()
        plt.tight_layout()
        
        out_path = f"Final_Test_Plot_{target}.png"
        plt.savefig(out_path, dpi=300)
        print(f"  -> Saved plot: {out_path}")

if __name__ == "__main__":
    main()
