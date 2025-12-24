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

from algae_fusion.config import IMG_SIZE, DEVICE, BACKBONE, NON_FEATURE_COLS
from algae_fusion.models.backbones import ResNetRegressor
from algae_fusion.models.moe import GatingNetwork
from algae_fusion.models.moe import GatingNetwork
from algae_fusion.data.dataset import MaskedImageDataset
from algae_fusion.models.lstm import MorphLSTM
from algae_fusion.engine.pipeline import prepare_lstm_tensor


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
        artifacts['lgb2'] = joblib.load(lgb_path)

    # 1.5 Load LSTM Expert
    # [REVERTED] LSTM removed.
    pass



    # 2. Load Visual Expert (CNN)
    cnn_path = f"{model_prefix}_cnn.pth"
    if os.path.exists(cnn_path):
        # Dynamically detect in_channels from the weight file
        state_dict = torch.load(cnn_path, map_location=DEVICE, weights_only=True)
        # Weight shape for conv1 is [out_channels, in_channels, k, k]
        actual_in_channels = state_dict['backbone.conv1.weight'].shape[1]
        
        artifacts['in_channels'] = actual_in_channels
        print(f"  [Info] Detected CNN in_channels: {actual_in_channels}")
        
        cnn = ResNetRegressor(BACKBONE, in_channels=actual_in_channels).to(DEVICE)
        cnn.load_state_dict(state_dict)
        cnn.eval()
        artifacts['cnn'] = cnn
    else:
        print(f"  [WARN] CNN weights not found: {cnn_path}. Expert will be missing.")

    # 3. Load Gating Network
    gating_path = f"{model_prefix}_gating.pth"
    if os.path.exists(gating_path):
        state_dict_g = torch.load(gating_path, map_location=DEVICE, weights_only=True)
        # Weight shape for net.0 is [hidden_dim, input_dim]
        actual_input_dim = state_dict_g['net.0.weight'].shape[1]
        
        print(f"  [Info] Gating Network input_dim: {actual_input_dim}")
        # Weight shape for net.0 is [hidden_dim, input_dim]
        actual_input_dim = state_dict_g['net.0.weight'].shape[1]
        
        # Infer num_experts from output layer 'net.6.weight' [num_experts, 32]
        if 'net.6.weight' in state_dict_g:
            num_experts = state_dict_g['net.6.weight'].shape[0]
        else:
            num_experts = 3 # fallback
        
        print(f"  [Info] Gating Network input_dim: {actual_input_dim}, num_experts: {num_experts}")
        g_net = GatingNetwork(input_dim=actual_input_dim, num_experts=num_experts).to(DEVICE)
        g_net.load_state_dict(state_dict_g)
        g_net.eval()
        artifacts['g_net'] = g_net
    else:
        print(f"  [WARN] Gating Network missing: {gating_path}")
            
    return artifacts

def predict_single_target(df, target, artifacts):
    """
    Runs the MoE inference for a single target.
    Returns: (final_predictions, expert_weights)
    """
    feature_cols = [c for c in artifacts['meta']['feature_cols'] if c != 'split_set']
    gating_cols = [c for c in artifacts['meta'].get('gating_cols', []) if c != 'split_set']
    
    # --- Prepare Tabular Inputs ---
    # Use reindex to handle missing columns gracefully, then filter for numeric
    X = df.reindex(columns=feature_cols).select_dtypes(include=[np.number]).fillna(0)
    
    # Check if we have the correct number of columns for XGB/LGB
    # If the network was trained on 364 features but X has 365, models might complain.
    # However, XGB/LGB models usually handle column sets appropriately if names match.
    
    # Layer 1: XGB1
    xgb1 = artifacts.get('xgb1')
    if xgb1:
        l1_feat = xgb1.predict(X)
    else:
        l1_feat = np.zeros(len(df))
    
    # Layer 2: Augment
    X_aug = X.copy()
    X_aug["XGB1_Feature"] = l1_feat

    # Layer 1.5: LSTM (Augment with Temporal Features)
    # [REVERTED]
    preds_map = {}

    # --- Run Experts ---



    
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
        in_channels = artifacts.get('in_channels', 3)
        ds = MaskedImageDataset(df, target, IMG_SIZE, transform=val_transform, labels=dummy_labels, in_channels=in_channels)
        
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
    
    if g_net:
        # Prepare Gating Input
        X_gate = df.reindex(columns=gating_cols).select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
        
        # Verify dimension
        if X_gate.shape[1] != g_net.net[0].weight.shape[1]:
            print(f"   [WARN] Gating input dimension mismatch! Data: {X_gate.shape[1]}, Model: {g_net.net[0].weight.shape[1]}")
            # This can happen if some 'numeric' columns in metadata aren't in this DF.
            # We don't have an easy fix but we'll try to proceed or use fallback.
            # Actually, reindex already padded with NaN/0, so only filtering by select_dtypes matters.
        
        if gating_scaler:
            X_gate = gating_scaler.transform(X_gate)
        
        X_gate_t = torch.tensor(X_gate, dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            weights = g_net(X_gate_t) # [N, 3]
            weights = weights.cpu().numpy()
            
    else:
        # Fallback: Simple Average of AVAILABLE experts
        print(f"  [WARN] Gating Network missing. Using simple average of active experts.")
        
        # Check which experts are active (artifacts loaded)
        # Order: XGB, LGB, [LSTM], CNN
        active_mask = [1 if artifacts.get('xgb2') else 0, 1 if artifacts.get('lgb2') else 0]
        if 'lstm' in artifacts: active_mask.append(1)
        active_mask.append(1 if artifacts.get('cnn') else 0)

        num_experts = len(active_mask)
        weights = np.zeros((len(df), num_experts))
        
        active_count = sum(active_mask)
        if active_count > 0:
            avg_weight = 1.0 / active_count
            for i in range(num_experts):
                if active_mask[i]:
                    weights[:, i] = avg_weight
        else:
             weights[:, 0] = 1.0 

    # --- Aggregate ---
    # --- Aggregate ---
    # --- Aggregate ---
    # Determine stacking based on Gating Weights dimension
    num_expected = weights.shape[1]
    
    # Base: XGB, LGB
    stack_list = [preds_map['xgb'], preds_map['lgb']]
    
    # If 4 experts: XGB, LGB, LSTM, CNN
    if num_expected == 4:
        lstm_p = preds_map.get('lstm', np.zeros(len(df)))
        stack_list.append(lstm_p)
        stack_list.append(preds_map['cnn'])
        
    # If 3 experts: XGB, LGB, CNN
    elif num_expected == 3:
        stack_list.append(preds_map['cnn'])
        
    # If 2 experts: XGB, LGB
    else:
        # Fallback or only 2
        pass
        
    # Ensure dimensions match
    # If mismatch (e.g. num_expected=5?), we might crash, but 2/3/4 covers our cases.

    
    E_preds = np.vstack(stack_list).T # [N, NumExperts]

    
    # Weighted Sum
    raw_final_pred = np.sum(weights * E_preds, axis=1)
    
    # --- [NEW] Return individual expert raw preds for analysis ---
    expert_raw = {
        'xgb': preds_map['xgb'],
        'lgb': preds_map['lgb'], 
        'cnn': preds_map.get('cnn', np.zeros(len(df)))
    }
    if 'lstm' in preds_map: expert_raw['lstm'] = preds_map['lstm']

    
    # --- Inverse Transform ---
    scaler = artifacts.get('scaler')
    if scaler:
        if hasattr(scaler, 'inverse_transform'):
            final_pred = scaler.inverse_transform(raw_final_pred.reshape(-1, 1)).flatten()
        else:
            final_pred = raw_final_pred
    else:
        final_pred = raw_final_pred
        
    return final_pred, weights, expert_raw

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
    
    # --- Pre-processing (Consistent with training pipeline) ---
    df['original_order'] = range(len(df))
    # df.loc[df['condition'] == 'Initial', 'condition'] = 'Light' # [REMOVED] Initial is now pre-split into Light/Dark in dataset
    
    # Sort and assign group_idx (Necessary for sequential matching)
    df = df.sort_values(by=['condition', 'time', 'file']).reset_index(drop=True)
    df['group_idx'] = df.groupby(['condition', 'time']).cumcount()
    
    # --- 1. Dynamic Features Context ---
    # Since the 20% test set is a complete slice of trajectories, 
    # we can compute features directly on the input.
    print("   [Info] Computing Stochastic Sliding Window on input data...")
    from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
    
    # Match pipeline.py logic: Automatically identify all morphological columns
    all_numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    morph_cols = [c for c in all_numeric_cols if c.startswith('cell_') and c not in NON_FEATURE_COLS]
    print(f"   [Info] Identified {len(morph_cols)} morphological features.")

    df_dynamic = compute_sliding_window_features_stochastic(df, window_size=2, morph_cols=morph_cols)
    
    # Ensure group_idx is retained (fix for previous drop bug)
    if 'group_idx' not in df_dynamic.columns:
        df_dynamic['group_idx'] = df_dynamic.groupby(['condition', 'time']).cumcount()

    results = df.copy()
    
    print("\nStarting Multi-Target MoE Inference...")
    print("======================================")
    
    for target in TARGETS:
        print(f"-> Processing Target: {target}")
        
    for target in TARGETS:
        print(f"-> Processing Target: {target}")
        
        # We need to process Light and Dark separately because they use different models now
        conditions = df['condition'].unique()
        combined_pred_static = []
        combined_pred_dynamic = []
        
        # We will iterate by condition subset
        for cond in conditions:
            if cond not in ["Light", "Dark"]:
                # If there are other conditions (e.g. Initial), we might map them or skip
                # Assuming 'Initial' is mapped or handled. If 'Initial' exists, usually mapped to Light in training?
                # In main.py we saw: # df.loc[df['condition'] == 'Initial', 'condition'] = 'Light' was commented out.
                # But typically Initial is treated as Light. Let's assume strict matching for now.
                # If the user trains with "Light" and "Dark", Initial (0h) might have been passed to both or one.
                # Let's try to map 'Initial' to 'Light' for inference if 'Light' model exists.
                working_cond = "Light" if cond == "Initial" else cond
            else:
                working_cond = cond
            
            print(f"   [Condition: {cond}] (Using Model: {working_cond})")
            subset_mask = df['condition'] == cond
            subset_df = df[subset_mask].copy()
            if subset_df.empty: continue
            
            # --- 1. Static Prediction ---
            model_path_static = f"weights/{target}_{working_cond}_mean"
            artifacts_static = load_moe_ensemble(target, model_path_static)
            
            if artifacts_static:
                p_s, w_s, exp_s = predict_single_target(subset_df, target, artifacts_static)
                
                # Store results aligned with index
                subset_df[f"Pred_{target}_Static"] = p_s
                subset_df[f"W_XGB_{target}_Static"] = w_s[:, 0]
                subset_df[f"W_LGB_{target}_Static"] = w_s[:, 1]
                subset_df[f"W_CNN_{target}_Static"] = w_s[:, 2]
                
                if exp_s['xgb'] is not None: subset_df[f"Pred_{target}_Static_XGB"] = exp_s['xgb']
                if exp_s['lgb'] is not None: subset_df[f"Pred_{target}_Static_LGB"] = exp_s['lgb']
                
                combined_pred_static.append(subset_df)
            else:
                print(f"      [WARN] Static model not found: {model_path_static}")
                # Append subset without preds locally, or handle missing
                combined_pred_static.append(subset_df) # Will lack cols

            # --- 2. Dynamic Prediction ---
            # Dynamic needs df_dynamic subset
            # Filter df_dynamic by file/time/condition matching the subset
            # Faster: use boolean mask on df_dynamic if indices align? No, separate DF.
            # We filter df_dynamic by the same condition
            subset_dyn_mask = df_dynamic['condition'] == cond
            subset_dyn = df_dynamic[subset_dyn_mask].copy()
            
            if not subset_dyn.empty:
                model_path_dynamic = f"weights/{target}_{working_cond}_stochastic"
                artifacts_dynamic = load_moe_ensemble(target, model_path_dynamic)
                
                if artifacts_dynamic:
                    p_d, w_d, exp_d = predict_single_target(subset_dyn, target, artifacts_dynamic)
                     # Create temp dataframe to merge
                    res_dyn = pd.DataFrame({
                        'file': subset_dyn['file'],
                        'condition': subset_dyn['condition'],
                        'time': subset_dyn['time'],
                        f"Pred_{target}_Dynamic": p_d,
                        f"W_XGB_{target}_Dynamic": w_d[:, 0],
                        f"W_LGB_{target}_Dynamic": w_d[:, 1],
                        f"W_CNN_{target}_Dynamic": w_d[:, 2]
                    })
                    if exp_d['xgb'] is not None: res_dyn[f"Pred_{target}_Dynamic_XGB"] = exp_d['xgb']
                    if exp_d['lgb'] is not None: res_dyn[f"Pred_{target}_Dynamic_LGB"] = exp_d['lgb']
                    
                    combined_pred_dynamic.append(res_dyn)
                else:
                    print(f"      [WARN] Dynamic model not found: {model_path_dynamic}")

        # --- Reassemble ---
        # 1. Static: Concatenate valid parts
        if combined_pred_static:
            full_static = pd.concat(combined_pred_static)
            # Update main results with static preds
            # We use 'file' as key or index
            # results is based on 'df'. full_static is subsets of 'df'.
            # We can map by index if we kept it? 
            # subset_df was a copy. Let's merge on file/condition/time.
            cols_to_merge = [c for c in full_static.columns if c.startswith("Pred_") or c.startswith("W_")]
            if cols_to_merge:
                full_static_cut = full_static[['file', 'condition', 'time'] + cols_to_merge]
                results = results.merge(full_static_cut, on=['file', 'condition', 'time'], how='left', suffixes=('', '_new'))
                # Handle possible duplicates if any (shouldn't be)
                
        # 2. Dynamic: Concatenate
        if combined_pred_dynamic:
            full_dynamic = pd.concat(combined_pred_dynamic)
            results = results.merge(full_dynamic, on=['file', 'condition', 'time'], how='left', suffixes=('', '_new'))

    # Calculate R2 if targets exist
    from sklearn.metrics import r2_score
    
    print("\n--------------------------------------")
    print("       FINAL TEST SET EVALUATION      ")
    print("--------------------------------------")
    
    for target in TARGETS:
        if target in df.columns:
            print(f"\n>>> Target: {target} <<<")
            
            # Static R2
            if f"Pred_{target}_Static" in results.columns:
                r2_static = r2_score(results[target], results[f"Pred_{target}_Static"])
                print(f"   [Static]  R2 Score: {r2_static:.4f}")
            
            # Dynamic R2
            if f"Pred_{target}_Dynamic" in results.columns:
                r2_dynamic = r2_score(results[target], results[f"Pred_{target}_Dynamic"])
                print(f"   [Dynamic] R2 Score: {r2_dynamic:.4f}")

    # Save
    out_dir = os.path.dirname(args.output)
    # Save Clean Version (Only IDs, Targets, Preds)
    clean_cols = ['file', 'time', 'condition'] + [t for t in TARGETS if t in results.columns]
    pred_cols = [c for c in results.columns if c.startswith("Pred_")]
    # Restore original order for output
    results = results.sort_values('original_order').drop(columns=['original_order'])
    
    results_clean = results[clean_cols + pred_cols].copy()
    results_clean.to_csv(args.output, index=False)
    
    # Also save full version for debugging
    full_path = args.output.replace(".csv", "_FULL.csv")
    results.to_csv(full_path, index=False)
    print(f"      (Full version with weights saved to: {full_path})")
    print(f"Done! MoE results saved to: {args.output}")
    
    # Visualize
    visualize_moe_results(results, TARGETS)

def visualize_moe_results(df, targets):
    from sklearn.metrics import r2_score
    for target in targets:
        pred_static_col = f"Pred_{target}_Static"
        pred_dynamic_col = f"Pred_{target}_Dynamic"
        
        has_static = pred_static_col in df.columns
        has_dynamic = pred_dynamic_col in df.columns
        has_truth = target in df.columns
        
        if not (has_static or has_dynamic):
            continue
            
        plt.figure(figsize=(14, 6))
        
        # 1. Scatter Plot
        plt.subplot(1, 2, 1)
        r2_str = ""
        
        if has_truth:
            if has_static:
                r2_s = r2_score(df[target], df[pred_static_col])
                sns.scatterplot(x=df[target], y=df[pred_static_col], alpha=0.5, label=f'Static (R2={r2_s:.3f})', color='blue')
                r2_str += f"S:{r2_s:.2f} "
            
            if has_dynamic:
                r2_d = r2_score(df[target], df[pred_dynamic_col])
                sns.scatterplot(x=df[target], y=df[pred_dynamic_col], alpha=0.5, label=f'Dynamic (R2={r2_d:.3f})', color='red', marker='X')
                r2_str += f"D:{r2_d:.2f}"
            
            lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
            plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
            plt.title(f"{target}: True vs Pred\n{r2_str}")
        else:
            plt.title(f"{target}: Predictions Distribution")
            if has_static: sns.histplot(df[pred_static_col], color='blue', alpha=0.3, label='Static')
            if has_dynamic: sns.histplot(df[pred_dynamic_col], color='red', alpha=0.3, label='Dynamic')
        
        plt.legend()
        
        # 3. Trajectory (Now 2)
        plt.subplot(1, 2, 2)
        if 'time' in df.columns and 'condition' in df.columns:
            df_sorted = df.sort_values('time')
            for cond, style in [('Light', '-'), ('Dark', '--')]:
                subset = df_sorted[df_sorted['condition'] == cond]
                if subset.empty: continue
                
                if has_truth:
                    true_means = subset.groupby('time')[target].mean()
                    plt.plot(true_means.index, true_means.values, style + 'o', label=f'True {cond}', color='black')
                
                if has_static:
                    stat_means = subset.groupby('time')[pred_static_col].mean()
                    plt.plot(stat_means.index, stat_means.values, style + 'x', label=f'Static {cond}', color='blue')
                
                if has_dynamic:
                    dyn_means = subset.groupby('time')[pred_dynamic_col].mean()
                    plt.plot(dyn_means.index, dyn_means.values, style + '*', label=f'Dynamic {cond}', color='red')
                    
            plt.title(f"{target} Population Dynamics")
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"MoE_Result_{target}.png", dpi=300)
        print(f"  -> Saved plot: MoE_Result_{target}.png")

if __name__ == "__main__":
    main()
