
import io
import os
import sys
import torch
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from algae_fusion.config import DEVICE, NON_FEATURE_COLS
from algae_fusion.models.ode import GrowthODE, ODEProjector
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch
from algae_fusion.engine.pipeline import Log1pScaler, StandardWrapper

def visualize_latent(target_name, condition):
    print(f"\n Visualizing ODE Latent Manifold for {target_name} ({condition})...")
    
    # 1. Load Data
    test_csv = "data/dataset_test.csv"
    if not os.path.exists(test_csv):
        print(f"Error: {test_csv} not found.")
        return
        
    df_test = pd.read_csv(test_csv)
    if condition != "All":
        df_test = df_test[df_test['condition'] == condition].reset_index(drop=True)
        if len(df_test) == 0:
            print(f"No data found for condition {condition}")
            return
            
    # 2. Load Scaler & Scale Targets
    # Note: weights are saved in 'weights/' directory
    scaler_path = f"weights/ode_{target_name}_{condition}_scaler.joblib"
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler not found at {scaler_path}. Please re-run training first.")
        return

    print(f"  -> Loading Scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    # Apply scaling to target col
    # We need to simulate the pipeline's behavior
    vals = df_test[target_name].values
    df_test[target_name] = scaler.transform(vals)
    
    # 3. Create Dataset
    cols = [c for c in df_test.columns if c not in NON_FEATURE_COLS + [target_name]]
    group_col = 'group_idx' if 'group_idx' in df_test.columns else 'file'
    
    ds = AlgaeTimeSeriesDataset(df_test, cols, target_name, group_col=group_col)
    loader = DataLoader(ds, batch_size=len(ds), collate_fn=collate_ode_batch)
    
    # 4. Load Model
    input_dim = len(cols)
    latent_dim = 32 # Matched with pipeline.py
    
    ode_core = GrowthODE(input_dim=input_dim, hidden_dim=latent_dim).to(DEVICE)
    model = ODEProjector(ode_core, latent_dim).to(DEVICE)
    
    model_path = f"weights/ode_{target_name}_{condition}.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return

    print(f"  -> Loading Model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # 5. Extract Latent States
    batch = next(iter(loader))
    x = batch['features'].to(DEVICE)
    mask = batch['mask'].to(DEVICE)
    # Create time indices [0, 1, 2, ...] corresponding to sequence length
    # Note: ODERNN expects time points. Here we assume uniform spacing.
    t = torch.arange(x.shape[1], dtype=torch.float32).to(DEVICE)
    
    print("  -> Extracting Latent States...")
    with torch.no_grad():
        # Get latent states [B, T, 32]
        h = model.get_latent(x, t, mask)
        
    # 6. Dimensionality Reduction (PCA)
    mask_np = mask.cpu().numpy().astype(bool)
    h_np = h.cpu().numpy()
    
    # Flatten valid states for PCA training
    # We only care about valid timepoints
    h_flat = h_np[mask_np]
    
    print(f"  -> Fitting PCA on {h_flat.shape} samples...")
    pca = PCA(n_components=2)
    h_pca_flat = pca.fit_transform(h_flat)
    
    # 7. Visualization
    plt.figure(figsize=(10, 8))
    
    # Plot background trajectories (gray lines)
    # We need to transform each sequence individually using the fitted PCA to keep structure
    cmap = plt.cm.viridis
    
    print("  -> Plotting Manifold...")
    
    # Scatter plot of all points first (for colorbar)
    # We use times from batch. Assumes padded_t.
    # We need strictly valid times.
    
    all_times = []
    all_pcs = []
    
    for i in range(h_np.shape[0]):
        length = int(mask[i].sum().item())
        seq_h = h_np[i, :length, :] # [T_i, 32]
        seq_pca = pca.transform(seq_h) # [T_i, 2]
        
        # Plot trajectory line
        plt.plot(seq_pca[:, 0], seq_pca[:, 1], color='gray', alpha=0.15, linewidth=1)
        
        # Store for scatter
        seq_times = batch['times'][i][:length].numpy()
        all_times.append(seq_times)
        all_pcs.append(seq_pca)
        
    all_times = np.concatenate(all_times)
    all_pcs = np.concatenate(all_pcs)
    
    sc = plt.scatter(all_pcs[:, 0], all_pcs[:, 1], c=all_times, cmap=cmap, s=15, alpha=0.8)
    
    # Draw Mean Trajectory
    # Bin times and calculate mean PC position
    # Assuming integer hours roughly
    unique_times = np.unique(np.round(all_times))
    mean_traj = []
    for t_val in unique_times:
        # Find points close to this time
        indices = np.where(np.abs(all_times - t_val) < 0.5)
        if len(indices[0]) > 0:
            center = np.mean(all_pcs[indices], axis=0)
            mean_traj.append(center)
    mean_traj = np.array(mean_traj)
    
    # Plot Mean Trajectory with Arrows
    if len(mean_traj) > 1:
        plt.plot(mean_traj[:, 0], mean_traj[:, 1], 'k--', linewidth=2, label='Mean Flow')
        # Add arrows
        for j in range(len(mean_traj)-1):
            if j % 2 == 0: # Every other step
                plt.arrow(mean_traj[j, 0], mean_traj[j, 1], 
                          mean_traj[j+1, 0]-mean_traj[j, 0], mean_traj[j+1, 1]-mean_traj[j, 1],
                          shape='full', lw=0, length_includes_head=True, head_width=0.02, color='k')

    cbar = plt.colorbar(sc)
    cbar.set_label('Time (h)')
    
    var_ratio = pca.explained_variance_ratio_
    plt.xlabel(f'Latent PC1 ({var_ratio[0]*100:.1f}%)')
    plt.ylabel(f'Latent PC2 ({var_ratio[1]*100:.1f}%)')
    plt.title(f'Neural ODE Growth Manifold: {target_name} ({condition})\nTotal Explained Variance: {var_ratio.sum()*100:.1f}%')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # ... (Manifold Plotting code remains same until savefig) ...
    
    os.makedirs("results/ode_manifold", exist_ok=True)
    save_path = f"results/ode_manifold/{target_name}_{condition}_manifold.png"
    plt.savefig(save_path, dpi=150)
    print(f"✅ Manifold visualization saved to {save_path}")
    plt.close()

    # === NEW: Feature Contribution Analysis ===
    print("\n  -> Analyzing Feature Contributions (Correlation with PCs)...")
    
    # 1. Flatten Inputs corresponding to Valid Latent States
    # x: [B, T, D] -> flattened to [N_valid, D]
    x_np = x.cpu().numpy()
    x_flat = x_np[mask_np] # [N_valid, Input_Dim]
    
    # h_pca_flat: [N_valid, 2] (PC1, PC2)
    
    # 2. Calculate Correlation
    n_features = x_flat.shape[1]
    feature_names = cols
    
    correlations = []
    for i in range(n_features):
        feat_vec = x_flat[:, i]
        # Skip constant features
        if np.std(feat_vec) < 1e-6:
            correlations.append((0, 0))
            continue
            
        # Pearson correlation with PC1 and PC2
        corr_pc1 = np.corrcoef(feat_vec, h_pca_flat[:, 0])[0, 1]
        corr_pc2 = np.corrcoef(feat_vec, h_pca_flat[:, 1])[0, 1]
        correlations.append((corr_pc1, corr_pc2))
        
    df_corr = pd.DataFrame(correlations, columns=['PC1', 'PC2'], index=feature_names)
    
    # 3. Top Contributors
    top_pc1 = df_corr['PC1'].abs().sort_values(ascending=False).head(10)
    top_pc2 = df_corr['PC2'].abs().sort_values(ascending=False).head(10)
    
    print("\n  [Top Features driving PC1 (Horizontal Axis - Growth?)]")
    print(df_corr.loc[top_pc1.index, 'PC1'])
    
    print("\n  [Top Features driving PC2 (Vertical Axis - State Shift?)]")
    print(df_corr.loc[top_pc2.index, 'PC2'])
    
    # 4. Plot Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # PC1 Plot
    y_pos = np.arange(len(top_pc1))
    axes[0].barh(y_pos, df_corr.loc[top_pc1.index, 'PC1'], align='center', color='skyblue')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(top_pc1.index)
    axes[0].invert_yaxis()  # labels read top-to-bottom
    axes[0].set_title(f"Top Drivers of PC1 (Explained Var: {pca.explained_variance_ratio_[0]*100:.1f}%)")
    axes[0].set_xlabel("Pearson Correlation")
    
    # PC2 Plot
    y_pos = np.arange(len(top_pc2))
    axes[1].barh(y_pos, df_corr.loc[top_pc2.index, 'PC2'], align='center', color='salmon')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_pc2.index)
    axes[1].invert_yaxis()
    axes[1].set_title(f"Top Drivers of PC2 (Explained Var: {pca.explained_variance_ratio_[1]*100:.1f}%)")
    axes[1].set_xlabel("Pearson Correlation")
    
    plt.tight_layout()
    importance_path = f"results/ode_manifold/{target_name}_{condition}_feature_importance.png"
    plt.savefig(importance_path, dpi=150)
    print(f"✅ Feature importance plot saved to {importance_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="Dry_Weight")
    parser.add_argument("--condition", default="Dark")
    args = parser.parse_args()
    
    visualize_latent(args.target, args.condition)
