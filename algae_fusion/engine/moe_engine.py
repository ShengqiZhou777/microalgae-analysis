import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from algae_fusion.models.moe import GatingNetwork
from algae_fusion.config import DEVICE

def train_and_fuse_experts(
    df_val, 
    target_name, 
    expert_preds_list, 
    gating_feature_cols, 
    y_val_orig,
    save_prefix=None
):
    """
    Trains a Gating Network to fuse predictions from multiple experts.
    
    Args:
        df_val (pd.DataFrame): Validation dataframe containing metadata/features.
        target_name (str): Name of the target variable.
        expert_preds_list (list of np.array): List of prediction arrays from experts [N,].
        gating_feature_cols (list): List of column names to use as input for the Gating Net.
        y_val_orig (np.array): Ground truth values [N,].
        save_prefix (str): If provided, saves the Gating Net and Scaler to this prefix.
        
    Returns:
        final_valid_pred (np.array): Fused predictions [N,].
        g_net (nn.Module): The trained Gating Network.
        gating_scaler (StandardScaler): The fitted scaler for gating features.
    """
    
    num_experts = len(expert_preds_list)
    print(f"  [Gating] Training Gating Network with {num_experts} experts...")
    
    # 1. Prepare Gating Input Features
    # Ensure cols are numeric and fill NaNs
    X_gating_val = df_val[gating_feature_cols].select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
    
    # Scale Gating Features
    gating_scaler = StandardScaler()
    X_gating_val = gating_scaler.fit_transform(X_gating_val)
    
    # 2. Prepare Tensors
    E_preds_val = torch.tensor(np.vstack(expert_preds_list).T, dtype=torch.float32).to(DEVICE)
    y_g_val = torch.tensor(y_val_orig, dtype=torch.float32).to(DEVICE).view(-1, 1)
    X_g_val = torch.tensor(X_gating_val, dtype=torch.float32).to(DEVICE)
    
    # 3. Initialize Network
    g_net = GatingNetwork(input_dim=X_gating_val.shape[1], num_experts=num_experts).to(DEVICE)
    opt_g = optim.Adam(g_net.parameters(), lr=0.001)
    crit_g = nn.MSELoss()
    
    # 4. Train Loop (Simple fit on validation set as meta-learner)
    g_net.train()
    for _ in range(50): # 50 Epochs usually sufficient for convergence on small gating task
        weights = g_net(X_g_val)
        y_p = torch.sum(weights * E_preds_val, dim=1).view(-1, 1)
        loss = crit_g(y_p, y_g_val)
        opt_g.zero_grad()
        loss.backward()
        opt_g.step()
    
    # 5. Predict (Fusion)
    g_net.eval()
    with torch.no_grad():
        final_weights = g_net(X_g_val).cpu().numpy()
        # Weighted sum of expert preds
        final_valid_pred = np.sum(final_weights * np.vstack(expert_preds_list).T, axis=1)
        
    # 6. Save if requested
    if save_prefix:
        torch.save(g_net.state_dict(), f"{save_prefix}_gating.pth")
        joblib.dump(gating_scaler, f"{save_prefix}_gating_scaler.joblib")
        
    return final_valid_pred, g_net, gating_scaler
