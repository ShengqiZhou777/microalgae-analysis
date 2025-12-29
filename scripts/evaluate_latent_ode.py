import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(os.getcwd()) # Ensure root is in path

from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from algae_fusion.config import *
from algae_fusion.data.latent_ode_dataset import LatentODEDataset, collate_set_ode
from algae_fusion.models.latent_ode import CellPopulationODE

# Helper Wrapper (Same as pipeline)
class StandardWrapper:
    def __init__(self, s): self.s = s
    def transform(self, y): return self.s.transform(y.reshape(-1, 1)).flatten()
    def inverse_transform(self, yp): return self.s.inverse_transform(yp.reshape(-1, 1)).flatten()

def evaluate(target_name="Chl_Per_Cell"):
    print(f"--- Evaluating Latent ODE for {target_name} ---")
    
    # 1. Load Data to Re-Fit Scaler and Get Ground Truth
    path_train = "data/dataset_train.csv"
    path_val = "data/dataset_val.csv"
    path_test = "data/dataset_test.csv"
    
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)
    
    # Re-Fit Scaler (as done in pipeline)
    y_train_orig = df_train[target_name].values
    scaler_target = StandardScaler()
    scaler_target.fit(y_train_orig.reshape(-1, 1))
    pt_fold = StandardWrapper(scaler_target)
    
    # Transform Data
    df_train[target_name] = pt_fold.transform(df_train[target_name].values)
    df_val[target_name] = pt_fold.transform(df_val[target_name].values)
    df_test[target_name] = pt_fold.transform(df_test[target_name].values)
    
    # Features
    input_sample = df_train.drop(columns=[target_name, 'split_set', 'Source_Path', 'file', 'group_idx'] + NON_FEATURE_COLS, errors='ignore')
    input_dim = len(input_sample.columns)
    feature_cols = input_sample.columns.tolist()
    
    print(f"  Feature Dim: {input_dim}")
    
    # Datasets
    # We evaluate on Train (to see fit), Val, and Test
    # [Correction] Set sizes match actual data availability
    train_ds = LatentODEDataset(df_train, feature_cols, target_name, set_size=200)
    val_ds = LatentODEDataset(df_val, feature_cols, target_name, set_size=25)
    test_ds = LatentODEDataset(df_test, feature_cols, target_name, set_size=25)
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=False, collate_fn=collate_set_ode)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, collate_fn=collate_set_ode)
    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, collate_fn=collate_set_ode)
    
    # Load Model
    model = CellPopulationODE(feature_dim=input_dim, set_embedding_dim=128, latent_dim=64).to(DEVICE)
    weights_path = f"weights/latent_ode_{target_name}.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return
        
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    # Prediction Loop
    time_scale = 100.0
    
    def predict_and_plot(loader, split_name):
        print(f"  Plotting {split_name}...")
        with torch.no_grad():
            for batch in loader:
                x = batch['features'].to(DEVICE)
                t = batch['times'].to(DEVICE) / time_scale
                targets = batch['targets'].numpy()
                conditions = batch['conditions'].numpy() # 1=Light, 0=Dark
                
                # Predict
                pred, _, _ = model(x, t)
                pred = pred.cpu().numpy() # (B, T, 1)
                
                # Plot each trajectory in batch
                batch_size = x.shape[0]
                for i in range(batch_size):
                    cond_lbl = "Light" if conditions[i] == 1.0 else "Dark"
                    
                    # Inverse Transform
                    y_true = pt_fold.inverse_transform(targets[i])
                    y_pred = pt_fold.inverse_transform(pred[i])
                    times = batch['times'][i].numpy()
                    
                    plt.figure(figsize=(8, 5))
                    plt.plot(times, y_true, 'o-', label='Ground Truth', color='black')
                    plt.plot(times, y_pred, 's--', label='Latent ODE Prediction', color='red')
                    
                    plt.title(f"{target_name} ({split_name}) - {cond_lbl}")
                    plt.xlabel("Time (h)")
                    plt.ylabel(target_name)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    os.makedirs("results/latent_ode_plots", exist_ok=True)
                    plt.savefig(f"results/latent_ode_plots/{target_name}_{split_name}_{cond_lbl}.png")
                    print(f"    Saved results/latent_ode_plots/{target_name}_{split_name}_{cond_lbl}.png")
                    plt.close()

    predict_and_plot(train_loader, "Train")
    predict_and_plot(val_loader, "Val")
    predict_and_plot(test_loader, "Test")

if __name__ == "__main__":
    evaluate()
