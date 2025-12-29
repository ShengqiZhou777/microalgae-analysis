
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from algae_fusion.config import DEVICE, IMG_SIZE, BATCH_SIZE, LR, EPOCHS, BACKBONE
from algae_fusion.models.cnn import ResNetRegressor
from algae_fusion.data.dataset import MaskedImageDataset
from algae_fusion.engine.trainer import train_epoch, eval_epoch

def run_image_training(df_train, df_val, target_name, train_transform, val_transform, 
                       y_train_scaled, y_val_scaled, target_scaler, 
                       mode="full", window_size=3, stochastic_window=False):
    """
    Runs the CNN training (Layer 2) and returns validation predictions.
    
    Args:
        df_train (pd.DataFrame): Training data.
        df_val (pd.DataFrame): Validation data.
        target_name (str): Target column name.
        train_transform (callable): Image transforms for training.
        val_transform (callable): Image transforms for validation.
        y_train_scaled (np.array): Scaled training targets.
        y_val_scaled (np.array): Scaled validation targets.
        target_scaler (object): Scaler for inverse transform.
        mode (str): Training mode.
        window_size (int): Sliding window size.
        stochastic_window (bool): Whether using stochastic history.
        
    Returns:
        results (dict): Dictionary prediction and model.
            - 'val_preds_cnn': Predictions on validation set (unscaled).
            - 'model': Trained CNN model (nn.Module).
    """
    val_preds_cnn = np.zeros(len(df_val))
    cnn_model = None
    
    if mode in ["full", "cnn_only"]:
        in_channels = (window_size + 1) * 3 if stochastic_window else 3
        
        train_ds = MaskedImageDataset(df_train, target_name, IMG_SIZE, train_transform, labels=y_train_scaled, in_channels=in_channels)
        val_ds = MaskedImageDataset(df_val, target_name, IMG_SIZE, val_transform, labels=y_val_scaled, in_channels=in_channels)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
        
        cnn_model = ResNetRegressor(BACKBONE, in_channels=in_channels).to(DEVICE)
        criterion = nn.HuberLoss()
        optimizer = optim.Adam(cnn_model.parameters(), lr=LR, weight_decay=1e-2)
        
        best_loss = float('inf')
        best_state = None
        patience, no_improve = 5, 0
        
        print(f"  [Image Engine] Training CNN (ResNet{BACKBONE}) for {EPOCHS} epochs...")
        
        for ep in range(EPOCHS):
            tr_loss = train_epoch(cnn_model, train_loader, criterion, optimizer)
            val_loss, _ = eval_epoch(cnn_model, val_loader, criterion)
            print(f"    Epoch {ep+1}/{EPOCHS} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            if val_loss < best_loss:
                best_loss, best_state, no_improve = val_loss, cnn_model.state_dict(), 0
            else:
                no_improve += 1
                
            if no_improve >= patience:
                print(f"    Early stopping at epoch {ep+1}")
                break
                
        # Load best state
        if best_state is not None:
            cnn_model.load_state_dict(best_state)
            
        # Generates predictions
        _, ps = eval_epoch(cnn_model, val_loader, criterion)
        val_preds_cnn = target_scaler.inverse_transform(ps)
        
    return {
        'val_preds_cnn': val_preds_cnn,
        'model': cnn_model
    }
