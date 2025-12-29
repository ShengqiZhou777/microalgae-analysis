
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from algae_fusion.utils.model_utils import Log1pScaler, StandardWrapper


def get_all_features(df_in):
    """
    Identify feature columns (morphology + history).
    """
    return [c for c in df_in.columns 
            if (c.startswith('cell_') or c.startswith('Prev')) 
            and not c.endswith('_file') 
            and not c.endswith('_Source_Path')
            and not c.endswith('_path')]

def get_morph_features(df_in):
    """
    Identify morphological columns (base features).
    """
    return [c for c in df_in.columns if c.startswith('cell_')]

def prepare_modeling_data(df_train, df_val, target_name):
    """
    Identifies feature columns and applies adaptive scaling to targets.
    
    Returns:
        X_train (pd.DataFrame): Training features
        X_val (pd.DataFrame): Validation features
        y_train_scaled (np.ndarray): Scaled training targets
        y_val_scaled (np.ndarray): Scaled validation targets
        target_scaler (object): Fitted scaler object (Log1pScaler or StandardWrapper)
        tab_cols (list): List of feature column names
    """
    tab_cols = get_all_features(df_train)
    X_train = df_train[tab_cols]
    X_val = df_val[tab_cols]
    
    y_train_orig = df_train[target_name].values
    y_val_orig = df_val[target_name].values
    
    y_train_min, y_train_max = y_train_orig.min(), y_train_orig.max()
    # Avoid div by zero
    denominator = (y_train_min + 1e-9) if y_train_min >= 0 else (abs(y_train_min) + 1e-9)
    ratio = y_train_max / denominator
    
    scaler_target = StandardScaler()
    
    if y_train_min >= 0 and ratio > 10:
        # Heavily skewed data -> Log1p + StandardScale
        y_train_transformed = np.log1p(y_train_orig)
        y_val_transformed = np.log1p(y_val_orig)
        
        y_train_scaled = scaler_target.fit_transform(y_train_transformed.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_target.transform(y_val_transformed.reshape(-1, 1)).flatten()
        target_scaler = Log1pScaler(scaler_target)
    else:
        # Uniform or negative data -> StandardScale only
        y_train_scaled = scaler_target.fit_transform(y_train_orig.reshape(-1, 1)).flatten()
        y_val_scaled = scaler_target.transform(y_val_orig.reshape(-1, 1)).flatten()
        target_scaler = StandardWrapper(scaler_target)
        
    return X_train, X_val, y_train_scaled, y_val_scaled, target_scaler, tab_cols

def load_and_split_data(condition=None):
    """
    Loads train/val/test CSVs and filters them by condition if provided.
    Returns: df_train, df_val, df_test
    """
    path_train = "data/dataset_train.csv"
    path_val = "data/dataset_val.csv"
    path_test = "data/dataset_test.csv"
    
    if not os.path.exists(path_train):
        raise FileNotFoundError(f"Dataset not found: {path_train}. Please run 'scripts/create_balanced_dataset.py' first.")
        
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val) if os.path.exists(path_val) else pd.DataFrame()
    df_test = pd.read_csv(path_test) if os.path.exists(path_test) else pd.DataFrame()
    
    # Filter by condition
    if condition is not None and condition != "All":
        df_train = df_train[df_train['condition'] == condition].reset_index(drop=True)
        if not df_val.empty:
            df_val = df_val[df_val['condition'] == condition].reset_index(drop=True)
        if not df_test.empty:
            df_test = df_test[df_test['condition'] == condition].reset_index(drop=True)
            
    # Standard cleanup for all
    for d in [df_train, df_val, df_test]:
        if not d.empty:
            if 'group_idx' not in d.columns:
                d['group_idx'] = d.groupby(['condition', 'time']).cumcount()
            d['Source_Path'] = d['Source_Path'].astype(str)
            d.sort_values(by=['condition', 'time'], kind='stable', inplace=True)
            d.reset_index(drop=True, inplace=True)
            
    return df_train, df_val, df_test
