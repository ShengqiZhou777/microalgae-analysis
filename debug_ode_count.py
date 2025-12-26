
import pandas as pd
import numpy as np
import torch
from algae_fusion.data.dataset import AlgaeTimeSeriesDataset, collate_ode_batch
from algae_fusion.config import NON_FEATURE_COLS

def debug_data():
    target_name = "Oxygen_Rate"
    
    # 1. Load Data
    train_df = pd.read_csv("data/dataset_train.csv")
    test_df = pd.read_csv("data/dataset_test.csv")
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 2. Sort
    df = df.sort_values(by=['condition', 'time'], kind='stable').reset_index(drop=True)
    
    # 3. Generate group_idx
    if 'group_idx' not in df.columns:
        print("Generating group_idx...")
        df['group_idx'] = df.groupby(['condition', 'time']).cumcount()
    
    # 4. Generate splits (simplified logic from pipeline)
    def assign_split_by_id(df_all):
        splits_map = {}
        for cond, sub in df_all.groupby('condition'):
            unique_ids = sorted(sub['group_idx'].unique())
            N = len(unique_ids)
            test_thresh = int(N * 0.8)
            for i, uid in enumerate(unique_ids):
                status = 'TRAIN'
                if i >= test_thresh: status = 'TEST'
                splits_map[(cond, uid)] = status
        
        split_df = pd.DataFrame([
            {'condition': c, 'group_idx': i, 'split_set': s} 
            for (c, i), s in splits_map.items()
        ], columns=['condition', 'group_idx', 'split_set'])
        
        if 'split_set' in df_all.columns:
            df_all = df_all.drop(columns=['split_set'])
        return pd.merge(df_all, split_df, on=['condition', 'group_idx'], how='left')

    df = assign_split_by_id(df)
    
    # 5. Check Counts for Light and Dark
    for condition in ["Light", "Dark"]:
        print(f"\n--- Checking {condition} ---")
        df_cond = df[df['condition'] == condition]
        df_test = df_cond[df_cond['split_set'] == 'TEST'].copy()
        
        print(f"Total Test Rows: {len(df_test)}")
        print(f"Unique Groups in Test: {df_test['group_idx'].nunique()}")
        
        # Create Dataset
        feature_cols = [c for c in df_test.columns if c not in NON_FEATURE_COLS + [target_name]]
        ds = AlgaeTimeSeriesDataset(df_test, feature_cols, target_name, group_col='group_idx')
        
        print(f"Dataset Length (Seq Groups): {len(ds)}")
        
        # Check actual values
        y_vals = []
        for i in range(len(ds)):
            batch = ds[i]
            y = batch['targets'] # tensor
            y_vals.extend(y.numpy())
        
        y_vals = np.array(y_vals)
        print(f"Total Points in Targets: {len(y_vals)}")
        print(f"Non-NaN Points: {np.sum(~np.isnan(y_vals))}")
        print(f"First 10 Targets: {y_vals[:10]}")

if __name__ == "__main__":
    debug_data()
