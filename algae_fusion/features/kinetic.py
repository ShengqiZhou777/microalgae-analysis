import numpy as np
import pandas as pd

def process_one_fold(df, pop_stats, time_map, valid_morph_cols):
    """Helper function to process a single fold's dataframe with kinetic features."""
    df = df.copy()
    df['Prev_Time'] = df['time'].map(time_map)
    
    df = df.merge(pop_stats, left_on=['Prev_Time', 'condition'], right_on=['time', 'condition'], how='left', suffixes=('', '_PrevPop'))
    df['dt'] = df['time'] - df['Prev_Time']
    
    for col in valid_morph_cols:
        pop_col = col + '_PopMean'
        if pop_col in df.columns:
            df[f'Rel_{col}'] = df[col] / (df[pop_col] + 1e-6)
            df.loc[df['Prev_Time'].isna(), f'Rel_{col}'] = 1.0
            
            df[f'Rate_{col}'] = (df[col] - df[pop_col]) / (df['dt'] + 1e-6)
            df.loc[df['Prev_Time'].isna(), f'Rate_{col}'] = 0.0
    
    drop_cols = ['Prev_Time', 'dt']
    for c in df.columns:
        if c.endswith('_PopMean') or c.endswith('_PrevPop'):
            drop_cols.append(c)
    if 'time_PrevPop' in df.columns:
        drop_cols.append('time_PrevPop')
    
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    return df

def compute_kinetic_features(df_train, df_val, morph_cols):
    """Compute population trajectory features WITHOUT data leakage."""
    df_train = df_train.copy()
    df_val = df_val.copy()
    
    valid_morph_cols = [c for c in morph_cols if c in df_train.columns]
    if not valid_morph_cols:
        return df_train, df_val
    
    pop_stats = df_train.groupby(['time', 'condition'])[valid_morph_cols].mean().add_suffix('_PopMean').reset_index()
    all_times = sorted(df_train['time'].unique())
    time_map = {t: all_times[i-1] if i > 0 else np.nan for i, t in enumerate(all_times)}
    
    df_train_aug = process_one_fold(df_train, pop_stats, time_map, valid_morph_cols)
    df_val_aug = process_one_fold(df_val, pop_stats, time_map, valid_morph_cols)
    
    return df_train_aug, df_val_aug
