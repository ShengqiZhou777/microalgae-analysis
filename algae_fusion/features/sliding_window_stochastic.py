import pandas as pd
import numpy as np

def compute_sliding_window_features_stochastic(df, window_size=3, morph_cols=None):
    """
    Generates 'History' features by picking ONE individual image from previous timepoints,
    matching them sequentially based on their index within the group. 
    This simulates a scenario where we don't have the full population mean.
    """
    if morph_cols is None:
        return df

    # Sort within each group to ensure 'sequential' picking
    df = df.sort_values(['time', 'condition', 'file']).reset_index(drop=True)
    df['group_idx'] = df.groupby(['time', 'condition']).cumcount()
    
    all_times = sorted(df['time'].unique())
    time_idx_map = {t: i for i, t in enumerate(all_times)}
    
    df_aug = df.copy()
    
    # Include identifier columns to allow Dataset to find historical images
    id_cols = ['file', 'Source_Path']
    
    for k in range(1, window_size + 1):
        shift_data = []
        for t in all_times:
            idx = time_idx_map[t]
            idx = time_idx_map[t]
            # Moved check down to allow for custom prev_idx mapping
            # [USER REQUEST] Special Window Logic for 1h (Image Index 1)
            # User wants: 1h -> Prev1=Self, Prev2=0h (Shifted window)
            if idx == 1:
                if k == 1:
                    prev_idx = idx      # Prev1 uses Self
                elif k == 2:
                    prev_idx = idx - 1  # Prev2 uses 0h
                else:
                    prev_idx = idx - k  # Fallback for k>2
            else:
                prev_idx = idx - k

            if prev_idx < 0:
                continue # No history available, leave as NaN (handled by fillna later)
                
            prev_t = all_times[prev_idx]
            
            # Get the pool of data from the previous timepoint
            prev_pool = df[df['time'] == prev_t].copy()
            
            for cond in ['Light', 'Dark', 'Initial']:
                curr_group = df[(df['time'] == t) & (df['condition'] == cond)]
                if curr_group.empty: continue
                
                prev_group = prev_pool[prev_pool['condition'] == (cond if cond != 'Initial' else 'Initial')]
                if prev_group.empty:
                    prev_group = prev_pool[prev_pool['condition'] == 'Initial'] # Fallback
                
                # Sequential Match: Match by group_idx strictly.
                if not prev_group.empty:
                    # Also set index for ID columns
                    prev_feats = prev_group.set_index('group_idx')[morph_cols + id_cols]
                    prev_mean = prev_group[morph_cols].mean().to_dict()
                    # Fallback for ID columns (not ideal, but handles edge cases)
                    first_ids = prev_group.iloc[0][id_cols].to_dict()
                    
                    for row_idx, row in curr_group.iterrows():
                        g_idx = row['group_idx']
                        
                        # Use strict index match if available, else fallback
                        if g_idx in prev_feats.index:
                            stats = prev_feats.loc[g_idx].to_dict()
                        else:
                            stats = {**prev_mean, **first_ids}
                        
                        aug_row = {'file': row['file'], 'time': t, 'condition': cond}
                        for m_col in morph_cols:
                            aug_row[f'Prev{k}_{m_col}'] = stats[m_col]
                        # Add historical file paths
                        for id_col in id_cols:
                            aug_row[f'Prev{k}_{id_col}'] = stats[id_col]
                            
                        shift_data.append(aug_row)
        
        df_shift = pd.DataFrame(shift_data)
        if not df_shift.empty:
            df_aug = df_aug.merge(df_shift, on=['file', 'time', 'condition'], how='left')
            
            # [USER REQUEST] Fill Start-of-Experiment (0h) with SELF values instead of NaN
            for m_col in morph_cols:
                prev_col = f"Prev{k}_{m_col}"
                if prev_col in df_aug.columns:
                    df_aug[prev_col] = df_aug[prev_col].fillna(df_aug[m_col])
                    
            # For ID columns, we can fill with current ID (it's the same file anyway)
            for id_col in id_cols:
                prev_id = f"Prev{k}_{id_col}"
                if prev_id in df_aug.columns:
                    df_aug[prev_id] = df_aug[prev_id].fillna(df_aug[id_col])
            
    return df_aug

