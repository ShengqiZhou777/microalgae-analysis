import pandas as pd

def compute_sliding_window_features(df, window_size=3, morph_cols=None):
    """Generates 'History' features based on population means of previous timepoints."""
    if morph_cols is None:
        return df
        
    pop_stats = df.groupby(['time', 'condition'])[morph_cols].mean().reset_index()
    stat_dict = {}
    for idx, row in pop_stats.iterrows():
        t, c = row['time'], row['condition']
        stat_dict[(t, c)] = row[morph_cols].to_dict()
        
    all_times = sorted(df['time'].unique())
    time_idx_map = {t: i for i, t in enumerate(all_times)}
    
    history_map = {}
    for t in all_times:
        idx = time_idx_map[t]
        prev_times = []
        for k in range(1, window_size + 1):
            prev_idx = idx - k
            if prev_idx >= 0:
                prev_times.append(all_times[prev_idx])
            else:
                prev_times.append(all_times[0]) 
        history_map[t] = prev_times

    df_aug = df.copy()
    for k in range(1, window_size + 1):
        shift_data = []
        for t in all_times:
            for c in ['Light', 'Dark']:
                prev_t = history_map[t][k-1]
                stats = stat_dict.get((prev_t, c))
                if stats is None:
                    stats = stat_dict.get((prev_t, 'Light'))
                
                if stats:
                    row = {'time': t, 'condition': c}
                    for m_col in morph_cols:
                        row[f'Prev{k}_{m_col}'] = stats[m_col]
                    shift_data.append(row)
                    
        df_shift = pd.DataFrame(shift_data)
        if not df_shift.empty:
            df_aug = df_aug.merge(df_shift, on=['time', 'condition'], how='left')
    
    return df_aug
