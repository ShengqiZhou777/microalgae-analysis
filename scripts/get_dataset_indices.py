import pandas as pd
import numpy as np
import os
import json

def get_indices_head_tail():
    # 1. Load Data
    train_path = "data/dataset_train.csv"
    test_path = "data/dataset_test.csv"
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found.")
        return

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()
    
    # Pool all data for the global head-tail split
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all.loc[df_all['condition'] == 'Initial', 'condition'] = 'Light'
    
    # 2. Sort by condition/time/file for stable group_idx
    df_all = df_all.sort_values(by=['condition', 'time', 'file']).reset_index(drop=True)
    df_all['group_idx'] = df_all.groupby(['condition', 'time']).cumcount()
    
    # 3. Apply deterministic splitting (80/20 & 80/20)
    # logic matches pipeline.py
    def assign_split(group):
        N = len(group)
        test_thresh = int(N * 0.8)
        val_thresh = int(test_thresh * 0.8)
        
        splits = ['TRAIN'] * N
        for i in range(val_thresh, test_thresh):
            splits[i] = 'VAL'
        for i in range(test_thresh, N):
            splits[i] = 'TEST'
        group['split_set'] = splits
        return group

    df_all = df_all.groupby(['condition', 'time'], group_keys=False).apply(assign_split)
    
    # 4. Generate Structured JSON
    report = {}
    for (cond, t), group in df_all.groupby(['condition', 'time']):
        t_str = str(int(t) if isinstance(t, (int, float)) else t)
        if cond not in report: report[cond] = {}
        if t_str not in report[cond]: report[cond][t_str] = {"TRAIN": [], "VAL": [], "TEST": []}
        
        for _, row in group.iterrows():
            report[cond][t_str][row['split_set']].append(int(row['group_idx']))

    # 5. Save Results
    os.makedirs("results", exist_ok=True)
    
    # Full mapping
    output_cols = ['condition', 'time', 'group_idx', 'file', 'split_set']
    df_all[output_cols].to_csv("results/dataset_group_indices_stable.csv", index=False)
    
    # JSON for programmatic access
    with open("results/dataset_group_indices_stable.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("Done! Deterministic Head-Tail indices extracted.")
    print("Files saved:")
    print("- results/dataset_group_indices_stable.csv (Full Readable Mapping)")
    print("- results/dataset_group_indices_stable.json (Nested {Condition -> Time -> Split -> [IDs]})")
    
    # Verify Counts
    print("\nSummary Counts:")
    print(df_all.groupby('split_set').size())

if __name__ == "__main__":
    get_indices_head_tail()
