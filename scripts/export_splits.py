#导出训练集、验证集和测试集
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import os

def export_split_data():
    print("Loading Original Data...")
    # 1. Load the main training pool
    df_pool = pd.read_csv("data/dataset_train.csv")
    df_pool['file'] = df_pool['file'].astype(str)
    
    # 2. Replicate the grouping and splitting logic from pipeline.py
    # Note: pipeline.py uses GroupShuffleSplit with n_splits=1, test_size=0.2 if max_folds=1
    groups = df_pool['file']
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df_pool, groups=groups))
    
    df_train = df_pool.iloc[train_idx].copy()
    df_val = df_pool.iloc[val_idx].copy()
    
    # 3. Load the independent test set 
    # (The one that we found shared some Source_Paths earlier)
    if os.path.exists("data/dataset_test.csv"):
        df_test = pd.read_csv("data/dataset_test.csv")
    else:
        df_test = pd.DataFrame()
        print("Warning: data/dataset_test.csv not found.")

    # 4. Save to files
    os.makedirs("debug_splits", exist_ok=True)
    
    df_train.to_csv("debug_splits/split_TRAIN_pool.csv", index=False)
    df_val.to_csv("debug_splits/split_VAL_internal.csv", index=False)
    
    if not df_test.empty:
        df_test.to_csv("debug_splits/split_TEST_independent.csv", index=False)
    
    print("\nFiles saved in 'debug_splits/':")
    print(f"- split_TRAIN_pool.csv ({len(df_train)} rows) -> 模型实际吃的'书'")
    print(f"- split_VAL_internal.csv ({len(df_val)} rows) -> 模型用来'期中跑分'的数据")
    print(f"- split_TEST_independent.csv ({len(df_test)} rows) -> 你最后给我的'高考卷'")

    # Extra info: check if any overlap
    train_files = set(df_train['file'])
    val_files = set(df_val['file'])
    overlap = train_files.intersection(val_files)
    
    print(f"\nValidation Check:")
    print(f"Overlap between Train-File and Val-File count: {len(overlap)} (Should be 0 if GroupSplit works)")

if __name__ == "__main__":
    export_split_data()
