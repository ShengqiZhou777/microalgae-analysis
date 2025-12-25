#取最少的作为数据集基准
import pandas as pd
import os

def create_balanced_dataset():
    input_path = "data/Final_Training_Data_With_Labels.csv"
    train_dest = "data/dataset_train.csv"
    test_dest = "data/dataset_test.csv"
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Configuration
    SAMPLES_PER_GROUP = 250
    TEST_RATIO = 0.2
    N_TEST = int(SAMPLES_PER_GROUP * TEST_RATIO) # 50
    N_TRAIN = SAMPLES_PER_GROUP - N_TEST      # 200
    
    balanced_rows_train = []
    balanced_rows_test = []
    
    # 2. Process Initial (0h) - Special Case: Duplication
    print("\nProcessing Initial (0h)...")
    initial_group = df[df['condition'] == 'Initial'].sort_values('file').reset_index(drop=True)
    
    if len(initial_group) < SAMPLES_PER_GROUP:
        raise ValueError(f"Initial group has {len(initial_group)} samples, fewer than {SAMPLES_PER_GROUP}!")
        
    initial_subset = initial_group.head(SAMPLES_PER_GROUP).copy()
    
    # Clone for Light and Dark
    for cond in ['Light', 'Dark']:
        # Create copies
        cloned = initial_subset.copy()
        cloned['condition'] = cond # Force Rename
        
        # Split Head-Tail
        # Train: 0 -> 200
        # Test: 200 -> 250
        train_part = cloned.iloc[:N_TRAIN].copy()
        test_part = cloned.iloc[N_TRAIN:].copy()
        
        # Assign split_set label for clarity (though we save to separate files)
        train_part['split_set'] = 'TRAIN' 
        # Note: Pipeline might split TRAIN into TRAIN/VAL later, but here we just define the main pool
        test_part['split_set'] = 'TEST'
        
        balanced_rows_train.append(train_part)
        balanced_rows_test.append(test_part)
        print(f"  -> Generated {cond} 0h: {len(train_part)} Train, {len(test_part)} Test")

    # 3. Process Time > 0
    print("\nProcessing Time > 0...")
    groups = df[df['condition'] != 'Initial'].groupby(['condition', 'time'])
    
    for (cond, time), group in groups:
        # Sort by file for deterministic behavior
        group_sorted = group.sort_values('file').reset_index(drop=True)
        
        if len(group_sorted) < SAMPLES_PER_GROUP:
             raise ValueError(f"Group {cond} {time}h has {len(group_sorted)} samples, fewer than {SAMPLES_PER_GROUP}!")
             
        # Take top 250
        subset = group_sorted.head(SAMPLES_PER_GROUP).copy()
        
        # Split Head-Tail
        train_part = subset.iloc[:N_TRAIN].copy()
        test_part = subset.iloc[N_TRAIN:].copy()
        
        train_part['split_set'] = 'TRAIN'
        test_part['split_set'] = 'TEST'
        
        balanced_rows_train.append(train_part)
        balanced_rows_test.append(test_part)
        print(f"  -> Processed {cond} {time}h: {len(train_part)} Train, {len(test_part)} Test")
        
    # 4. Concatenate and Save
    df_train_final = pd.concat(balanced_rows_train).reset_index(drop=True)
    df_test_final = pd.concat(balanced_rows_test).reset_index(drop=True)
    
    # Backup old files
    if os.path.exists(train_dest):
        os.rename(train_dest, train_dest + ".bak_balanced")
    if os.path.exists(test_dest):
        os.rename(test_dest, test_dest + ".bak_balanced")
        
    df_train_final.to_csv(train_dest, index=False)
    df_test_final.to_csv(test_dest, index=False)
    
    print("\n" + "="*40)
    print("BALANCED DATASET GENERATED SUCCESSFULLY")
    print("="*40)
    print(f"Train Set: {len(df_train_final)} rows (Saved to {train_dest})")
    print(f"Test Set:  {len(df_test_final)} rows (Saved to {test_dest})")
    print(f"Total:     {len(df_train_final) + len(df_test_final)} (Expected: 4500)")
    print("="*40)

if __name__ == "__main__":
    create_balanced_dataset()
