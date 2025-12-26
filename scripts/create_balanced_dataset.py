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
    TEST_RATIO = 0.1 # 10% for Independent Test
    # To get 8:1:1, Val needs to be 10% of TOTAL.
    # Since we already removed 10% (Test), Remaining is 90%.
    # Val should be 1/9 of Remaining to be 10% of Total.
    VAL_RATIO_INTERNAL = 1/9 
    
    # Calculate Counts
    N_TEST = int(SAMPLES_PER_GROUP * TEST_RATIO)             # 25 (10%)
    N_REMAINING = SAMPLES_PER_GROUP - N_TEST                 # 225
    N_VAL = int(N_REMAINING * VAL_RATIO_INTERNAL)            # 25 (10%)
    N_TRAIN = N_REMAINING - N_VAL                            # 200 (80%)
    
    print(f"Sampling per Group: {SAMPLES_PER_GROUP}")
    print(f"  -> Test: {N_TEST}")
    print(f"  -> Val:  {N_VAL}")
    print(f"  -> Train: {N_TRAIN}")
    
    balanced_rows_train = []
    balanced_rows_val = []
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
        
        # Split Head-Tail-Tail
        # Train -> Val -> Test
        train_part = cloned.iloc[:N_TRAIN].copy()
        val_part = cloned.iloc[N_TRAIN:N_TRAIN+N_VAL].copy()
        test_part = cloned.iloc[N_TRAIN+N_VAL:].copy()
        
        train_part['split_set'] = 'TRAIN' 
        val_part['split_set'] = 'VAL'
        test_part['split_set'] = 'TEST'
        
        # [NEW] Add group_idx for trajectory linkage
        # Ensure indices align 0..N across all 3 sets
        train_part['group_idx'] = range(len(train_part))
        val_part['group_idx'] = range(len(train_part), len(train_part) + len(val_part))
        test_part['group_idx'] = range(len(train_part) + len(val_part), len(train_part) + len(val_part) + len(test_part))
        
        balanced_rows_train.append(train_part)
        balanced_rows_val.append(val_part)
        balanced_rows_test.append(test_part)
        print(f"  -> Generated {cond} 0h: {len(train_part)} Train, {len(val_part)} Val, {len(test_part)} Test")

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
        
        # Split
        train_part = subset.iloc[:N_TRAIN].copy()
        val_part = subset.iloc[N_TRAIN:N_TRAIN+N_VAL].copy()
        test_part = subset.iloc[N_TRAIN+N_VAL:].copy()
        
        train_part['split_set'] = 'TRAIN'
        val_part['split_set'] = 'VAL'
        test_part['split_set'] = 'TEST'
        
        # [NEW] Add group_idx for trajectory linkage
        train_part['group_idx'] = range(len(train_part))
        val_part['group_idx'] = range(len(train_part), len(train_part) + len(val_part))
        test_part['group_idx'] = range(len(train_part) + len(val_part), len(train_part) + len(val_part) + len(test_part))
        
        balanced_rows_train.append(train_part)
        balanced_rows_val.append(val_part)
        balanced_rows_test.append(test_part)
        print(f"  -> Processed {cond} {time}h: {len(train_part)} Train, {len(val_part)} Val, {len(test_part)} Test")
        
    # 4. Concatenate and Save
    df_train_final = pd.concat(balanced_rows_train).reset_index(drop=True)
    df_val_final = pd.concat(balanced_rows_val).reset_index(drop=True)
    df_test_final = pd.concat(balanced_rows_test).reset_index(drop=True)
    
        
    df_train_final.to_csv("data/dataset_train.csv", index=False)
    df_val_final.to_csv("data/dataset_val.csv", index=False)
    df_test_final.to_csv("data/dataset_test.csv", index=False)
    
    print("\n" + "="*40)
    print("BALANCED DATASET WITH VALIDATION GENERATED")
    print("="*40)
    print(f"Train Set: {len(df_train_final)} rows (Saved to data/dataset_train.csv)")
    print(f"Val Set:   {len(df_val_final)} rows (Saved to data/dataset_val.csv)")
    print(f"Test Set:  {len(df_test_final)} rows (Saved to data/dataset_test.csv)")
    print(f"Total:     {len(df_train_final) + len(df_val_final) + len(df_test_final)} (Expected: 4500)")
    print("="*40)

if __name__ == "__main__":
    create_balanced_dataset()
