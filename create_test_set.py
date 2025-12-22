import pandas as pd
import os

def create_diverse_test_set(n_per_group=5):
    input_path = "data/Final_Training_Data_With_Labels.csv"
    output_path = "test_samples.csv"
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    df = pd.read_csv(input_path)
    
    # Stratified Split: Take n_per_group for TEST, keep rest for TRAIN
    print(f"Splitting data: {n_per_group} samples per group for TEST, rest for TRAIN...")
    
    test_indices = []
    
    # Iterate groups to pick test indices
    for name, group in df.groupby(['time', 'condition']):
        # Sample n_per_group (or all if fewer)
        n = min(len(group), n_per_group)
        sampled = group.sample(n=n, random_state=42)
        test_indices.extend(sampled.index.tolist())
        
    # Create disjoint datasets
    df_test = df.loc[test_indices].copy().reset_index(drop=True)
    df_train = df.drop(index=test_indices).copy().reset_index(drop=True)
    
    # Save both
    test_path = "data/dataset_test.csv"
    train_path = "data/dataset_train.csv"
    
    df_test.to_csv(test_path, index=False)
    df_train.to_csv(train_path, index=False)
    
    print(f"Stats:")
    print(f"  Original: {len(df)}")
    print(f"  Train:    {len(df_train)} (Saved to {train_path})")
    print(f"  Test:     {len(df_test)}  (Saved to {test_path})")
    print("Test set strictly excluded from Training set.")

if __name__ == "__main__":
    create_diverse_test_set(n_per_group=5)
