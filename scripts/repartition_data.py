
import pandas as pd
import os

def repartition():
    print("Loading Data...")
    # Read the combined data (for re-partitioning, we should pull from the total pool if possible, 
    # but the user likely wants to pull from dataset_train.csv which has 4911 samples)
    df = pd.read_csv("data/dataset_train.csv")
    
    test_rows = []
    train_rows = []
    
    # Group by condition and time to pick 10 from each
    groups = df.groupby(['condition', 'time'])
    
    print(f"Total groups found: {len(groups)}")
    
    for (cond, time), group in groups:
        # Shuffle group
        shuffled = group.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Take 10 for test
        test_part = shuffled.head(10)
        train_part = shuffled.tail(len(shuffled) - 10)
        
        test_rows.append(test_part)
        train_rows.append(train_part)
        
        print(f"  Group ({cond}, {time}h): {len(group)} -> Train: {len(train_part)}, Test: 10")

    df_test_new = pd.concat(test_rows).reset_index(drop=True)
    df_train_new = pd.concat(train_rows).reset_index(drop=True)
    
    # Save with backup
    if os.path.exists("data/dataset_train_OLD.csv"):
        print("Backup already exists. Overwriting current data...")
    else:
        os.rename("data/dataset_train.csv", "data/dataset_train_OLD.csv")
        if os.path.exists("data/dataset_test.csv"):
            os.rename("data/dataset_test.csv", "data/dataset_test_OLD.csv")
    
    df_train_new.to_csv("data/dataset_train.csv", index=False)
    df_test_new.to_csv("data/dataset_test.csv", index=False)
    
    print("\nRepartition Complete!")
    print(f"New Train Set: {len(df_train_new)} rows")
    print(f"New Test Set: {len(df_test_new)} rows (17 groups * 10 = 170)")

if __name__ == "__main__":
    repartition()
