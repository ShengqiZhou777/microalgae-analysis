
import pandas as pd

def check_overlap():
    print("Loading datasets...")
    try:
        df_train = pd.read_csv("data/dataset_train.csv")
        df_test = pd.read_csv("data/dataset_test.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    print(f"Train size: {len(df_train)}")
    print(f"Test size: {len(df_test)}")

    # 1. Check Image Overlap (Source_Path)
    # Assuming 'Source_Path' is the unique identifier for an image
    train_imgs = set(df_train['Source_Path'])
    test_imgs = set(df_test['Source_Path'])
    
    img_overlap = train_imgs.intersection(test_imgs)
    print(f"\n[1] Exact Image Overlap (Source_Path):")
    print(f"    Count: {len(img_overlap)}")
    if len(img_overlap) > 0:
        print(f"    WARNING: {len(img_overlap)} images are in BOTH Train and Test set!")
        print(f"    Example: {list(img_overlap)[0]}")
    else:
        print("    OK: No identical images found.")

    # 2. Check Flask/Replicate Overlap (file)
    train_files = set(df_train['file'])
    test_files = set(df_test['file'])
    
    file_overlap = train_files.intersection(test_files)
    print(f"\n[2] Biological Replicate Overlap (file ID):")
    print(f"    Count: {len(file_overlap)}")
    print(f"    Test Files Count: {len(test_files)}")
    print(f"    Overlap Rate: {len(file_overlap) / len(test_files) * 100:.1f}%")
    
    if len(file_overlap) > 0:
        print(f"    NOTE: The test set shares {len(file_overlap)} biological replicates (flasks) with the training set.")
        print("          This confirms the 'Same Flask' hypothesis.")

    # 3. Check for Duplicate Rows (based on features)
    # Let's check a few key morph columns to see if numbers are exactly identical
    feat_cols = ['cell_mean_area', 'cell_mean_mean_intensity', 'cell_mean_eccentricity']
    valid_cols = [c for c in feat_cols if c in df_train.columns and c in df_test.columns]
    
    if valid_cols:
        print(f"\n[3] Exact Feature Duplication Check (on {valid_cols}):")
        # specific check: combine file+features to see if row is identical
        train_sigs = df_train[valid_cols].astype(str).agg('-'.join, axis=1)
        test_sigs = df_test[valid_cols].astype(str).agg('-'.join, axis=1)
        
        sig_overlap = set(train_sigs).intersection(set(test_sigs))
        print(f"    Rows with identical feature values: {len(sig_overlap)}")
        if len(sig_overlap) > 0:
            print("    WARNING: Some rows have exactly identical numerical features.")

if __name__ == "__main__":
    check_overlap()
