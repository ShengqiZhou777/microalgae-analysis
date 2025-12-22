import pandas as pd
import numpy as np
import re

TIMECOURSE_PATH = "data/TIMECOURSE.csv"

def parse_value(val_str):
    """
    Parses strings like '19762.21 (649.95...)' or '20508.16' into float.
    Returns np.nan if conversion fails.
    """
    if pd.isna(val_str) or val_str == "":
        return np.nan
    try:
        # Take the part before the first space or parenthesis
        clean_str = str(val_str).split(' ')[0].split('(')[0]
        return float(clean_str)
    except:
        return np.nan

def load_omics_features(csv_path=TIMECOURSE_PATH):
    """
    Parses TIMECOURSE.csv to extract Transcriptome and Proteome global stats.
    Returns a DataFrame indexed by ['time', 'condition'].
    """
    # Read the file without header first to handle custom structure
    raw_df = pd.read_csv(csv_path, header=None)
    
    # Define Timepoints and their Avg column indices (0-based)
    # 0h: Avg is col 6. Step is 4.
    # Timepoints: 0, 1, 2, 3, 6, 12, 24, 48, 72
    timepoints = [0, 1, 2, 3, 6, 12, 24, 48, 72]
    # Indices corresponding to 'Avg (Std)' columns for each timepoint
    # 0h -> 6, 1h -> 10, ...
    col_indices = [6 + i*4 for i in range(len(timepoints))]
    
    # Rows definitions (0-based in raw_df)
    # Transcriptome: Lines 20-39 (Row index 19-38 in 0-indexed pandas)
    # Proteome: Lines 40-57 (Row index 39-56)
    
    # Note: Using strict line numbers from inspection. 
    # CSV Line 20 is index 19.
    
    features = {} # Key: (time, condition), Value: {feat_name: val}
    
    # Initialize dictionary for all time/condition pairs
    for t in timepoints:
        for c in ['Light', 'Dark']:
            features[(t, c)] = {}

    def process_rows(start_idx, end_idx, prefix):
        for i in range(start_idx, end_idx + 1):
            row = raw_df.iloc[i]
            
            # Col 1 is the feature name (e.g., "FPKM最大值")
            feat_name_raw = str(row[1]).strip()
            if not feat_name_raw or feat_name_raw == 'nan':
                continue
                
            # Create a simplified feature name (English)
            # We assume the order is consistent or we map by chinese name
            # For simplicity and universality, we append specific suffixes based on row index or content
            # But let's try to make it readable.
            
            # Simple mapping based on known keywords
            if '最大' in feat_name_raw or 'Max' in feat_name_raw: suffix = 'Max'
            elif '最小' in feat_name_raw or 'Min' in feat_name_raw: suffix = 'Min'
            elif '均值' in feat_name_raw or 'Mean' in feat_name_raw: suffix = 'Mean'
            elif '中位数' in feat_name_raw or 'Median' in feat_name_raw: suffix = 'Median'
            elif '<1' in feat_name_raw: suffix = 'Count_LT1'
            elif '1-10' in feat_name_raw: suffix = 'Count_1_10'
            elif '10-100' in feat_name_raw: suffix = 'Count_10_100'
            elif '100-1000' in feat_name_raw: suffix = 'Count_100_1k'
            elif '1000-1w' in feat_name_raw: suffix = 'Count_1k_10k'
            elif '>1w' in feat_name_raw or '>10000' in feat_name_raw: suffix = 'Count_GT10k'
            elif '1w-10w' in feat_name_raw: suffix = 'Count_10k_100k'
            elif '10w-100w' in feat_name_raw: suffix = 'Count_100k_1m'
            elif '100w-1000w' in feat_name_raw: suffix = 'Count_1m_10m'
            elif '>1000w' in feat_name_raw: suffix = 'Count_GT10m'
            else: suffix = f"Feat_{i}" # Fallback
            
            feat_key = f"{prefix}_{suffix}"
            
            # Check Condition (Col 2: Light or Dark)
            # Note: The structure is staggered. 
            # Row 19 is Light. Row 20 is Dark.
            # Row 21 is Light (Min). Row 22 is Dark (Min).
            # The CSV has merged cells logic for "Feature Name", but "Condition" is explicit on each row?
            # Let's check the file content again.
            # Line 20: ..., FPKM最大值, Light, ...
            # Line 21: ..., (empty), Dark, ... (Col 1 is empty, implies same feature)
            
            cond = str(row[2]).strip()
            
            # If feature name is empty in this row, inherit from previous (handled by manual pairing or state)
            # Actually, looking at the file (step 48):
            # Line 20: "转录组...", "FPKM最大值", "Light", ...
            # Line 21: nan, nan, "Dark", ...
            
            # So we need to track the "current feature name"
            
            pass 

    # Robust iteration
    current_prefix = ""
    current_base_feat = ""
    
    # We iterate through the whole Omics block
    max_rows = len(raw_df)
    for idx_row in range(19, max_rows): # Line 20 to End
        row = raw_df.iloc[idx_row]
        
        # Detect Section
        col0 = str(row[0])
        if '转录组' in col0: 
            current_prefix = 'Trans'
        elif '蛋白组' in col0:
            current_prefix = 'Prot'
            
        if not current_prefix:
            continue
            
        # Detect Feature Name (Col 1)
        col1 = str(row[1])
        if col1 and col1 != 'nan':
            # normalize chinese chars
            if '最大' in col1: current_base_feat = 'Max'
            elif '最小' in col1: current_base_feat = 'Min'
            elif '均值' in col1: current_base_feat = 'Mean'
            elif '中维' in col1 or '中位' in col1: current_base_feat = 'Median' # Typo tolerance
            elif '<1' in col1: current_base_feat = 'Count_LT1'
            elif '1-10' in col1 and '1w' not in col1: current_base_feat = 'Count_1_10'
            elif '10-100' in col1: current_base_feat = 'Count_10_100'
            elif '100-1000' in col1: current_base_feat = 'Count_100_1k'
            elif '1000-1w' in col1: current_base_feat = 'Count_1k_10k'
            elif '>1w' in col1: current_base_feat = 'Count_GT10k'
            elif '1w-10w' in col1: current_base_feat = 'Count_10k_100k'
            elif '10w-100w' in col1: current_base_feat = 'Count_100k_1m'
            elif '100w-1000w' in col1: current_base_feat = 'Count_1m_10m'
            elif '>1000w' in col1: current_base_feat = 'Count_GT10m'
            else: current_base_feat = f"Feat_{idx_row}"

        final_feat_name = f"{current_prefix}_{current_base_feat}"
        
        # Detect Condition (Col 2)
        condition = str(row[2]).strip()
        if condition not in ['Light', 'Dark']:
            continue
            
        # Extract values for each timepoint
        for i, time in enumerate(timepoints):
            col_idx = col_indices[i]
            # Safety check bounds
            if col_idx < len(row):
                val_raw = row[col_idx]
                val = parse_value(val_raw)
                features[(time, condition)][final_feat_name] = val
                
    # --- Imputation Logic ---
    # "0h makes no distinction... it is the seed"
    # Logic: For Time=0, if 'Dark' values are missing/NaN, copy from 'Light'.
    # Actually, regardless of missing, we should probably force them to be identical 
    # if 0h is truly the shared seed.
    # But let's check if Dark 0h is indeed empty in the Dict.
    
    # Get 0h features
    feats_0h_light = features.get((0, 'Light'), {})
    feats_0h_dark  = features.get((0, 'Dark'), {})
    
    # Update Dark 0h with Light 0h values where Dark is missing
    # Or overwrite entirely to ensure consistency? 
    # "0h makes no distinction" -> They should be identical.
    features[(0, 'Dark')] = feats_0h_light.copy()
    
    # Convert to DataFrame
    data_list = []
    for (t, c), feats in features.items():
        row = {'time': t, 'condition': c}
        row.update(feats)
        data_list.append(row)
        
    df_omics = pd.DataFrame(data_list)
    return df_omics

if __name__ == "__main__":
    df = load_omics_features()
    print("Parsed Omics Data Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nSample (0h):")
    print(df[df['time'] == 0])
    print("\nSample (72h):")
    print(df[df['time'] == 72])
