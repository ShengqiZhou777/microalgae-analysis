
import pandas as pd
import sys
import os

# Mocking config and pipeline to load data
# We just need to load the dataframe as pipeline does
try:
    from algae_fusion.engine.pipeline import load_and_preprocess_data, CONFIG_PATH
    import algae_fusion.config as config
    
    print(f"Loading data from {config.DATA_PATH}...")
    # Typically pipeline loads and merges. Let's try to replicate valid loading or just read CSV if known.
    # Looking at pipeline.py, it calls load_and_preprocess_data(target_name)
    
    # Alternatively, just look at the CSV directly if possible.
    # But pipeline does some merging.
    
    # Let's peek at the raw dataframe first.
    df = pd.read_csv(config.DATA_PATH)
    print("Columns:", df.columns.tolist())
    print("\n--- First 5 'file' entries ---")
    print(df['file'].head())
    print("\n--- 'file' count checking ---")
    print(f"Total rows: {len(df)}")
    print(f"Unique 'file' values: {df['file'].nunique()}")
    
    if len(df) == df['file'].nunique():
        print("ALERT: 'file' column is UNIQUE for every row. Grouping by it will yield length-1 sequences.")
    else:
        print(f"Okay, 'file' is not unique. Average group size: {len(df)/df['file'].nunique():.2f}")

except Exception as e:
    print(e)
