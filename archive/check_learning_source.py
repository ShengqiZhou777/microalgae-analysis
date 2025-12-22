import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

def check_importance():
    # Load the results and raw data
    df = pd.read_csv("data/Final_Training_Data_With_Labels.csv")
    from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic as compute_sw
    
    morph_cols = ['cell_mean_area', 'cell_mean_mean_intensity', 'cell_mean_eccentricity', 'cell_mean_solidity']
    
    # 1. Prepare data with history
    print("Preparing data with 012 sliding window...")
    df_aug = compute_sw(df, window_size=3, morph_cols=morph_cols)
    
    # Select features (current + history)
    target = 'Dry_Weight'
    non_feat = ['file', 'time', 'condition', 'Dry_Weight', 'Chl_Per_Cell', 'Fv_Fm', 'Oxygen_Rate', 'Total_Chl']
    features = [c for c in df_aug.columns if c not in non_feat and df_aug[c].dtype != 'object']
    
    X = df_aug[features]
    y = df_aug[target]
    
    # 2. Train a simple model to inspect
    model = XGBRegressor(n_estimators=100)
    model.fit(X, y)
    
    # 3. Get Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    # Categorize features
    def get_cat(f):
        if 'Prev' in f: return 'Historical (Past)'
        return 'Current (Present)'
    
    importance['Category'] = importance['Feature'].apply(get_cat)
    
    # 4. Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', hue='Category', data=importance.head(15))
    plt.title(f"What did the model learn? (Top 15 Features for {target})")
    plt.tight_layout()
    plt.savefig("feature_importance_evidence.png")
    
    print("\n--- LEARNING EVIDENCE ---")
    hist_total = importance[importance['Category'] == 'Historical (Past)']['Importance'].sum()
    curr_total = importance[importance['Category'] == 'Current (Present)']['Importance'].sum()
    print(f"Total Importance of PAST features: {hist_total:.2%}")
    print(f"Total Importance of CURRENT features: {curr_total:.2%}")
    print("--------------------------")
    print("Result saved to 'feature_importance_evidence.png'")

if __name__ == "__main__":
    check_importance()
