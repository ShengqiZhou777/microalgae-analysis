
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca_selection():
    print("Loading Data...")
    df = pd.read_csv("data/Final_Training_Data_With_Labels.csv")
    
    # 1. Identify Morphological Candidates
    # Filter for 'cell_' columns, exclude any potential existing Prev_ columns or meta info
    morph_cols = [c for c in df.columns if c.startswith("cell_") and "Prev" not in c]
    
    print(f"Found {len(morph_cols)} morphological candidates.")
    
    # 2. Preprocessing
    X = df[morph_cols].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Run PCA
    n_components = 5
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # 4. Analyze Loadings
    loadings = pd.DataFrame(
        pca.components_.T, 
        columns=[f"PC{i+1}" for i in range(n_components)], 
        index=morph_cols
    )
    
    print("\n" + "="*50)
    print("PCA Feature Selection Analysis")
    print("="*50)
    
    selected_features = []
    
    for i in range(n_components):
        pc = f"PC{i+1}"
        # Find feature with max absolute loading on this PC
        best_feat = loadings[pc].abs().idxmax()
        loading_val = loadings.loc[best_feat, pc]
        explained_var = pca.explained_variance_ratio_[i] * 100
        
        print(f"\n>>> {pc} (Explains {explained_var:.1f}% Variance)")
        print(f"    Top Feature: {best_feat} (Loading: {loading_val:.3f})")
        
        # Verify orthogonality (check what else is high)
        top_3 = loadings[pc].abs().sort_values(ascending=False).head(3).index.tolist()
        print(f"    Top 3 Correlated: {', '.join(top_3)}")
        
        if best_feat not in selected_features:
            selected_features.append(best_feat)
            
    print("\n" + "="*50)
    print("Recommended Features based on PCA:")
    print("="*50)
    print(selected_features)
    
    # Compare with current manual selection
    current_selection = [
        'cell_mean_area', 
        'cell_mean_mean_intensity', 
        'cell_mean_eccentricity', 
        'cell_mean_solidity'
    ]
    
    print("\nCurrent Manual Selection:")
    print(current_selection)
    
    # Check overlap
    overlap = set(selected_features).intersection(set(current_selection))
    print(f"\nOverlap: {len(overlap)}/{len(current_selection)}")
    
if __name__ == "__main__":
    run_pca_selection()
