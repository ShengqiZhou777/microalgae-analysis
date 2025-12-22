import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def check_consistency(csv_path):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Exclude metadata and target columns to find all 18x5 morphology features
    exclude_cols = [
        'file', 'time', 'condition', 'Source_Path', 
        'Dry_Weight', 'Chl_Per_Cell', 'Fv_Fm', 'Oxygen_Rate', 'Total_Chl'
    ]
    morph_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.int64]]
    
    print(f"Detected {len(morph_cols)} morphological feature columns.")
    
    # Group by time and condition
    grouped = df.groupby(['time', 'condition'])
    
    # Calculate Mean, Std, and CV
    stats_mean = grouped[morph_cols].mean()
    stats_std = grouped[morph_cols].std()
    
    # Calculate CV (Coefficient of Variation) and handle division by zero
    cv_df = (stats_std / (stats_mean + 1e-9)) * 100
    
    # Summary of CVs across all features
    print("\n--- CV (%) Distribution Summary (Across all features) ---")
    cv_summary = cv_df.describe().T[['mean', 'min', '50%', 'max']]
    cv_summary.columns = ['Avg_CV', 'Min_CV', 'Median_CV', 'Max_CV']
    print(cv_summary.sort_values(by='Avg_CV', ascending=False).head(20)) # Show top 20 most variable
    
    # Save full CV table
    cv_df.to_csv("comprehensive_cv_results.csv")
    print(f"\nFull results saved to 'comprehensive_cv_results.csv'")

    # Flag features with high CV (> 30%)
    high_cv_threshold = 30
    high_cv_mask = cv_df > high_cv_threshold
    if high_cv_mask.any().any():
        print(f"\n[WARNING] Features with CV > {high_cv_threshold}% detected:")
        high_vars = high_cv_mask.any()
        print(high_vars[high_vars].index.tolist())
    else:
        print(f"\n[INFO] All features have CV < {high_cv_threshold}% across all groups.")

    # Categorize by Physical Meaning
    # Geometry: area, perimeter, circularity, aspect_ratio, solidity, major_axis, minor_axis, eccentricity
    # Intensity: mean_intensity, std_intensity, skew_intensity, kurt_intensity, max_intensity, min_intensity
    # Texture: texture_contrast, texture_homogeneity, texture_energy, texture_correlation
    
    physical_groups = {
        'Geometry': ['area', 'perimeter', 'circularity', 'aspect_ratio', 'solidity', 'axis', 'eccentricity'],
        'Intensity': ['intensity'],
        'Texture': ['texture']
    }
    
    # We focus only on "mean" version of these physical features to address user's specific concern
    phys_avg_cv = pd.DataFrame()
    for name, keywords in physical_groups.items():
        matched_cols = []
        for kw in keywords:
            matched_cols.extend([c for c in cv_df.columns if kw in c and '_mean_' in c])
        if matched_cols:
            phys_avg_cv[name] = cv_df[matched_cols].mean(axis=1)
            
    plt.figure(figsize=(10, 8))
    sns.heatmap(phys_avg_cv, annot=True, cmap='coolwarm', center=20)
    plt.title('Average CV (%) of "Mean" Features by Physical Group')
    plt.savefig("physical_feature_cv.png")
    print("Saved physical category heatmap to 'physical_feature_cv.png'")
    
    # Correlation Analysis: Do these "noisy" features have a signal?
    target_cols = ['Dry_Weight', 'Chl_Per_Cell', 'Fv_Fm', 'Oxygen_Rate']
    available_targets = [t for t in target_cols if t in df.columns]
    
    # Calculate correlation of population means across timepoints
    pop_means = df.groupby(['time', 'condition'])[morph_cols + available_targets].mean()
    correlations = pop_means.corr()[available_targets].drop(available_targets)
    
    print("\n--- Predictive Signal Check: Correlation of Feature Means with Targets ---")
    # Show features that are both high CV but high Correlation
    for target in available_targets:
        print(f"\nTop 5 features correlating with {target}:")
        top_corr = correlations[target].abs().sort_values(ascending=False).head(5)
        for feat, score in top_corr.items():
            cv_val = cv_df[feat].mean()
            print(f"  - {feat}: Corr={score:.3f}, Avg_CV={cv_val:.1f}%")

    # Final Decision logic:
    # Feature is "Good" if CV is low OR (CV is moderate AND Correlation is very high)
    print("\n--- Feature Quality Recommendation ---")
    recommendations = []
    for feat in morph_cols:
        avg_cv = cv_df[feat].mean()
        max_corr = correlations.loc[feat].abs().max()
        if avg_cv < 15:
            status = "KEEP (Stable)"
        elif max_corr > 0.8:
            status = "KEEP (High Signal despite Noise)"
        elif avg_cv > 100:
            status = "DISCARD (Pure Noise)"
        else:
            status = "CAUTION (Check Importance)"
        recommendations.append({'Feature': feat, 'CV': avg_cv, 'Max_Corr': max_corr, 'Status': status})
    
    rec_df = pd.DataFrame(recommendations)
    print(rec_df.sort_values(by='CV', ascending=False).head(20))
    rec_df.to_csv("feature_recommendations.csv", index=False)

if __name__ == "__main__":
    csv_path = "data/Final_Training_Data_With_Labels.csv"
    if os.path.exists(csv_path):
        check_consistency(csv_path)
    else:
        print(f"Error: {csv_path} not found.")
