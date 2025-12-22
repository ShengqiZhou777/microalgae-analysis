import pandas as pd
import numpy as np
from algae_fusion.features.sliding_window import compute_sliding_window_features
from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
import matplotlib.pyplot as plt

def run_comparison():
    csv_path = "data/Final_Training_Data_With_Labels.csv"
    df = pd.read_csv(csv_path)
    morph_cols = ['cell_mean_area', 'cell_mean_mean_intensity', 'cell_mean_eccentricity']
    
    # 1. Original Population Mean Approach
    df_mean = compute_sliding_window_features(df.copy(), window_size=1, morph_cols=morph_cols)
    
    # 2. Your Proposed Sequential Approach
    df_seq = compute_sliding_window_features_stochastic(df.copy(), window_size=1, morph_cols=morph_cols)
    
    # Analyze the variance of the "Previous" features
    feat = 'Prev1_cell_mean_area'
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns_plot = plt.scatter(df_mean['time'], df_mean[feat], alpha=0.1, color='blue')
    plt.title("Population Mean (Current Logic)\nPrev1 Feature is constant per group")
    plt.xlabel("Current Time")
    plt.ylabel("Historical Feature Value")

    plt.subplot(1, 2, 2)
    plt.scatter(df_seq['time'], df_seq[feat], alpha=0.1, color='green')
    plt.title("Sequential Matching (Proposed Logic)\nPrev1 Feature has individual noise")
    plt.xlabel("Current Time")
    plt.ylabel("Historical Feature Value")
    
    plt.tight_layout()
    plt.savefig("logic_comparison.png")
    print("Generated 'logic_comparison.png' to show the difference in data distribution.")

if __name__ == "__main__":
    # Check if seaborn is available for plotting
    try:
        import seaborn as sns
        run_comparison()
    except ImportError:
        print("Please ensure seaborn/matplotlib are installed.")
