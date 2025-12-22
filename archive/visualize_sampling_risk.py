import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report_viz(csv_path):
    df = pd.read_csv(csv_path)
    
    # Select two representative features
    # 1. Stable (Area)
    # 2. Unstable but Signal-rich (Skew Intensity)
    features = ['cell_mean_area', 'cell_mean_skew_intensity']
    titles = ['Geometric Feature (Stable)', 'Distribution Feature (Unstable)']
    
    plt.figure(figsize=(16, 8))
    
    all_times = sorted(df['time'].unique())
    time_to_idx = {t: i for i, t in enumerate(all_times)}
    
    for i, feature in enumerate(features):
        plt.subplot(1, 2, i+1)
        
        # Plot individual points with hue to show variance WITHIN each condition
        sns.stripplot(x='time', y=feature, hue='condition', data=df, alpha=0.1, 
                      palette={'Light': 'orange', 'Dark': 'blue', 'Initial': 'gray'}, 
                      dodge=True, order=all_times)
        
        # Plot population means for each condition
        for cond, color in [('Initial', 'gray'), ('Light', 'red'), ('Dark', 'darkblue')]:
            cond_df = df[df['condition'] == cond]
            if not cond_df.empty:
                means = cond_df.groupby('time')[feature].mean()
                x_pos = [time_to_idx[t] for t in means.index]
                plt.plot(x_pos, means.values, marker='o', color=color, linewidth=2, markersize=8, label=f'{cond} Mean')
        
        # Professional Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=6, alpha=0.3, label='Individual (Light)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=6, alpha=0.3, label='Individual (Dark)'),
            Line2D([0], [0], marker='o', color='red', linewidth=2, label='Light Trend'),
            Line2D([0], [0], marker='o', color='darkblue', linewidth=2, label='Dark Trend'),
            Line2D([0], [0], marker='o', color='gray', linewidth=2, label='Initial Trend')
        ]
        
        plt.title(f'Signal vs. Noise: {titles[i]}', fontsize=14)
        plt.xlabel('Time (h)', fontsize=12)
        plt.ylabel(feature, fontsize=12)
        plt.legend(handles=legend_elements, loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig("sampling_risk_comparison.png", dpi=300)
    print("Saved high-res comparison to 'sampling_risk_comparison.png'")

if __name__ == "__main__":
    generate_report_viz("data/Final_Training_Data_With_Labels.csv")
