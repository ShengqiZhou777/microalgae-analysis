import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score

def visualize_results(csv_path, target_name, output_dir="."):
    """
    Visualizes metrics for regression results:
    1. Scatter plot of True vs Predicted (Accuracy)
    2. Trajectory plots (Mean +/- Std) over time for each condition
    """
    if not os.path.exists(csv_path):
        print(f"[Warn] Visualization skipped. File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if f"Predicted_{target_name}" not in df.columns:
        print(f"[Warn] Visualization failed. 'Predicted_{target_name}' column missing.")
        return

    # Increase height to accommodate split plots
    plt.figure(figsize=(18, 12)) 
    
    # Subplot 1: All Samples Scatter Plot (Left Half)
    plt.subplot(1, 2, 1)
    y_true = df[target_name]
    y_pred = df[f"Predicted_{target_name}"]
    
    # Calculate R2
    score = r2_score(y_true, y_pred)
    
    # Handle hue only if condition exists
    if 'condition' in df.columns:
        sns.scatterplot(x=y_true, y=y_pred, hue=df['condition'], alpha=0.5)
    else:
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    
    # Diagonal line
    min_limit = min(y_true.min(), y_pred.min())
    max_limit = max(y_true.max(), y_pred.max())
    lims = [min_limit, max_limit]
    plt.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
    
    plt.title(f"Sample-level Accuracy (R2: {score:.4f})")
    plt.xlabel(f"True {target_name}")
    plt.ylabel(f"Predicted {target_name}")
    
    # Helper for trajectory plotting
    def plot_trajectory(ax, condition, color):
        if 'condition' not in df.columns or 'time' not in df.columns:
            return

        subset = df[df['condition'] == condition]
        if subset.empty: return
        
        # Group by time and calculate mean/std
        true_stats = subset.groupby('time')[target_name].agg(['mean', 'std'])
        pred_stats = subset.groupby('time')[f"Predicted_{target_name}"].agg(['mean', 'std'])
        
        # Plot True
        ax.plot(true_stats.index, true_stats['mean'], '--o', color=color, alpha=0.6, label=f'True {condition}')
        # Plot Pred
        ax.plot(pred_stats.index, pred_stats['mean'], '-s', color=color, label=f'Pred {condition}', linewidth=2)
        
        # Fill Error
        ax.fill_between(true_stats.index, true_stats['mean'] - true_stats['std'], 
                         true_stats['mean'] + true_stats['std'], color=color, alpha=0.1)
        
        ax.set_title(f"{target_name} Population Trajectory ({condition})")
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(target_name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    # Subplot 2: Light Trajectory (Top Right)
    ax2 = plt.subplot(2, 2, 2)
    plot_trajectory(ax2, 'Light', 'red')
    
    # Subplot 3: Dark Trajectory (Bottom Right)
    ax3 = plt.subplot(2, 2, 4)
    plot_trajectory(ax3, 'Dark', 'blue')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f"validation_plot.png")
    plt.savefig(viz_path, dpi=300)
    plt.close()
    print(f"  [SUCCESS] Saved result visualization to: {viz_path}")
