import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_test_results(csv_path="multi_timepoint_prediction.csv"):
    df = pd.read_csv(csv_path)
    target = 'Dry_Weight'
    pred_col = f'Predicted_{target}'
    
    if pred_col not in df.columns:
        # Fallback if the naming was different
        cols = [c for c in df.columns if 'Predicted' in c]
        if cols:
            pred_col = cols[0]
        else:
            print(f"Error: Prediction column not found in {csv_path}")
            return

    # Calculate Metrics
    y_true = df[target]
    y_pred = df[pred_col]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    print("\n" + "="*30)
    print("TEST SET EVALUATION RESULTS")
    print("="*30)
    print(f"R2 Score:  {r2:.4f}")
    print(f"MAE:       {mae:.4f}")
    print(f"RMSE:      {rmse:.4f}")
    print("="*30)

    # Plotting
    plt.figure(figsize=(16, 6))
    
    # 1. Scatter Plot
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df, x=target, y=pred_col, hue='condition', style='time', s=100, alpha=0.8)
    
    # Diagonal line
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0)
    
    plt.xlabel(f"True {target}")
    plt.ylabel(f"Predicted {target}")
    plt.title(f"Test Set: True vs Predicted (R2: {r2:.4f})")
    
    # 2. Trajectory Plot
    plt.subplot(1, 2, 2)
    # Aggregate by time and condition
    agg_df = df.groupby(['time', 'condition']).agg({
        target: ['mean', 'std'],
        pred_col: ['mean', 'std']
    }).reset_index()
    
    for cond, color in [('Light', 'red'), ('Dark', 'blue')]:
        sub = agg_df[agg_df['condition'] == cond]
        if sub.empty: continue
        
        plt.errorbar(sub['time'], sub[target]['mean'], yerr=sub[target]['std'], 
                     fmt='--o', color=color, alpha=0.4, label=f'True {cond}')
        plt.errorbar(sub['time'], sub[pred_col]['mean'], yerr=sub[pred_col]['std'], 
                     fmt='-s', color=color, label=f'Pred {cond}')

    plt.xlabel("Time (h)")
    plt.ylabel(target)
    plt.title("Population Trajectory Match on Test Samples")
    plt.legend()
    
    plt.tight_layout()
    output_img = "Test_Set_Analysis.png"
    plt.savefig(output_img, dpi=300)
    print(f"\n[SUCCESS] Visual report saved to: {output_img}")

if __name__ == "__main__":
    evaluate_test_results()
