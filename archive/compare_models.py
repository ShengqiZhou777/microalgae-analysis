import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os

def evaluate(df, name):
    target = 'Dry_Weight'
    pred_col = f'Predicted_{target}'
    if pred_col not in df.columns: return None
    
    y_true = df[target]
    y_pred = df[pred_col]
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'y_true': y_true, 'y_pred': y_pred}

def compare():
    files = {
        'Dynamic (History)': 'prediction_dynamic.csv',
        'Static (No History)': 'prediction_static.csv'
    }
    
    results = {}
    for name, fpath in files.items():
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            results[name] = evaluate(df, name)
        else:
            print(f"Warning: {fpath} not found.")

    if not results:
        print("No predictions found.")
        return

    # Print Table
    print("\n" + "="*45)
    print(f"{'Model':<20} | {'R2':<8} | {'MAE':<8} | {'RMSE':<8}")
    print("-" * 45)
    for name, res in results.items():
        print(f"{name:<20} | {res['R2']:.4f}   | {res['MAE']:.4f}   | {res['RMSE']:.4f}")
    print("=" * 45)

    # Visualization
    plt.figure(figsize=(14, 6))
    
    # 1. Scatter Comparison
    for i, (name, res) in enumerate(results.items()):
        plt.subplot(1, 2, i+1)
        sns.scatterplot(x=res['y_true'], y=res['y_pred'], alpha=0.6)
        
        # Diagonal
        lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
        plt.plot(lims, lims, 'k--', alpha=0.5)
        
        plt.title(f"{name}\nR2: {res['R2']:.4f}")
        plt.xlabel("True Dry Weight")
        plt.ylabel("Predicted Dry Weight")
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("Model_Comparison.png", dpi=300)
    print("\n[SUCCESS] Comparison plot saved to Model_Comparison.png")

if __name__ == "__main__":
    compare()
