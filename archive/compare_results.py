import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def evaluate(csv_path, name):
    try:
        df = pd.read_csv(csv_path)
        # Check if we have predictions
        target_col = [c for c in df.columns if "Predicted_" not in c and c not in ['file', 'time', 'condition', '_orig_file']][-1]
        pred_col = f"Predicted_{target_col}"
        
        y_true = df[target_col]
        y_pred = df[pred_col]
        
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        print(f"--- {name} ---")
        print(f"R2 Score: {r2:.4f}")
        print(f"RMSE:     {rmse:.4f}")
        return r2
    except Exception as e:
        print(f"--- {name} ---")
        print(f"Error evaluating: {e}")
        return 0

print("\n=== FINAL MODEL SHOWDOWN ===\n")
score_dynamic = evaluate("prediction_dynamic.csv", "DYNAMIC MODEL (With History)")
score_static = evaluate("prediction_static.csv",  "STATIC MODEL  (No History)")

print("\n============================")
if score_dynamic > score_static:
    print(f"🏆 WINNER: DYNAMIC MODEL (by {score_dynamic - score_static:.4f} R2)")
else:
    print(f"🏆 WINNER: STATIC MODEL (by {score_static - score_dynamic:.4f} R2)")
print("============================\n")
