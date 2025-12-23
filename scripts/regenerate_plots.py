import pandas as pd
from predict_multitask import visualize_moe_results, TARGETS
import os

def main():
    csv_path = "output/Final_MoE_Test_Predictions_v7.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Loading results...")
    results = pd.read_csv(csv_path)
    
    # Ensure time/condition types (handling potential issues if loaded as object)
    if 'time' in results.columns:
        # If time is something like '0h', '12h', strip 'h' and convert to int for sorting if needed
        # But predict_multitask usually handles `df_sorted = df.sort_values('time')`.
        # Just to be safe, let's inspect unique values if needed, but plotting function should handle it 
        # as long as it sorts meaningfully.
        pass

    print("Regenerating plots (Without Expert Weights)...")
    visualize_moe_results(results, TARGETS)
    
    print("Moving plots to output/...")
    os.system("mv MoE_Result_*.png output/")
    print("Done!")

if __name__ == "__main__":
    main()
