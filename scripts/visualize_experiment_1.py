import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import os

# Config
INPUT_FILE = "results/Final_MoE_Verification.csv"
OUTPUT_PLOT = "results/Experiment1_BlindBaseline.png"
TARGETS = ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    # Setup Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    sns.set_style("whitegrid")

    for i, target in enumerate(TARGETS):
        ax = axes[i]
        
        # Check columns
        static_col = f"Pred_{target}_Static"
        dyn_col = f"Pred_{target}_Dynamic"
        
        if target not in df.columns or static_col not in df.columns or dyn_col not in df.columns:
            ax.text(0.5, 0.5, f"Missing Data for {target}", ha='center', va='center')
            continue

        # Calculate Scores
        r2_static = r2_score(df[target], df[static_col])
        r2_dynamic = r2_score(df[target], df[dyn_col])
        gap = r2_dynamic - r2_static
        
        # Plot Dynamic First (Background)
        sns.scatterplot(
            x=df[target], y=df[dyn_col], 
            ax=ax, color='red', alpha=0.3, label=f'Dynamic (R²={r2_dynamic:.3f})', s=40
        )
        
        # Plot Static Second (Foreground)
        sns.scatterplot(
            x=df[target], y=df[static_col], 
            ax=ax, color='blue', alpha=0.3, label=f'Static (R²={r2_static:.3f})', marker='^', s=40
        )
        
        # Diagonal Line
        lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.75, lw=1.5)
        
        # Formatting
        ax.set_title(f"{target}\nGap: +{gap:.3f}", fontsize=14, fontweight='bold')
        ax.set_xlabel("True Value")
        ax.set_ylabel("Predicted Value")
        ax.legend()
        
        # Add Interpretation Text Box
        text_str = (
            f"Image Alone (Static): {r2_static:.2f}\n"
            f"+ Context (Dynamic): {r2_dynamic:.2f}\n"
            f"Gain from Context: {gap*100:.1f}%"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

    plt.suptitle("Experiment 1: The 'Blind' Baseline Test\n(Quantifying the Value of 'Cheating'/Context)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plt.savefig(OUTPUT_PLOT, dpi=300)
    print(f"Saved plot to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
