import matplotlib.pyplot as plt
import numpy as np

def draw_window_logic():
    timepoints = [0, 1, 2, 3, 6, 12, 24, 48, 72]
    window_columns = ['Current', 'Prev 1', 'Prev 2', 'Prev 3']
    
    # Calculate mapping logic
    data = []
    for i, t in enumerate(timepoints):
        row = [f"{t}h"]
        for k in range(1, 4):
            prev_idx = max(0, i - k)
            row.append(f"{timepoints[prev_idx]}h")
        data.append(row)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Table styling
    table = ax.table(cellText=data, colLabels=window_columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    
    # Highlight Padding (Initial values)
    for i in range(3): # 0h, 1h, 2h rows
        for j in range(1, 4):
            if i == 0: # 0h
               color = '#ff9999' # Reddish for full padding
            else:
               color = '#ffe5e5' if data[i][j] == "0h" else '#e5ffe5'
            table[(i+1, j)].set_facecolor(color)

    # Highlight Sliding (Later values)
    for i in range(3, len(timepoints)):
        for j in range(1, 4):
            table[(i+1, j)].set_facecolor('#e5ffe5') # Greenish for real history

    plt.title("Sliding Window Logic: Why We Don't Lose Data (Padding '000')", fontsize=16, pad=20)
    
    # Add text explanation
    plt.figtext(0.1, 0.05, 
                "Red/Pink: Padding (Missing history, forced to match 0h)\n"
                "Green: Real History (Sliding window logic)\n"
                "Result: All files are used; 0h samples use themselves as history.", 
                ha="left", fontsize=10, bbox={"facecolor":"white", "alpha":0.8})

    plt.savefig("window_mapping_logic.png", dpi=300, bbox_inches='tight')
    print("Saved 'window_mapping_logic.png'")

if __name__ == "__main__":
    draw_window_logic()
