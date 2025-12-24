import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2

# Settings
CSV_PATH = "data/dataset_train.csv"
OUTPUT_PLOT = "Intensity_Trend_Verification.png"
OUTPUT_MONTAGE = "Intensity_Visual_Proof.png"
FEATURE = "cell_median_mean_intensity" # The verified top feature
PATH_PREFIX = "data/" # Adjust based on where script runs from (root)

print("Loading Data...")
if not os.path.exists(CSV_PATH):
    print("Train csv not found, trying test...")
    CSV_PATH = "data/dataset_test.csv"

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} samples.")

# Filter strictly for Light/Dark
df = df[df['condition'].isin(['Light', 'Dark'])].copy()
features_present = FEATURE in df.columns
if not features_present:
    print(f"Feature {FEATURE} not found! checking columns...")
    # fallback to similar
    cols = [c for c in df.columns if 'intensity' in c]
    if cols:
        FEATURE = cols[0]
        print(f"Using fallback: {FEATURE}")
    else:
        print("No intensity features found.")
        exit()

# ===========================
# 1. QUANTITATIVE PLOT
# ===========================
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='time', y=FEATURE, hue='condition', style='condition', markers=True, dashes=False, err_style="band")
plt.title(f"Population Trend: {FEATURE} over Time")
plt.xlabel("Time (h)")
plt.ylabel("Intensity Value (Normalized/Raw)")
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_PLOT, dpi=150)
print(f"Saved trend plot to {OUTPUT_PLOT}")

# ===========================
# 2. FALSE COLOR MONTAGE & RIDGE PLOT
# ===========================
# 2.1 Ridge Plot (KDE) - Statistical Proof
plt.figure(figsize=(10, 6))
# Filter for Dark condition as it shows the trend best (based on previous analysis)
subset_dark = df[df['condition'] == 'Dark']
sns.kdeplot(data=subset_dark, x=FEATURE, hue='time', palette='viridis', common_norm=False, fill=True, alpha=0.3, linewidth=2)
plt.title(f"Intensity Distribution Shift over Time (Dark Condition)")
plt.xlabel("Cell Mean Intensity")
plt.ylabel("Density")
plt.grid(True, alpha=0.2)
plt.savefig("Intensity_Distribution_Shift.png", dpi=150)
print(f"Saved distribution shift plot to Intensity_Distribution_Shift.png")

# 2.2 False Color Montage - Visual Proof
montage_imgs = []
# Pick specific intervals from available times
target_times = sorted(df['time'].unique())
display_times = [t for t in [0, 6, 24, 48, 72] if t in target_times]

print(f"Generating False Color montage for Dark at times: {display_times}")

for t in display_times:
    # Get random sample close to median
    subset = df[(df['condition'] == 'Dark') & (df['time'] == t)]
    if subset.empty: continue
    
    median_val = subset[FEATURE].median()
    subset['diff'] = abs(subset[FEATURE] - median_val)
    sample = subset.sort_values('diff').iloc[0]
    
    pass_suffix = "_mask.png"
    base_name = sample['file'].replace(pass_suffix, "").replace(".png", "")
    
    # Path construction
    src_clean = sample['Source_Path'].strip().replace("./", "")
    parent_dir = os.path.dirname(src_clean)
    img_dir = os.path.join(PATH_PREFIX, parent_dir, "images")
    mask_dir = os.path.join(PATH_PREFIX, parent_dir, "masks")
    
    img_path = os.path.join(img_dir, base_name + ".jpg")
    mask_path = os.path.join(mask_dir, base_name + "_mask.png")
    
    if not os.path.exists(img_path):
        continue
        
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Read as grayscale for colormap
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None: continue

    if mask is not None:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        img = cv2.bitwise_and(img, img, mask=mask)
        
        # Crop
        ys, xs = np.where(mask > 0)
        if len(ys) > 0:
            y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
            pad = 10
            y1, y2 = max(0, y1-pad), min(img.shape[0], y2+pad)
            x1, x2 = max(0, x1-pad), min(img.shape[1], x2+pad)
            img = img[y1:y2, x1:x2]

    # Resize
    img = cv2.resize(img, (128, 128))
    
    # APPLY FALSE COLOR (INFERNO is good for intensity)
    # Normalize img to 0-255 strictly for consistency if needed, but imread does it.
    heatmap = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    
    # Add text label
    # Draw a black box for text readability
    cv2.rectangle(heatmap, (0, 0), (128, 30), (0,0,0), -1)
    
    cv2.putText(heatmap, f"{t}h", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    val_str = f"Int:{sample[FEATURE]:.1f}"
    
    # Bottom text
    cv2.rectangle(heatmap, (0, 108), (128, 128), (0,0,0), -1)
    cv2.putText(heatmap, val_str, (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    montage_imgs.append(heatmap)

if montage_imgs:
    montage = np.hstack(montage_imgs)
    cv2.imwrite("Intensity_FalseColor_Proof.png", montage)
    print(f"Saved false color montage to Intensity_FalseColor_Proof.png")
else:
    print("Could not generate montage.")
