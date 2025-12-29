
import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import matplotlib

# Add project root
sys.path.append(os.getcwd())

# Configuration
TEST_DATA_PATH = "data/dataset_test.csv"
OUTPUT_DIR = "results/ode_explanation"

def visualize_area_importance(condition="Dark", feature_name="Area"):
    print(f"=== Visualizing {feature_name} Features for {condition} Condition ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(TEST_DATA_PATH):
        print("Dataset not found.")
        return
        
    df = pd.read_csv(TEST_DATA_PATH)
    df = df[df['condition'] == condition].reset_index(drop=True)
    
    if df.empty:
        print("No data found for this condition.")
        return
        
    # 2. Select Representative Samples
    times = sorted(df['time'].unique())
    selected_times = [times[0], times[len(times)//2], times[-1]] # Start, Middle, End
    
    for t in selected_times:
        print(f"Processing Time: {t}h")
        
        # Robust Image Finding
        row = df[df['time'] == t].iloc[0]
        
        # CSV Path: ./TIMECOURSE/0h/images... -> Data Path: data/TIMECOURSE/0h/images...
        raw_path = row['Source_Path']
        if raw_path.startswith("./"):
            raw_path = raw_path[2:]
        img_path = os.path.join("data", raw_path)
        
        # If directory, pick first jpg
        if os.path.isdir(img_path):
            files = os.listdir(img_path)
            jpgs = [f for f in files if f.endswith(".jpg")]
            if jpgs:
                img_path = os.path.join(img_path, jpgs[0])
                print(f"  Path is dir, using child: {img_path}")
            else:
                print(f"  No jpgs in dir: {img_path}")
                continue
        
        # If file missing, try sibling
        if not os.path.exists(img_path):
            folder_path = os.path.dirname(img_path)
            if os.path.exists(folder_path):
                files = os.listdir(folder_path)
                jpgs = [f for f in files if f.endswith(".jpg")]
                if jpgs:
                    img_path = os.path.join(folder_path, jpgs[0])
                    print(f"  Exact file missing, using sibling: {img_path}")
                else:
                    print(f"  No jpgs found in {folder_path}")
                    continue
            else:
                 print(f"  Folder not found: {folder_path}")
                 continue
            
        # 3. Load Image
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"Failed to load image at: {img_path}")
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # 4. Load Mask (Pre-existing)
        # Image Path: .../images/filename.jpg -> Mask Path: .../masks/filename_mask.png
        mask_path = img_path.replace("/images/", "/masks/").replace(".jpg", "_mask.png")
        
        if not os.path.exists(mask_path):
            print(f"  Mask not found: {mask_path}, falling back to thresholding")
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY_INV, 25, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        else:
            # Load Mask (Instance Mask: each cell has unique ID)
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"  Failed to load mask: {mask_path}")
                continue
            
            # 5. Extract Contours for each Instance
            contours = []
            unique_ids = np.unique(mask)
            
            # Skip background (0)
            if 0 in unique_ids:
                unique_ids = unique_ids[unique_ids != 0]
                
            for uid in unique_ids:
                # Create binary mask for this single cell ID
                cell_mask = (mask == uid).astype(np.uint8) * 255
                
                # Find contour for this ID
                cts, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cts:
                    # Take the largest contour for this ID (usually just one)
                    c = max(cts, key=cv2.contourArea)
                    contours.append(c)

        # (Skip debug mask saving for instance mask loop)
        
        # 6. Calculate Area Features
        areas = []
        valid_contours = []
        img_area = img_bgr.shape[0] * img_bgr.shape[1]
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise and large borders
            if area > 20 and area < (0.9 * img_area):
                areas.append(area)
                valid_contours.append(cnt)
                
        if not areas:
            print(f"  No valid contours found in {img_path}")
            continue
            
        areas = np.array(areas)
        print(f"  Found {len(areas)} cells. Area Range: {areas.min():.1f} - {areas.max():.1f}")
        
        # 7. Select Feature for Visualization
        if feature_name == 'Area':
            values = areas
            label = 'Cell Area (px)'
            # Top 20% = High
            threshold = np.percentile(values, 80)
            high_condition = lambda v: v > threshold
        else:
            # Intensity Mode
            # Calculate Mean Intensity for each cell
            intensities = []
            gray_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            for cnt in valid_contours:
                # Create mask for this single cell
                mask_single = np.zeros_like(gray_img)
                cv2.drawContours(mask_single, [cnt], -1, 255, -1)
                # Mean intensity inside
                mean_val = cv2.mean(gray_img, mask=mask_single)[0]
                intensities.append(mean_val)
            values = np.array(intensities)
            label = 'Mean Intensity (0-255)'
            # For Intensity: Low Intensity usually means high density/chlorophyll (Darker) in brightfield?
            # Or High Intensity means fluorescence? 
            # Let's assume prediction correlates with feature importance direction.
            # Importance was +0.0009 for max_intensity. Positive importance means shuffling it hurts.
            # Let's just visualize the raw value distribution first.
            # Let's highlight Extremes (Very Dark or Very Bright).
            # Usually biological interest is Low Intensity (Dark) in BF.
            # So High Yield might be Low Intensity.
            threshold = np.percentile(values, 20) # Bottom 20% (Darkest)
            high_condition = lambda v: v < threshold
            
        vis_img = img_rgb.copy()
        high_yield_count = 0
        
        # Colormap for background (Optional) or just Binary Decision
        # Let's stick to Binary Decision Map style for clarity as requested
        
        for cnt, val in zip(valid_contours, values):
            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            if high_condition(val):
                # High Yield (Large or Dark): Red
                color = (255, 0, 0)
                thickness = 3
                high_yield_count += 1
                cv2.drawContours(vis_img, [cnt], -1, color, thickness)
                cv2.putText(vis_img, "HIGH", (cX - 20, cY - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                # Low Yield: Gray/Blue
                color = (0, 100, 255) # Orange-ish
                thickness = 1
                cv2.drawContours(vis_img, [cnt], -1, color, thickness)

        # 8. Add Dashboard Summary
        ratio = high_yield_count / len(values)
        status = "MATURE" if ratio > 0.3 else "GROWING"
        
        cv2.rectangle(vis_img, (0, 0), (350, 120), (0, 0, 0), -1)
        cv2.putText(vis_img, f"Time: {t}h | Mode: {feature_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, f"Target Cells: {high_yield_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(vis_img, f"Status: {status}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 9. Save Plot
        plt.figure(figsize=(12, 6))
        plt.imshow(vis_img)
        plt.title(f"Production Map ({feature_name}) - t={t}h")
        plt.axis('off')
        
        save_path = f"{OUTPUT_DIR}/decision_vis_{condition}_{feature_name}_{t}h.png"
        if os.path.exists(save_path):
            os.remove(save_path) # Fix PermissionError by removing first
            
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', type=str, default='Dark')
    parser.add_argument('--feature', type=str, default='Area', choices=['Area', 'Intensity'])
    args = parser.parse_args()
    
    visualize_area_importance(args.condition, args.feature)
