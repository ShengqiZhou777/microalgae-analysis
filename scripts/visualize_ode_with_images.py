
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2

def load_image(path):
    # Mapping logic: ./TIMECOURSE/0h/images/foo.jpg -> data/TIMECOURSE/0h/images/foo.jpg
    # Only if path starts with ./TIMECOURSE
    if path.startswith("./TIMECOURSE"):
        real_path = path.replace("./TIMECOURSE", "data/TIMECOURSE")
    else:
        real_path = path
    
    if not os.path.exists(real_path):
        print(f"  [DEBUG] Failed to find: {real_path} (Orig: {path})")
        return None

    img = cv2.imread(real_path)
    if img is None: 
        print(f"  [DEBUG] cv2 failed to read: {real_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"  [SUCCESS] Loaded: {real_path}")
    return img

def visualize_with_images(target_list=None):
    print("Generating Image-Annotated Trajectories...")
    
    if not os.path.exists("data/result.csv"):
        print("Missing data/result.csv")
        return
    if not os.path.exists("data/dataset_test.csv"):
        print("Missing data/dataset_test.csv")
        return

    df_curve = pd.read_csv("data/result.csv")
    df_data = pd.read_csv("data/dataset_test.csv")
    
    # Pre-process Curve (Initial -> Light/Dark)
    if 'Initial' in df_curve['condition'].values:
        df_init = df_curve[df_curve['condition'] == 'Initial'].copy()
        df_others = df_curve[df_curve['condition'] != 'Initial'].copy()
        df_l = df_init.copy(); df_l['condition'] = 'Light'
        df_d = df_init.copy(); df_d['condition'] = 'Dark'
        df_curve = pd.concat([df_others, df_l, df_d], ignore_index=True).sort_values(['condition', 'time'])

    targets = target_list if target_list else ["Dry_Weight", "Chl_Per_Cell", "Fv_Fm", "Oxygen_Rate"]
    
    output_dir = "results/ode_physical_visuals"
    os.makedirs(output_dir, exist_ok=True)
    
    # Times to annotate
    annotate_times = [0, 24, 48, 72]

    for target in targets:
        if target not in df_curve.columns: continue
        
        plt.figure(figsize=(12, 8))
        
        # 1. Plot Curves
        for cond, color in [('Light', 'gold'), ('Dark', 'navy')]:
            sub = df_curve[df_curve['condition'] == cond]
            # sub = sub.groupby('time')[target].mean().reset_index() # Ensure unique line if result.csv has multiples
            plt.plot(sub['time'], sub[target], color=color, linewidth=3, label=cond)
            
            # 2. Annotate Images
            for t in annotate_times:
                # Find point on curve
                row_curve = sub[sub['time'] == t]
                if len(row_curve) == 0: continue
                y_curve = row_curve[target].mean()
                
                # Find matching samples in dataset (Real Evidence)
                if t == 0:
                    # 0h is usually all 'Initial' or mapped to condition. 
                    # In dataset_test, condition might be 'Initial'.
                    # Let's search broadly for time=t
                    candidates = df_data[df_data['time'] == t]
                else:
                    candidates = df_data[(df_data['time'] == t) & (df_data['condition'] == cond)]
                
                if len(candidates) == 0: continue
                
                # Find sample closest to the curve value
                # This ensures the image we show is 'Representative' of what the model predicts/tracks
                candidates = candidates.copy()
                candidates['diff'] = (candidates[target] - y_curve).abs()
                best_sample = candidates.sort_values('diff').iloc[0]
                
                # Load Image
                # CSV has: Source_Path='./TIMECOURSE/0h/images', file='12345_mask.png'
                # Disk has: 'data/TIMECOURSE/0h/images/12345.jpg'
                
                sp_col = 'Source_Path' if 'Source_Path' in candidates.columns else 'file' # Fallback
                file_col = 'file' if 'file' in candidates.columns else None
                
                src_path = best_sample[sp_col]
                filename = best_sample[file_col] if file_col else ""
                
                # Logic to reconstruct real path
                if str(src_path).startswith("./TIMECOURSE"):
                    # 1. Map dir
                    real_dir = str(src_path).replace("./TIMECOURSE", "data/TIMECOURSE")
                    
                    # 2. Fix filename (remove _mask.png -> .jpg)
                    # Check if file has _mask
                    real_filename = filename.replace("_mask.png", ".jpg").replace("_mask.tif", ".jpg")
                    
                    img_path = os.path.join(real_dir, real_filename)
                else:
                    # Fallback
                    img_path = src_path

                img = load_image(img_path)
                
                if img is not None:
                    # Add Image
                    imagebox = OffsetImage(img, zoom=0.15)
                    
                    # Offset logic (Increased spacing)
                    # Push Light UP, Dark DOWN further
                    offset_y = 70 if cond == 'Light' else -70
                    
                    ab = AnnotationBbox(imagebox, (t, y_curve),
                                        xycoords='data',
                                        boxcoords="offset points",
                                        xybox=(0, offset_y),
                                        pad=0.3,
                                        arrowprops=dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90,rad=3", color='gray', alpha=0.5))
                    plt.gca().add_artist(ab)
                    
                    # [NEW] Add Text Annotation (Morphology Stats)
                    # Pick key features to show 'Why' it changes
                    area = best_sample.get('cell_mean_area', 0)
                    texture = best_sample.get('cell_mean_texture_contrast', 0)
                    
                    # Format text
                    stats_text = f"Area: {area:.0f}\nTex: {texture:.2f}"
                    
                    # Text box below/above the image
                    # If Light (Top), text ABOVE image? Or below?
                    # Let's put text further away from the curve to avoid curve overlap
                    text_offset = 40 if cond == 'Light' else -45
                    
                    plt.annotate(stats_text, 
                                 xy=(t, y_curve), 
                                 xytext=(0, offset_y + text_offset),
                                 textcoords="offset points",
                                 ha='center', va='center',
                                 fontsize=9,
                                 fontweight='bold',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=1.0))
        
        plt.title(f"{target}: Physical Trajectory (Curve = Model, Insets = Real Cells)")
        plt.xlabel("Time (h)")
        plt.ylabel(target)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = f"{output_dir}/{target}_annotated.png"
        plt.savefig(save_path, dpi=150)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    visualize_with_images()
