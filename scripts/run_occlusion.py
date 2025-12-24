
import os
import sys
import torch
import numpy as np
import cv2
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns

# Fix Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from algae_fusion.models.backbones import ResNetRegressor
from algae_fusion.data.dataset import MaskedImageDataset
from algae_fusion.config import IMG_SIZE, DEVICE

def denormalize_image(tensor):
    # Mean=[0.5]*3, Std=[0.5]*3 -> x * 0.5 + 0.5
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def occlusion_sensitivity(model, x_tensor, target_idx=None, patch_size=32, stride=8):
    """
    Sliding window occlusion.
    x_tensor: (C, H, W)
    """
    model.eval()
    width = x_tensor.shape[2]
    height = x_tensor.shape[1]
    
    # Base prediction
    with torch.no_grad():
        base_pred = model(x_tensor.unsqueeze(0)).item()
    
    heatmap = np.zeros((height, width))
    
    # Occlusion value: 0 (normalized space for 0.5 is 0)
    # The background is black (0 in uint8) -> -1 in normalized.
    # We want to occlusion with "Mean Grey" which is 0.5 in uint8 -> 0 in normalized.
    # This removes visual information.
    OCCLUSION_VAL = 0.0 # corresponds to 0.5 in [0,1] floating point
    
    # Slide
    # We focus on the last 3 channels (current image) if multiple frames
    # But we should occlude ALL frames at same position if we want to remove "spatial feature"
    # Or just the current one. Ideally spatial feature is consistent.
    
    # Simply iterate
    for h in range(0, height - patch_size + 1, stride):
        for w in range(0, width - patch_size + 1, stride):
            
            # Clone
            input_curr = x_tensor.clone()
            
            # Apply patch
            # Occlude all channels at this spatial location
            input_curr[:, h:h+patch_size, w:w+patch_size] = OCCLUSION_VAL
            
            with torch.no_grad():
                pred = model(input_curr.unsqueeze(0)).item()
            
            # Importance = |Change| or just Drop?
            # We want to know what contributes POSITIVELY to the value.
            # If we block a high-value region, prediction should DROP (Base > Pred).
            # Diff = Base - Pred
            diff = base_pred - pred
            
            heatmap[h:h+patch_size, w:w+patch_size] += diff

    # Normalize heatmap
    heatmap = heatmap / (heatmap.max() + 1e-7)
    return heatmap, base_pred

def main():
    target = "Chl_Per_Cell"
    condition = "Light"
    variance = "stochastic" 
    
    # 1. Load Model (Same as before)
    weights_path = f"weights/{target}_{condition}_{variance}_cnn.pth"
    if not os.path.exists(weights_path):
        variance = "mean"
        weights_path = f"weights/{target}_{condition}_{variance}_cnn.pth"
    
    print(f"Loading weights from {weights_path}...")
    in_channels = 9 if variance == "stochastic" else 3
    model = ResNetRegressor(backbone="resnet34", in_channels=in_channels).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    
    # 2. Load Samples (High Value preferably to see what drives it)
    df_train = pd.read_csv("data/dataset_train.csv")
    df = df_train[df_train['condition'] == condition].reset_index(drop=True)
    df = df.sort_values(by=target)
    
    # Pick the High Value sample (Index 2 in previous script)
    # The user questioned the previous result, so let's stick to the same samples
    indices = [len(df)-1] # Max value
    samples = df.iloc[indices].reset_index(drop=True)
    
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    ds = MaskedImageDataset(pd.DataFrame(samples), target, IMG_SIZE, val_transform, in_channels=in_channels)
    
    os.makedirs("output/occlusion", exist_ok=True)
    
    for i in range(len(samples)):
        x, y = ds[i]
        x_tensor = x.to(DEVICE)
        
        # Run Occlusion
        print(f"Running Occlusion Sensitivity on Sample {i} (Val: {samples.iloc[i][target]:.2f})...")
        # Smaller stride = smoother heatmap
        heatmap, pred_val = occlusion_sensitivity(model, x_tensor, patch_size=32, stride=2)
        
        # Visualization
        if in_channels > 3:
            img_tensor = x[-3:, :, :] 
        else:
            img_tensor = x
        
        orig_img = denormalize_image(img_tensor)
        
        # Resize heatmap to match image strictly
        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        
        # Plot
        plt.figure(figsize=(10, 4))
        
        # Original
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Original (Pred: {pred_val:.2f})")
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap_resized, cmap='seismic', vmin=-1, vmax=1)
        plt.title("Sensitivity (Blue=Neg, Red=Pos)")
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.4)
        plt.title("Overlay")
        plt.axis('off')
        
        out_path = f"output/occlusion/Occlusion_{target}_{condition}_{i}.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
