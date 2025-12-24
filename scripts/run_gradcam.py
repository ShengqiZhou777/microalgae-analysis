
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Fix Path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from algae_fusion.models.backbones import ResNetRegressor
from algae_fusion.data.dataset import MaskedImageDataset
from algae_fusion.config import IMG_SIZE, DEVICE

# Quick GradCAM Implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple
        self.gradients = grad_output[0]
        
    def __call__(self, x):
        # Forward
        self.model.zero_grad()
        output = self.model(x)
        
        # Backward (w.r.t prediction)
        output.backward(retain_graph=True)
        
        # Pool the gradients across the channels
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the activations
        activations = self.activations.detach().clone()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize
        heatmap /= torch.max(heatmap) + 1e-7
        
        return heatmap.cpu().numpy()

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        def first_layer_hook(module, grad_in, grad_out):
            return (torch.clamp(grad_in[0], min=0.0),)

        def relu_hook(module, grad_in, grad_out):
            # Guided Backprop logic: Clamp gradients > 0
            if isinstance(module, torch.nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        # Register hook to all ReLUs
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.ReLU):
                self.hooks.append(module.register_backward_hook(relu_hook))
    
    def __call__(self, x):
        self.model.zero_grad()
        x.requires_grad = True
        output = self.model(x)
        output.backward()
        
        # Gradient at input
        return x.grad.cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def generate_guided_gradcam(gradcam_map, guided_prop):
    # gradcam_map: (H, W) [0,1]
    # guided_prop: (3, H, W) non-normalized gradients
    
    # Normalize Guided Prop
    grads = guided_prop.transpose(1, 2, 0) # H,W,C
    
    # Simple Guided GradCAM: Element-wise multiply
    # We need to resize gradcam to match
    cam_resized = cv2.resize(gradcam_map, (grads.shape[1], grads.shape[0]))
    
    # Multi-channel multiply
    ggcam = grads * cam_resized[..., None]
    
    return ggcam

def visualize_saliency(tensor):
    # Visualize gradients: usually abs, max across channels, normalize
    # tensor shape (H, W, C)
    grads = np.abs(tensor)
    grads = np.max(grads, axis=2) # Take max across channels
    
    # Normalize to 0-255
    grads = (grads - grads.min()) / (grads.max() - grads.min() + 1e-8)
    grads = np.uint8(grads * 255)
    return grads

def denormalize_image(tensor):
    # Mean=[0.5]*3, Std=[0.5]*3 -> x * 0.5 + 0.5
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * 0.5 + 0.5
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)

def main():
    target = "Chl_Per_Cell"
    condition = "Light"
    variance = "stochastic" 
    
    # 1. Load Model
    # Note: Check contribution script said "CNN 38.3%" for this combo
    weights_path = f"weights/{target}_{condition}_{variance}_cnn.pth"
    if not os.path.exists(weights_path):
        # Fallback to mean if stochastic not found
        variance = "mean"
        weights_path = f"weights/{target}_{condition}_{variance}_cnn.pth"
        if not os.path.exists(weights_path):
            print("Model weights not found!")
            return

    print(f"Loading weights from {weights_path}...")
    
    # Check in_channels: 9 if stochastic (3 frames * 3ch), 3 if static
    in_channels = 9 if variance == "stochastic" else 3
    # Use resnet34 as per config
    model = ResNetRegressor(backbone="resnet34", in_channels=in_channels).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()
    
    # Target Layer: Last Conv Block of ResNet18
    # model.backbone.layer4[-1] is the last BasicBlock
    # We want the output of the conv2, or just the block output
    target_layer = model.backbone.layer4[-1]
    
    grad_cam = GradCAM(model, target_layer)
    
    # 2. Load Data (Validation Samples)
    df_train = pd.read_csv("data/dataset_train.csv")
    df = df_train[df_train['condition'] == condition].reset_index(drop=True)
    
    # Pick 3 samples: Low, Med, High Target Value
    # Ensure we sort by target to get range
    df = df.sort_values(by=target)
    
    # Indices
    indices = [0, len(df)//2, len(df)-1]
    samples = df.iloc[indices].reset_index(drop=True)
    
    print(f"Selected samples with values: {[s[target] for _, s in samples.iterrows()]}")
    
    # Transform
    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    
    # We need to construct the dataset to handle history loading if stochastic
    # But for visualization, we can hack it for 'stochastic' by just using current image repeated if lazy,
    # OR better: use the Dataset class properly.
    
    # Create a mini dataset
    ds = MaskedImageDataset(pd.DataFrame(samples), target, IMG_SIZE, val_transform, in_channels=in_channels)
    
    os.makedirs("output/gradcam", exist_ok=True)
    
    for i in range(len(samples)):
        x, y = ds[i]
        x_tensor = x.unsqueeze(0).to(DEVICE)
        x_tensor.requires_grad = True
        
        # Get Heatmap
        heatmap = grad_cam(x_tensor)
        
        # Guided Backprop
        gbp = GuidedBackprop(model)
        gradients = gbp(x_tensor)
        gbp.remove_hooks()
        
        # We focus on the current image channels (last 3)
        if in_channels > 3:
            grads_curr = gradients[0, -3:, :, :] 
            img_tensor = x[-3:, :, :] 
        else:
            grads_curr = gradients[0]
            img_tensor = x
            
        orig_img = denormalize_image(img_tensor) # (H, W, 3) 0-255 uint8
        
        # Visualize Guided Backprop (Saliency)
        saliency = visualize_saliency(grads_curr.transpose(1, 2, 0))
        saliency_color = cv2.applyColorMap(saliency, cv2.COLORMAP_HOT)
        
        # Guided Grad-CAM
        ggcam = generate_guided_gradcam(heatmap, grads_curr)
        ggcam_viz = visualize_saliency(ggcam) # Normalize
        
        # Save
        sample_val = samples.iloc[i][target]
        base_name = f"output/gradcam/Viz_{target}_{condition}_{i}_Val_{sample_val:.3f}"
        
        # 1. GradCAM Overlay
        heatmap_resized = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(f"{base_name}_GradCAM.jpg", overlay)
        
        # 2. Guided Backprop (Saliency)
        cv2.imwrite(f"{base_name}_GuidedBP.jpg", saliency)
        
        # 3. Guided GradCAM (High Res Class-Discriminative)
        cv2.imwrite(f"{base_name}_GuidedGradCAM.jpg", ggcam_viz)
        
        print(f"Saved visualizations for sample {i}")

if __name__ == "__main__":
    main()
