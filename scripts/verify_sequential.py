
import pandas as pd
import torch
import numpy as np
from algae_fusion.data.dataset import MaskedImageDataset
from algae_fusion.models.backbones import ResNetRegressor
from algae_fusion.features.sliding_window_stochastic import compute_sliding_window_features_stochastic
from torchvision import transforms

# 1. Create Dummy Data
data = {
    'file': ['1.png', '2.png', '3.png', '4.png'],
    'time': [0, 1, 2, 3],
    'condition': ['Light', 'Light', 'Light', 'Light'],
    'Source_Path': ['path', 'path', 'path', 'path'],
    'cell_mean_area': [10, 20, 30, 40],
    'Dry_Weight': [1.0, 2.0, 3.0, 4.0]
}
df = pd.DataFrame(data)

# 2. Run Feature Engineering
print("--- Testing Feature Engineering ---")
df_aug = compute_sliding_window_features_stochastic(df, window_size=2, morph_cols=['cell_mean_area'])
print("Augmented Columns:", [c for c in df_aug.columns if 'Prev' in c])

# 3. Test Dataset
print("\n--- Testing Dataset Stacking ---")
# Mock _load_and_mask_image to avoid file system dependency in this test
def mock_load(self, src, file):
    from PIL import Image
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

MaskedImageDataset._load_and_mask_image = mock_load

ds = MaskedImageDataset(df_aug, target_col='Dry_Weight', img_size=(224, 224), transform=transforms.ToTensor())
tensor, label = ds[2] # t=2 should have Prev1 and Prev2

print("Stacked Tensor Shape:", tensor.shape)
assert tensor.shape == (9, 224, 224), f"Wrong shape: {tensor.shape}"

# 4. Test Model
print("\n--- Testing Model Forward Pass ---")
model = ResNetRegressor(backbone="resnet18", in_channels=9)
output = model(tensor.unsqueeze(0))
print("Model Output Shape:", output.shape)
assert output.shape == (1, 1), f"Wrong output shape: {output.shape}"

print("\n[SUCCESS] Sequential verification passed!")
