import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from algae_fusion.config import PATH_PREFIX

class MaskedImageDataset(Dataset):
    def __init__(self, df, target_col, img_size=(224, 224), transform=None, labels=None, in_channels=3):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.img_size = img_size
        self.transform = transform
        self.labels = labels
        self.in_channels = in_channels

    def __len__(self):
        return len(self.df)

    def _load_and_mask_image(self, source_path, file_name):
        """Helper to load and mask a single image."""
        base = file_name.replace("_mask.png", "").replace(".png", "")
        # source_path might already have PATH_PREFIX joined if it was processed before, 
        # but in __getitem__ we pass row["Source_Path"] which is relative to data/
        full_source = os.path.join(PATH_PREFIX, source_path)
        parent_folder = os.path.dirname(full_source)
        image_path = os.path.join(parent_folder, "images", base + ".jpg")
        mask_path = os.path.join(parent_folder, "masks", base + "_mask.png")

        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((self.img_size[0], self.img_size[1], 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.ones(img.shape[:2], np.uint8) * 255

        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
        mask_norm = (mask / 255.0).astype(np.float32)
        masked_img = (img.astype(np.float32) * mask_norm[..., None]).astype(np.uint8)
        return Image.fromarray(masked_img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        all_tensors = []
        all_tensors = []
        # Dynamic History Logic
        # Total Channels = (History + Current) * 3
        # History Frames = (In_Channels / 3) - 1
        # Example: 12 channels -> (12/3) - 1 = 3 history frames (Prev3, Prev2, Prev1)
        if self.in_channels > 3:
            num_history = (self.in_channels // 3) - 1
            # Sequence: [t-N, ..., t-2, t-1, t]
            # We iterate backwards from N down to 1
            history_keys = [f"Prev{k}_file" for k in range(num_history, 0, -1)]
            
            # 1. Load History
            for key in history_keys:
                if key in row and pd.notna(row[key]):
                    # Load historical frame
                    hist_file = row[key]
                    hist_src  = row[key.replace("_file", "_Source_Path")]
                    pil = self._load_and_mask_image(hist_src, hist_file)
                else:
                    # Padding: If no history, repeat the current image (t)
                    pil = self._load_and_mask_image(row["Source_Path"], row["file"])
                
                t = self.transform(pil) if self.transform else transforms.ToTensor()(pil)
                all_tensors.append(t)
                
        # 2. Load Current Image (t)
        curr_pil = self._load_and_mask_image(row["Source_Path"], row["file"])
        curr_t = self.transform(curr_pil) if self.transform else transforms.ToTensor()(curr_pil)
        all_tensors.append(curr_t)
        
        
        # 3. Stack along Channel dimension
        if len(all_tensors) > 1:
            stacked_tensor = torch.cat(all_tensors, dim=0)
        else:
            stacked_tensor = all_tensors[0]

        if self.labels is not None:
            label_val = self.labels[idx]
        else:
            label_val = self.df.iloc[idx][self.target_col]
            
        label = torch.tensor(label_val, dtype=torch.float32)
        return stacked_tensor, label
