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
        # We need to respect that structure.
        image_dir = os.path.join(PATH_PREFIX, source_path)
        # Assuming masks are in a sibling folder named 'masks' relative to 'images'
        # e.g. TIMECOURSE/0h/images -> TIMECOURSE/0h/masks
        mask_dir = image_dir.replace("images", "masks")
        
        image_path = os.path.join(image_dir, base + ".jpg")
        mask_path = os.path.join(mask_dir, base + "_mask.png")

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

class ODETimeSeriesDataset(Dataset):
    """
    Unified Dataset for Neural ODE / ODE-RNN.
    Supports two modes:
    1. mode='trajectory' (default): Returns full biological sequences (0h -> 24h).
    2. mode='window': Chunks sequences into sliding windows for step-wise forecasting.
    """
    def __init__(self, df, feature_cols, target, time_col='time', group_col='file', mode='trajectory', window_size=3):
        # Allow target to be a column name (str) or an array (np.ndarray/pd.Series)
        self.target_is_array = not isinstance(target, str)
        local_df = df.copy()
        
        if self.target_is_array:
             # Inject the external target array into the dataframe for consistent sorting/grouping
             target_col_name = "_internal_target"
             local_df[target_col_name] = target
             self.target_col = target_col_name
        else:
             self.target_col = target
             
        self.df = local_df.sort_values([group_col, time_col]).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.time_col = time_col
        self.group_col = group_col
        self.mode = mode
        self.window_size = window_size
        
        self.samples = []
        
        # 1. Group by biological replicate
        grouped = self.df.groupby(group_col)
        
        for _, group in grouped:
            times = group[self.time_col].values.astype(np.float32)
            feats = group[self.feature_cols].values.astype(np.float32)
            targets = group[self.target_col].values.astype(np.float32)
            
            # Condition (Assume consistent within group)
            cond_str = group['condition'].iloc[0] if 'condition' in group.columns else 'Unknown'
            cond_label = 1.0 if cond_str == 'Light' else 0.0

            n = len(times)
            if n == 0: continue

            if self.mode == 'trajectory':
                # --- Mode 1: Full Trajectory ---
                # Simply wrap the whole group as one sample
                self.samples.append({
                    'times': torch.tensor(times),
                    'features': torch.tensor(feats),
                    'targets': torch.tensor(targets),
                    'mask': torch.ones(n), # All present
                    'condition': torch.tensor(cond_label)
                })

            elif self.mode == 'window':
                # --- Mode 2: Sliding Window Chunks ---
                if self.window_size is None:
                    raise ValueError("window_size must be provided for mode='window'")

                # Safety Check: Group must differ in length to form at least one window? 
                # Actually, if n <= window_size, we might not have enough history for the strict definition.
                # Here we loop range(1, n).
                if n < 2: 
                    continue # Need at least 2 points to predict something (t=1 from t=0)
                
                # We want to predict index 'i' using 'i-window_size' to 'i-1'
                for i in range(1, n):
                    # Start index (inclusive)
                    start_idx = max(0, i - self.window_size)
                    # End index (inclusive for slicing needs +1)
                    end_idx = i + 1
                    
                    chunk_times = times[start_idx : end_idx]
                    
                    # Additional Safety: If chunk is too small?
                    # Current logic allows short chunks at start (padding implicit by short length).
                    # But if chunk len = 1, it means we only have target, no history?
                    # i=1, start=0, end=2 -> [0, 1]. Len=2. Mask=[-1]=0. History=[0]. Target=[1]. OK.
                    # As long as len > 1, we are good.
                    if len(chunk_times) < 2:
                        continue

                    chunk_feats = feats[start_idx : end_idx].copy()
                    chunk_targets = targets[start_idx : end_idx]
                    
                    # Construct Mask: 1 for history AND current target (we want to predict t)
                    mask = np.ones(len(chunk_times))
                    # mask[-1] = 0 # OLD: Mask out target. NEW: We want to predict target, so mask=1.
                    
                    # Do NOT mask future features (we want to use Features(t) to predict Target(t))
                    # chunk_feats[-1, :] = 0 
                    
                    self.samples.append({
                        'times': torch.tensor(chunk_times),
                        'features': torch.tensor(chunk_feats),
                        'targets': torch.tensor(chunk_targets),
                        'mask': torch.tensor(mask),
                        'condition': torch.tensor(cond_label)
                    })
                    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_ode_batch(batch):
    """
    Custom collate function to handle variable length time series.
    Pads sequences to max length in batch.
    """
    # 1. Sort by length (descending) for packing if needed
    batch.sort(key=lambda x: len(x['times']), reverse=True)
    
    max_len = max([len(x['times']) for x in batch])
    feature_dim = batch[0]['features'].shape[1]
    
    padded_x = []
    padded_y = []
    padded_t = []
    padded_mask = []
    conditions = []
    
    for item in batch:
        l = len(item['times'])
        
        # Features
        px = torch.zeros(max_len, feature_dim)
        px[:l] = item['features']
        padded_x.append(px)
        
        # Targets
        py = torch.zeros(max_len)
        py[:l] = item['targets']
        padded_y.append(py)
        
        # Times
        pt = torch.zeros(max_len)
        pt[:l] = item['times']
        padded_t.append(pt)
        
        # Mask
        pm = torch.zeros(max_len)
        pm[:l] = 1
        padded_mask.append(pm)
        
        conditions.append(item['condition'])
        
    return {
        'features': torch.stack(padded_x), # [B, T, D]
        'targets': torch.stack(padded_y),  # [B, T]
        'times': torch.stack(padded_t),    # [B, T]
        'mask': torch.stack(padded_mask),   # [B, T]
        'conditions': torch.stack(conditions) # [B]
    }

# Backward compatibility alias
AlgaeTimeSeriesDataset = ODETimeSeriesDataset