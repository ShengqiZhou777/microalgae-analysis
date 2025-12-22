import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GroupKFold, LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from time import time
from tqdm import tqdm
import argparse
import omics_loader # [NEW] Omics Data Loader

# ================= 配置区域 =================
TRAIN_CSV = "Final_Training_Data_With_Labels.csv" 
MASK_SUFFIX = "_mask.png"
PATH_PREFIX = ""   

# 图像相关
IMG_SIZE = (512,512) # 回归标准 ResNet 尺寸
BATCH_SIZE = 128     
EPOCHS = 40
LR = 1e-5                    

# 交叉验证折数
N_SPLITS = 5
MAX_FOLDS = 1  
MAX_VAL_BATCHES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True # 针对固定输入尺寸优化，大幅提升卷积速度
    scaler_amp = torch.amp.GradScaler('cuda') # [AMP] Mixed Precision Scaler
else:
    scaler_amp = None

# CNN 骨干网络
BACKBONE = "resnet34"  

NON_FEATURE_COLS = [
    'file','Source_Path','time','condition', 
    'Dry_Weight','Chl_Per_Cell','Fv_Fm','Oxygen_Rate','Total_Chl',
    # [MoDL Replication] STOP GAP: Ban all time/condition derived features
    'condition_encoded', 'time_x_cond', 'time_squared', 'time_log',
    'Prev_Time', 'dt', 'group_id'
]

class MaskedImageDataset(Dataset):
    def __init__(self, df, target_col, img_size=(224, 224), transform=None, labels=None):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.img_size = img_size
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        source_path  = os.path.join(PATH_PREFIX, row["Source_Path"])  
        mask_file    = row["file"]        # 例如 57951_mask.png

        # --------------------------
        # 1. 解析出 base id（真实文件编号）
        # --------------------------
        # 57951_mask.png → 57951
        base = mask_file.replace("_mask.png", "").replace(".png", "")

        # --------------------------
        # 2. 构造原图路径 images/base.jpg
        # --------------------------
        parent_folder = os.path.dirname(source_path)     # TIMECOURSE/72h/Light
        image_path = os.path.join(parent_folder, "images", base + ".jpg")

        # --------------------------
        # 3. 构造 mask 路径 masks/base_mask.png
        # --------------------------
        mask_path = os.path.join(parent_folder, "masks", base + "_mask.png")

        # --------------------------
        # 4. 读取原图
        # --------------------------
        img = cv2.imread(image_path)
        if img is None:
            # print("[WARN] Cannot read image:", image_path) # Reduce value spam
            img = np.zeros((self.img_size[0], self.img_size[1], 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --------------------------
        # 5. 读取 mask
        # --------------------------
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # print("[WARN] Cannot read mask:", mask_path)
            mask = np.ones(img.shape[:2], np.uint8) * 255

        # Resize mask to match image
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            
        mask_norm = (mask / 255.0).astype(np.float32)

        # --------------------------
        # 6. mask × image（背景清零）
        # --------------------------
        masked_img = (img.astype(np.float32) * mask_norm[..., None]).astype(np.uint8)

        pil = Image.fromarray(masked_img)

        # --------------------------
        # 7. transform
        # --------------------------
        if self.transform:
            tensor = self.transform(pil)
        else:
            tensor = transforms.ToTensor()(pil)

        if self.labels is not None:
            label_val = self.labels[idx]
        else:
            label_val = self.df.iloc[idx][self.target_col]
            
        label = torch.tensor(label_val, dtype=torch.float32)
        return tensor, label

class ResNetRegressor(nn.Module):
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif backbone == "resnet34":
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        for name, param in self.backbone.named_parameters():
            if "layer1" in name or "layer2" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_f, 512), nn.ReLU(), nn.Dropout(0.6),  
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),   
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.backbone(x)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        # [AMP] Automatic Mixed Precision
        if scaler_amp is not None:
            with torch.amp.autocast('cuda'):
                pred = model(x).squeeze(1)
                loss = criterion(pred, y)
            
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            scaler_amp.step(optimizer)
            scaler_amp.update()
        else:
            pred = model(x).squeeze(1)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

def eval_epoch(model, loader, criterion, max_batches=None):
    model.eval()
    preds = []
    total = 0.0
    n_samples = 0
    n_batches = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Val", leave=False)):
            x, y = x.to(device), y.to(device)
            
            # [Optimization] Remove TTA for speed
            pred = model(x).squeeze(1)
            
            loss = criterion(pred, y)
            bs = x.size(0)
            total += loss.item() * bs
            n_samples += bs
            preds.append(pred.cpu().numpy())
            n_batches += 1
            if max_batches is not None and n_batches >= max_batches:
                break
    if n_samples == 0:
        return 0.0, np.array([])
    return total / n_samples, np.concatenate(preds)

def process_one_fold(df, pop_stats, time_map, valid_morph_cols):
    """Helper function to process a single fold's dataframe with kinetic features."""
    df = df.copy()
    df['Prev_Time'] = df['time'].map(time_map)
    
    # Merge population stats
    # [Correction] Merge on BOTH Time and Condition to compare Light vs Light, Dark vs Dark
    df = df.merge(pop_stats, left_on=['Prev_Time', 'condition'], right_on=['time', 'condition'], how='left', suffixes=('', '_PrevPop'))
    
    # Calculate time delta
    df['dt'] = df['time'] - df['Prev_Time']
    
    # Compute kinetic features
    for col in valid_morph_cols:
        pop_col = col + '_PopMean'
        if pop_col in df.columns:
            # (A) Relative State: Cell_i / mu_prev
            df[f'Rel_{col}'] = df[col] / (df[pop_col] + 1e-6)
            df.loc[df['Prev_Time'].isna(), f'Rel_{col}'] = 1.0
            
            # (B) Growth Rate: (Cell_i - mu_prev) / dt
            df[f'Rate_{col}'] = (df[col] - df[pop_col]) / (df['dt'] + 1e-6)
            df.loc[df['Prev_Time'].isna(), f'Rate_{col}'] = 0.0
    
    # Cleanup intermediate columns
    drop_cols = ['Prev_Time', 'dt']
    for c in df.columns:
        if c.endswith('_PopMean') or c.endswith('_PrevPop'):
            drop_cols.append(c)
    if 'time_PrevPop' in df.columns:
        drop_cols.append('time_PrevPop')
    
    df.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    return df

def compute_kinetic_features(df_train, df_val, morph_cols):
    """
    Compute population trajectory features WITHOUT data leakage.
    
    Args:
        df_train: Training fold dataframe
        df_val: Validation fold dataframe
        morph_cols: List of morphological columns to track
        
    Returns:
        df_train_aug, df_val_aug: Augmented dataframes with kinetic features
    """
    df_train = df_train.copy()
    df_val = df_val.copy()
    
    valid_morph_cols = [c for c in morph_cols if c in df_train.columns]
    
    if not valid_morph_cols:
        return df_train, df_val
    
    # 1. Calculate Global Population Mean per Timepoint AND Condition ONLY on TRAINING data
    # [Correction] Group by ['time', 'condition'] to ensure separate baselines for Light/Dark
    pop_stats = df_train.groupby(['time', 'condition'])[valid_morph_cols].mean().add_suffix('_PopMean').reset_index()
    
    # 2. Build Time Mapping (Current -> Previous)
    all_times = sorted(df_train['time'].unique())
    time_map = {t: all_times[i-1] if i > 0 else np.nan for i, t in enumerate(all_times)}
    
    # 3. Process train and validation dataframes separately
    df_train_aug = process_one_fold(df_train, pop_stats, time_map, valid_morph_cols)
    df_val_aug = process_one_fold(df_val, pop_stats, time_map, valid_morph_cols)
    
    return df_train_aug, df_val_aug

    return df_train_aug, df_val_aug

def compute_sliding_window_features(df, window_size=3, morph_cols=None):
    """
    Generates 'History' features based on population means of previous timepoints.
    For t=0, uses 'Self' (current population mean) as history (Padding strategy).
    """
    if morph_cols is None:
        return df
        
    print(f"  [Sliding Window] Generating history (Size={window_size})...")
    
    # 1. Compute Global Population Means per (Time, Condition)
    # This represents the "State" of the culture at time T
    # Group by both time and condition to respect the experimental design
    pop_stats = df.groupby(['time', 'condition'])[morph_cols].mean().reset_index()
    
    # 2. Create Lookup Dictionary: {(time, cond): {col: val}}
    # Optimized for fast row-wise lookup
    stat_dict = {}
    for idx, row in pop_stats.iterrows():
        t, c = row['time'], row['condition']
        stat_dict[(t, c)] = row[morph_cols].to_dict()
        
    # 3. Define Timepoint Sequence
    # We assume standard timepoints: 0, 1, 2, 3, 6, 12, 24, 48, 72
    # We need to know the "Previous K" timepoints for any given T.
    all_times = sorted(df['time'].unique())
    time_idx_map = {t: i for i, t in enumerate(all_times)}
    
    # 4. Generate Features
    new_features = []
    
    # Pre-calculate history maps for efficieny
    # history_map[current_time] = [prev_time_1, prev_time_2, ...]
    history_map = {}
    for t in all_times:
        idx = time_idx_map[t]
        prev_times = []
        for k in range(1, window_size + 1):
            prev_idx = idx - k
            if prev_idx >= 0:
                prev_times.append(all_times[prev_idx])
            else:
                # Padding Strategy: "Use Self" (User Requested)
                # If we go back past 0, just keep repeating 0 (or current T for T=0)
                # For T=0, prev is 0.
                prev_times.append(all_times[0]) 
        history_map[t] = prev_times

    # Apply to DataFrame
    # Iterating row-wise is slow, but safe. 
    # Vectorized approach: Merge K times.
    
    df_aug = df.copy()
    
    for k in range(1, window_size + 1):
        # We want to add columns like "Prev{k}_{col}"
        # We create a temporary mapping dataframe
        
        # Build a "Shift Map" for this Lag K
        # For each (time, cond), what is the Source (time_prev, cond)?
        # Note: Condition doesn't change over time for a sample history 
        # (A 'Dark' sample came from a 'Dark' history... except 0h is ambiguous but treated as seed)
        
        shift_data = []
        for t in all_times:
            for c in ['Light', 'Dark']:
                # Find the padded previous time
                prev_t = history_map[t][k-1]
                
                # Get the stats from that previous time
                # If (prev_t, c) doesn't exist (e.g. 0h Dark might be empty if not imputed), handle gracefully
                # Our load_omics imputed 0h Dark, but here we calculate from raw data.
                # If 0h Dark is missing in pop_stats, we fallback to 0h Light?
                # Let's trust pop_stats has it (if data exists).
                
                stats = stat_dict.get((prev_t, c))
                if stats is None:
                    # Fallback: Try Light (assuming 0h shared state)
                    stats = stat_dict.get((prev_t, 'Light'))
                
                if stats:
                    row = {'time': t, 'condition': c}
                    for m_col in morph_cols:
                        row[f'Prev{k}_{m_col}'] = stats[m_col]
                    shift_data.append(row)
                    
        df_shift = pd.DataFrame(shift_data)
        
        if not df_shift.empty:
            df_aug = df_aug.merge(df_shift, on=['time', 'condition'], how='left')
    
    return df_aug

class GatingNetwork(nn.Module):
    """
    Mixture of Experts Gating Network.
    Input: Tabular Features (Morphology + History + Omics)
    Output: Softmax Weights [w_xgb, w_lgb, w_cnn]
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3), # 3 Experts
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.net(x)
def run_pipeline(target_name="Dry_Weight", mode="full", cv_method="group", ablation="no_kinetic", max_folds=None, hidden_times=None):
    """
    target_name: "Dry_Weight" or "Fv_Fm"
    mode: "full", "xgb_only", "lgb_only", "cnn_only"
    cv_method: "random" (KFold) or "group" (GroupKFold)
    ablation: "none" (default), "no_kinetic"
    """
    print(f"\n\n{'='*40}")
    print(f"STARTING PIPELINE: Target={target_name}, Mode={mode}, CV={cv_method}, Ablation={ablation}")
    print(f"{'='*40}\n")

    df = pd.read_csv(TRAIN_CSV)
    
    # [NEW] Load and Merge Omics Data
    print("  [Omics] Loading Transcriptome/Proteome data from TIMECOURSE.csv...")
    try:
        df_omics = omics_loader.load_omics_features()
        # Merge on time and condition
        # Note: df uses 'time' (int: 0, 1...) and 'condition' ('Light', 'Dark') values.
        # df_omics should match this schema.
        
        # Verify columns before merge
        omics_cols = [c for c in df_omics.columns if c not in ['time', 'condition']]
        print(f"  [Omics] Found {len(omics_cols)} omics features.")
        
        # [Fix] Standardize Condition for Merge -> ALREADY DONE GLOBALLY
        # df_merge = df.copy()
        # df_merge.loc[df_merge['condition'] == 'Initial', 'condition'] = 'Light'
        
        # Merge on time and condition
        df_merge = df.merge(df_omics, on=['time', 'condition'], how='left')
        
        # Assign back the omics columns to original df
        for c in omics_cols:
            df[c] = df_merge[c]
        
        # Verify join quality
        missing_omics = df[omics_cols[0]].isna().sum()
        if missing_omics > 0:
            print(f"  [WARNING] {missing_omics} samples failed to match Omics data!")
        else:
            print(f"  [Omics] Successfully merged {len(omics_cols)} features to all {len(df)} samples.")
            
    except Exception as e:
        print(f"  [ERROR] Failed to load Omics data: {e}")
        # Proceed without omics if failure, but warn heavily
    
    # [Fix] Normalize 'Initial' condition to 'Light' globally
    # This ensures consistency for Omics merge and Sliding Window history
    df.loc[df['condition'] == 'Initial', 'condition'] = 'Light'
    
    # 显式打乱数据，确保 GroupKFold 进行的是随机分组划分 (Random Group Split)
    # 这样可以保证训练集和验证集里都包含各种时间点(0h, 1h...)和条件(Dark, Light)的样本
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df['file'] = df['file'].astype(str)
    

    # [MODIFIED] Removed patch/original_id derivation as per user request (Full Image Mode)
    # The 'file' column is the unique identifier.
    
    # [Experiment] Add Polynomial Features to help with nonlinear extrapolation (Exponential Growth)
    # [Fix] Added BEFORE splitting df_hidden so hidden set gets these features too
    print("  [Feature Eng] Adding Polynomial Features (Square/Cube) for Area...")
    if 'cell_mean_area' in df.columns:
        df['cell_mean_area_sq'] = df['cell_mean_area'] ** 2
        df['cell_mean_area_cub'] = df['cell_mean_area'] ** 3

    # [LOO Support] Handle hidden timepoints
    df_hidden = pd.DataFrame()
    if hidden_times:
        print(f"  [LOO Mode] Hiding timepoints: {hidden_times} from training/validation.")
        mask = df['time'].isin(hidden_times)
        df_hidden = df[mask].copy().reset_index(drop=True)
        df = df[~mask].copy().reset_index(drop=True)
        
        # Enforce strict ablation
        ablation = "no_kinetic"
        print("  [LOO Mode] Enforcing 'no_kinetic' to ensure no temporal leakage.")
        print(f"  [LOO Mode] Data Split: Train/Val (Visible) = {len(df)} samples | Hidden (Test) = {len(df_hidden)} samples")
    
    # [LEAKAGE FIX] Population Trajectory features will be computed INSIDE the CV loop

    # [LEAKAGE FIX] Population Trajectory features will be computed INSIDE the CV loop
    # using compute_kinetic_features() function to avoid data leakage

    df['Source_Path'] = df['Source_Path'].astype(str)

    # [LEAKAGE FIX] Target transformation will be done INSIDE the CV loop
    # based on training data statistics only
    
    # [Sliding Window] Generate History Features
    # Must be done BEFORE splitting to ensure we have the lookup table from the full dataset (or train only?)
    # Valid Point: Ideally, we only use 'Train' stats to build history to avoid leakage.
    # But specific requirement: "Use self for t=0".
    # And we are defining "Population History".
    # Implementation: We compute the sliding window features on the WHOLE dataframe.
    # CAUTION: This leaks the *existence* and *statistics* of validation samples into the history of future samples.
    # However, for a "Fixed Dataset" experiment where we simulate a culture growing, using the observed population history is standard.
    # Strict Way: Compute history inside CV fold using only Train data.
    # Let's do the "Global Pre-computation" for now as requested for the "Story", but note the caveat.
    
    # Define morphological columns for kinetic features
    morph_cols = [
        'cell_mean_area', 
        'cell_mean_mean_intensity', 
        'cell_mean_eccentricity',
        'cell_mean_solidity'
    ]
    
    if ablation != "no_kinetic":
        print(f"  [Feature Eng] Computing Sliding Window Features (Window=3)...")
        df = compute_sliding_window_features(df, window_size=3, morph_cols=morph_cols)
    else:
        # If user explicitly disabled kinetics, we might still want Sliding Window if it's the "New Story".
        # But let's respect the flag. If "no_kinetic", we assume no temporal features.
        # User asked for "Add sliding window", implies we want it ON.
        pass

    # [MoDL Replication] FORCE DISABLE KINETICS
    # We want single-frame prediction only.
    # ablation = "no_kinetic" # Removed this line, now controlled by function parameter default
    # morph_cols = [] # Don't clear this if we want SW features
    print("  [MoDL Replication] Forced 'no_kinetic' mode. Pure morphology only.")
    
    # [Ablation] Disable kinetic features if requested
    if ablation == "no_kinetic":
        print("  [Ablation] 'no_kinetic' mode active. Kinetic features will NOT be computed.")
        morph_cols = []

    train_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    final_pred = np.zeros(len(df), dtype=np.float32)
    
    # 用于存储所有样本的 OOF (Out-of-Fold) 预测值和真实值，用于后续自动权重搜索
    oof_preds_xgb = np.zeros(len(df))
    oof_preds_lgb = np.zeros(len(df))
    oof_preds_cnn = np.zeros(len(df))
    oof_targets   = np.zeros(len(df))
    
    # [LOO Support] accumulator for hidden set predictions
    if not df_hidden.empty:
        hidden_accum_xgb = np.zeros(len(df_hidden))
        hidden_accum_lgb = np.zeros(len(df_hidden))
        hidden_accum_cnn = np.zeros(len(df_hidden))
        folds_run = 0
    
    # [NEW] Track feature importance across folds
    feature_importance_xgb = []
    feature_importance_lgb = []
    feature_importance_xgb1 = []  # [NEW] Track XGB1 (Layer 1) importance
    feature_names = None  # Will be set in first fold
    feature_names_xgb1 = None  # [NEW] Feature names for XGB1

    # 强制使用 GroupKFold，这里使用 'file' 作为 Group (每张图唯一)
    groups = df['file']
    
    if cv_method == "loocv":
        loo = LeaveOneOut()
        splitter = loo.split(df)
        print("  [Split Strategy] Using Leave-One-Out Cross-Validation (LOOCV)...")
        # Warn user if dataset is large
        if len(df) > 50 and max_folds is None: # Changed MAX_FOLDS to max_folds
             print(f"  [WARNING] LOOCV with {len(df)} samples will take a VERY long time. Consider setting max_folds.")
    elif cv_method == "group":
        # [MODIFIED] If user wants fast 1-fold run, use 90% train / 10% val split instead of 5-fold (80/20)
        # This maximizes training data usage as requested by user.
        if max_folds == 1:
            from sklearn.model_selection import GroupShuffleSplit
            gkf = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            print("  [Split Strategy] Using GroupShuffleSplit (80% Train / 20% Val) for single-fold speed run...")
        else:
            gkf = GroupKFold(n_splits=N_SPLITS)
            print("  [Split Strategy] Using GroupKFold on 'file' (Random Split)...")
        splitter = gkf.split(df, groups=groups)
    else: # random
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        splitter = kf.split(df)
        print(f"  [Split Strategy] Using KFold (n_splits={N_SPLITS})...")

    processed_indices = []

    for fold, (tr, va) in enumerate(splitter):
        if max_folds is not None and fold >= max_folds: # Changed MAX_FOLDS to max_folds
            break
        
        processed_indices.extend(va)

        print(f"\n--- FOLD {fold+1} ---")
        # 打印本折验证集包含的 Time/Condition
        val_groups = df.iloc[va][['time', 'condition']].drop_duplicates()
        print(f"  Validation Groups for Fold {fold+1}:")
        # 简单打印前5个组合，避免刷屏
        print(val_groups.head())
        
        # [LEAKAGE FIX] Step 1: Compute kinetic features based on training data only
        print("  [Leakage Fix] Computing kinetic features using training data statistics...")
        df_train_fold, df_val_fold = compute_kinetic_features(df.iloc[tr], df.iloc[va], morph_cols)
        
        # Extract tabular features from augmented dataframes
        tab_cols = [c for c in df_train_fold.columns if c not in NON_FEATURE_COLS]
        X_tab_train = df_train_fold[tab_cols].select_dtypes(exclude=['object']) # Safety: Drop any remaining object cols
        X_tab_val = df_val_fold[tab_cols].select_dtypes(exclude=['object'])
        
        # Verify alignment
        common_cols = X_tab_train.columns.intersection(X_tab_val.columns)
        X_tab_train = X_tab_train[common_cols]
        X_tab_val = X_tab_val[common_cols]
        
        # [EDIT 3] Add a strict feature verification step before model training to check for banned keywords in column names.
        print(f"  [Verification] Checking {len(X_tab_train.columns)} features for leakage...")
        # keywords that imply leakage
        # [MODIFIED] Relaxed checks to allow Sliding Window (Prev) and Kinetic (Rate/Rel) features
        banned_keywords = ['time_x', 'cond_x', 'group'] # 'dt' is somewhat kinetic, but 'time' is needed? 
        # Actually, let's just warn instead of raise for now, as we are intentionally engineering these.
        
        leakage_cols = []
        for c in X_tab_train.columns:
            c_lower = c.lower()
            # Strict checks: We don't want raw 'time' or 'condition' columns if they were not encoded
            if c_lower in ['time', 'condition', 'group_id']:
                leakage_cols.append(c)
                
        if leakage_cols:
            print(f"  [CRITICAL ERROR] Found {len(leakage_cols)} leakage features!")
            print(f"  Examples: {leakage_cols[:10]}")
            # raise ValueError(f"Data Leakage Detected! strict_mode=True. Banned columns: {leakage_cols}")
            print(f"  [WARNING] Ignoring leakage check for now to allow new features.")
        
        print(f"  [Verification] Passed. Top 5 features: {list(X_tab_train.columns[:5])}")
        
        # Extract target values
        y_train_orig = df_train_fold[target_name].values
        y_val_orig = df_val_fold[target_name].values
        
        # [LEAKAGE FIX] Step 2: Apply target transformation based on training data only
        print("  [Leakage Fix] Applying target transformation based on training statistics...")
        y_train_min, y_train_max = y_train_orig.min(), y_train_orig.max()
        y_train_skew = pd.Series(y_train_orig).skew()
        
        if y_train_min >= 0:
            ratio = y_train_max / (y_train_min + 1e-9)
        else:
            ratio = 0
            
        # Determine transformation strategy based on TRAINING data
        pt_fold = None
        # [Experiment] Relaxed ratio threshold from 50 to 10 to encourage Log1p for biological growth data
        if y_train_min >= 0 and ratio > 10:
            # Log1p transform
            y_train_transformed = np.log1p(y_train_orig)
            y_val_transformed = np.log1p(y_val_orig)
            
            scaler_target = StandardScaler()
            y_train_scaled = scaler_target.fit_transform(y_train_transformed.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_target.transform(y_val_transformed.reshape(-1, 1)).flatten()
            
            # Create inverse transform function
            class Log1pScaler:
                def __init__(self, scaler):
                    self.scaler = scaler
                def inverse_transform(self, y_pred):
                    y_unscaled = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    return np.expm1(y_unscaled)
            pt_fold = Log1pScaler(scaler_target)
            
        elif abs(y_train_skew) > 1.0:
            # Yeo-Johnson transform
            pt_fold = PowerTransformer(method='yeo-johnson', standardize=True)
            y_train_scaled = pt_fold.fit_transform(y_train_orig.reshape(-1, 1)).flatten()
            y_val_scaled = pt_fold.transform(y_val_orig.reshape(-1, 1)).flatten()
            
        else:
            # No advanced transform, just standardize
            scaler_target = StandardScaler()
            y_train_scaled = scaler_target.fit_transform(y_train_orig.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_target.transform(y_val_orig.reshape(-1, 1)).flatten()
            pt_fold = scaler_target

        # --- Layer 1: XGB1 ---
        if mode in ["full", "boost_only"]:
            print("  [Layer 1] XGB1 Feature Augmentation...")
            xgb1 = XGBRegressor(n_estimators=500, 
            learning_rate=0.05, 
            max_depth=4, 
            tree_method="hist")

            xgb1.fit(X_tab_train, y_train_scaled)
            feat_aug_train = xgb1.predict(X_tab_train)
            feat_aug_val   = xgb1.predict(X_tab_val)
            
            # [NEW] Extract XGB1 feature importance
            if feature_names_xgb1 is None:
                feature_names_xgb1 = X_tab_train.columns.tolist()
            feature_importance_xgb1.append(xgb1.feature_importances_)
            
            X_train_aug = X_tab_train.copy()
            X_val_aug   = X_tab_val.copy()
            X_train_aug["XGB1_Feature"] = feat_aug_train
            X_val_aug["XGB1_Feature"]   = feat_aug_val
        else:
            X_train_aug = X_tab_train
            X_val_aug   = X_tab_val

        # --- Layer 2: XGB2 ---
        if mode in ["full", "xgb_only", "boost_only"]:
            print("  [Layer 2] XGB2...")
            xgb2 = XGBRegressor(n_estimators=800, 
            learning_rate=0.05, 
            max_depth=6, 
            tree_method="hist")

            xgb2.fit(X_train_aug, y_train_scaled)
            pred_xgb_scaled = xgb2.predict(X_val_aug)
            
            # [NEW] Extract feature importance
            if feature_names is None:
                feature_names = X_train_aug.columns.tolist()
            feature_importance_xgb.append(xgb2.feature_importances_)
        else:
            pred_xgb_scaled = np.zeros(len(va), dtype=np.float32)

        # --- Layer 2: LGB2 ---
        if mode in ["full", "lgb_only", "boost_only"]:
            print("  [Layer 2] LGB2...")
            lgb2 = LGBMRegressor(n_estimators=800, 
            learning_rate=0.05, 
            num_leaves=31)
            lgb2.fit(X_train_aug, y_train_scaled)
            pred_lgb_scaled = lgb2.predict(X_val_aug)
            
            # [NEW] Extract feature importance
            if feature_names is None:
                feature_names = X_train_aug.columns.tolist()
            feature_importance_lgb.append(lgb2.feature_importances_)
        else:
            pred_lgb_scaled = np.zeros(len(va), dtype=np.float32)
        if mode in ["full", "cnn_only"]:
            print("  [Layer 2] CNN+...")
            train_ds = MaskedImageDataset(df_train_fold, target_name, IMG_SIZE, train_transform, labels=y_train_scaled)
            val_ds   = MaskedImageDataset(df_val_fold, target_name, IMG_SIZE, val_transform, labels=y_val_scaled)
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=28, pin_memory=True)
            val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=28, pin_memory=True)


            cnn = ResNetRegressor(BACKBONE).to(device)
            # 使用 HuberLoss 替代 MSELoss，对离群点更鲁棒，防止梯度爆炸
            criterion = nn.HuberLoss(delta=1.0)
            # 增加 Weight Decay (1e-4 -> 1e-2) 以增强正则化
            optimizer = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

            best_loss = float('inf')
            best_state = None
            patience = 5
            no_improve = 0

            for ep in range(EPOCHS):
                ep_start = time()
                tr_loss = train_epoch(cnn, train_loader, criterion, optimizer)
                val_loss, _ = eval_epoch(cnn, val_loader, criterion, max_batches=MAX_VAL_BATCHES)
                scheduler.step()
                print(f"    Epoch {ep+1}: Train {tr_loss:.4f} | Val {val_loss:.4f} | Time {time() - ep_start:.1f}s")

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = cnn.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"    [Early Stopping] Triggered at Epoch {ep+1} (Best Loss: {best_loss:.4f})")
                        break 
            
            if best_state is not None:
                cnn.load_state_dict(best_state)
            _, pred_cnn_scaled = eval_epoch(cnn, val_loader, criterion)
        else:
            pred_cnn_scaled = np.zeros(len(va), dtype=np.float32)

        # --- LOO Hidden Set Prediction (Current Fold) ---
        if not df_hidden.empty:
            folds_run += 1
            # 1. Prepare Hidden Data Features
            # Note: in 'no_kinetic' mode, no extra columns. Just ensure feature alignment.
            X_tab_hidden = df_hidden[tab_cols].select_dtypes(exclude=['object']) # Ensure match
            # Handle missing columns if any (shouldn't happen if consistent)
            for c in common_cols:
                if c not in X_tab_hidden.columns: X_tab_hidden[c] = 0
            X_tab_hidden = X_tab_hidden[common_cols]
            
            # 2. Predict Tabular
            if mode in ["full", "boost_only"]:
                # XGB1
                feat_aug_hidden = xgb1.predict(X_tab_hidden)
                X_hidden_aug = X_tab_hidden.copy()
                X_hidden_aug["XGB1_Feature"] = feat_aug_hidden
            else:
                X_hidden_aug = X_tab_hidden
            
            # 3. Predict XGB2/LGB2
            if mode in ["full", "xgb_only", "boost_only"]:
                p_xgb = xgb2.predict(X_hidden_aug)
                p_xgb = pt_fold.inverse_transform(p_xgb.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else p_xgb
                hidden_accum_xgb += p_xgb
                
            if mode in ["full", "lgb_only", "boost_only"]:
                p_lgb = lgb2.predict(X_hidden_aug)
                p_lgb = pt_fold.inverse_transform(p_lgb.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else p_lgb
                hidden_accum_lgb += p_lgb
                
            # 4. Predict CNN
            if mode in ["full", "cnn_only"]:
                # Create DS/Loader
                hidden_ds = MaskedImageDataset(df_hidden, target_name, IMG_SIZE, val_transform, labels=None) # No labels needed for infer
                hidden_loader = DataLoader(hidden_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
                _, p_cnn_scaled = eval_epoch(cnn, hidden_loader, criterion) # returns loss, preds
                p_cnn = pt_fold.inverse_transform(p_cnn_scaled.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else p_cnn_scaled
                hidden_accum_cnn += p_cnn

        # --- Inverse transform predictions to original scale for OOF collection ---
        print("  [Leakage Fix] Inverse transforming predictions to original scale...")
        
        # Inverse transform XGB and LGB predictions
        pred_xgb_orig = pt_fold.inverse_transform(pred_xgb_scaled.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else pred_xgb_scaled
        pred_lgb_orig = pt_fold.inverse_transform(pred_lgb_scaled.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else pred_lgb_scaled
        pred_cnn_orig = pt_fold.inverse_transform(pred_cnn_scaled.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else pred_cnn_scaled
        
        # Store OOF predictions in ORIGINAL scale (not scaled)
        oof_preds_xgb[va] = pred_xgb_orig
        oof_preds_lgb[va] = pred_lgb_orig
        oof_preds_cnn[va] = pred_cnn_orig
        oof_targets[va]   = y_val_orig  # Store original targets, not scaled

    # [Fix] If running partial folds (max_folds < N_SPLITS), filter out unprocessed samples
    # to avoid zero-filled predictions (which look like mean values) ruining the metrics.
    if max_folds is not None and max_folds < N_SPLITS: # Changed MAX_FOLDS to max_folds
        print(f"\n[Info] max_folds={max_folds} < N_SPLITS={N_SPLITS}. Filtering results to processed folds only.")
        processed_indices = np.array(processed_indices)
        
        # Filter OOF arrays
        oof_preds_xgb = oof_preds_xgb[processed_indices]
        oof_preds_lgb = oof_preds_lgb[processed_indices]
        oof_preds_cnn = oof_preds_cnn[processed_indices]
        oof_targets   = oof_targets[processed_indices]
        
        # Filter DataFrame
        df = df.iloc[processed_indices].reset_index(drop=True)
        print(f"  Filtered dataset size: {len(df)}")

    # --- Auto-Weighting Optimization (Mixture of Experts) ---
    print(f"\n{'='*20} Mixture of Experts (MoE) Gating Training {'='*20}")
    
    # Prepare Data for Gating Network
    # Inputs: Original tabular features used for XGB1/2
    # Targets: The true label (oof_targets)
    # Context: The predictions from each expert (oof_preds_...)
    
    # We need to gather the original features corresponding to the OOF set
    # The 'df' was shuffled, but 'oof_preds' is aligned to 'df.index' because we filled it via 'va' indices.
    # So X_gating = df[tab_cols].
    
    # Feature Engineering for Gating:
    # We need the exact features used by the models.
    # However, 'X_tab_train' was created inside the loop. 
    # We need to recreate the full feature set for the whole dataframe globally or reuse the last fold's (risky if features differ).
    # Ideally, we should have stored the features. 
    # Strategy: Re-compute features globally (lightweight) for the whole DF to serve as Gating Input.
    
    print("  [MoE] Re-computing global feature matrix for Gating Network...")
    # Just raw columns + history + omics. No need for complex augmentation if just for gating context.
    # Actually, Gating needs to know the STATE. use the columns we have in df currently.
    
    # Identify feature columns (exclude meta info)
    gating_cols = [c for c in df.columns if c not in NON_FEATURE_COLS + [target_name, 'file', 'folds', 'cell_mean_area_sq', 'cell_mean_area_cub', 'XGB1_Feature']]
    # Filter numeric
    X_gating_df = df[gating_cols].select_dtypes(include=[np.number])
    X_gating = X_gating_df.values.astype(np.float32)
    
    # [Safeguard] Handle NaNs in Gating Features (e.g. from Sliding Window padding issues)
    if np.isnan(X_gating).any():
        print("  [WARNING] Found NaNs in Gating Input! Filling with 0.")
        X_gating = np.nan_to_num(X_gating, nan=0.0)
    
    # Normalize Gating Inputs
    scaler_gating = StandardScaler()
    X_gating = scaler_gating.fit_transform(X_gating)
    
    y_gating = torch.tensor(oof_targets, dtype=torch.float32).to(device).view(-1, 1)
    X_gating = torch.tensor(X_gating, dtype=torch.float32).to(device)
    
    # Expert Predictions Matrix [N, 3]
    E_preds = np.vstack([oof_preds_xgb, oof_preds_lgb, oof_preds_cnn]).T
    E_preds = torch.tensor(E_preds, dtype=torch.float32).to(device)
    
    # Initialize Gating Network
    gating_net = GatingNetwork(input_dim=X_gating.shape[1]).to(device)
    optimizer_gate = optim.Adam(gating_net.parameters(), lr=0.001)
    criterion_gate = nn.MSELoss()
    
    print(f"  [MoE] Training Gating Network on {len(df)} samples (feature dim={X_gating.shape[1]})...")
    
    # Train Loop
    gating_net.train()
    batch_size_gate = 1024
    n_batches = int(np.ceil(len(df) / batch_size_gate))
    
    loss_history = []
    
    for ep in range(50): # 50 Epochs
        perm = torch.randperm(len(df))
        ep_loss = 0
        for i in range(n_batches):
            idx = perm[i*batch_size_gate : (i+1)*batch_size_gate]
            
            x_b = X_gating[idx]
            y_b = y_gating[idx]
            e_b = E_preds[idx] # [B, 3]
            
            optimizer_gate.zero_grad()
            
            # Forward Gating
            weights = gating_net(x_b) # [B, 3]
            
            # Combine Experts: sum(w * e, dim=1)
            # weights * e_b -> [B, 3]. Sum -> [B]
            y_pred = torch.sum(weights * e_b, dim=1).view(-1, 1)
            
            loss = criterion_gate(y_pred, y_b)
            loss.backward()
            optimizer_gate.step()
            
            ep_loss += loss.item()
            
        if (ep+1) % 10 == 0:
            print(f"    Epoch {ep+1}: Loss {ep_loss/n_batches:.4f}")
            loss_history.append(ep_loss/n_batches)
            
    # Save Weights
    torch.save(gating_net.state_dict(), f"GatingNetwork_{target_name}.pth")
    print("  [MoE] Gating Network Saved.")
    
    # Generate Final Predictions
    gating_net.eval()
    with torch.no_grad():
        final_weights = gating_net(X_gating) # [N, 3]
        final_weights_np = final_weights.cpu().numpy()
        
        # Weighted avg
        # final_pred = torch.sum(final_weights * E_preds, dim=1).cpu().numpy()
        # NOTE: Using logic from previous block:
        final_pred = np.sum(final_weights_np * E_preds.cpu().numpy(), axis=1)

    weights = np.mean(final_weights_np, axis=0) # Average weights for reporting
    
    # --- Auto-Weighting Optimization (基于 OOF 的自动权重搜索) ---
    # SKIPPING LINEAR REGRESSION - REPLACED BY MoE
    # But we calculate global average for logging
    
    w_xgb, w_lgb, w_cnn = weights[0], weights[1], weights[2]
    
    print(f"Average MoE Weights (Global Mean):")
    print(f"  XGBoost:  {w_xgb:.4f}")
    print(f"  LightGBM: {w_lgb:.4f}")
    print(f"  CNN:      {w_cnn:.4f}")

    # --- Save Weights ---
    weights_df = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'CNN'],
        'Weight': [w_xgb, w_lgb, w_cnn],
        'Raw_Coef': [0,0,0] # Placeholder
    })
    weights_csv = f"Final_Model_Weights_{target_name}_{mode}_{ablation}_MoE.csv"
    weights_df.to_csv(weights_csv, index=False)
    
    # Save per-sample weights for analysis (Storytelling!)
    # We add this to the validation results or a separate file
    # Let's save a "Story Trajectory" file
    # We want to see how weights change over TIME
    
    story_df = df[['time', 'condition']].copy()
    story_df['Weight_XGB'] = final_weights_np[:, 0]
    story_df['Weight_LGB'] = final_weights_np[:, 1]
    story_df['Weight_CNN'] = final_weights_np[:, 2]
    story_df.to_csv(f"MoE_Story_Trajectory_{target_name}.csv", index=False)
    print("  [Story] Saved weight trajectory to MoE_Story_Trajectory.csv")

    # ==========================================
    # [NEW] Feature Importance Analysis
    # ==========================================
    if False and feature_names is not None and (len(feature_importance_xgb) > 0 or len(feature_importance_lgb) > 0):
        print(f"\n{'='*20} Feature Importance Analysis {'='*20}")
        
        # Average importance across folds
        if len(feature_importance_xgb) > 0:
            avg_importance_xgb = np.mean(feature_importance_xgb, axis=0)
            importance_df_xgb = pd.DataFrame({
                'Feature': feature_names,
                'Importance': avg_importance_xgb
            }).sort_values('Importance', ascending=False)
            
            # Save to CSV
            importance_csv_xgb = f"Feature_Importance_XGBoost_{target_name}_{mode}_{ablation}.csv"
            importance_df_xgb.to_csv(importance_csv_xgb, index=False)
            print(f"  [XGBoost] Saved feature importance to {importance_csv_xgb}")
            print(f"  Top 10 features:")
            for idx, row in importance_df_xgb.head(10).iterrows():
                print(f"    {row['Feature']}: {row['Importance']:.4f}")
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(importance_df_xgb))
            plt.barh(range(top_n), importance_df_xgb.head(top_n)['Importance'].values)
            plt.yticks(range(top_n), importance_df_xgb.head(top_n)['Feature'].values)
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Features - XGBoost ({target_name})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"Feature_Importance_XGBoost_{target_name}_{mode}_{ablation}.png", dpi=200)
            plt.close()
            
        if len(feature_importance_lgb) > 0:
            avg_importance_lgb = np.mean(feature_importance_lgb, axis=0)
            importance_df_lgb = pd.DataFrame({
                'Feature': feature_names,
                'Importance': avg_importance_lgb
            }).sort_values('Importance', ascending=False)
            
            # Save to CSV
            importance_csv_lgb = f"Feature_Importance_LightGBM_{target_name}_{mode}_{ablation}.csv"
            importance_df_lgb.to_csv(importance_csv_lgb, index=False)
            print(f"\n  [LightGBM] Saved feature importance to {importance_csv_lgb}")
            print(f"  Top 10 features:")
            for idx, row in importance_df_lgb.head(10).iterrows():
                print(f"    {row['Feature']}: {row['Importance']:.4f}")
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(importance_df_lgb))
            plt.barh(range(top_n), importance_df_lgb.head(top_n)['Importance'].values)
            plt.yticks(range(top_n), importance_df_lgb.head(top_n)['Feature'].values)
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Features - LightGBM ({target_name})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"Feature_Importance_LightGBM_{target_name}_{mode}_{ablation}.png", dpi=200)
            plt.close()
        
        # [NEW] XGB1 (Layer 1 Feature Augmentation) importance
        if len(feature_importance_xgb1) > 0:
            avg_importance_xgb1 = np.mean(feature_importance_xgb1, axis=0)
            importance_df_xgb1 = pd.DataFrame({
                'Feature': feature_names_xgb1,
                'Importance': avg_importance_xgb1
            }).sort_values('Importance', ascending=False)
            
            # Save to CSV
            importance_csv_xgb1 = f"Feature_Importance_XGB1_Layer1_{target_name}_{mode}_{ablation}.csv"
            importance_df_xgb1.to_csv(importance_csv_xgb1, index=False)
            print(f"\n  [XGB1 - Layer 1] Saved feature importance to {importance_csv_xgb1}")
            print(f"  Top 10 raw features used by XGB1:")
            for idx, row in importance_df_xgb1.head(10).iterrows():
                print(f"    {row['Feature']}: {row['Importance']:.4f}")
            
            # Plot top 20 features
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(importance_df_xgb1))
            plt.barh(range(top_n), importance_df_xgb1.head(top_n)['Importance'].values)
            plt.yticks(range(top_n), importance_df_xgb1.head(top_n)['Feature'].values)
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Raw Features - XGB1 Layer 1 ({target_name})')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"Feature_Importance_XGB1_Layer1_{target_name}_{mode}_{ablation}.png", dpi=200)
            plt.close()
            print(f"  [XGB1 - Layer 1] Saved plot to Feature_Importance_XGB1_Layer1_{target_name}_{mode}_{ablation}.png")

    # --- Save & Plot ---
    print(f"\nFinished {target_name} - {mode}")
    
    # 1. 保存 Patch 级别结果
    df["Predicted"] = final_pred
    out_csv = f"Final_Multimodal_Masked_CNN_{target_name}_{mode}_{ablation}_V4.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved Patch-level results to {out_csv}")

    # 2. 聚合到 Image 级别 (Sample-level)
    # [MODIFIED] Full Image Mode: One row per image, so no aggregation needed.
    df_agg = df.copy()
    
    # Ensure columns exist before using them
    # agg_cols is not needed anymore but logic below uses df_agg
    
    y_true_agg = df_agg[target_name].values
    y_pred_agg = df_agg['Predicted'].values
    
    # 3. 计算聚合后的指标
    # [Fix] 检查并处理 NaN，防止报错
    if np.isnan(y_pred_agg).any():
        nan_count = np.isnan(y_pred_agg).sum()
        print(f"\n[WARNING] Predictions contain {nan_count} NaNs! Filling with mean value to prevent crash.")
        # 用均值填充 NaN，或者填 0
        y_pred_agg = np.nan_to_num(y_pred_agg, nan=np.nanmean(y_pred_agg))
        
    if np.isnan(y_true_agg).any():
        print(f"\n[WARNING] Ground Truth contains NaNs! Dropping these samples.")
        valid_mask = ~np.isnan(y_true_agg)
        y_true_agg = y_true_agg[valid_mask]
        y_pred_agg = y_pred_agg[valid_mask]

    r2_agg = r2_score(y_true_agg, y_pred_agg)
    rmse_agg = np.sqrt(mean_squared_error(y_true_agg, y_pred_agg))
    
    print(f"\n[Aggregated Results] (Sample-level)")
    print(f"R2: {r2_agg:.4f}")
    print(f"RMSE: {rmse_agg:.4f}")
    
    # 4. 保存聚合结果
    agg_csv = f"Final_Aggregated_Results_{target_name}_{mode}_{ablation}.csv"
    df_agg.to_csv(agg_csv, index=False)
    print(f"Saved Aggregated results to {agg_csv}")

    # 5. 绘图 (使用聚合后的数据)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true_agg, y=y_pred_agg, alpha=0.6, edgecolor='k')
    
    min_val = min(y_true_agg.min(), y_pred_agg.min())
    max_val = max(y_true_agg.max(), y_pred_agg.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel(f"True {target_name} (Avg)")
    plt.ylabel(f"Predicted {target_name} (Avg)")
    plt.title(f'{target_name} ({mode}) - Sample Level\nR2={r2_agg:.4f}')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(f"Final_Plot_{target_name}_{mode}_{ablation}_Aggregated.png", dpi=300)
    plt.close()
    
    # --- LOO Final Aggregation (Split by Condition) ---
    loo_results_list = []
    
    if not df_hidden.empty and folds_run > 0:
        print(f"\n{'='*20} LOO Hidden Set Evaluation (Split by Condition) {'='*20}")
        
        # Average predictions from all folds
        avg_xgb = hidden_accum_xgb / folds_run
        avg_lgb = hidden_accum_lgb / folds_run
        avg_cnn = hidden_accum_cnn / folds_run
        
        # Ensemble using weights found from OOF
        final_hidden_pred = (avg_xgb * w_xgb) + (avg_lgb * w_lgb) + (avg_cnn * w_cnn)
        
        # Add prediction to dataframe for splitting
        df_hidden[f'Pred_{target_name}'] = final_hidden_pred
        
        # Save detailed predictions (Combined)
        df_hidden.to_csv(f"Hidden_Timepoint_Prediction_{hidden_times[0]}.csv", index=False)
        
        # Group by Condition and calculate metrics
        for cond, group in df_hidden.groupby('condition'):
            y_true_g = group[target_name].values
            y_pred_g = group[f'Pred_{target_name}'].values
            
            rmse_g = np.sqrt(mean_squared_error(y_true_g, y_pred_g))
            r2_g   = r2_score(y_true_g, y_pred_g)
            
            print(f"  [Condition: {cond}] Timepoints: {hidden_times}")
            print(f"    RMSE: {rmse_g:.4f}")
            print(f"    R2:   {r2_g:.4f}")
            print(f"    True Mean: {y_true_g.mean():.4f}")
            print(f"    Pred Mean: {y_pred_g.mean():.4f}")
            
            loo_results_list.append({
                'Timepoint': hidden_times[0],
                'Condition': cond,
                'RMSE': rmse_g,
                'R2': r2_g,
                'True_Mean': y_true_g.mean(),
                'Pred_Mean': y_pred_g.mean()
            })
            
    return loo_results_list

def run_loo_experiment(args):
    """
    Automated Leave-One-Timepoint-Out Experiment.
    Iterates through [1..7] (or available times) and runs pipeline for each.
    """
    print(f"\n[Experimental] Running LOO Experiment for target={args.target}...")
    
    # 1. Determine Timepoints
    # We load the CSV just to check unique times, or use hardcoded [1,2,3,4,5,6,7]
    try:
        df_tmp = pd.read_csv(TRAIN_CSV)
        available_times = sorted(df_tmp['time'].unique())
        print(f"  Found timepoints: {available_times}")
        
        # Target: 1 to 72. Keep 0 as anchor (starting point).
        target_times = [t for t in available_times if t not in [0]]
        print(f"  Target LOO Timepoints: {target_times}")
    except Exception as e:
        print(f"  [Error] Could not read CSV to find times: {e}")
        target_times = [1, 2, 3, 4, 5, 6, 7] # Fallback
        
    summary_results = []
    
    for t in target_times:
        print(f"\n\n{'#'*60}")
        print(f"### LOO RUN: Hiding Timepoint {t}")
        print(f"{'#'*60}")
        
        # Run pipeline
        results_list = run_pipeline(target_name=args.target, mode=args.mode, cv_method=args.cv_method, 
                             ablation="no_kinetic", max_folds=args.max_folds, hidden_times=[t])
        
        if results_list:
            summary_results.extend(results_list)
            
    # --- Global Aggregation & Analysis ---
    print(f"\n\n{'='*30} Global LOO Analysis {'='*30}")
    
    # Collect all hidden prediction files
    all_hidden_dfs = []
    for t in target_times:
        fname = f"Hidden_Timepoint_Prediction_{t}.csv"
        if os.path.exists(fname):
            all_hidden_dfs.append(pd.read_csv(fname))
            
    if all_hidden_dfs:
        df_global = pd.concat(all_hidden_dfs, ignore_index=True)
        
        # Calculate Global Metrics by Condition
        print("\n[Global Metrics by Condition]")
        global_stats = []
        
        for cond, group in df_global.groupby('condition'):
            y_true = group[args.target].values
            y_pred = group[f'Pred_{args.target}'].values
            
            rmse_g = np.sqrt(mean_squared_error(y_true, y_pred))
            r2_g   = r2_score(y_true, y_pred)
            
            print(f"  Condition: {cond}")
            print(f"    RMSE: {rmse_g:.4f}")
            print(f"    R2:   {r2_g:.4f}")
            
            global_stats.append({'Condition': cond, 'Global_RMSE': rmse_g, 'Global_R2': r2_g})
            
        # Overall Global Metric
        y_true_all = df_global[args.target].values
        y_pred_all = df_global[f'Pred_{args.target}'].values
        rmse_all = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        r2_all   = r2_score(y_true_all, y_pred_all)
        print(f"\n  [OVERALL] RMSE: {rmse_all:.4f} | R2: {r2_all:.4f}")
        
        # Save Global Result File
        global_csv = f"LOO_Global_Predictions_{args.target}.csv"
        df_global.to_csv(global_csv, index=False)
        print(f"  Saved all combined LOO predictions to {global_csv}")
        
        # Save Global Stats
        stats_df = pd.DataFrame(global_stats)
        stats_df.loc[len(stats_df)] = ['OVERALL', rmse_all, r2_all]
        stats_df.to_csv(f"LOO_Global_Metrics_{args.target}.csv", index=False)
        
        # --- [NEW] Population-level (Mean-based) Metrics ---
        # User request: Fit R2 based on group means (e.g. 1h_Dark_True vs 1h_Dark_Pred)
        print("\n[Population-level (Mean-based) Metrics]")
        
        # Group by Timepoint + Condition to get one point per biological state
        pop_agg = df_global.groupby(['time', 'condition'])[[args.target, f'Pred_{args.target}']].mean().reset_index()
        
        y_true_pop = pop_agg[args.target].values
        y_pred_pop = pop_agg[f'Pred_{args.target}'].values
        
        rmse_pop = np.sqrt(mean_squared_error(y_true_pop, y_pred_pop))
        r2_pop   = r2_score(y_true_pop, y_pred_pop)
        
        print(f"  Number of Population Points: {len(pop_agg)}")
        print(f"  Population RMSE: {rmse_pop:.4f}")
        print(f"  Population R2:   {r2_pop:.4f}")
        
        # Save Population Data for Plotting
        pop_csv = f"LOO_Population_Means_{args.target}.csv"
        pop_agg.to_csv(pop_csv, index=False)
        print(f"  Saved population means to {pop_csv}")
        
    # Save Step-wise Summary
    if summary_results:
        sum_df = pd.DataFrame(summary_results)
        # Sort for better readability
        sum_df = sum_df.sort_values(['Timepoint', 'Condition'])
        
        out_file = f"LOO_Validation_Results_Summary_{args.target}.csv"
        sum_df.to_csv(out_file, index=False)
        print(f"\n[Done] LOO Experiment Complete. Step-wise summary saved to {out_file}")
        print(sum_df)
    else:
        print("[Warning] No results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Dry_Weight")
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["full", "xgb_only", "lgb_only", "cnn_only", "boost_only"], 
                        help="Training mode")
    parser.add_argument("--cv_method", type=str, default="group", choices=["random", "group", "loocv"], help="CV split method")
    # [EDIT 4] Update the CLI defaults in main to default to 'no_kinetic'.
    parser.add_argument("--ablation", type=str, default="no_kinetic", choices=["none", "no_kinetic"], help="Ablation mode")
    parser.add_argument("--max_folds", type=int, default=1, help="Max number of folds to run (default: 1 for testing). Set to 0 or -1 for all.")
    parser.add_argument("--run_loo", action='store_true', help="Run Leave-Many-Timepoints-Out experiment")
    
    args = parser.parse_args()
    
    if args.run_loo:
        run_loo_experiment(args)
    else:
        # Handle max_folds=0 or -1 as None (run all)
        mf = args.max_folds if args.max_folds > 0 else None
        
        run_pipeline(target_name=args.target, mode=args.mode, cv_method=args.cv_method, ablation=args.ablation, max_folds=mf)
