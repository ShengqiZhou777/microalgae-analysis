import os
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, GroupKFold
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

# ================= 配置区域 =================
TRAIN_CSV = "Final_Training_Data_Patched_Features.csv" 
MASK_SUFFIX = "_mask.png"
PATH_PREFIX = ""   

# 图像相关
IMG_SIZE = (224,224) # 回归标准 ResNet 尺寸
BATCH_SIZE = 512     
EPOCHS = 10
LR = 1e-5                    

# 交叉验证折数
N_SPLITS = 5
MAX_FOLDS = 1  
MAX_VAL_BATCHES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True # 针对固定输入尺寸优化，大幅提升卷积速度
    scaler_amp = torch.cuda.amp.GradScaler() # [AMP] 混合精度缩放器
else:
    scaler_amp = None

# CNN 骨干网络
BACKBONE = "resnet34"  

NON_FEATURE_COLS = [
    'file','label','Source_Folder','Source_Path','time','condition', 
    'Dry_Weight','Chl_Per_Cell','Fv_Fm','Oxygen_Rate','Total_Chl',
    'Original_Image_ID', 'Patch_File', 'Patch_Path', 'Patch_Mask_Path' 
]

class MaskedImageDataset(Dataset):
    def __init__(self, df, target_col, img_size=(224, 224), transform=None, labels=None):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        self.img_size = img_size
        self.transform = transform
        self.labels = labels
        # [Optimization] Remove cache to save RAM and avoid OOM
        # self._cache = {} 

    def __len__(self):
        return len(self.df)

    def _load_one(self, idx):
        row = self.df.iloc[idx]
        # 使用 Patch 路径
        image_path = row["Patch_Path"]
        mask_path  = row["Patch_Mask_Path"]

        img = cv2.imread(image_path)
        if img is None:
            img = np.zeros((self.img_size[0], self.img_size[1], 3), np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.ones(img.shape[:2], np.uint8) * 255

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask_norm = (mask / 255.0).astype(np.float32)
        masked_img = (img.astype(np.float32) * mask_norm[..., None]).astype(np.uint8)
        return masked_img.astype(np.uint8)

    def __getitem__(self, idx):
        # [Optimization] Direct load, no cache
        img_array = self._load_one(idx)

        pil = Image.fromarray(img_array)
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
    # Merge population stats
    # [Correction] Merge on BOTH Time and Condition to compare Light vs Light, Dark vs Dark
    if pop_stats is not None and not pop_stats.empty:
        df = df.merge(pop_stats, left_on=['Prev_Time', 'condition'], right_on=['time', 'condition'], how='left', suffixes=('', '_PrevPop'))
    else:
        # If pop_stats is None (e.g. no valid morph cols), create empty columns to avoid KeyErrors
        for col in valid_morph_cols:
            df[col + '_PopMean'] = np.nan
    
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
        pop_stats: Population statistics dataframe (for applying to hidden set)
        time_map: Time mapping dictionary (for applying to hidden set)
    """
    df_train = df_train.copy()
    df_val = df_val.copy()
    
    valid_morph_cols = [c for c in morph_cols if c in df_train.columns]
    
    if not valid_morph_cols:
        # [Fix] Return empty dict instead of None to prevent TypeError later
        return df_train, df_val, None, {}
    
    # 1. Calculate Global Population Mean per Timepoint AND Condition ONLY on TRAINING data
    # [Correction] Group by ['time', 'condition'] to ensure separate baselines for Light/Dark
    pop_stats = df_train.groupby(['time', 'condition'])[valid_morph_cols].mean().add_suffix('_PopMean').reset_index()
    
    # 2. Build Time Mapping (Current -> Previous)
    all_times = sorted(df_train['time'].unique())
    time_map = {t: all_times[i-1] if i > 0 else np.nan for i, t in enumerate(all_times)}
    
    # 3. Process train and validation dataframes separately
    df_train_aug = process_one_fold(df_train, pop_stats, time_map, valid_morph_cols)
    df_val_aug = process_one_fold(df_val, pop_stats, time_map, valid_morph_cols)
    
    return df_train_aug, df_val_aug, pop_stats, time_map

def run_pipeline(target_name="Dry_Weight", mode="full", cv_method="random", ablation="none", max_folds=None, hidden_times=None):
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
    
    # [Filter] Remove images with low cell count (< 4) as requested
    if 'number_of_cells' in df.columns:
        print(f"  [Filter] Removing images with number_of_cells < 4...")
        before = len(df)
        df = df[df['number_of_cells'] >= 4].reset_index(drop=True)
        print(f"  [Filter] Dropped {before - len(df)} patches from low-count images.")
    elif 'cell_count' in df.columns:
         print(f"  [Filter] Removing images with cell_count < 4...")
         before = len(df)
         df = df[df['cell_count'] >= 4].reset_index(drop=True)
         print(f"  [Filter] Dropped {before - len(df)} patches from low-count images.")
    
    # 显式打乱数据，确保 GroupKFold 进行的是随机分组划分 (Random Group Split)
    # 这样可以保证训练集和验证集里都包含各种时间点(0h, 1h...)和条件(Dark, Light)的样本
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df['file'] = df['file'].astype(str)
    
    # 自动从文件名解析 Original_Image_ID (如果 CSV 里没有)
    # 假设文件名格式: {base_name}_p{patch_id}.jpg
    if 'Original_Image_ID' not in df.columns:
        print("  [Info] 'Original_Image_ID' column missing. Deriving from filenames...")
        # 使用 rsplit 确保只分割最后一个 _p 部分
        df['Original_Image_ID'] = df['file'].apply(lambda x: x.rsplit('_p', 1)[0] if '_p' in x else x)

    # [Modified] Blacklist removed as per user request (Total_Chl is no longer target)
    # blacklist_ids = [...] 
    df['Original_Image_ID'] = df['Original_Image_ID'].astype(str)


    # [新增] 自动补全 Patch_Path 和 Patch_Mask_Path (如果 CSV 里没有)
    # 假设 merge.py 生成的 CSV 只有 Source_Path，我们需要构造 Patch 路径
    if 'Patch_Path' not in df.columns:
        print("  [Info] 'Patch_Path' column missing. Constructing from Source_Path...")
        # Source_Path 是 .../feature_patch
        # Patch_Path 应该是 .../images_patch/{file}
        # Patch_Mask_Path 应该是 .../masks_patch/{file_mask}
        
        def get_patch_path(row):
            # row['Source_Path'] 类似 .../TIMECOURSE/0h/feature_patch
            # 我们需要 .../TIMECOURSE/0h/images_patch/filename
            base_dir = os.path.dirname(row['Source_Path']) # .../TIMECOURSE/0h
            # row['file'] 是 mask 文件名 (e.g., xxx_p0_mask.png)
            # 我们需要对应的 image 文件名 (e.g., xxx_p0.jpg)
            image_filename = row['file'].replace("_mask.png", ".jpg")
            return os.path.join(base_dir, "images_patch", image_filename)

        def get_mask_path(row):
            base_dir = os.path.dirname(row['Source_Path'])
            # row['file'] 已经是 mask 文件名
            return os.path.join(base_dir, "masks_patch", row['file'])

        df['Patch_Path'] = df.apply(get_patch_path, axis=1)
        df['Patch_Mask_Path'] = df.apply(get_mask_path, axis=1)

    # [LOO Support] Handle hidden timepoints
    df_hidden = pd.DataFrame()
    if hidden_times:
        print(f"  [LOO Experiment] Hiding timepoints: {hidden_times}")
        mask_hidden = df['time'].isin(hidden_times)
        df_hidden = df[mask_hidden].copy().reset_index(drop=True)
        df = df[~mask_hidden].copy().reset_index(drop=True)
        print(f"  [LOO] Hidden Set Size: {len(df_hidden)} patches")
        print(f"  [LOO] Visible Set Size: {len(df)} patches")

    df['Source_Path'] = df['Source_Path'].astype(str)

    # [LEAKAGE FIX] Target transformation will be done INSIDE the CV loop
    # based on training data statistics only
    
    # Define morphological columns for kinetic features
    morph_cols = [
        'cell_mean_area', 
        'cell_mean_mean_intensity', 
        'cell_mean_eccentricity',
        'cell_mean_solidity'
    ]
    
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
    
    # [NEW] Track feature importance across folds
    feature_importance_xgb = []
    feature_importance_lgb = []
    feature_importance_xgb1 = []  # [NEW] Track XGB1 (Layer 1) importance
    feature_names = None  # Will be set in first fold
    feature_names_xgb1 = None  # [NEW] Feature names for XGB1

    # [LOO] Initialize Hidden Accumulators
    hidden_accum_xgb = np.zeros(len(df_hidden))
    hidden_accum_lgb = np.zeros(len(df_hidden))
    hidden_accum_cnn = np.zeros(len(df_hidden))
    hidden_count = 0

    # 强制使用 GroupKFold，但 Group 是 Original_Image_ID
    # 这样保证同一张大图的所有 Patch 要么都在训练集，要么都在验证集（防泄露）
    # 同时实现了 Random Split 的效果（因为 Image ID 是随机分布的）
    groups = df['Original_Image_ID']
    gkf = GroupKFold(n_splits=N_SPLITS)
    splitter = gkf.split(df, groups=groups)
    print("  [Split Strategy] Using GroupKFold on 'Original_Image_ID' (Leakage-Free Random Split)...")

    processed_indices = []

    for fold, (tr, va) in enumerate(splitter):
        if MAX_FOLDS is not None and fold >= MAX_FOLDS:
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
        df_train_fold, df_val_fold, pop_stats, time_map = compute_kinetic_features(df.iloc[tr], df.iloc[va], morph_cols)
        
        # Extract tabular features from augmented dataframes
        tab_cols = [c for c in df_train_fold.columns if c not in NON_FEATURE_COLS]
        X_tab_train = df_train_fold[tab_cols]
        X_tab_val = df_val_fold[tab_cols]
        
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
        if y_train_min >= 0 and ratio > 50:
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
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
            val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)


            cnn = ResNetRegressor(BACKBONE).to(device)
            # 使用 HuberLoss 替代 MSELoss，对离群点更鲁棒，防止梯度爆炸
            criterion = nn.HuberLoss(delta=1.0)
            # 增加 Weight Decay (1e-4 -> 1e-2) 以增强正则化
            optimizer = optim.Adam(cnn.parameters(), lr=LR, weight_decay=1e-2)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

            best_loss = float('inf')
            best_state = None
            patience = 8
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

        # ==========================================================
        # [LOO] Predict on Hidden Set (if exists)
        # ==========================================================
        if len(df_hidden) > 0:
            print("  [LOO] Predicting on hidden set for this fold...")
            # 1. Process Hidden Set (Apply kinetic features from THIS fold's training stats)
            
            # [FIX] Update time_map for hidden timepoints so they link to correct prev time
            train_times = sorted(df.iloc[tr]['time'].unique())
            hidden_t_vals = df_hidden['time'].unique()
            for ht in hidden_t_vals:
                # Find prev time (largest time in train that is smaller than ht)
                prevs = [t for t in train_times if t < ht]
                if prevs:
                    time_map[ht] = max(prevs)
                else:
                    time_map[ht] = np.nan
            
            # [FIX] Ensure valid_morph_cols is defined based on Training Data Columns
            valid_morph_cols = [c for c in morph_cols if c in df.columns]
            
            df_hidden_fold = process_one_fold(df_hidden, pop_stats, time_map, valid_morph_cols)
            X_tab_hidden = df_hidden_fold[tab_cols]
            
            # 2. Layer 1: XGB1 (Augmentation)
            if mode in ["full", "boost_only"]:
                feat_aug_hidden = xgb1.predict(X_tab_hidden)
                X_hidden_aug = X_tab_hidden.copy()
                X_hidden_aug["XGB1_Feature"] = feat_aug_hidden
            else:
                X_hidden_aug = X_tab_hidden
                
            # 3. Layer 2: XGB2
            if mode in ["full", "xgb_only", "boost_only"]:
                p_xgb_hidden = xgb2.predict(X_hidden_aug)
                # Apply inverse transform (if Log1p/PowerTransformer was used)
                p_xgb_hidden = pt_fold.inverse_transform(p_xgb_hidden.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else p_xgb_hidden
                hidden_accum_xgb += p_xgb_hidden
                
            # 4. Layer 2: LGB2
            if mode in ["full", "lgb_only", "boost_only"]:
                p_lgb_hidden = lgb2.predict(X_hidden_aug)
                p_lgb_hidden = pt_fold.inverse_transform(p_lgb_hidden.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else p_lgb_hidden
                hidden_accum_lgb += p_lgb_hidden
                
            # 5. Layer 2: CNN
            if mode in ["full", "cnn_only"]:
                hidden_ds = MaskedImageDataset(df_hidden_fold, target_name, IMG_SIZE, val_transform, labels=None)
                hidden_loader = DataLoader(hidden_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True)
                _, p_cnn_scaled_hidden = eval_epoch(cnn, hidden_loader, criterion)
                p_cnn_hidden = pt_fold.inverse_transform(p_cnn_scaled_hidden.reshape(-1, 1)).flatten() if hasattr(pt_fold, 'inverse_transform') else p_cnn_scaled_hidden
                hidden_accum_cnn += p_cnn_hidden

            hidden_count += 1


    # [Fix] If running partial folds (MAX_FOLDS < N_SPLITS), filter out unprocessed samples
    # to avoid zero-filled predictions (which look like mean values) ruining the metrics.
    if MAX_FOLDS is not None and MAX_FOLDS < N_SPLITS:
        print(f"\n[Info] MAX_FOLDS={MAX_FOLDS} < N_SPLITS={N_SPLITS}. Filtering results to processed folds only.")
        processed_indices = np.array(processed_indices)
        
        # Filter OOF arrays
        oof_preds_xgb = oof_preds_xgb[processed_indices]
        oof_preds_lgb = oof_preds_lgb[processed_indices]
        oof_preds_cnn = oof_preds_cnn[processed_indices]
        oof_targets   = oof_targets[processed_indices]
        
        # Filter DataFrame
        df = df.iloc[processed_indices].reset_index(drop=True)
        print(f"  Filtered dataset size: {len(df)}")

    # --- Auto-Weighting Optimization (基于 OOF 的自动权重搜索) ---
    print(f"\n{'='*20} Auto-Weighting Optimization {'='*20}")
    
    # 构造元特征矩阵 [N_samples, 3]
    # OOF predictions are now in ORIGINAL scale (no scaling applied)
    X_meta = np.vstack([oof_preds_xgb, oof_preds_lgb, oof_preds_cnn]).T
    y_meta = oof_targets  # Also in original scale
    
    # 使用非负约束的线性回归 (Non-Negative Least Squares)
    # Working in original scale now, so fit_intercept can be useful
    meta_model = LinearRegression(positive=True, fit_intercept=True) 
    meta_model.fit(X_meta, y_meta)
    
    weights = meta_model.coef_
    # 归一化权重，使其和为 1 (方便观察，实际预测时直接用 coef_ 即可，通常和也接近 1)
    weights_sum = np.sum(weights) + 1e-8
    w_xgb, w_lgb, w_cnn = weights / weights_sum
    
    print(f"Optimal Weights Found (Normalized):")
    print(f"  XGBoost:  {w_xgb:.4f}")
    print(f"  LightGBM: {w_lgb:.4f}")
    print(f"  CNN:      {w_cnn:.4f}")

    # --- Save Weights ---
    weights_df = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'CNN'],
        'Weight': [w_xgb, w_lgb, w_cnn],
        'Raw_Coef': [weights[0], weights[1], weights[2]]
    })
    weights_csv = f"Final_Model_Weights_{target_name}_{mode}_{ablation}.csv"
    weights_df.to_csv(weights_csv, index=False)
    print(f"  [Info] Model weights saved to {weights_csv}")
    
    # 应用最优权重生成最终预测 (already in original scale, no inverse transform needed)
    final_pred = meta_model.predict(X_meta)
    
    # ==========================================
    # [NEW] Feature Importance Analysis
    # ==========================================
    if feature_names is not None and (len(feature_importance_xgb) > 0 or len(feature_importance_lgb) > 0):
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
    # 按照 Original_Image_ID 分组，计算预测值的均值
    # 注意：真实标签对于同一张图的所有 Patch 是一样的，取均值即可
    agg_cols = {target_name: 'mean', 'Predicted': 'mean'}
    # 如果有 time, condition 等列，也可以保留
    if 'time' in df.columns: agg_cols['time'] = 'first'
    if 'condition' in df.columns: agg_cols['condition'] = 'first'
    
    df_agg = df.groupby('Original_Image_ID').agg(agg_cols).reset_index()
    
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

    # ==========================================
    # [LOO] Finalize Hidden Set Predictions
    # ==========================================
    if hidden_times and hidden_count > 0:
        print(f"\n[LOO] Finalizing Hidden Set Predictions (Averaged over {hidden_count} folds)...")
        
        # Average the accumulated predictions
        avg_pred_xgb = hidden_accum_xgb / hidden_count
        avg_pred_lgb = hidden_accum_lgb / hidden_count
        avg_pred_cnn = hidden_accum_cnn / hidden_count
        
        # Create meta-features for hidden set
        X_meta_hidden = np.vstack([avg_pred_xgb, avg_pred_lgb, avg_pred_cnn]).T
        
        # Apply Meta-Learner (trained on OOFs)
        final_pred_hidden = meta_model.predict(X_meta_hidden)
        
        # Save predictions
        df_hidden[f'Pred_{target_name}'] = final_pred_hidden
        fname = f"Hidden_Timepoint_Prediction_{hidden_times[0]}.csv"
        df_hidden.to_csv(fname, index=False)
        print(f"  Saved hidden set predictions to {fname}")
        
        # Calculate Metrics per Condition
        loo_results_list = []
        print("\n[LOO Hidden Set Performance]")
        for cond, group in df_hidden.groupby('condition'):
            y_true_g = group[target_name].values
            y_pred_g = group[f'Pred_{target_name}'].values
            
            rmse_g = np.sqrt(mean_squared_error(y_true_g, y_pred_g))
            r2_g   = r2_score(y_true_g, y_pred_g)
            
            print(f"  [Condition: {cond}] Timepoints: {hidden_times}")
            print(f"    RMSE: {rmse_g:.4f}")
            print(f"    R2:   {r2_g:.4f}")
            
            loo_results_list.append({
                'Timepoint': hidden_times[0],
                'Condition': cond,
                'RMSE': rmse_g,
                'R2': r2_g,
                'True_Mean': y_true_g.mean(),
                'Pred_Mean': y_pred_g.mean()
            })
            
        return loo_results_list
        
    return None

def run_loo_experiment(args):
    """
    Automated Leave-One-Timepoint-Out Experiment (Backported from FusionV7).
    Iterates through available timepoints and runs pipeline for each.
    """
    print(f"\\n[Experimental] Running LOO Experiment for target={args.target}...")
    
    # 1. Determine Timepoints
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
        print(f"\\n\\n{'#'*60}")
        print(f"### LOO RUN: Hiding Timepoint {t}")
        print(f"{'#'*60}")
        
        # Run pipeline
        results_list = run_pipeline(target_name=args.target, mode=args.mode, cv_method=args.cv_method, 
                             ablation=args.ablation, max_folds=args.max_folds, hidden_times=[t])
        
        if results_list:
            summary_results.extend(results_list)
            
    # --- Global Aggregation & Analysis ---
    print(f"\\n\\n{'='*30} Global LOO Analysis {'='*30}")
    
    # Collect all hidden prediction files
    all_hidden_dfs = []
    for t in target_times:
        fname = f"Hidden_Timepoint_Prediction_{t}.csv"
        if os.path.exists(fname):
            all_hidden_dfs.append(pd.read_csv(fname))
            
    if all_hidden_dfs:
        df_global = pd.concat(all_hidden_dfs, ignore_index=True)
        
        # Calculate Global Metrics by Condition
        print("\\n[Global Metrics by Condition]")
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
        print(f"\\n  [OVERALL] RMSE: {rmse_all:.4f} | R2: {r2_all:.4f}")
        
        # Save Global Result File
        global_csv = f"LOO_Global_Predictions_{args.target}_V5.csv"
        df_global.to_csv(global_csv, index=False)
        print(f"  Saved all combined LOO predictions to {global_csv}")
        
        # Save Global Stats
        stats_df = pd.DataFrame(global_stats)
        stats_df.loc[len(stats_df)] = ['OVERALL', rmse_all, r2_all]
        stats_df.to_csv(f"LOO_Global_Metrics_{args.target}_V5.csv", index=False)
        
    # Save Step-wise Summary
    if summary_results:
        sum_df = pd.DataFrame(summary_results)
        # Sort for better readability
        sum_df = sum_df.sort_values(['Timepoint', 'Condition'])
        
        out_file = f"LOO_Validation_Results_Summary_{args.target}_V5.csv"
        sum_df.to_csv(out_file, index=False)
        print(f"\\n[Done] LOO Experiment Complete. Step-wise summary saved to {out_file}")
        print(sum_df)
    else:
        print("[Warning] No results collected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Dry_Weight")
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["full", "xgb_only", "lgb_only", "cnn_only", "boost_only"], 
                        help="Training mode")
    parser.add_argument("--cv_method", type=str, default="group", choices=["random", "group"], help="CV split method")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_kinetic"], help="Ablation mode")
    parser.add_argument("--max_folds", type=int, default=N_SPLITS, help="Max number of folds to run")
    parser.add_argument("--run_loo", action='store_true', help="Run Leave-One-Timepoint-Out experiment")
    
    args = parser.parse_args()
    
    if args.run_loo:
        run_loo_experiment(args)
    else:
        # Handle max_folds override
        mf = args.max_folds if args.max_folds < N_SPLITS else None
        run_pipeline(target_name=args.target, mode=args.mode, cv_method=args.cv_method, ablation=args.ablation, max_folds=mf)

