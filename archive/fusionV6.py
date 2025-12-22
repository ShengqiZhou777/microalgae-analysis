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

# ================= 配置区域 =================
TRAIN_CSV = "Final_Training_Data_With_Labels.csv" 
MASK_SUFFIX = "_mask.png"
PATH_PREFIX = ""   

# 图像相关
IMG_SIZE = (512,512) # 回归标准 ResNet 尺寸
BATCH_SIZE = 128     
EPOCHS = 30
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
    'Dry_Weight','Chl_Per_Cell','Fv_Fm','Oxygen_Rate','Total_Chl'
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

def run_pipeline(target_name="Dry_Weight", mode="full", cv_method="random", ablation="none"):
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
    
    # 显式打乱数据，确保 GroupKFold 进行的是随机分组划分 (Random Group Split)
    # 这样可以保证训练集和验证集里都包含各种时间点(0h, 1h...)和条件(Dark, Light)的样本
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df['file'] = df['file'].astype(str)
    

    # [MODIFIED] Removed patch/original_id derivation as per user request (Full Image Mode)
    # The 'file' column is the unique identifier.

    # [LEAKAGE FIX] Population Trajectory features will be computed INSIDE the CV loop
    # using compute_kinetic_features() function to avoid data leakage

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

    # 强制使用 GroupKFold，这里使用 'file' 作为 Group (每张图唯一)
    groups = df['file']
    
    if cv_method == "loocv":
        loo = LeaveOneOut()
        splitter = loo.split(df)
        print("  [Split Strategy] Using Leave-One-Out Cross-Validation (LOOCV)...")
        # Warn user if dataset is large
        if len(df) > 50 and MAX_FOLDS is None:
             print(f"  [WARNING] LOOCV with {len(df)} samples will take a VERY long time. Consider setting MAX_FOLDS.")
    elif cv_method == "group":
        gkf = GroupKFold(n_splits=N_SPLITS)
        splitter = gkf.split(df, groups=groups)
        print("  [Split Strategy] Using GroupKFold on 'file' (Random Split)...")
    else: # random
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        splitter = kf.split(df)
        print(f"  [Split Strategy] Using KFold (n_splits={N_SPLITS})...")

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
        df_train_fold, df_val_fold = compute_kinetic_features(df.iloc[tr], df.iloc[va], morph_cols)
        
        # Extract tabular features from augmented dataframes
        tab_cols = [c for c in df_train_fold.columns if c not in NON_FEATURE_COLS]
        X_tab_train = df_train_fold[tab_cols].select_dtypes(exclude=['object']) # Safety: Drop any remaining object cols
        X_tab_val = df_val_fold[tab_cols].select_dtypes(exclude=['object'])
        
        # Verify alignment
        common_cols = X_tab_train.columns.intersection(X_tab_val.columns)
        X_tab_train = X_tab_train[common_cols]
        X_tab_val = X_tab_val[common_cols]
        
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Dry_Weight")
    parser.add_argument("--mode", type=str, default="full", 
                        choices=["full", "xgb_only", "lgb_only", "cnn_only", "boost_only"], 
                        help="Training mode")
    parser.add_argument("--cv_method", type=str, default="group", choices=["random", "group", "loocv"], help="CV split method")
    parser.add_argument("--ablation", type=str, default="none", choices=["none", "no_kinetic"], help="Ablation mode")
    
    args = parser.parse_args()
    
    run_pipeline(target_name=args.target, mode=args.mode, cv_method=args.cv_method, ablation=args.ablation)

