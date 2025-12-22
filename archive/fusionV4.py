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
BATCH_SIZE = 256      
EPOCHS = 50
LR = 1e-5                    

# 交叉验证折数
N_SPLITS = 5
MAX_FOLDS = 5
MAX_VAL_BATCHES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True # 针对固定输入尺寸优化，大幅提升卷积速度

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
        self._cache = {}

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
        if idx in self._cache:
            img_array = self._cache[idx]
        else:
            img_array = self._load_one(idx)
            self._cache[idx] = img_array

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
            
            # [TTA] Test Time Augmentation: 原始 + 水平翻转 + 垂直翻转 平均
            p1 = model(x).squeeze(1)
            p2 = model(torch.flip(x, [3])).squeeze(1) # Horizontal Flip
            p3 = model(torch.flip(x, [2])).squeeze(1) # Vertical Flip
            pred = (p1 + p2 + p3) / 3.0
            
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

def run_pipeline(target_name="Dry_Weight", mode="full", cv_method="random"):
    """
    target_name: "Dry_Weight" or "Fv_Fm"
    mode: "full", "xgb_only", "lgb_only", "cnn_only"
    cv_method: "random" (KFold) or "group" (GroupKFold)
    """
    print(f"\n\n{'='*40}")
    print(f"STARTING PIPELINE: Target={target_name}, Mode={mode}, CV={cv_method}")
    print(f"{'='*40}\n")

    df = pd.read_csv(TRAIN_CSV)
    
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

    # 剔除已知异常样本 (黑名单)
    # 这些样本在 Total_Chl 预测中表现出极端的离群行为，可能会污染其他指标的训练
    blacklist_ids = ['65190', '65198', '65015', '64973', '65001', '65178', '64937', '65191', '65073', '65023', '65143', '64910', '65193'] 
    # 确保 ID 类型一致 (转为字符串比较)
    df['Original_Image_ID'] = df['Original_Image_ID'].astype(str)
    initial_len = len(df)
    df = df[~df['Original_Image_ID'].isin(blacklist_ids)].reset_index(drop=True)
    if len(df) < initial_len:
        print(f"  [Data Cleaning] Removed {initial_len - len(df)} patches belonging to blacklist images: {blacklist_ids}")

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

    df['Source_Path'] = df['Source_Path'].astype(str)

    tab_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X_tab = df[tab_cols]
    y_orig = df[target_name].values
    y = y_orig.copy()

    # --- Auto-Target Transform Logic (PowerTransformer) ---
    pt = None
    y_min, y_max = y.min(), y.max()
    y_skew = pd.Series(y).skew()
    
    # Heuristic: High skew OR large dynamic range. 
    # We use Yeo-Johnson which is safe for positive/negative and similar to Box-Cox for positive.
    if abs(y_skew) > 1.0 or (y_min > 0 and y_max / y_min > 20):
        print(f"  [Auto-Transform] Target '{target_name}' Skew={y_skew:.2f}, Range=[{y_min:.4f}, {y_max:.4f}]. Applying PowerTransformer(Yeo-Johnson).")
        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        y = pt.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        print(f"  [Auto-Transform] Target '{target_name}' Skew={y_skew:.2f}, Range=[{y_min:.4f}, {y_max:.4f}]. No transform.")

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

    # 强制使用 GroupKFold，但 Group 是 Original_Image_ID
    # 这样保证同一张大图的所有 Patch 要么都在训练集，要么都在验证集（防泄露）
    # 同时实现了 Random Split 的效果（因为 Image ID 是随机分布的）
    groups = df['Original_Image_ID']
    gkf = GroupKFold(n_splits=N_SPLITS)
    splitter = gkf.split(df, groups=groups)
    print("  [Split Strategy] Using GroupKFold on 'Original_Image_ID' (Leakage-Free Random Split)...")

    for fold, (tr, va) in enumerate(splitter):
        if MAX_FOLDS is not None and fold >= MAX_FOLDS:
            break

        print(f"\n--- FOLD {fold+1} ---")
        # 打印本折验证集包含的 Time/Condition
        val_groups = df.iloc[va][['time', 'condition']].drop_duplicates()
        print(f"  Validation Groups for Fold {fold+1}:")
        # 简单打印前5个组合，避免刷屏
        print(val_groups.head())
        
        scaler = StandardScaler()
        y_train_raw = y[tr].reshape(-1, 1)
        y_val_raw   = y[va].reshape(-1, 1)
        y_train_scaled = scaler.fit_transform(y_train_raw).flatten()
        y_val_scaled   = scaler.transform(y_val_raw).flatten()

        # --- Layer 1: XGB1 ---
        if mode in ["full", "boost_only"]:
            print("  [Layer 1] XGB1 Feature Augmentation...")
            xgb1 = XGBRegressor(n_estimators=500, 
            learning_rate=0.05, 
            max_depth=4, 
            tree_method="hist")

            xgb1.fit(X_tab.iloc[tr], y_train_scaled)
            feat_aug_train = xgb1.predict(X_tab.iloc[tr])
            feat_aug_val   = xgb1.predict(X_tab.iloc[va])
            
            X_train_aug = X_tab.iloc[tr].copy()
            X_val_aug   = X_tab.iloc[va].copy()
            X_train_aug["XGB1_Feature"] = feat_aug_train
            X_val_aug["XGB1_Feature"]   = feat_aug_val
        else:
            X_train_aug = X_tab.iloc[tr]
            X_val_aug   = X_tab.iloc[va]

        # --- Layer 2: XGB2 ---
        if mode in ["full", "xgb_only", "boost_only"]:
            print("  [Layer 2] XGB2...")
            xgb2 = XGBRegressor(n_estimators=800, 
            learning_rate=0.05, 
            max_depth=6, 
            tree_method="hist")

            xgb2.fit(X_train_aug, y_train_scaled)
            pred_xgb_scaled = xgb2.predict(X_val_aug)
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
        else:
            pred_lgb_scaled = np.zeros(len(va), dtype=np.float32)
        if mode in ["full", "cnn_only"]:
            print("  [Layer 2] CNN+...")
            train_ds = MaskedImageDataset(df.iloc[tr], target_name, IMG_SIZE, train_transform, labels=y_train_scaled)
            val_ds   = MaskedImageDataset(df.iloc[va], target_name, IMG_SIZE, val_transform, labels=y_val_scaled)
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
                        break
            
            if best_state is not None:
                cnn.load_state_dict(best_state)
            _, pred_cnn_scaled = eval_epoch(cnn, val_loader, criterion)
        else:
            pred_cnn_scaled = np.zeros(len(va), dtype=np.float32)

        # --- 收集 OOF 预测值 (不在此处做最终融合) ---
        oof_preds_xgb[va] = pred_xgb_scaled
        oof_preds_lgb[va] = pred_lgb_scaled
        oof_preds_cnn[va] = pred_cnn_scaled
        oof_targets[va]   = y_val_scaled

    # --- Auto-Weighting Optimization (基于 OOF 的自动权重搜索) ---
    print(f"\n{'='*20} Auto-Weighting Optimization {'='*20}")
    
    # 构造元特征矩阵 [N_samples, 3]
    # 注意：如果某个 mode 没开 (比如 cnn_only)，那一列全是 0，权重自然会是 0
    X_meta = np.vstack([oof_preds_xgb, oof_preds_lgb, oof_preds_cnn]).T
    y_meta = oof_targets
    
    # 使用非负约束的线性回归 (Non-Negative Least Squares)
    # fit_intercept=False 因为我们已经做过 Scaler 了，不需要偏置项
    meta_model = LinearRegression(positive=True, fit_intercept=False) 
    meta_model.fit(X_meta, y_meta)
    
    weights = meta_model.coef_
    # 归一化权重，使其和为 1 (方便观察，实际预测时直接用 coef_ 即可，通常和也接近 1)
    weights_sum = np.sum(weights) + 1e-8
    w_xgb, w_lgb, w_cnn = weights / weights_sum
    
    print(f"Optimal Weights Found (Normalized):")
    print(f"  XGBoost:  {w_xgb:.4f}")
    print(f"  LightGBM: {w_lgb:.4f}")
    print(f"  CNN:      {w_cnn:.4f}")
    
    # 应用最优权重生成最终预测
    # 注意：这里使用原始 weights (未归一化) 以获得最小二乘的最优解
    pred_scaled = (
        weights[0] * oof_preds_xgb + 
        weights[1] * oof_preds_lgb + 
        weights[2] * oof_preds_cnn
    )
    
    final_pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        
    if pt is not None:
        final_pred = pt.inverse_transform(final_pred.reshape(-1, 1)).flatten()

    # --- Save & Plot ---
    print(f"\nFinished {target_name} - {mode}")
    
    # 1. 保存 Patch 级别结果
    df["Predicted"] = final_pred
    out_csv = f"Final_Multimodal_Masked_CNN_{target_name}_{mode}_V4.csv"
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
    agg_csv = f"Final_Aggregated_Results_{target_name}_{mode}.csv"
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
    
    plt.savefig(f"Final_Plot_{target_name}_{mode}_Aggregated.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="Dry_Weight")
    parser.add_argument("--mode", type=str, default="full")
    parser.add_argument("--cv_method", type=str, default="random", choices=["random", "group"], help="random=KFold(Shuffle), group=GroupKFold")
    args = parser.parse_args()
    run_pipeline(args.target, args.mode, args.cv_method)
