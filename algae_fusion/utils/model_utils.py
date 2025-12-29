import torch
import joblib
import json
import os
import random
import numpy as np
from algae_fusion.config import RANDOM_SEED

def set_seed(seed=RANDOM_SEED):
    """设置全局随机种子以保证实验可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  [Info] Random Seed set to {seed}")

class Log1pScaler:
    """对目标值进行 log1p 变换并标准化的包装类"""
    def __init__(self, s): self.s = s
    def transform(self, y): return self.s.transform(np.log1p(y).reshape(-1, 1)).flatten()
    def inverse_transform(self, yp): return np.expm1(self.s.inverse_transform(yp.reshape(-1, 1)).flatten())

class StandardWrapper:
    """标准 Scikit-learn Scaler 的简单包装类"""
    def __init__(self, s): self.s = s
    def transform(self, y): return self.s.transform(y.reshape(-1, 1)).flatten()
    def inverse_transform(self, yp):
        if yp.ndim == 1: yp = yp.reshape(-1, 1)
        return self.s.inverse_transform(yp).flatten()

def save_pipeline_models(mode, model_prefix, xgb=None, lgb=None, cnn=None, target_scaler=None, metadata=None):
    """
    Save all trained models and metadata from the pipeline.
    
    Args:
        mode (str): Training mode (full, boost_only, cnn_only, etc.)
        model_prefix (str): Path prefix for saving files (e.g. weights/Target_Condition)
        xgb: XGBoost model (optional)
        lgb: LightGBM model (optional)
        cnn: PyTorch CNN model (optional)
        target_scaler: Target scaler (optional)
        metadata (dict): Dictionary of metadata to save
    """
    
    # 1. Save Tabular Models
    if mode in ["full", "boost_only"]:
        if xgb: xgb.save(f"{model_prefix}_xgb.json")
    
    if mode in ["full", "lgb_only", "boost_only"]:
        if lgb: lgb.save(f"{model_prefix}_lgb.joblib")
    
    if mode in ["full", "cnn_only"]:
        if cnn: torch.save(cnn.state_dict(), f"{model_prefix}_cnn.pth")
        
    # 2. Save Target Scaler
    if target_scaler:
        print(f"  [Saving] Target Scaler: {model_prefix}_target_scaler.joblib")
        joblib.dump(target_scaler, f"{model_prefix}_target_scaler.joblib")
    
    # 3. Save Metadata
    if metadata:
        with open(f"{model_prefix}_metadata.json", "w") as f:
            json.dump(metadata, f)
    
    print(f"  [SUCCESS] All models saved to: {model_prefix}_*")
