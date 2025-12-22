import torch

# ================= 数据路径 =================
TRAIN_CSV = "data/Final_Training_Data_With_Labels.csv" 
MASK_SUFFIX = "_mask.png"
PATH_PREFIX = "data/"   

# ================= 训练参数 =================
IMG_SIZE = (512, 512) 
BATCH_SIZE = 64     
EPOCHS = 15
LR = 1e-5                    

# ================= 验证策略 =================
N_SPLITS = 5
MAX_FOLDS = 1  
MAX_VAL_BATCHES = 10

# ================= 硬件配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BACKBONE = "resnet34"  

# ================= 特征黑名单 =================
NON_FEATURE_COLS = [
    'file', 'Source_Path', 'time', 'condition', 
    'Dry_Weight', 'Chl_Per_Cell', 'Fv_Fm', 'Oxygen_Rate', 'Total_Chl',
    'condition_encoded', 'time_x_cond', 'time_squared', 'time_log',
    'Prev_Time', 'dt', 'group_id',
    'Prev1_file', 'Prev1_Source_Path', 'Prev2_file', 'Prev2_Source_Path', 'Prev3_file', 'Prev3_Source_Path'
]
