---
description: 如何在 Docker 容器中运行模块化微藻预测流程
---

# 运行微藻预测流程 (Docker)

此项目已模块化，建议通过 Docker 容器运行以确保环境一致性。

### 1. 构建镜像
如果你对 `Dockerfile` 做了修改，请运行：
```bash
docker build -t algae-fusion:v7 .
```

### 2. 运行实验
你可以直接在容器内运行 `main.py`。请确保挂载了数据目录。

#### 快速测试 (Boost-only 模式, 1折)
```bash
docker run --gpus all -v $(pwd):/workspace algae-fusion:v7 \
    python3 main.py --mode boost_only --max_folds 1
```

#### 完整 MoE 训练 (Dry_Weight)
```bash
docker run --gpus all -v $(pwd):/workspace algae-fusion:v7 \
    python3 main.py --target Dry_Weight --mode full --max_folds 5
```

#### 运行 LOO (Leave-One-Timepoint-Out) 实验
```bash
docker run --gpus all -v $(pwd):/workspace algae-fusion:v7 \
    python3 main.py --target Dry_Weight --loo
```

### 3. 查看结果
结果将保存为 `.csv` 和 `.png` 文件并同步到你的本地目录。
