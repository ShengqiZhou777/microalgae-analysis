---
description: 如何在 Docker 容器中完整运行微藻预测流程（训练与推理）
---

# 运行微藻预测流程 (Docker)

本项目提供了完整的训练和推理工具链。推荐在 Docker 容器中运行以确保环境一致性。

### 1. 构建镜像
首次运行前，请确保构建镜像：
```bash
docker build -t algae-fusion:v7 .
```

### 2. 训练模型

#### 选项 A：一键多目标训练 (推荐)
使用提供的脚本自动训练所有目标变量 (Dry_Weight, Chl_Per_Cell, Fv_Fm, Oxygen_Rate)，同时包含静态和动态模型。

```bash
docker run --gpus all --ipc=host --rm -it -v $(pwd):/workspace algae-fusion:v7 \
    bash run_multitask_training.sh
```

#### 选项 B：单目标单独训练 (用于调试)
您可以直接运行 `main.py` 来训练特定目标。
*   `--mode cnn_only`: 训练静态 CNN-XGBoost 模型
*   `--mode cnn_only --stochastic`: 训练动态模型

```bash
# 示例：仅训练细胞干重 (Dry_Weight) 的静态模型
docker run --gpus all --ipc=host --rm -it -v $(pwd):/workspace algae-fusion:v7 \
    python3 main.py --target Dry_Weight --mode cnn_only --max_folds 5
```

### 3. 模型推理 (预测)

训练完成后，使用 `predict_multitask.py` 对新数据进行预测。该脚本会自动加载 saved weights 并生成结果。

**前提**：确保 `weights/` 目录下有训练好的模型文件。

```bash
# 对测试集进行预测
docker run --gpus all --ipc=host --rm -it -v $(pwd):/workspace algae-fusion:v7 \
    python3 predict_multitask.py --input data/dataset_test.csv --output Final_Test_Results.csv
```

结果 (`Final_Test_Results.csv`) 和可视化图表 (`Final_Test_Plot_*.png`) 将保存到当前目录。
