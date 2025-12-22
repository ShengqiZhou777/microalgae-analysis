FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip python3-venv \
    git wget libgl1-mesa-glx libglib2.0-0 ffmpeg && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \\
    # also provide `python` command for scripts that call `python`
    ln -sf /usr/bin/python3 /usr/bin/python

RUN pip install --upgrade pip setuptools wheel

# PyTorch + CUDA11.8
RUN pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# TensorFlow GPU
RUN pip install tensorflow==2.10.1

RUN pip install \
    cellpose \
    segment-anything \
    scikit-image scikit-learn scipy pandas \
    xgboost lightgbm catboost \
    opencv-python-headless \
    tqdm loguru fastremap roifile imgviz seaborn matplotlib
WORKDIR /workspace
