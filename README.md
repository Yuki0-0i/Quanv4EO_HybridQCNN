# Quanv4EO Hybrid Quantum-Classical CNN

> 基于 Quanv4EO (TGRS 2025) 的混合量子-经典遥感图像分类模型现代化重构与改进

[![CI](https://github.com/Yuki0-0i/Quanv4EO_HybridQCNN/actions/workflows/ci.yml/badge.svg)](https://github.com/Yuki0-0i/Quanv4EO_HybridQCNN/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PennyLane 0.42](https://img.shields.io/badge/PennyLane-0.42-purple.svg)](https://pennylane.ai/)

## 项目简介

本项目以 Quanv4EO 量子卷积神经网络为基础，将其从 2020 年的旧依赖（JAX 0.2 + PennyLane 0.14）现代化重写为现代 PennyLane 0.42 + PyTorch 2.11 架构，并系统化探索量子参数 / 经典后端 / 特征融合的贡献。

**核心成果**：
- 4 量子配置 × 4 实验 × 5-seed = 80 个数据点
- 项目最佳: **Exp3 融合方案 (QConv + RGB + ResNet-18) = 0.807 ± 0.007**，距经典 ResNet-18 RGB (0.827) 仅 -0.020
- 5-fold CV 验证数字稳健性

## 项目结构

```
Quanv4EO_HybridQCNN/
├── README.md                  # 本文件
├── .gitignore                 # 忽略大文件/数据
├── requirements.txt            # 环境依赖
├── run_all.sh                 # 一键复现脚本 (5-seed 评估)
│
├── quanv4eo_modern/            # 现代化重写的核心代码
│   ├── quantum/
│   │   └── qconv2d.py          # QConv2D 类 (frozen + trainable lite)
│   ├── classical/
│   │   └── cnn_baselines.py    # 3-CNN / ResNet-18 / Fusion
│   └── data/
│       └── dataset.py          # EuroSAT 加载器
│
├── experiments/                # 实验脚本 (按功能分)
│   ├── smoke_test.py           # 环境烟测
│   ├── run_pipeline_500.py     # 基础 QConv 提取 (无 chunk)
│   ├── run_pipeline_5000_chunked.py  # 分批 5000 张 (推荐)
│   ├── improve_cnn.py          # Baseline 3-CNN 训练
│   ├── improve_cnn_2000.py     # 2000 张 CNN 训练
│   ├── classic_cnn_baseline.py # 经典纯 CNN 对照
│   ├── quantum_configs_5seed.py # 4 配置 5-seed
│   ├── scale_test_5000.py      # 3 数据规模 5-seed
│   ├── exp1_trainable.py       # Exp1: Trainable QConv + 3-CNN
│   ├── exp2_resnet.py          # Exp2: Trainable + ResNet-18
│   ├── exp3_fusion.py          # Exp3: QConv + RGB + ResNet-18
│   ├── exp_matrix.py           # 4 配置 × 3 实验 × 5-seed
│   ├── kfold_3configs.py       # 5-fold CV 验证
│   ├── write_report_345.py     # 自动写 docx 报告
│   ├── plot_figures.py         # 4 张关键实验图
│   └── ...
│
├── reports/                    # Markdown 报告
│   ├── all_experiments_summary.md
│   ├── 4configs_4exps_full_matrix.md
│   ├── exp1_trainable.md
│   ├── exp2_resnet.md
│   ├── exp3_fusion.md
│   ├── kfold_cv.md
│   ├── kfold_3configs.md
│   ├── quantum_configs_1000_5seed.md
│   ├── classic_cnn_baseline.md
│   └── ...
│
├── source/                     # 原版源码 (参考)
│   ├── quanv4eo-main/          # Quanv4EO 原仓库
│   └── QNN4EO-main/            # QNN4EO 原仓库
│
└── venv/                       # Python 环境 (gitignore)
```

## 快速开始

### 1. 创建环境

```bash
conda create -n plenv_gpu python=3.10 -y
conda activate plenv_gpu

# 清华 pip 源 (推荐)
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据集准备

下载 EuroSAT (27,000 张 64×64 RGB JPG, 10 类)，解压到 `datasets/EuroSAT/<Class>/*.jpg`。
- 来源 1: https://github.com/phelber/EuroSAT (EuroSAT.zip, ~90MB)
- 来源 2: torchvision `EuroSAT` 数据集类

### 3. 跑实验

```bash
# 烟测
python experiments/smoke_test.py

# 5000 张 QConv 提取 (主流程, 约 4-5 小时, 8 worker)
python experiments/run_pipeline_5000_chunked.py \
    --max_per_class 500 --n_jobs 8 --qubits 4 --n_layers 2 --encoding ry \
    --chunk_size 1000 --tag baseline_5000_4q2l_ry

# 4 配置 × 4 实验 × 5-seed (推荐后台 tmux 跑, 约 1 小时)
tmux new-session -d -s exp_matrix "python experiments/exp_matrix.py"

# 画图
python experiments/plot_figures.py
```

## 实验矩阵（5-Seed Mean ± Std, 5000 张 EuroSAT）

| 量子配置 | Baseline | Exp1 | Exp2 | **Exp3** |
|---|---|---|---|---|
| Baseline 4q 2l RY | 0.656±0.015 | 0.654±0.009 | 0.628±0.029 | **0.807±0.007** |
| Qubit=6  6q 2l RY | 0.664±0.014 | 0.655±0.017 | 0.612±0.023 | 0.780±0.007 |
| Depth=4  4q 4l RY | 0.653±0.013 | 0.649±0.008 | 0.620±0.024 | 0.796±0.014 |
| Encoding RX+RY    | 0.645±0.011 | 0.645±0.007 | 0.631±0.029 | 0.772±0.013 |

- **Baseline**: Frozen QConv + 3-CNN
- **Exp1**: Trainable QConv (Lite) + 3-CNN
- **Exp2**: Trainable QConv (Lite) + ResNet-18
- **Exp3**: Trainable QConv (Lite) + RGB 融合 + ResNet-18

## 关键发现

1. **Exp3 融合方案是最大突破**（所有配置下稳定涨 12-16pp）
2. **量子参数边际效应有限**（Qubit/Depth/Encoding 对最终 acc 影响 < 4pp）
3. **ResNet-18 杀鸡用牛刀**（4×63×63 量子特征，3-CNN 更匹配，ResNet 反而差 2-5pp）
4. **Lite 可训练参数效果有限**（Exp1 vs Baseline 差异 < 1σ）
5. **K-fold CV 验证数字稳健**（差异 < 1pp，标准差 < 0.005）

## 数据来源

- **QNN4EO (JSTARS 2021)**: A. Sebastianelli et al. "On Circuit-based Hybrid Quantum Neural Networks for Remote Sensing Imagery Classification."
- **Quanv4EO (TGRS 2025)**: A. Sebastianelli et al. "Quanv4EO: Empowering Earth Observation by Means of Quanvolutional Neural Networks." doi:10.1109/TGRS.2025.3556335
- **EuroSAT**: P. Helber et al. (arXiv:1709.00029)

## 仓库信息

- 作者: Yuki0-0i
- 平台: Linux + 2× RTX 5090
- 文档: `26项目报告1.docx` (上级目录)
- License: MIT

## 引用本项目

```bibtex
@misc{quanv4eo_hybridqcnn_2026,
  author = {Yuki0-0i},
  title = {Quanv4EO Hybrid Quantum-Classical CNN: Modernized Implementation and Improvement},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Yuki0-0i/Quanv4EO_HybridQCNN}}
}
```
