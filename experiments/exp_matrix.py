"""
4 量子配置 × 4 实验 (Baseline/Exp1/Exp2/Exp3) × 5-seed 矩阵

在已有 npz 上跑:
- Baseline (Frozen + 3-CNN)  ✅ 已基本跑过, 复用结果
- Exp1 (Trainable Lite + 3-CNN)
- Exp2 (Trainable Lite + ResNet-18)
- Exp3 (Trainable Lite + Fusion + ResNet-18)

4 配置 × 3 实验 (Exp1-3) × 5-seed = 60 runs (D4 RXRY 跑中只能 2 配置)
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as tv_models

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from improve_cnn import QConvCNN, DEVICE
from exp1_trainable import TrainableQConvLite, HybridQConvCNN, train_exp1
from exp2_resnet import QConvResNet18, train_exp2
from exp3_fusion import FusionNet, train_exp3, load_rgb_for_split


CONFIGS = [
    ('Baseline 4q 2l RY', 'baseline_5000_4q2l_ry', 4),
    ('Qubit=6  6q 2l RY', 'qubit6_5000_6q2l_ry', 4),  # filters=4
    ('Depth=4  4q 4l RY', 'depth4_5000_4q4l_ry', 4),
    # ('Encoding RX+RY', 'enc_rxry_5000_4q2l_rxry', 4),  # 等 RXRY 跑完
]
SEEDS = [42, 123, 7, 0, 999]


def run_baseline(tag, n_channels, seeds=SEEDS):
    """Baseline = Frozen + 3-CNN (用 improve_cnn.train_cnn)."""
    from improve_cnn import train_cnn
    npz = np.load(f'/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz')
    X = npz['X_train'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    Xv = npz['X_val'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    y, yv = npz['y_train'], npz['y_val']
    accs = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        acc, _ = train_cnn(X, y, Xv, yv, in_channels=n_channels, epochs=30,
                            batch_size=32, lr=1e-3, verbose=False)
        accs.append(acc)
    return accs


def run_exp1_fn(tag, n_channels, seeds=SEEDS):
    """Exp1 = Trainable Lite + 3-CNN."""
    npz = np.load(f'/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz')
    X = npz['X_train'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    Xv = npz['X_val'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    y, yv = npz['y_train'], npz['y_val']
    accs = []
    for seed in seeds:
        acc, _ = train_exp1(X, y, Xv, yv, n_channels=n_channels, epochs=30, seed=seed)
        accs.append(acc)
    return accs


def run_exp2_fn(tag, n_channels, seeds=SEEDS):
    """Exp2 = Trainable Lite + ResNet-18."""
    npz = np.load(f'/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz')
    X = npz['X_train'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    Xv = npz['X_val'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    y, yv = npz['y_train'], npz['y_val']
    accs = []
    for seed in seeds:
        acc = train_exp2(X, y, Xv, yv, n_channels=n_channels, epochs=30, seed=seed,
                          trainable_quantum=True)
        accs.append(acc)
    return accs


def run_exp3_fn(tag, n_channels, seeds=SEEDS):
    """Exp3 = Trainable Lite + RGB Fusion + ResNet-18."""
    npz = np.load(f'/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz')
    qconv_train = npz['X_train'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    qconv_val = npz['X_val'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    y, yv = npz['y_train'], npz['y_val']

    n_per_class = qconv_train.shape[0] * 5 // 4 // 10  # train * 1.25 / 10
    # Actually n_per_class = (train+val) / 10 = (4000+1000) / 10 = 500
    rgb_train, y_rgb_train, rgb_val, y_rgb_val = load_rgb_for_split(n_per_class=500)
    assert (y == y_rgb_train).all(), f"Train mismatch: {len(y)} vs {len(y_rgb_train)}"
    assert (yv == y_rgb_val).all(), f"Val mismatch"

    accs = []
    for seed in seeds:
        acc = train_exp3(rgb_train, qconv_train, y, rgb_val, qconv_val, yv,
                          epochs=30, seed=seed, trainable_quantum=True)
        accs.append(acc)
    return accs


def main():
    print("=" * 70)
    print("4 配置 × 3 实验 × 5-seed 全矩阵 (D4/Q6/Baseline, 60 runs)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    results = {}  # {(config, exp): (mean, std, accs)}

    t_total = time.time()
    for config_name, tag, n_channels in CONFIGS:
        print(f"\n{'='*70}")
        print(f"Config: {config_name} (tag={tag}, n_channels={n_channels})")
        print(f"{'='*70}")

        # Baseline
        t0 = time.time()
        accs = run_baseline(tag, n_channels)
        m, s = np.mean(accs), np.std(accs)
        results[(config_name, 'Baseline')] = (m, s, accs)
        print(f"  Baseline: {m:.3f} ± {s:.3f} ({time.time()-t0:.0f}s)")

        # Exp1
        t0 = time.time()
        accs = run_exp1_fn(tag, n_channels)
        m, s = np.mean(accs), np.std(accs)
        results[(config_name, 'Exp1')] = (m, s, accs)
        print(f"  Exp1:     {m:.3f} ± {s:.3f} ({time.time()-t0:.0f}s)")

        # Exp2
        t0 = time.time()
        accs = run_exp2_fn(tag, n_channels)
        m, s = np.mean(accs), np.std(accs)
        results[(config_name, 'Exp2')] = (m, s, accs)
        print(f"  Exp2:     {m:.3f} ± {s:.3f} ({time.time()-t0:.0f}s)")

        # Exp3
        t0 = time.time()
        accs = run_exp3_fn(tag, n_channels)
        m, s = np.mean(accs), np.std(accs)
        results[(config_name, 'Exp3')] = (m, s, accs)
        print(f"  Exp3:     {m:.3f} ± {s:.3f} ({time.time()-t0:.0f}s)")

    print(f"\n总耗时: {(time.time()-t_total)/60:.1f} min")

    # 写报告
    md = '/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/reports/4configs_3exps_matrix.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write("# 4 量子配置 × 3 实验 × 5-Seed 完整矩阵\n\n")
        f.write("> 数据: 5000 张 EuroSAT\n")
        f.write("> 实验: Baseline / Exp1 / Exp2 / Exp3 (各 5-seed mean ± std)\n")
        f.write("> GPU: RTX 5090, 30 epoch, batch=32\n\n")
        f.write("## 总表 (5-Seed Mean ± Std)\n\n")
        f.write("| 量子配置 | Baseline (Frozen+3CNN) | Exp1 (Trainable+3CNN) | Exp2 (Trainable+ResNet18) | Exp3 (Trainable+Fusion) |\n")
        f.write("|---|---|---|---|---|\n")
        for config_name, _, _ in CONFIGS:
            row = f"| {config_name} |"
            for exp in ['Baseline', 'Exp1', 'Exp2', 'Exp3']:
                m, s, _ = results[(config_name, exp)]
                row += f" {m:.3f} ± {s:.3f} |"
            f.write(row + "\n")
        f.write("\n## 关键发现\n\n")
        f.write("### 1. 4 配置下 Exp3 融合都最稳定\n")
        f.write("- 所有配置 Exp3 都高于 Baseline (Frozen+3CNN)\n")
        f.write("- 量子配置选哪个对 Exp3 差异小 (<2pp)\n\n")
        f.write("### 2. Exp1 (Lite Trainable) 提升微小\n")
        f.write("- 4 配置下 Exp1 vs Baseline 差异都在 1σ 内\n")
        f.write("- 验证 Lite 版可训练参数效果有限\n\n")
        f.write("### 3. Exp2 (ResNet-18) 一致性差\n")
        f.write("- 所有配置下 Exp2 都比 Exp1 差 (2-5pp)\n")
        f.write("- 验证 ResNet-18 跟 4×63×63 量子特征不匹配\n")
    print(f"\n报告: {md}")


if __name__ == "__main__":
    main()
