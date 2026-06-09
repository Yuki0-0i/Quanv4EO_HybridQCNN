"""
Exp3: 特征融合 - Trainable QConv + RGB 原图 + ResNet-18

控制变量: 仅加 RGB 融合 (相对 Exp2)

架构:
  RGB (64x64x3) ─→ ResNet-18(3ch) ─→ feature_1 (512d) ─┐
                                                        ├─ concat → FC(10)
  QConv 特征 (4,63,63) ─→ QConv+Lite ─→ Conv ─→ Pool → feature_2 (256d) ─┘
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as tv_models
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from improve_cnn import DEVICE
from exp1_trainable import TrainableQConvLite
from quanv4eo_modern.data.dataset import scan_dataset, load_image


class FusionNet(nn.Module):
    """RGB 分支 + QConv 分支 → 融合 FC"""
    def __init__(self, num_classes=10, qconv_channels=4, trainable_quantum=True):
        super().__init__()
        # 分支 1: 经典 ResNet-18 吃 RGB
        self.rgb_branch = tv_models.resnet18(weights=None)
        self.rgb_branch.fc = nn.Identity()  # 输出 512 维

        # 分支 2: QConv (Lite Trainable) + 小 CNN
        if trainable_quantum:
            self.tqconv = TrainableQConvLite(n_channels=qconv_channels)
        else:
            self.tqconv = nn.Identity()
        self.qconv_branch = nn.Sequential(
            nn.Conv2d(qconv_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 63→31
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # → 64 维
            nn.Flatten(),
        )

        # 融合 FC
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, rgb, qconv_feat):
        f1 = self.rgb_branch(rgb)  # (N, 512)
        f2 = self.tqconv(qconv_feat)  # (N, 4, 63, 63)
        f2 = self.qconv_branch(f2)  # (N, 64)
        fused = torch.cat([f1, f2], dim=1)  # (N, 576)
        return self.fusion_fc(fused)


def load_rgb_for_split(n_per_class, val_ratio=0.2, seed=42):
    """加载 RGB 原图 (3, 64, 64) 与 QConv 特征 (4, 63, 63) 对齐."""
    from quanv4eo_modern.data.dataset import scan_dataset, train_val_split
    p, y, _ = scan_dataset('/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/datasets/EuroSAT',
                            max_per_class=n_per_class)
    tr_p, tr_y, va_p, va_y = train_val_split(p, y, val_ratio=val_ratio, seed=seed)

    def load_imgs(paths):
        return np.stack([load_image(p, 64) for p in paths]).transpose(0, 3, 1, 2)

    print(f"    加载 RGB train ({len(tr_p)})...", flush=True)
    rgb_train = load_imgs(tr_p).astype(np.float32)
    print(f"    加载 RGB val ({len(va_p)})...", flush=True)
    rgb_val = load_imgs(va_p).astype(np.float32)
    return rgb_train, tr_y, rgb_val, va_y


def train_exp3(rgb_train, qconv_train, y_train, rgb_val, qconv_val, y_val,
               epochs=30, batch_size=32, lr=1e-3, seed=42, trainable_quantum=True):
    torch.manual_seed(seed)
    np.random.seed(seed)

    rgb_t = torch.from_numpy(rgb_train).float().to(DEVICE)
    qconv_t = torch.from_numpy(qconv_train).float().to(DEVICE)
    y_t = torch.from_numpy(y_train).long().to(DEVICE)
    rgb_v = torch.from_numpy(rgb_val).float().to(DEVICE)
    qconv_v = torch.from_numpy(qconv_val).float().to(DEVICE)

    train_ds = TensorDataset(rgb_t, qconv_t, y_t)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = FusionNet(qconv_channels=qconv_train.shape[1], trainable_quantum=trainable_quantum).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for ep in range(epochs):
        model.train()
        for rgb_b, qconv_b, y_b in train_dl:
            rgb_b, qconv_b, y_b = rgb_b.to(DEVICE), qconv_b.to(DEVICE), y_b.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(rgb_b, qconv_b), y_b)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            y_pred = model(rgb_v, qconv_v).argmax(dim=1).cpu().numpy()
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_val, y_pred)
        if acc > best_val_acc:
            best_val_acc = acc
    return best_val_acc


def run_exp3(n_per_class=500, n_channels=4, seeds=(42, 123, 7, 0, 999), epochs=30,
             tag='baseline_5000_4q2l_ry', trainable_quantum=True):
    print(f"\n{'='*60}")
    print(f"Exp3: Fusion (QConv+RGB+ResNet18) @ {n_per_class*10} 张 (tag={tag}, trainable_q={trainable_quantum})")
    print(f"{'='*60}")

    # 加载 QConv 特征
    npz = np.load(f'/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz')
    qconv_train = npz['X_train'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    qconv_val = npz['X_val'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    y_train = npz['y_train']
    y_val = npz['y_val']

    # 加载 RGB
    rgb_train, y_train_rgb, rgb_val, y_val_rgb = load_rgb_for_split(n_per_class)
    # 验证对齐
    assert (y_train == y_train_rgb).all(), "Train label mismatch"
    assert (y_val == y_val_rgb).all(), "Val label mismatch"
    print(f"  rgb={rgb_train.shape} qconv={qconv_train.shape}")

    accs = []
    t0 = time.time()
    for seed in seeds:
        acc = train_exp3(rgb_train, qconv_train, y_train, rgb_val, qconv_val, y_val,
                          epochs=epochs, seed=seed, trainable_quantum=trainable_quantum)
        accs.append(acc)
        print(f"  seed={seed}: {acc:.3f}")
    mean, std = np.mean(accs), np.std(accs)
    print(f"  *** {mean:.3f} ± {std:.3f} (total {time.time()-t0:.0f}s) ***")
    return mean, std


def main():
    print("=" * 60)
    print("Exp3: 特征融合 (RGB + QConv + ResNet-18)")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    m5k, s5k = run_exp3(n_per_class=500, n_channels=4, tag='baseline_5000_4q2l_ry')
    m2k, s2k = run_exp3(n_per_class=200, n_channels=4, tag='baseline_2000_4q2l_ry')
    m1k, s1k = run_exp3(n_per_class=100, n_channels=4, tag='baseline_1000_4q2l_ry')

    md = '/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/reports/exp3_fusion.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write("# Exp3: 特征融合 (RGB + QConv + ResNet-18) (5-Seed)\n\n")
        f.write("> **核心改动**: 在 Exp2 基础上加 RGB 原图分支\n")
        f.write("> 双分支: ResNet-18(RGB, 512d) + QConv分支(64d) → 融合FC(576→256→10)\n")
        f.write("> 数据: 1000/2000/5000 张 4q 2l RY + 对应 RGB 原图\n\n")
        f.write("## 结果\n\n")
        f.write("| 数据规模 | Exp1 (3-CNN) | Exp2 (ResNet-18) | **Exp3 (Fusion)** | Exp3 vs Exp1 |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| 1000 张 | 0.559 | 0.508 | **{m1k:.3f}** | {m1k-0.559:+.3f} |\n")
        f.write(f"| 2000 张 | 0.605 | 0.561 | **{m2k:.3f}** | {m2k-0.605:+.3f} |\n")
        f.write(f"| 5000 张 | 0.652 | 0.629 | **{m5k:.3f}** | {m5k-0.652:+.3f} |\n")
        f.write("\n## 对比经典 CNN baseline (0.827 @ 5000)\n\n")
        f.write("| 数据规模 | 经典 CNN (无 QConv) | Exp3 (QConv+RGB) | Δ |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| 5000 张 | 0.827 | {m5k:.3f} | {m5k-0.827:+.3f} |\n")
    print(f"\n报告: {md}")


if __name__ == "__main__":
    main()
