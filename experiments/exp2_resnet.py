"""
Exp2: Trainable QConv (Lite) + ResNet-18

控制变量: 仅换后端 (3-CNN → ResNet-18)
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


class QConvResNet18(nn.Module):
    """QConv (Lite trainable) + ResNet-18 后端."""
    def __init__(self, n_channels=4, num_classes=10, trainable_quantum=True):
        super().__init__()
        if trainable_quantum:
            self.tqconv = TrainableQConvLite(n_channels=n_channels)
        else:
            self.tqconv = nn.Identity()
        # 改 ResNet-18 第一层 Conv2d: 3 → n_channels
        self.resnet = tv_models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(n_channels, 64, 7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.tqconv(x)
        return self.resnet(x)


def train_exp2(X, y, Xv, yv, n_channels=4, epochs=30, batch_size=32, lr=1e-3, seed=42,
               trainable_quantum=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xt = torch.from_numpy(X).float().to(DEVICE)
    yt = torch.from_numpy(y).long().to(DEVICE)
    Xv_t = torch.from_numpy(Xv).float().to(DEVICE)

    train_ds = TensorDataset(Xt, yt)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = QConvResNet18(n_channels=n_channels, trainable_quantum=trainable_quantum).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            y_pred = model(Xv_t).argmax(dim=1).cpu().numpy()
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(yv, y_pred)
        if acc > best_val_acc:
            best_val_acc = acc
    return best_val_acc


def run_exp2(n_per_class=500, n_channels=4, seeds=(42, 123, 7, 0, 999), epochs=30,
             tag='baseline_5000_4q2l_ry', trainable_quantum=True):
    print(f"\n{'='*60}")
    print(f"Exp2: QConv + ResNet-18 @ {n_per_class*10} 张 (tag={tag}, trainable_q={trainable_quantum})")
    print(f"{'='*60}")
    npz = np.load(f'/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz')
    X = npz['X_train'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    Xv = npz['X_val'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    y = npz['y_train']
    yv = npz['y_val']
    print(f"  X={X.shape}")

    accs = []
    t0 = time.time()
    for seed in seeds:
        acc = train_exp2(X, y, Xv, yv, n_channels=n_channels, epochs=epochs,
                          seed=seed, trainable_quantum=trainable_quantum)
        accs.append(acc)
        print(f"  seed={seed}: {acc:.3f}")
    mean, std = np.mean(accs), np.std(accs)
    print(f"  *** {mean:.3f} ± {std:.3f} (total {time.time()-t0:.0f}s) ***")
    return mean, std


def main():
    print("=" * 60)
    print("Exp2: QConv + ResNet-18 5-Seed 评估")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # 5000 张: trainable + ResNet-18
    m5k, s5k = run_exp2(n_per_class=500, n_channels=4, tag='baseline_5000_4q2l_ry',
                        trainable_quantum=True)

    # 2000 张
    m2k, s2k = run_exp2(n_per_class=200, n_channels=4, tag='baseline_2000_4q2l_ry',
                        trainable_quantum=True)

    # 1000 张
    m1k, s1k = run_exp2(n_per_class=100, n_channels=4, tag='baseline_1000_4q2l_ry',
                        trainable_quantum=True)

    # 报告
    md = '/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/reports/exp2_resnet.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write("# Exp2: QConv (Lite Trainable) + ResNet-18 (5-Seed)\n\n")
        f.write("> **核心改动**: 后端从 3-layer CNN 替换为 ResNet-18\n")
        f.write("> ResNet-18 第一层 Conv2d 改为接 4 通道 (QConv 特征)\n")
        f.write("> 数据: 1000/2000/5000 张 4q 2l RY\n")
        f.write("> 训练: 30 epoch, batch=32, Adam(lr=1e-3)\n\n")
        f.write("## 结果\n\n")
        f.write("| 数据规模 | Exp1 (3-CNN Trainable) | **Exp2 (ResNet-18 Trainable)** | Δ vs 3-CNN |\n")
        f.write("|---|---|---|---|\n")
        f.write(f"| 1000 张 | 0.559 ± 0.025 | **{m1k:.3f} ± {s1k:.3f}** | {m1k-0.559:+.3f} |\n")
        f.write(f"| 2000 张 | 0.605 ± 0.019 | **{m2k:.3f} ± {s2k:.3f}** | {m2k-0.605:+.3f} |\n")
        f.write(f"| 5000 张 | 0.652 ± 0.016 | **{m5k:.3f} ± {s5k:.3f}** | {m5k-0.652:+.3f} |\n")
        f.write("\n## 关键发现\n\n")
        f.write("- ResNet-18 替换 3-CNN 后端对 QConv 特征的影响\n")
        f.write("- 跟 Exp1 相比的提升反映后端 CNN 强度对量子特征利用的贡献\n")
    print(f"\n报告: {md}")


if __name__ == "__main__":
    main()
