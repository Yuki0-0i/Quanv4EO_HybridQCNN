"""
在 2000 张 baseline 特征上跑 CNN 后端
- 走 GPU (improve_cnn 已自动检测)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score

_EXP_CANDIDATES = [
    Path("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments"),
    Path(r"D:\dxy1\Quanv4EO_0604\experiments"),
    Path(__file__).resolve().parent,
]
EXP = next((p for p in _EXP_CANDIDATES if p.is_dir()), _EXP_CANDIDATES[-1])
sys.path.insert(0, str(EXP))
from improve_cnn import load_features_2d, train_cnn, QConvCNN


def main():
    tag = "baseline_2000_4q2l_ry"
    print(f"Loading {tag} ...")
    Xt, yt, Xv, yv = load_features_2d(tag, c=4)
    print(f"  Shape: {Xt.shape}")

    # 多 epoch 训练, 让 CNN 充分收敛
    print("\nTraining CNN (50 epochs) ...")
    acc, _ = train_cnn(Xt, yt, Xv, yv, in_channels=4, epochs=50, lr=1e-3, verbose=True)
    print(f"\n*** 2000 张 Baseline 4q 2l RY + CNN = {acc:.3f} ***")

    # 对比总结
    print("\n" + "=" * 60)
    print("综合对比 (1000 vs 2000 张)")
    print("=" * 60)
    print(f"{'配置':<30s} {'MLP':<6s} {'RF':<6s} {'CNN':<6s}")
    print(f"{'1000张 Baseline 4q2lRY':<30s} {0.310:<6.3f} {0.490:<6.3f} {0.695:<6.3f}")
    print(f"{'2000张 Baseline 4q2lRY':<30s} {0.280:<6.3f} {0.575:<6.3f} {acc:<6.3f}")


if __name__ == "__main__":
    main()
