"""
8q/16q 5000 张 4 实验 × 5-seed (用已有 npz)
"""
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments")
sys.path.insert(0, "/hdd/Dengxuanyu/dxy1/Quanv4EO_0604")
from improve_cnn import train_cnn
from exp1_trainable import train_exp1
from exp2_resnet import train_exp2
from exp3_fusion import train_exp3, load_rgb_for_split


CONFIGS = [
    ('8q 2l RY', 'q8_5000_8q2l_ry', 8),
    ('16q 2l RY', 'q16_5000_16q2l_ry', 16),
]
SEEDS = [42, 123, 7, 0, 999]


def main():
    for config_name, tag, n_channels in CONFIGS:
        print(f"\n=== {config_name} 5000 张 4 实验 5-seed ===", flush=True)
        npz = np.load(f"/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz")
        X = npz['X_train'].reshape(-1, n_channels, 63, 63).astype(np.float32)
        Xv = npz['X_val'].reshape(-1, n_channels, 63, 63).astype(np.float32)
        y, yv = npz['y_train'], npz['y_val']

        # Baseline
        t0 = time.time()
        accs = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            acc, _ = train_cnn(X, y, Xv, yv, in_channels=n_channels, epochs=30, batch_size=32, lr=1e-3, verbose=False)
            accs.append(acc)
        print(f"  Baseline: {np.mean(accs):.3f} ± {np.std(accs):.3f} ({time.time()-t0:.0f}s)", flush=True)

        # Exp1
        t0 = time.time()
        accs = []
        for seed in SEEDS:
            acc, _ = train_exp1(X, y, Xv, yv, n_channels=n_channels, epochs=30, seed=seed)
            accs.append(acc)
        print(f"  Exp1:     {np.mean(accs):.3f} ± {np.std(accs):.3f} ({time.time()-t0:.0f}s)", flush=True)

        # Exp2
        t0 = time.time()
        accs = []
        for seed in SEEDS:
            acc = train_exp2(X, y, Xv, yv, n_channels=n_channels, epochs=30, seed=seed, trainable_quantum=True)
            accs.append(acc)
        print(f"  Exp2:     {np.mean(accs):.3f} ± {np.std(accs):.3f} ({time.time()-t0:.0f}s)", flush=True)

        # Exp3
        t0 = time.time()
        print("  Loading RGB...", flush=True)
        rgb_train, y_tr_rgb, rgb_val, y_va_rgb = load_rgb_for_split(500)
        assert (y == y_tr_rgb).all() and (yv == y_va_rgb).all()
        accs = []
        for seed in SEEDS:
            acc = train_exp3(rgb_train, X, y, rgb_val, Xv, yv, epochs=30, seed=seed, trainable_quantum=True)
            accs.append(acc)
        print(f"  Exp3:     {np.mean(accs):.3f} ± {np.std(accs):.3f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
