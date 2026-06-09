"""
4 个量子配置 (1000 张) GPU CNN 5-seed 对比
- Baseline 4q 2l RY
- Qubit=6  6q 2l RY
- Depth=4  4q 4l RY
- Encoding RX+RY  4q 2l RX+RY
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from improve_cnn import train_cnn, DEVICE


def main():
    experiments = [
        ("Baseline 4q 2l RY", "baseline_1000_4q2l_ry", 4),
        ("Qubit=6  6q 2l RY", "qubit6_1000_6q2l_ry", 6),
        ("Depth=4  4q 4l RY", "depth4_1000_4q4l_ry", 4),
        ("Encoding RX+RY", "enc_rxry_1000_4q2l_rxry", 4),
    ]

    results = []
    for label, tag, c in experiments:
        npz = np.load(f'/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz')
        Xt = npz['X_train'].reshape(-1, c, 63, 63).astype(np.float32)
        Xv = npz['X_val'].reshape(-1, c, 63, 63).astype(np.float32)
        yt, yv = npz['y_train'], npz['y_val']

        print(f"\n=== {label} ({tag})  Xt={Xt.shape} Xv={Xv.shape} ===")
        accs = []
        for seed in [42, 123, 7, 0, 999]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            acc, _ = train_cnn(Xt, yt, Xv, yv, in_channels=c, epochs=30, batch_size=32, lr=1e-3, verbose=False)
            accs.append(acc)
            print(f"  seed={seed}: {acc:.3f}")
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        print(f"  *** {mean_acc:.3f} ± {std_acc:.3f} ***")
        results.append({"label": label, "tag": tag, "c": c, "mean": mean_acc, "std": std_acc})

    # 写报告
    md_path = Path('/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/reports/quantum_configs_1000_5seed.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# 4 量子配置 × 5 seeds (Windows 1000 张数据)\n\n")
        f.write("> 数据: Windows 原版 1000 张 EuroSAT (每类 100)\n")
        f.write("> 后端: CNN (3 conv + 2 FC, 30 epoch, Adam lr=1e-3)\n")
        f.write("> GPU: RTX 5090, PyTorch 2.11\n")
        f.write("> 5 seeds: 42, 123, 7, 0, 999\n\n")
        f.write("## 结果\n\n")
        f.write("| 量子配置 | channels | 5-seed mean ± std |\n")
        f.write("|---|---|---|\n")
        for r in results:
            f.write(f"| {r['label']} | {r['c']} | **{r['mean']:.3f} ± {r['std']:.3f}** |\n")
        f.write("\n## Δ vs Baseline\n\n")
        base = results[0]['mean']
        for r in results[1:]:
            delta = (r['mean'] - base) * 100
            f.write(f"- {r['label']}: {delta:+.1f}pp\n")
    print(f"\n报告: {md_path}")


if __name__ == "__main__":
    main()
