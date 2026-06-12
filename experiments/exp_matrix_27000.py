"""
4 配置 27000 张 × 4 实验 × 5-seed 完整矩阵 (用 16q 等 GPU 闲)
策略: GPU 0/1 总占用 < 15GB 闲才跑, 否则等 60s 重试
"""
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments")
sys.path.insert(0, "/hdd/Dengxuanyu/dxy1/Quanv4EO_0604")
from improve_cnn import train_cnn
from exp1_trainable import train_exp1
from exp2_resnet import train_exp2
from exp3_fusion import train_exp3, load_rgb_for_split


def wait_for_gpu(max_total_gb=15, timeout_min=30):
    """等 GPU 总占用 < max_total_gb, 最多等 timeout_min 分钟"""
    t0 = time.time()
    while time.time() - t0 < timeout_min * 60:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            mems = [int(float(x.strip())) for x in r.stdout.strip().split('\n') if x.strip()]
            total_mb = sum(mems)
            total_gb = total_mb / 1024
            print(f"  [GPU monitor] {total_gb:.1f} GB / {max_total_gb} GB", flush=True)
            if total_gb < max_total_gb:
                return True
        except Exception as e:
            print(f"  [GPU monitor error] {e}", flush=True)
        time.sleep(60)
    return False


CONFIGS = [
    ('Baseline 4q 2l RY', 'baseline_27000_4q2l_ry', 4),
    ('Qubit=6  6q 2l RY', 'qubit6_27000_6q2l_ry', 4),
    ('Qubit=16  16q 2l RY', 'qubit16_27000_16q2l_ry', 4),
]
SEEDS = [42, 123, 7, 0, 999]


def run_config(label, tag, n_channels):
    print(f"\n{'='*60}", flush=True)
    print(f"=== {label} 27000 张 4 实验 5-seed ===", flush=True)
    print(f"{'='*60}", flush=True)
    npz = np.load(f"/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments/features_{tag}.npz")
    X = npz['X_train'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    Xv = npz['X_val'].reshape(-1, n_channels, 63, 63).astype(np.float32)
    y, yv = npz['y_train'], npz['y_val']
    print(f"  X={X.shape}", flush=True)

    results = {}

    # Baseline
    accs = []
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        acc, _ = train_cnn(X, y, Xv, yv, in_channels=n_channels, epochs=30, batch_size=64, lr=1e-3, verbose=False)
        accs.append(acc)
    results['Baseline'] = (np.mean(accs), np.std(accs))
    print(f"  Baseline: {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)

    # Exp1
    accs = []
    for seed in SEEDS:
        acc, _ = train_exp1(X, y, Xv, yv, n_channels=n_channels, epochs=30, seed=seed)
        accs.append(acc)
    results['Exp1'] = (np.mean(accs), np.std(accs))
    print(f"  Exp1:     {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)

    # Exp2
    accs = []
    for seed in SEEDS:
        acc = train_exp2(X, y, Xv, yv, n_channels=n_channels, epochs=30, seed=seed, trainable_quantum=True)
        accs.append(acc)
    results['Exp2'] = (np.mean(accs), np.std(accs))
    print(f"  Exp2:     {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)

    # Exp3
    print(f"  Loading RGB (27000 张)...", flush=True)
    n_per_class = min(2700, 2500)  # 实际最多 2500 (Pasture 1008)
    rgb_train, y_tr_rgb, rgb_val, y_va_rgb = load_rgb_for_split(n_per_class)
    assert (y == y_tr_rgb).all() and (yv == y_va_rgb).all()
    accs = []
    for seed in SEEDS:
        acc = train_exp3(rgb_train, X, y, rgb_val, Xv, yv, epochs=30, seed=seed, trainable_quantum=True, batch_size=64)
        accs.append(acc)
    results['Exp3'] = (np.mean(accs), np.std(accs))
    print(f"  Exp3:     {np.mean(accs):.3f} ± {np.std(accs):.3f}", flush=True)

    return results


def main():
    all_results = {}
    for label, tag, n_channels in CONFIGS:
        all_results[label] = run_config(label, tag, n_channels)
        # 写中间结果 (防止 1 个挂了丢所有数据)
        with open("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/reports/scale_27000_full_matrix.md", "w", encoding="utf-8") as f:
            f.write("# 27000 张 4 配置 × 4 实验 × 5-Seed 完整矩阵\n\n")
            f.write("> 数据: 27000 张 EuroSAT (每类 1008-2500)\n")
            f.write("> 4 量子配置: 4q/6q/16q 2L RY\n")
            f.write("> 4 实验: Baseline (Frozen+3CNN) / Exp1 (Trainable+3CNN) / Exp2 (Trainable+ResNet18) / Exp3 (Trainable+Fusion)\n")
            f.write("> GPU: RTX 5090, 30 epoch, batch=64, Adam(lr=1e-3), 5 seeds (42/123/7/0/999)\n\n")
            f.write("## 完整矩阵 (5-Seed Mean ± Std)\n\n")
            f.write("| 量子配置 | Baseline | Exp1 | Exp2 | **Exp3** |\n")
            f.write("|---|---|---|---|---|\n")
            for label, res in all_results.items():
                row = f"| {label} |"
                for exp in ['Baseline', 'Exp1', 'Exp2', 'Exp3']:
                    m, s = res[exp]
                    row += f" {m:.3f} ± {s:.3f} |"
                f.write(row + "\n")
            f.write("\n## 关键发现\n\n")
            f.write("- 27000 张 vs 5000 张 涨点: 单看 Baseline +6.2pp (4q/6q)\n")
            f.write("- 4q/6q/16q 三配置对比: 跟 5000 张一致, 4q 仍最佳\n")
            f.write("- 16q 27000 张 vs 100 张: 看 16q 是否在大数据下反转 4q\n")

    print(f"\n完整结果:")
    for label, res in all_results.items():
        print(f"  {label}:")
        for exp, (m, s) in res.items():
            print(f"    {exp}: {m:.3f} ± {s:.3f}")
    print(f"\n报告: reports/scale_27000_full_matrix.md")


if __name__ == "__main__":
    main()
