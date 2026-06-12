"""
8q / 16q QConv 100 张评估 (CPU)

8q 2L 100 张 8 worker: ~2.3h
16q 2L 100 张 8 worker: ~3h

目的: 看 8q / 16q 相对 4q 6q 是否有表达力突破
"""
import sys
import os
import time
from pathlib import Path
import numpy as np

ROOT = Path("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604")
sys.path.insert(0, str(ROOT))

# joblib tuning (CPU 不抢 GPU)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TMP", "/tmp")
os.environ.setdefault("TEMP", "/tmp")
os.environ.setdefault("TMPDIR", "/tmp")

from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import torch

from quanv4eo_modern.data.dataset import scan_dataset, load_image, train_val_split, CLASS_NAMES
from quanv4eo_modern.quantum.qconv2d import QConv2D


def _process_one(args):
    (path, label, qc_kwargs) = args
    qc = QConv2D(**qc_kwargs)
    img = load_image(path, target_size=64)
    feat = qc.apply(img).reshape(-1)
    return feat, label


def run_extraction(tag, n_per_class, qubits, n_layers, encoding, n_jobs=8, chunk_size=100):
    """小数据集 QConv 提取 (500 张 = 80 train + 20 val)."""
    npz_path = ROOT / "experiments" / f"features_{tag}.npz"
    if npz_path.exists():
        print(f"  [SKIP] {tag} npz 已存在")
        return
    p, y, c2i = scan_dataset(ROOT / "datasets" / "EuroSAT", max_per_class=n_per_class)
    tr_p, tr_y, va_p, va_y = train_val_split(p, y, val_ratio=0.2)
    print(f"  train={len(tr_p)}, val={len(va_p)}", flush=True)
    qc_kwargs = dict(qubits=qubits, filters=qubits, kernel_size=2, stride=1,
                      n_layers=n_layers, encoding=encoding, seed=758493)
    print(f"  qc_kwargs: {qc_kwargs}", flush=True)

    def extract_chunk(paths, labels):
        args_list = [(pp, yy, qc_kwargs) for pp, yy in zip(paths, labels)]
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_one)(a) for a in args_list
        )
        return (np.stack([r[0] for r in results]),
                np.array([r[1] for r in results], dtype=np.int64))

    t0 = time.time()
    train_feats, train_labels = extract_chunk(tr_p, tr_y)
    print(f"  train 提取: {train_feats.shape}, {time.time()-t0:.0f}s", flush=True)
    val_feats, val_labels = extract_chunk(va_p, va_y)
    print(f"  val 提取:   {val_feats.shape}, {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(
        npz_path,
        X_train=train_feats, y_train=train_labels,
        X_val=val_feats, y_val=val_labels,
        class_names=np.array(CLASS_NAMES),
        qc_kwargs=np.array(str(qc_kwargs)),
    )
    print(f"  saved: {npz_path}", flush=True)


def run_5seed_cnn(tag, n_channels, seeds=[42, 123, 7, 0, 999]):
    """加载 npz, 5-seed CNN 训练."""
    from improve_cnn_2000 import train_cnn  # noqa
    # 避免 relative import, 直接 inline train_cnn 逻辑
    sys.path.insert(0, str(ROOT / "experiments"))
    from improve_cnn import train_cnn  # type: ignore
    npz = np.load(ROOT / "experiments" / f"features_{tag}.npz")
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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, required=True, choices=[8, 16])
    parser.add_argument("--n_per_class", type=int, default=10)  # 100 张
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--encoding", type=str, default="ry")
    parser.add_argument("--n_jobs", type=int, default=8)
    args = parser.parse_args()

    tag = f"q{args.qubits}_{args.n_per_class*10}_{args.qubits}q{args.n_layers}l_{args.encoding}"

    print("=" * 60)
    print(f"{args.qubits}q {args.n_layers}l {args.encoding} @ {args.n_per_class*10} 张")
    print("=" * 60)

    # 1. 提取 QConv 特征
    print(f"\n[1/2] QConv 提取 ({args.qubits}q {args.n_layers}l, {args.n_per_class*10} 张)...")
    t0 = time.time()
    run_extraction(tag, args.n_per_class, args.qubits, args.n_layers, args.encoding,
                    n_jobs=args.n_jobs)
    print(f"\nQConv 提取总耗时: {(time.time()-t0)/60:.1f} min\n")

    # 2. 5-seed CNN
    print(f"[2/2] 5-seed CNN 训练...")
    accs = run_5seed_cnn(tag, n_channels=args.qubits)
    mean, std = np.mean(accs), np.std(accs)
    print(f"\n*** {args.qubits}q 100 张 5-seed: {mean:.3f} ± {std:.3f} ***\n")

    # 写报告
    md = ROOT / "reports" / f"{args.qubits}q_5seed.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write(f"# {args.qubits}q 量子配置 5-Seed 评估 (100 张)\n\n")
        f.write(f"> 数据: {args.n_per_class*10} 张 EuroSAT ({args.n_per_class}/类)\n")
        f.write(f"> 配置: {args.qubits}q {args.n_layers}l {args.encoding}\n")
        f.write(f"> GPU: RTX 5090 (CNN 训练), CPU (QConv 提取, {args.n_jobs} worker)\n\n")
        f.write(f"## 5-Seed 结果\n\n")
        f.write(f"| seed | val acc |\n|---|---|\n")
        for seed, acc in zip([42, 123, 7, 0, 999], accs):
            f.write(f"| {seed} | {acc:.3f} |\n")
        f.write(f"\n**5-seed mean ± std: {mean:.3f} ± {std:.3f}**\n\n")
        f.write("## 对比 4q/6q 100 张\n\n")
        f.write("| 配置 | 5-seed (100 张) |\n|---|---|\n")
        f.write("| 4q 2l RY | 0.560 |\n")
        f.write("| 6q 2l RY | 0.499 |\n")
        f.write(f"| {args.qubits}q {args.n_layers}l {args.encoding} | {mean:.3f} |\n")
    print(f"报告: {md}")


if __name__ == "__main__":
    main()
