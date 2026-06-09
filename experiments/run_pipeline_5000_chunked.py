"""
分批版 pipeline: 5000 张 EuroSAT
- 5000 张分 5 批, 每批 1000 张
- 每批用 8 worker (避免多 worker hang)
- 每批单独保存 npz, 最后合并

Usage:
  python run_pipeline_5000_chunked.py
"""
from __future__ import annotations
import sys
import os
import time
from pathlib import Path

ROOT = next(
    (p for p in [
        Path("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604"),
        Path(r"D:\dxy1\Quanv4EO_0604"),
    ] if (p / "quanv4eo_modern").is_dir()),
    Path(__file__).resolve().parents[1],
)
sys.path.insert(0, str(ROOT))

# joblib tuning
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_TMP = "/tmp"
os.makedirs(_TMP, exist_ok=True)
os.environ.setdefault("TMP", _TMP)
os.environ.setdefault("TEMP", _TMP)
os.environ.setdefault("TMPDIR", _TMP)

import numpy as np
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from quanv4eo_modern.data.dataset import scan_dataset, load_image, train_val_split, CLASS_NAMES


def _process_one(args):
    (path, label, qc_kwargs) = args
    from quanv4eo_modern.quantum.qconv2d import QConv2D
    qc = QConv2D(**qc_kwargs)
    img = load_image(path, target_size=64)
    feat = qc.apply(img).reshape(-1)
    return feat, label


def extract_chunk(paths, labels, qc_kwargs, n_jobs=8, verbose=True):
    n = len(paths)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    args_list = [(p, y, qc_kwargs) for p, y in zip(paths, labels)]
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_one)(a) for a in args_list
    )
    elapsed = time.time() - t0
    feats = np.stack([r[0] for r in results], axis=0)
    labs = np.array([r[1] for r in results], dtype=np.int64)
    if verbose:
        print(f"  chunk done: {n} imgs in {elapsed:.1f}s ({elapsed/n:.2f}s/img, n_jobs={n_jobs})", flush=True)
    return feats, labs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_class", type=int, default=500)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--encoding", type=str, default="ry")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--tag", type=str, default="baseline_5000_4q2l_ry")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Chunked pipeline: {args.max_per_class}/类, {args.qubits}q {args.n_layers}l {args.encoding}")
    print(f"  chunk_size={args.chunk_size}, n_jobs={args.n_jobs}")
    print("=" * 60)

    p, y, c2i = scan_dataset(
        ROOT / "datasets" / "EuroSAT",
        max_per_class=args.max_per_class,
    )
    print(f"Loaded {len(p)} imgs, classes={len(c2i)}", flush=True)

    tr_p, tr_y, va_p, va_y = train_val_split(p, y, val_ratio=0.2)
    print(f"Split: train={len(tr_p)}, val={len(va_p)}", flush=True)

    qc_kwargs = dict(
        qubits=args.qubits, filters=args.filters, kernel_size=2, stride=1,
        n_layers=args.n_layers, encoding=args.encoding,
    )

    # ---- Train: 分批 ----
    print(f"\n[Train features: {len(tr_p)} imgs in {(len(tr_p)+args.chunk_size-1)//args.chunk_size} chunks]", flush=True)
    t_total = time.time()
    train_chunks = []
    for ci in range(0, len(tr_p), args.chunk_size):
        chunk_paths = tr_p[ci:ci + args.chunk_size]
        chunk_labels = tr_y[ci:ci + args.chunk_size]
        print(f"  chunk {ci//args.chunk_size + 1}/{(len(tr_p)+args.chunk_size-1)//args.chunk_size}: {len(chunk_paths)} imgs", flush=True)
        X, lab = extract_chunk(chunk_paths, chunk_labels, qc_kwargs, n_jobs=args.n_jobs)
        train_chunks.append((X, lab))
    X_train = np.concatenate([c[0] for c in train_chunks], axis=0)
    y_train = np.concatenate([c[1] for c in train_chunks], axis=0)
    print(f"  X_train: {X_train.shape}, train total time: {(time.time()-t_total)/60:.1f} min", flush=True)

    # ---- Val: 分批 ----
    print(f"\n[Val features: {len(va_p)} imgs in {(len(va_p)+args.chunk_size-1)//args.chunk_size} chunks]", flush=True)
    t_val = time.time()
    val_chunks = []
    for ci in range(0, len(va_p), args.chunk_size):
        chunk_paths = va_p[ci:ci + args.chunk_size]
        chunk_labels = va_y[ci:ci + args.chunk_size]
        print(f"  chunk {ci//args.chunk_size + 1}/{(len(va_p)+args.chunk_size-1)//args.chunk_size}: {len(chunk_paths)} imgs", flush=True)
        X, lab = extract_chunk(chunk_paths, chunk_labels, qc_kwargs, n_jobs=args.n_jobs)
        val_chunks.append((X, lab))
    X_val = np.concatenate([c[0] for c in val_chunks], axis=0)
    y_val = np.concatenate([c[1] for c in val_chunks], axis=0)
    print(f"  X_val: {X_val.shape}, val total time: {(time.time()-t_val)/60:.1f} min", flush=True)

    # ---- 归一化 + 训练 ----
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print("\n[MLP training]", flush=True)
    mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42, early_stopping=True)
    t0 = time.time()
    mlp.fit(X_train_s, y_train)
    acc_mlp = accuracy_score(y_val, mlp.predict(X_val_s))
    print(f"  MLP val acc: {acc_mlp:.3f} (fit {time.time()-t0:.1f}s)", flush=True)

    print("\n[RF training]", flush=True)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train_s, y_train)
    acc_rf = accuracy_score(y_val, rf.predict(X_val_s))
    print(f"  RF val acc: {acc_rf:.3f} (fit {time.time()-t0:.1f}s)", flush=True)

    majority = np.bincount(y_train).argmax()
    majority_acc = (y_val == majority).mean()
    print(f"\n[Majority baseline: {CLASS_NAMES[majority]}]: {majority_acc:.3f}", flush=True)

    # ---- 保存 ----
    feat_path = ROOT / "experiments" / f"features_{args.tag}.npz"
    np.savez_compressed(
        feat_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        class_names=np.array(CLASS_NAMES),
        qc_kwargs=np.array(str(qc_kwargs)),
    )
    print(f"\nFeatures saved: {feat_path}", flush=True)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")
    print(f"Feature dim: {X_train.shape[1]}")
    print(f"MLP acc: {acc_mlp:.3f}")
    print(f"RF  acc: {acc_rf:.3f}")
    print(f"Majority baseline: {majority_acc:.3f}")


if __name__ == "__main__":
    main()
