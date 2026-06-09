"""
并行版 pipeline: 500 张 EuroSAT
- 4 worker 并行跑 QConv2D
- 每 worker 独立 QConv2D 实例（避免 pickle qnode）
- 把 500 张图先 split 成 n_jobs 份
- 跨平台路径: Windows (D:\) / Linux (/hdd/)
"""
from __future__ import annotations
import sys
import os
import time
from pathlib import Path

ROOT_CANDIDATES = [
    Path("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604"),
    Path(r"D:\dxy1\Quanv4EO_0604"),
    Path(__file__).resolve().parents[1],
]
ROOT = next((p for p in ROOT_CANDIDATES if (p / "quanv4eo_modern").is_dir()), ROOT_CANDIDATES[-1])
sys.path.insert(0, str(ROOT))

# joblib 性能调优: 限制 BLAS 多线程 + 指向 ASCII 路径
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
_TMP = os.environ.get("TMPDIR", "/tmp")
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
    """worker 函数: 跑一张图的 QConv 并返回特征向量."""
    (path, label, qc_kwargs) = args
    # 延迟 import 防止 fork 问题
    from quanv4eo_modern.quantum.qconv2d import QConv2D
    qc = QConv2D(**qc_kwargs)
    img = load_image(path, target_size=64)
    feat = qc.apply(img).reshape(-1)
    return feat, label


def extract_features_parallel(
    paths: np.ndarray,
    labels: np.ndarray,
    qc_kwargs: dict,
    n_jobs: int = 4,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """并行提取 QConv 特征."""
    n = len(paths)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    args_list = [(p, y, qc_kwargs) for p, y in zip(paths, labels)]

    t0 = time.time()
    # Linux 上 multiprocessing 后端容易死锁, 改用 loky
    backend = os.environ.get("JOBLIB_BACKEND", "loky")
    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_process_one)(a) for a in args_list
    )
    elapsed = time.time() - t0

    feats = np.stack([r[0] for r in results], axis=0)
    labs = np.array([r[1] for r in results], dtype=np.int64)
    if verbose:
        print(f"  Total: {elapsed:.1f}s, {n} images, {elapsed/n:.2f}s/image (n_jobs={n_jobs}, backend={backend})", flush=True)
    return feats, labs


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_per_class", type=int, default=100, help="每类最多取多少张")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--qubits", type=int, default=4)
    parser.add_argument("--filters", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--encoding", type=str, default="ry")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--tag", type=str, default=None, help="特征文件名后缀")
    args = parser.parse_args()

    if args.tag is None:
        args.tag = f"{args.max_per_class}q{args.qubits}l{args.n_layers}_{args.encoding}"

    print("=" * 60)
    print(f"Pipeline (parallel): {args.max_per_class} 张/类 EuroSAT, {args.qubits}q {args.n_layers}l {args.encoding}")
    print(f"  n_jobs={args.n_jobs}, stride={args.stride}, kernel=2")
    print("=" * 60)

    # 1) 加载
    p, y, c2i = scan_dataset(
        ROOT / "datasets" / "EuroSAT",
        max_per_class=args.max_per_class,
    )
    print(f"Loaded {len(p)} images, classes={len(c2i)}")

    # 2) 切分
    tr_p, tr_y, va_p, va_y = train_val_split(p, y, val_ratio=0.2)
    print(f"Split: train={len(tr_p)}, val={len(va_p)}")

    # 3) QConv 配置
    qc_kwargs = dict(
        qubits=args.qubits, filters=args.filters, kernel_size=2, stride=args.stride,
        n_layers=args.n_layers, encoding=args.encoding,
    )
    print(f"QConv2D: {qc_kwargs}")

    # 4) 提取 train 特征
    print("\n[Train features]")
    X_train, y_train = extract_features_parallel(tr_p, tr_y, qc_kwargs, n_jobs=args.n_jobs)
    print(f"  X_train: {X_train.shape}")

    # 5) 提取 val 特征
    print("\n[Val features]")
    X_val, y_val = extract_features_parallel(va_p, va_y, qc_kwargs, n_jobs=args.n_jobs)
    print(f"  X_val: {X_val.shape}")

    # 6) 归一化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # 7) MLP
    print("\n[MLP training]")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        max_iter=300,
        random_state=42,
        early_stopping=True,
    )
    t0 = time.time()
    mlp.fit(X_train_s, y_train)
    print(f"  MLP fit: {time.time()-t0:.1f}s, n_iter={mlp.n_iter_}")
    y_pred = mlp.predict(X_val_s)
    acc_mlp = accuracy_score(y_val, y_pred)
    print(f"  MLP val accuracy: {acc_mlp:.3f}")

    # 8) RF
    print("\n[RF training]")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train_s, y_train)
    print(f"  RF fit: {time.time()-t0:.1f}s")
    y_pred_rf = rf.predict(X_val_s)
    acc_rf = accuracy_score(y_val, y_pred_rf)
    print(f"  RF val accuracy: {acc_rf:.3f}")

    # 9) Majority baseline
    majority = np.bincount(y_train).argmax()
    majority_acc = (y_val == majority).mean()
    print(f"\n[Majority baseline: {CLASS_NAMES[majority]}]")
    print(f"  Majority class accuracy: {majority_acc:.3f}")

    # 10) 保存特征 npy
    feat_path = ROOT / "experiments" / f"features_{args.tag}.npz"
    np.savez_compressed(
        feat_path,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        class_names=np.array(CLASS_NAMES),
        qc_kwargs=np.array(str(qc_kwargs)),
    )
    print(f"\nFeatures saved: {feat_path}")

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
