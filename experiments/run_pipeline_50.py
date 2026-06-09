"""
End-to-end pipeline test: 50 张 EuroSAT
1. 加载 50 张图
2. 跑 QConv2D
3. 保存 npy 特征
4. 训 sklearn MLP / RandomForest
5. 输出 baseline 准确率
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from quanv4eo_modern.data.dataset import scan_dataset, load_image, train_val_split, CLASS_NAMES
from quanv4eo_modern.quantum.qconv2d import QConv2D


def extract_features(
    paths: np.ndarray,
    labels: np.ndarray,
    qc: QConv2D,
    target_size: int = 64,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """对一组图片跑 QConv2D，特征 = 拉平的 (H_out, W_out, filters)."""
    n = len(paths)
    feats_list = []
    t0 = time.time()
    for i, p in enumerate(paths):
        img = load_image(p, target_size=target_size)
        feat_map = qc.apply(img)  # (H_out, W_out, filters)
        feat_vec = feat_map.reshape(-1)  # flatten
        feats_list.append(feat_vec)
        if verbose and (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  [{i+1}/{n}] elapsed={elapsed:.1f}s ETA={eta:.1f}s")
    X = np.stack(feats_list, axis=0)
    y = labels.copy()
    return X, y


def main():
    print("=" * 60)
    print("Pipeline test: 50 张 EuroSAT, 4q 2layers RY")
    print("=" * 60)

    # 1) 加载 50 张 (每类 5 张)
    p, y, c2i = scan_dataset(
        ROOT / "datasets" / "EuroSAT",
        max_per_class=5,
    )
    print(f"Loaded {len(p)} images, classes={len(c2i)}")

    # 2) 切分
    tr_p, tr_y, va_p, va_y = train_val_split(p, y, val_ratio=0.2)
    print(f"Split: train={len(tr_p)}, val={len(va_p)}")

    # 3) 建 QConv2D
    qc = QConv2D(qubits=4, filters=4, kernel_size=2, stride=1, n_layers=2, encoding="ry")
    print(f"QConv2D: qubits={qc.qubits} filters={qc.filters} ks={qc.kernel_size} stride={qc.stride} layers={qc.n_layers} enc={qc.encoding}")

    # 4) 提取特征
    print("\n[Train features]")
    X_train, y_train = extract_features(tr_p, tr_y, qc)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    print("\n[Val features]")
    X_val, y_val = extract_features(va_p, va_y, qc)
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")

    # 5) 归一化
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # 6) MLP
    print("\n[MLP training]")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64,),
        max_iter=200,
        random_state=42,
        early_stopping=True,
    )
    t0 = time.time()
    mlp.fit(X_train_s, y_train)
    print(f"  MLP fit: {time.time()-t0:.1f}s, n_iter={mlp.n_iter_}")
    y_pred = mlp.predict(X_val_s)
    acc_mlp = accuracy_score(y_val, y_pred)
    print(f"  MLP val accuracy: {acc_mlp:.3f}")

    # 7) Random Forest
    print("\n[RF training]")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    t0 = time.time()
    rf.fit(X_train_s, y_train)
    print(f"  RF fit: {time.time()-t0:.1f}s")
    y_pred_rf = rf.predict(X_val_s)
    acc_rf = accuracy_score(y_val, y_pred_rf)
    print(f"  RF val accuracy: {acc_rf:.3f}")

    # 8) Baseline: 多数类
    majority = np.bincount(y_train).argmax()
    majority_acc = (y_val == majority).mean()
    print(f"\n[Baseline: majority class {CLASS_NAMES[majority]}]")
    print(f"  Majority class accuracy: {majority_acc:.3f}")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Feature dim      : {X_train.shape[1]}")
    print(f"MLP val acc      : {acc_mlp:.3f}")
    print(f"RandomForest acc : {acc_rf:.3f}")
    print(f"Majority baseline: {majority_acc:.3f}")


if __name__ == "__main__":
    main()
