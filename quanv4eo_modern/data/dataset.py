"""
EuroSAT 数据加载器
- 路径扫描 + 类别标签
- 限制每类最大数量（用于小样本）
- 划分 train/val
- PIL 读 jpg（不用 cv2，避坑 numpy<2.0）
"""
from __future__ import annotations
from pathlib import Path
import random
from typing import List, Tuple, Dict, Union

import numpy as np
from PIL import Image


CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]


def scan_dataset(
    root: Union[str, Path],
    max_per_class: int = None,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, Dict[str, int]]:
    """扫描数据集根目录，返回 (paths, labels, class_to_idx).

    Args:
        root: 包含 10 个类别子目录的根目录.
        max_per_class: 每类最多取多少张（None=全量）.
        seed: 随机种子.

    Returns:
        paths: 全部图片绝对路径列表
        labels: shape=(N,) int 标签
        class_to_idx: 类别名 -> 索引 字典
    """
    root = Path(root)
    rng = random.Random(seed)

    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    paths: List[str] = []
    labels: List[int] = []

    for cname in CLASS_NAMES:
        cdir = root / cname
        if not cdir.is_dir():
            print(f"[WARN] missing class dir: {cdir}")
            continue
        files = sorted(cdir.glob("*.jpg"))
        if max_per_class is not None and len(files) > max_per_class:
            rng.shuffle(files)
            files = files[:max_per_class]
        for f in files:
            paths.append(str(f.resolve()))
            labels.append(class_to_idx[cname])

    paths_arr = np.array(paths, dtype=object)
    labels_arr = np.array(labels, dtype=np.int64)
    return paths_arr, labels_arr, class_to_idx


def load_image(path: str, target_size: int = None) -> np.ndarray:
    """读 jpg -> float32 (H, W, 3) in [0, 1].

    PIL handles EuroSAT jpg well, doesn't need rasterio for jpg.
    """
    img = Image.open(path).convert("RGB")
    if target_size is not None and img.size != (target_size, target_size):
        # Pillow 9.1+ has Image.Resampling.BILINEAR; older versions use Image.BILINEAR
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR
        img = img.resize((target_size, target_size), resample)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr  # (H, W, 3), [0, 1]


def train_val_split(
    paths: np.ndarray,
    labels: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层划分 train/val."""
    rng = np.random.default_rng(seed)
    train_p, train_y, val_p, val_y = [], [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_p.extend(paths[val_idx])
        val_y.extend(labels[val_idx])
        train_p.extend(paths[train_idx])
        train_y.extend(labels[train_idx])
    return (
        np.array(train_p, dtype=object),
        np.array(train_y, dtype=np.int64),
        np.array(val_p, dtype=object),
        np.array(val_y, dtype=np.int64),
    )


if __name__ == "__main__":
    # 快速验证
    from pathlib import Path
    _DS_CANDS = [
        Path("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/datasets/EuroSAT"),
        Path(r"D:\dxy1\Quanv4EO_0604\datasets\EuroSAT"),
    ]
    root = next((p for p in _DS_CANDS if p.is_dir()), _DS_CANDS[-1])
    p, y, c2i = scan_dataset(root, max_per_class=50)
    print(f"Total: {len(p)} images")
    for cname, cidx in c2i.items():
        cnt = int((y == cidx).sum())
        print(f"  {cname:25s}: {cnt}")
    img = load_image(p[0], target_size=64)
    print(f"\nSample load: shape={img.shape}, dtype={img.dtype}, range=[{img.min():.3f}, {img.max():.3f}]")

    tr_p, tr_y, va_p, va_y = train_val_split(p, y, val_ratio=0.2)
    print(f"\nSplit: train={len(tr_p)}, val={len(va_p)}")
