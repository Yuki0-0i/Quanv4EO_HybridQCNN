"""
环境烟测脚本 v3: Linux/Windows 跨平台
"""
import sys
import time
import numpy as np
import torch
from pathlib import Path

# 跨平台数据集路径
_DS_CANDIDATES = [
    Path("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/datasets/EuroSAT"),
    Path(r"D:\dxy1\Quanv4EO_0604\datasets\EuroSAT"),
]
DS_ROOT = next((p for p in _DS_CANDIDATES if p.is_dir()), _DS_CANDIDATES[-1])

print("=" * 60)
print(f"Plenv 环境烟测 v3 - dataset: {DS_ROOT}")
print("=" * 60)
print(f"Python : {sys.version.split()[0]}")
print(f"numpy  : {np.__version__}")
print(f"torch  : {torch.__version__} (cuda={torch.cuda.is_available()})")
print()

# 1. PennyLane + lightning.qubit
print("[1/4] PennyLane import ...")
import pennylane as qml
print(f"  PL        : {qml.__version__}")
import pennylane_lightning
print(f"  Lightning : loaded OK")
try:
    import pennylane_lightning_gpu
    print(f"  Lightning-GPU: loaded OK")
except Exception as e:
    print(f"  Lightning-GPU err (用 CPU 即可): {e}")

# 2. 数据读取
print("\n[2/4] rasterio 读 EuroSAT 1 张图 ...")
import rasterio
sample = next(DS_ROOT.rglob("*.jpg"))
print(f"  样本: {sample.parent.name}/{sample.name}")
with rasterio.open(sample) as src:
    img = src.read()  # (C, H, W)
    print(f"  shape: {img.shape}, dtype: {img.dtype}")
    print(f"  range: [{img.min()}, {img.max()}]")

# 3. 量子电路烟测
print("\n[3/4] 4-qubit RY + RandomLayers 烟测 ...")
n_qubits = 4
n_layers = 2
dev = qml.device("lightning.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(x, weights):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    qml.RandomLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

rng = np.random.default_rng(42)
x = rng.random(n_qubits)
w = rng.random((n_layers, n_qubits))

t0 = time.time()
out = circuit(x, w)
t1 = time.time()
print(f"  输出: {np.round(out, 3)}")
print(f"  单次耗时: {(t1-t0)*1000:.2f} ms")

# 4. 批量跑 1000 次估速度（更准）
print("\n[4/4] 批量 1000 次估时 ...")
N = 1000
t0 = time.time()
for _ in range(N):
    _ = circuit(x, w)
t1 = time.time()
avg_ms = (t1-t0) / N * 1000
print(f"  {N} 次: {(t1-t0)*1000:.0f} ms,  平均 {avg_ms:.3f} ms/次")

# 估算 EuroSAT 全量子预处理的耗时
patches_per_image = 63 * 63  # kernel=2, stride=1, 64x64 图
print(f"\n估算: 1 张 64x64 图, kernel=2 stride=1 -> {patches_per_image} patches")
est_per_img = patches_per_image * avg_ms / 1000
print(f"  预计单图耗时: {est_per_img:.1f} 秒 ({est_per_img/60:.1f} 分钟)")
print(f"  500 张图:     {500*est_per_img/60:.1f} 分钟")
print(f"  1000 张图:    {1000*est_per_img/60:.1f} 分钟")

print("\n" + "=" * 60)
print("烟测完成")
print("=" * 60)
