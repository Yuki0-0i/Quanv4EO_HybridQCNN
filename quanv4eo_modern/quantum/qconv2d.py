"""
Quantum Convolutional 2D layer
- PyTorch-free 设计: 直接吃/吐 numpy
- PennyLane 0.42 + lightning.qubit (CPU C++) / lightning.gpu (CUDA, 可选)
- 完全照搬原版 quanv4eo 的量子线路结构:
    RY/RX/RZ encoding (像素 × π) -> RandomLayers -> ⟨Z⟩ measure
- 支持 5 种 encoding: ry / rx / rz / rxry / rxyz
- 设备选择: device='cpu' (默认, 推荐) / 'gpu' (仅大电路/大批量才用)

Reference: D:/dxy1/Quanv4EO_0604/source/quanv4eo-main/circuits/random.py
"""
from __future__ import annotations
import os
import numpy as np
import pennylane as qml


def select_backend(device: str | None = None) -> str:
    """选择 PennyLane 后端.
    - 'cpu' / 'lightning.qubit' -> CPU C++ 后端 (推荐, 4-6 qubit 最快)
    - 'gpu' / 'lightning.gpu'  -> CUDA 后端 (需 custatevec-cu12)
    - None / 'auto'            -> 默认 CPU
    """
    env = (device or os.environ.get("QCONV_DEVICE", "cpu")).lower()
    if env in ("cpu", "lightning.qubit"):
        return "lightning.qubit"
    if env in ("gpu", "lightning.gpu"):
        return "lightning.gpu"
    raise ValueError(f"Unknown device: {device!r} (use 'cpu' or 'gpu')")


# ============================================================
# Encoding functions
# ============================================================
def _ry_encoding(phi, wires):
    """RY 编码: 用 patch 像素旋转前 min(|phi|, |wires|) 个 qubit.
    多余的 qubit 保持 |0⟩, 参与 RandomLayers 但不参与 encoding."""
    n_data = min(len(phi), len(wires))
    for j in range(n_data):
        qml.RY(np.pi * phi[j], wires=wires[j])


def _rx_encoding(phi, wires):
    n_data = min(len(phi), len(wires))
    for j in range(n_data):
        qml.RX(np.pi * phi[j], wires=wires[j])


def _rz_encoding(phi, wires):
    n_data = min(len(phi), len(wires))
    for j in range(n_data):
        qml.RZ(np.pi * phi[j], wires=wires[j])


def _rxry_encoding(phi, wires):
    """RX+RY 交替编码 (实验 3 用)."""
    n_data = min(len(phi), len(wires))
    for j in range(n_data):
        if j % 2 == 0:
            qml.RY(np.pi * phi[j], wires=wires[j])
        else:
            qml.RX(np.pi * phi[j], wires=wires[j])


def _rxyz_encoding(phi, wires):
    """混合编码: RY(3k) + RX(3k+1) + RZ(3k+2)"""
    n_data = min(len(phi), len(wires))
    for j in range(n_data):
        idx = j % 3
        if idx == 0:
            qml.RY(np.pi * phi[j], wires=wires[j])
        elif idx == 1:
            qml.RX(np.pi * phi[j], wires=wires[j])
        else:
            qml.RZ(np.pi * phi[j], wires=wires[j])


ENCODING_FNS = {
    "ry": _ry_encoding,
    "rx": _rx_encoding,
    "rz": _rz_encoding,
    "rxry": _rxry_encoding,
    "rxyz": _rxyz_encoding,
}


# ============================================================
# QConv2D layer
# ============================================================
class QConv2D:
    """Quantum 2D Convolution layer (PennyLane 0.42 modern rewrite).

    Args:
        qubits:    量子比特数
        filters:   输出通道数 (must <= qubits)
        kernel_size: 卷积核大小
        stride:    步长
        n_layers:  RandomLayers 层数
        encoding:  像素编码方式 'ry' | 'rx' | 'rz' | 'rxry' | 'rxyz'
        seed:      随机种子 (固定 RandomLayers 参数，保证 spatial-invariant)
        device:    'cpu' (默认) | 'gpu' | None (走 QCONV_DEVICE 环境变量)
    """

    def __init__(
        self,
        qubits: int = 4,
        filters: int = 4,
        kernel_size: int = 2,
        stride: int = 1,
        n_layers: int = 2,
        encoding: str = "ry",
        seed: int = 758493,
        device: str | None = None,
    ):
        if filters > qubits:
            raise ValueError(f"filters ({filters}) must <= qubits ({qubits})")
        if kernel_size ** 2 > qubits:
            raise ValueError(
                f"kernel_size**2 ({kernel_size**2}) must <= qubits ({qubits})"
            )
        if encoding not in ENCODING_FNS:
            raise ValueError(f"encoding must be one of {list(ENCODING_FNS)}")

        self.qubits = qubits
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_layers = n_layers
        self.encoding = encoding
        self.seed = seed
        self.device = select_backend(device)

        # Build device and qnode
        self.dev = qml.device(self.device, wires=self.qubits)

        # Fixed RandomLayers params (shared across all spatial positions)
        rng = np.random.default_rng(self.seed)
        self.rand_params = rng.uniform(0, 2 * np.pi, size=(self.n_layers, self.qubits))

        # Build qnode via decorator (PL 0.38 best practice)
        enc_fn = ENCODING_FNS[self.encoding]
        rand_params = self.rand_params
        n_filters = self.filters

        @qml.qnode(self.dev)
        def circuit(phi):
            enc_fn(phi, wires=range(self.qubits))
            qml.RandomLayers(rand_params, wires=range(self.qubits))
            return [qml.expval(qml.PauliZ(j)) for j in range(n_filters)]

        self.circuit = circuit

        # Pre-compute output shape
        # Will be set on first apply() call
        self._out_h = None
        self._out_w = None

    def _qconv_single(self, image: np.ndarray) -> np.ndarray:
        """Apply QConv to single channel image (H, W) -> (H_out, W_out, filters)."""
        h, w = image.shape
        ks = self.kernel_size
        st = self.stride
        h_out = (h - ks) // st + 1
        w_out = (w - ks) // st + 1

        out = np.zeros((h_out, w_out, self.filters), dtype=np.float32)

        for j in range(h_out):
            for i in range(w_out):
                patch = image[j * st : j * st + ks, i * st : i * st + ks]
                phi = patch.reshape(-1)  # length = ks*ks
                out[j, i, :] = self.circuit(phi)

        return out

    def apply(self, image: np.ndarray, verbose: bool = False) -> np.ndarray:
        """Apply QConv to (H, W, C) image -> (H_out, W_out, filters).

        Multi-channel: each channel runs through circuit, results averaged.
        """
        if image.ndim != 3:
            raise ValueError(f"expect (H, W, C), got {image.shape}")

        h, w, c = image.shape
        per_ch = [self._qconv_single(image[..., ch]) for ch in range(c)]
        # Stack: (C, H_out, W_out, filters) -> mean over C -> (H_out, W_out, filters)
        return np.mean(np.stack(per_ch, axis=0), axis=0).astype(np.float32)


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    import time
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.dataset import load_image, scan_dataset

    DS = Path("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/datasets/EuroSAT")
    p, y, c2i = scan_dataset(DS, max_per_class=10)
    print(f"Loaded {len(p)} images, classes: {len(c2i)}")

    qc = QConv2D(qubits=4, filters=4, kernel_size=2, stride=1, n_layers=2, encoding="ry")
    print(f"QConv2D: backend={qc.device} qubits={qc.qubits} filters={qc.filters} ks={qc.kernel_size} stride={qc.stride} layers={qc.n_layers} enc={qc.encoding}")

    img = load_image(p[0], target_size=64)
    print(f"Input image: {img.shape} range=[{img.min():.3f}, {img.max():.3f}]")

    t0 = time.time()
    out = qc.apply(img, verbose=True)
    t1 = time.time()
    print(f"Output: {out.shape}, range=[{out.min():.3f}, {out.max():.3f}]")
    print(f"Single image time: {t1-t0:.2f} sec")
    print(f"Estimated 500 images: {(t1-t0)*500/60:.1f} min")
    print(f"Estimated 1000 images: {(t1-t0)*1000/60:.1f} min")
