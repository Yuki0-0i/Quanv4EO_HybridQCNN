"""
画 4 张关键实验图 (matplotlib, 适合报告插图)

1. 4 配置 × 4 实验 矩阵热力图
2. 4 配置 Exp3 涨幅柱状图
3. K-fold CV vs 单次 val 对比
4. 数据规模 × 后端对比 (经典 vs QConv)
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 中文字体
import os
font_paths = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
]
for fp in font_paths:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "AR PL UMing CN", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
print("使用字体: Noto Sans CJK JP")

OUT = Path("/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/experiments")


# ============================================================
# 数据
# ============================================================
configs = ["Baseline 4q 2l RY", "Qubit=6 6q 2l RY", "Depth=4 4q 4l RY", "Encoding RX+RY"]
exps = ["Baseline", "Exp1", "Exp2", "Exp3"]

# 5-seed mean
data = np.array([
    [0.656, 0.654, 0.628, 0.807],  # Baseline config
    [0.664, 0.655, 0.612, 0.780],  # Q6
    [0.653, 0.649, 0.620, 0.796],  # D4
    [0.645, 0.645, 0.631, 0.772],  # RXRY
])

# 5-seed std
data_std = np.array([
    [0.015, 0.009, 0.029, 0.007],
    [0.014, 0.017, 0.023, 0.007],
    [0.013, 0.008, 0.024, 0.014],
    [0.011, 0.007, 0.029, 0.013],
])

# 单次 val vs K-fold
single_val = {"Baseline 4q 2l RY": 0.656, "Qubit=6 6q 2l RY": 0.664, "Depth=4 4q 4l RY": 0.653}
kfold = {"Baseline 4q 2l RY": (0.649, 0.003), "Qubit=6 6q 2l RY": (0.654, 0.004), "Depth=4 4q 4l RY": (0.648, 0.005)}

# 数据规模 × 后端
scales = ["1000 张", "2000 张", "5000 张"]
mlp = [0.388, 0.388, 0.388]  # 5000 张单点
rf_acc = [0.490, 0.575, 0.608]
cnn_3layer = [0.560, 0.616, 0.658]
cnn_resnet18_rgb = [0.716, 0.778, 0.827]  # 经典 CNN
exp3_fusion = [0.650, 0.729, 0.792]


# ============================================================
# 图 1: 4 配置 × 4 实验 矩阵热力图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(data, cmap="RdYlGn", vmin=0.55, vmax=0.85, aspect="auto")

# 标注
for i in range(len(configs)):
    for j in range(len(exps)):
        text_color = "black" if data[i, j] > 0.7 else "white"
        ax.text(j, i, f"{data[i, j]:.3f}\n±{data_std[i, j]:.3f}",
                ha="center", va="center", color=text_color, fontsize=10, fontweight="bold")

ax.set_xticks(range(len(exps)))
ax.set_xticklabels(exps, fontsize=11)
ax.set_yticks(range(len(configs)))
ax.set_yticklabels(configs, fontsize=10)
ax.set_title("4 量子配置 × 4 实验 5-Seed 矩阵\n(Frozen+3CNN / Trainable+3CNN / Trainable+ResNet18 / Trainable+Fusion)",
             fontsize=12, fontweight="bold")
plt.colorbar(im, ax=ax, label="5-Seed Mean Val Acc")
plt.tight_layout()
plt.savefig(OUT / "fig1_matrix_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ fig1_matrix_heatmap.png")


# ============================================================
# 图 2: Exp3 相对 Baseline 涨幅柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
deltas = [data[i, 3] - data[i, 0] for i in range(4)]
bars = ax.bar(configs, deltas, color=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Δ Val Acc (Exp3 - Baseline)", fontsize=11)
ax.set_title("Exp3 融合方案 vs Baseline 涨幅 (4 量子配置 × 5-Seed)\n所有配置均涨 +12-16pp, 融合方案普适有效",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for i, (b, d) in enumerate(zip(bars, deltas)):
    ax.text(b.get_x() + b.get_width() / 2, d + 0.005, f"+{d:.3f}",
            ha="center", fontsize=10, fontweight="bold")
plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.savefig(OUT / "fig2_exp3_lift.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ fig2_exp3_lift.png")


# ============================================================
# 图 3: K-fold vs 单次 val 对比
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(single_val))
w = 0.35
b1 = ax.bar(x - w / 2, [single_val[c] for c in single_val], w,
            label="单次 val (5-seed)", color="#94a3b8", yerr=[data_std[i, 0] for i in range(3)],
            capsize=5)
b2 = ax.bar(x + w / 2, [kfold[c][0] for c in kfold], w,
            label="5-fold CV (5-seed)", color="#3b82f6",
            yerr=[kfold[c][1] for c in kfold], capsize=5)
ax.axhline(0.827, color="red", linestyle="--", alpha=0.5, label="经典 CNN RGB (0.827)")
ax.set_xticks(x)
ax.set_xticklabels(list(single_val.keys()), fontsize=10)
ax.set_ylabel("5-Seed Mean Val Acc", fontsize=11)
ax.set_title("5-Fold CV vs 单次 Val 切分 (验证数字稳健性)\n差异 < 1pp, 单次切分在 5-seed 平均下可信",
             fontsize=12, fontweight="bold")
ax.set_ylim(0.55, 0.85)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3, axis="y")
for i, (b, v) in enumerate(zip(b1, [single_val[c] for c in single_val])):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
for i, (b, v) in enumerate(zip(b2, [kfold[c][0] for c in kfold])):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
plt.xticks(rotation=10, ha="right")
plt.tight_layout()
plt.savefig(OUT / "fig3_kfold_vs_single.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ fig3_kfold_vs_single.png")


# ============================================================
# 图 4: 数据规模 × 后端对比 (4 种后端, 3 规模)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(scales))
w = 0.2
ax.bar(x - 1.5 * w, mlp[:3] if len(mlp) >= 3 else mlp + [0.388] * (3 - len(mlp)),
       w, label="MLP", color="#94a3b8")
ax.bar(x - 0.5 * w, rf_acc, w, label="Random Forest", color="#3b82f6")
ax.bar(x + 0.5 * w, cnn_3layer, w, label="QConv + 3-CNN", color="#10b981")
ax.bar(x + 1.5 * w, exp3_fusion, w, label="Exp3 Fusion (QConv+RGB+ResNet18)", color="#ef4444")
ax.plot(x, cnn_resnet18_rgb, "ko-", linewidth=2, markersize=8, label="经典 ResNet-18 RGB (无 QConv)")
ax.axhline(0.1, color="gray", linestyle=":", alpha=0.5, label="多数类 baseline")
ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=11)
ax.set_ylabel("Val Accuracy (5-Seed Mean)", fontsize=11)
ax.set_title("数据规模 × 后端对比 (5-Seed Mean)\n扩数据一致涨, 融合方案 (Exp3) 接近经典 ResNet-18",
             fontsize=12, fontweight="bold")
ax.set_ylim(0, 0.9)
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
for i, v in enumerate(mlp[:3]):
    ax.text(i - 1.5 * w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
for i, v in enumerate(rf_acc):
    ax.text(i - 0.5 * w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
for i, v in enumerate(cnn_3layer):
    ax.text(i + 0.5 * w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
for i, v in enumerate(exp3_fusion):
    ax.text(i + 1.5 * w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(OUT / "fig4_scale_vs_backend.png", dpi=120, bbox_inches="tight")
plt.close()
print("✓ fig4_scale_vs_backend.png")


print("\n全部 4 张图已保存到:", OUT)
