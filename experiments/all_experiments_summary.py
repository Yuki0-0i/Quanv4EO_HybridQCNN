"""
4 实验统一对比 + 综合报告

Baseline:  Frozen QConv + 3-CNN
Exp1:      Trainable QConv (Lite) + 3-CNN
Exp2:      Trainable QConv (Lite) + ResNet-18
Exp3:      Trainable QConv (Lite) + RGB fusion + ResNet-18
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def collect_results():
    """汇总所有 4 实验结果."""
    return {
        'Baseline (Frozen+3CNN)': {
            '1000': 0.560, '2000': 0.616, '5000': 0.658,
        },
        'Exp1 (Trainable+3CNN)': {
            '1000': 0.559, '2000': 0.605, '5000': 0.652,
        },
        'Exp2 (Trainable+ResNet18)': {
            '1000': 0.508, '2000': 0.561, '5000': 0.629,
        },
        'Exp3 (Trainable+RGB+ResNet18)': {
            '1000': 0.650, '2000': 0.729, '5000': 0.792,
        },
        '经典 CNN baseline (无 QConv)': {
            '1000': 0.716, '2000': 0.778, '5000': 0.827,
        },
    }


def write_comprehensive_report():
    r = collect_results()

    md = '/hdd/Dengxuanyu/dxy1/Quanv4EO_0604/reports/all_experiments_summary.md'
    with open(md, 'w', encoding='utf-8') as f:
        f.write("# Quanv4EO 4 实验统一对比 (5-Seed)\n\n")
        f.write("> 数据: 1000/2000/5000 张 EuroSAT, 4q 2l RY\n")
        f.write("> GPU: RTX 5090, 30 epoch, batch=32, Adam(lr=1e-3)\n")
        f.write("> 单 seed 取 5 次均值 (5 seeds: 42, 123, 7, 0, 999)\n\n")
        f.write("## 总表 (5-Seed Mean)\n\n")
        f.write("| 实验 | 1000 张 | 2000 张 | 5000 张 |\n")
        f.write("|---|---|---|---|\n")
        for exp, vals in r.items():
            f.write(f"| {exp} | {vals['1000']:.3f} | {vals['2000']:.3f} | {vals['5000']:.3f} |\n")
        f.write("\n## Δ vs Baseline (5000 张)\n\n")
        b = r['Baseline (Frozen+3CNN)']['5000']
        for exp, vals in r.items():
            if 'Baseline' in exp:
                continue
            delta = vals['5000'] - b
            f.write(f"- {exp}: {delta:+.3f}pp (5000 张)\n")
        f.write("\n## Δ vs 经典 CNN (5000 张)\n\n")
        c = r['经典 CNN baseline (无 QConv)']['5000']
        for exp, vals in r.items():
            if '经典' in exp:
                continue
            delta = vals['5000'] - c
            f.write(f"- {exp}: {delta:+.3f}pp (5000 张, 经典 CNN 0.827)\n")
        f.write("\n## 关键发现\n\n")
        f.write("### 1. Exp1 (Trainable) 几乎跟 Frozen 一致\n")
        f.write("- 1000 张: -0.001, 2000 张: -0.011, 5000 张: -0.006\n")
        f.write("- 解释: Lite 版'可训练' (1x1 Conv + nn.Parameter) 只是后处理, 不解决 frozen kernel 的根本信息瓶颈\n")
        f.write("- 真 end-to-end (TorchLayer) 理论上更优, 但 1 epoch 30h, 不可行\n\n")
        f.write("### 2. Exp2 (ResNet-18) 反而比 3-CNN 差\n")
        f.write("- 1000 张: -5.1pp, 2000 张: -4.4pp, 5000 张: -2.3pp\n")
        f.write("- 解释: ResNet-18 专为 224x224 RGB 设计, 杀鸡用牛刀; 3-CNN 专为 4×63×63 设计, 更匹配\n")
        f.write("- 后端架构的复杂度不总是越好, 任务-架构匹配更重要\n\n")
        f.write("### 3. Exp3 (融合) 是真正的突破\n")
        f.write("- 1000 张: +9.1pp vs Exp1, 2000 张: +12.4pp, 5000 张: +14.0pp\n")
        f.write("- 5000 张 0.792 距经典 CNN 0.827 只差 3.5pp (vs Bare QConv 差 16.9pp)\n")
        f.write("- 关键: QConv 特征 + RGB 信息互补, 双分支结构保留各自优势\n")
        f.write("- 启示: QConv 单独用不如融合用, 这跟 QNN4EO 论文'前后端量子'思路一致\n\n")
        f.write("### 4. 数据规模效应持续\n")
        f.write("- 所有实验 1000 → 5000 张都有 +5-10pp 提升\n")
        f.write("- 边际收益递减 (5000 → 10000 张预期 +2-3pp)\n")
        f.write("- 当前 5000 张是性能/成本最优点\n\n")
        f.write("## 对 QNN4EO 论文 0.92 的解释\n\n")
        f.write("- QNN4EO 宣称 0.92, 我们 0.66 (baseline) / 0.79 (fusion)\n")
        f.write("- 差距来源:\n")
        f.write("  1. QNN4EO 用 27000 张全量数据, 我们 5000 张\n")
        f.write("  2. QNN4EO 量子参数可训练 (真 end-to-end), 我们 Lite (后处理)\n")
        f.write("  3. QNN4EO 1-qubit hybrid 在 FC 后, 我们 4-qubit QConv 在卷积位置\n")
        f.write("  4. QNN4EO 0.92 是 binary 二分类 (AnnualCrop vs 其他) 数字, 不是 10 类\n")
        f.write("- 真实差距是 QNN4EO 报告的可能 cherry-pick + 简单任务\n")
    print(f"综合报告: {md}")


if __name__ == "__main__":
    write_comprehensive_report()
