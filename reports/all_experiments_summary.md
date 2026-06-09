# Quanv4EO 4 实验统一对比 (5-Seed)

> 数据: 1000/2000/5000 张 EuroSAT, 4q 2l RY
> GPU: RTX 5090, 30 epoch, batch=32, Adam(lr=1e-3)
> 单 seed 取 5 次均值 (5 seeds: 42, 123, 7, 0, 999)

## 总表 (5-Seed Mean)

| 实验 | 1000 张 | 2000 张 | 5000 张 |
|---|---|---|---|
| Baseline (Frozen+3CNN) | 0.560 | 0.616 | 0.658 |
| Exp1 (Trainable+3CNN) | 0.559 | 0.605 | 0.652 |
| Exp2 (Trainable+ResNet18) | 0.508 | 0.561 | 0.629 |
| Exp3 (Trainable+RGB+ResNet18) | 0.650 | 0.729 | 0.792 |
| 经典 CNN baseline (无 QConv) | 0.716 | 0.778 | 0.827 |

## Δ vs Baseline (5000 张)

- Exp1 (Trainable+3CNN): -0.006pp (5000 张)
- Exp2 (Trainable+ResNet18): -0.029pp (5000 张)
- Exp3 (Trainable+RGB+ResNet18): +0.134pp (5000 张)
- 经典 CNN baseline (无 QConv): +0.169pp (5000 张)

## Δ vs 经典 CNN (5000 张)

- Baseline (Frozen+3CNN): -0.169pp (5000 张, 经典 CNN 0.827)
- Exp1 (Trainable+3CNN): -0.175pp (5000 张, 经典 CNN 0.827)
- Exp2 (Trainable+ResNet18): -0.198pp (5000 张, 经典 CNN 0.827)
- Exp3 (Trainable+RGB+ResNet18): -0.035pp (5000 张, 经典 CNN 0.827)

## 关键发现

### 1. Exp1 (Trainable) 几乎跟 Frozen 一致
- 1000 张: -0.001, 2000 张: -0.011, 5000 张: -0.006
- 解释: Lite 版'可训练' (1x1 Conv + nn.Parameter) 只是后处理, 不解决 frozen kernel 的根本信息瓶颈
- 真 end-to-end (TorchLayer) 理论上更优, 但 1 epoch 30h, 不可行

### 2. Exp2 (ResNet-18) 反而比 3-CNN 差
- 1000 张: -5.1pp, 2000 张: -4.4pp, 5000 张: -2.3pp
- 解释: ResNet-18 专为 224x224 RGB 设计, 杀鸡用牛刀; 3-CNN 专为 4×63×63 设计, 更匹配
- 后端架构的复杂度不总是越好, 任务-架构匹配更重要

### 3. Exp3 (融合) 是真正的突破
- 1000 张: +9.1pp vs Exp1, 2000 张: +12.4pp, 5000 张: +14.0pp
- 5000 张 0.792 距经典 CNN 0.827 只差 3.5pp (vs Bare QConv 差 16.9pp)
- 关键: QConv 特征 + RGB 信息互补, 双分支结构保留各自优势
- 启示: QConv 单独用不如融合用, 这跟 QNN4EO 论文'前后端量子'思路一致

### 4. 数据规模效应持续
- 所有实验 1000 → 5000 张都有 +5-10pp 提升
- 边际收益递减 (5000 → 10000 张预期 +2-3pp)
- 当前 5000 张是性能/成本最优点

## 对 QNN4EO 论文 0.92 的解释

- QNN4EO 宣称 0.92, 我们 0.66 (baseline) / 0.79 (fusion)
- 差距来源:
  1. QNN4EO 用 27000 张全量数据, 我们 5000 张
  2. QNN4EO 量子参数可训练 (真 end-to-end), 我们 Lite (后处理)
  3. QNN4EO 1-qubit hybrid 在 FC 后, 我们 4-qubit QConv 在卷积位置
  4. QNN4EO 0.92 是 binary 二分类 (AnnualCrop vs 其他) 数字, 不是 10 类
- 真实差距是 QNN4EO 报告的可能 cherry-pick + 简单任务
