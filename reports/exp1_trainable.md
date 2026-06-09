# Exp1: Trainable QConv + 3-CNN (5-Seed)

> **核心改动**: 量子层从 frozen 改为可训练 (lite 版, 1x1 Conv + nn.Parameter 量子参数)
> **数据**: 1000/2000/5000 张 EuroSAT 4q 2l RY, 加 Q6 5000 张
> **CNN**: 3 conv + 2 FC, 30 epoch, batch=32, Adam(lr=1e-3)
> **GPU**: RTX 5090

## 实验结果 (5-seed mean ± std)

| 配置 | 数据规模 | Exp1 (Trainable) | Baseline (Frozen) | Δ |
|---|---|---|---|---|
| 4q 2l RY | 1000 张 | 0.559 ± 0.025 | 0.560 ± 0.031 | -0.001 |
| 4q 2l RY | 2000 张 | 0.605 ± 0.019 | 0.616 ± 0.032 | -0.011 |
| 4q 2l RY | 5000 张 | 0.652 ± 0.016 | 0.658 ± 0.017 | -0.006 |
| 6q 2l RY | 5000 张 | 0.651 ± 0.008 | 0.662 ± 0.008 | -0.011 |

## 实现说明

**Lite 版实现 (本次实验)**:
- 预提取 QConv frozen 特征 (1次, 4.5h)
- 加 nn.Parameter 作为'量子参数', 通过 1x1 Conv 调整 QConv 特征
- CNN 端到端训练, 量子参数 + CNN 权重同时更新
- 实际意义: 模拟'可训练量子'对特征的后处理效应

**真 end-to-end 版 (理论上更优, 实际 1 epoch 30h, 不可行)**:
- 用 PennyLane TorchLayer 把量子电路嵌入 PyTorch
- 每次 forward 跑 QConv (每个 patch 一次, 一张图 11907 次)
- 量子参数 autograd 直接更新
