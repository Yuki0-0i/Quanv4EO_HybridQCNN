# Exp2: QConv (Lite Trainable) + ResNet-18 (5-Seed)

> **核心改动**: 后端从 3-layer CNN 替换为 ResNet-18
> ResNet-18 第一层 Conv2d 改为接 4 通道 (QConv 特征)
> 数据: 1000/2000/5000 张 4q 2l RY
> 训练: 30 epoch, batch=32, Adam(lr=1e-3)

## 结果

| 数据规模 | Exp1 (3-CNN Trainable) | **Exp2 (ResNet-18 Trainable)** | Δ vs 3-CNN |
|---|---|---|---|
| 1000 张 | 0.559 ± 0.025 | **0.508 ± 0.031** | -0.051 |
| 2000 张 | 0.605 ± 0.019 | **0.561 ± 0.017** | -0.043 |
| 5000 张 | 0.652 ± 0.016 | **0.629 ± 0.020** | -0.023 |

## 关键发现

- ResNet-18 替换 3-CNN 后端对 QConv 特征的影响
- 跟 Exp1 相比的提升反映后端 CNN 强度对量子特征利用的贡献
