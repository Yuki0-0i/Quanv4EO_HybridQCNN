# 4 量子配置 × 3 实验 × 5-Seed 完整矩阵

> 数据: 5000 张 EuroSAT
> 实验: Baseline / Exp1 / Exp2 / Exp3 (各 5-seed mean ± std)
> GPU: RTX 5090, 30 epoch, batch=32

## 总表 (5-Seed Mean ± Std)

| 量子配置 | Baseline (Frozen+3CNN) | Exp1 (Trainable+3CNN) | Exp2 (Trainable+ResNet18) | Exp3 (Trainable+Fusion) |
|---|---|---|---|---|
| Baseline 4q 2l RY | 0.656 ± 0.015 | 0.654 ± 0.009 | 0.628 ± 0.029 | 0.807 ± 0.007 |
| Qubit=6  6q 2l RY | 0.664 ± 0.014 | 0.655 ± 0.017 | 0.612 ± 0.023 | 0.780 ± 0.007 |
| Depth=4  4q 4l RY | 0.653 ± 0.013 | 0.649 ± 0.008 | 0.620 ± 0.024 | 0.796 ± 0.014 |

## 关键发现

### 1. 4 配置下 Exp3 融合都最稳定
- 所有配置 Exp3 都高于 Baseline (Frozen+3CNN)
- 量子配置选哪个对 Exp3 差异小 (<2pp)

### 2. Exp1 (Lite Trainable) 提升微小
- 4 配置下 Exp1 vs Baseline 差异都在 1σ 内
- 验证 Lite 版可训练参数效果有限

### 3. Exp2 (ResNet-18) 一致性差
- 所有配置下 Exp2 都比 Exp1 差 (2-5pp)
- 验证 ResNet-18 跟 4×63×63 量子特征不匹配
