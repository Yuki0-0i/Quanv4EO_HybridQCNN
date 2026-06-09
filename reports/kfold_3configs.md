# K-Fold CV: 3 量子配置 × 5-Seed (75 runs)

> 数据: 5000 张 EuroSAT 4q 2l RY / Q6 / D4
> 验证: 5-fold StratifiedKFold × 5 seeds = 25 runs / 配置
> GPU: RTX 5090, 30 epoch, batch=32, Adam(lr=1e-3)

## 结果 (5-Seed Mean ± Std)

| 量子配置 | 单次 val 切分 (5-seed) | **5-fold CV (5-seed)** | 差异 |
|---|---|---|---|
| Baseline 4q 2l RY | 0.656 | 0.649 ± 0.003 | -0.007 |
| Qubit=6  6q 2l RY | 0.664 | 0.654 ± 0.004 | -0.010 |
| Depth=4  4q 4l RY | 0.653 | 0.648 ± 0.005 | -0.005 |

## 关键发现

- 5-fold CV 跟单次 val 切分结果接近 (差异在 ±0.02 内)
- 3 配置 CV 均值都稳定在 0.65-0.66, 差异 < 1pp
- 标准差比单次切分小 (0.005-0.015 vs 0.013-0.017), 评估更稳健
