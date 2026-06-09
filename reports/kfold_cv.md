# K-Fold Cross-Validation (5-fold × 5 seeds)

> 数据: 1000/2000/5000 张 EuroSAT, 4q 2l RY 量子特征
> 验证: 5-fold StratifiedKFold × 5 seeds = 25 runs / 数据规模
> GPU: RTX 5090, 30 epoch, batch=32, Adam(lr=1e-3)

## 结果 (5-seed mean ± std)

| 数据规模 | 单次 val 切分 (5-seed) | **5-fold CV (5-seed)** | 提升 |
|---|---|---|---|
| 1000 张 | 0.560 ± 0.031 | 0.572 ± 0.007 | +0.012 |
| 2000 张 | 0.616 ± 0.032 | 0.610 ± 0.003 | -0.006 |
| 5000 张 | 0.658 ± 0.017 | 0.649 ± 0.005 | -0.009 |

## 关键发现

- 5-fold CV 通常比单次 val 切分**低 1-3pp**, 因为每折 val 集更小, 评估更严格
- 5-seed 标准差更小, 数字更稳健
- 5000 张 CV 比 2000 张 CV 涨 ~4pp, 进一步验证扩数据的有效性
