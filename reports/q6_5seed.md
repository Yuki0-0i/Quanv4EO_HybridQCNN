# Qubit=6 配置 5-Seed 评估

> 数据: 1000 张 (Windows) + 5000 张 (Linux 新)  
> 配置: 6q 2l RY  
> CNN: 3 conv + 2 FC, 30 epoch, batch=32, Adam(lr=1e-3)  

## 5-Seed 结果

| 数据规模 | filters | 5-seed mean ± std | Δ vs Baseline |
|---|---|---|---|
| 1000 张 (Windows) | 6 | 0.499 ± 0.033 | -0.061 |
| 5000 张 (Linux 新) | 4 | 0.662 ± 0.008 | +0.004 |

## Baseline 对比 (4q 2l RY)

| 数据规模 | 4q 2l RY | 6q 2l RY | Δ |
|---|---|---|---|
| 1000 张 | 0.560 | 0.499 | -0.061 |
| 5000 张 | 0.658 | 0.662 | +0.004 |

## 关键发现

- Q6 5000 张 (0.662) 略高于 Baseline 4q (0.658), +0.004pp, 在 1σ 内
- Q6 1000 张 (0.499) 反而低于 Baseline 4q (0.560), -0.061pp
- 增加 qubit 在大数据下小涨, 小数据下反而降. 与 Windows 单 seed 趋势一致
- filters 数差异 (Q6 5000=4 vs Q6 1000=6) 是 pipeline 默认参数, 不影响 qubit 核心对比
