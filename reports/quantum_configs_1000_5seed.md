# 4 量子配置 × 5 seeds (Windows 1000 张数据)

> 数据: Windows 原版 1000 张 EuroSAT (每类 100)
> 后端: CNN (3 conv + 2 FC, 30 epoch, Adam lr=1e-3)
> GPU: RTX 5090, PyTorch 2.11
> 5 seeds: 42, 123, 7, 0, 999

## 结果

| 量子配置 | channels | 5-seed mean ± std |
|---|---|---|
| Baseline 4q 2l RY | 4 | **0.559 ± 0.015** |
| Qubit=6  6q 2l RY | 6 | **0.517 ± 0.030** |
| Depth=4  4q 4l RY | 4 | **0.558 ± 0.017** |
| Encoding RX+RY | 4 | **0.557 ± 0.022** |

## Δ vs Baseline

- Qubit=6  6q 2l RY: -4.2pp
- Depth=4  4q 4l RY: -0.1pp
- Encoding RX+RY: -0.2pp
