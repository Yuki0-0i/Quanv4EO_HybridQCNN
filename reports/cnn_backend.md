# CNN 后端改进结果

> 架构: QConv(63,63,4/6) → Conv(3x3,16) → MaxPool → Conv(3x3,32) → MaxPool → Conv(3x3,64) → MaxPool → FC(128) → FC(10)
> 训练: Adam(lr=1e-3) + CrossEntropy, 30 epoch, batch=32, dropout=0.5
> 特征 reshape: (N, 15876) → (N, C, 63, 63) 喂给 Conv2D

## 实验结果

| 实验 | MLP | RF | **CNN** | Δ vs RF (pp) |
|---|---|---|---|---|
| Baseline 4q 2l RY | 0.310 | 0.490 | **0.700** | +21.0 |
| Qubit=6 | 0.280 | 0.470 | **0.675** | +20.5 |
| Depth=4 | 0.345 | 0.460 | **0.655** | +19.5 |
| Encoding RX+RY | 0.320 | 0.525 | **0.640** | +11.5 |

## 结论

- CNN 后端利用 QConv 特征图的 **空间结构** (vs flatten 后丢空间信息)
- 与 RF/MLP 对比, CNN 在 QConv 特征上能进一步提升
- 训练快 (~10s/epoch), 可作为最终分类器
