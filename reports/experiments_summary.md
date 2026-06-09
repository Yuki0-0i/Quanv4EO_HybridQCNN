# Quanv4EO 4 实验对比汇总

> 数据集: EuroSAT 1000 张 (每类 100 张, 800 train / 200 val)
> 量子后端: PennyLane 0.38 + lightning.qubit (CPU C++ 后端)
> 特征维度: 取决于 qubits × filters (4q=15876, 6q=23814)
> 训练: StandardScaler + MLP(128,) / RandomForest(200)

## 实验结果

| 实验 | 配置 | 特征维度 | MLP Val Acc | RF Val Acc |
|---|---|---|---|---|
| Baseline 4q 2l RY | 4q 2l RY | 15876 | 0.310 | 0.490 |
| Qubit=6 | 6q 2l RY | 23814 | 0.280 | 0.470 |
| Depth=4 | 4q 4l RY | 15876 | 0.345 | 0.460 |
| Encoding RX+RY | 4q 2l RX+RY | 15876 | 0.320 | 0.525 |
| (Baseline) Majority | - | - | - | 0.100 |

## 关键发现

1. **Encoding RX+RY > RY**: RF acc 0.525 vs 0.490 (+7%), 最有意义的创新点
2. **Qubit=6 略降**: 4q=0.490 → 6q=0.470 (-4%), 态空间扩大在小数据下稀释表达力
3. **Depth=4 持平**: MLP 略升 (+3%) 但 RF 略降 (-3%), 4-layer 边际收益小
4. **所有实验都远超多数类 baseline (0.100)**, 量子特征有真实信号

## 改进点 & 未来工作

- **加 PCA 降维**: 15876 维 / 800 样本 = 20:1 比例, 严重欠拟合
- **加经典 CNN 后端**: 在 QConv 特征后再接 1-2 层 Conv + FC
- **扩数据量到 5000-10000 张**: 当前 0.49 还没到 QConv 应有的水平
- **多次实验取均值**: 当前单次运行, 应该 3-5 次取均值和方差

## 速度记录

| 实验 | 单图时间 (8 worker) | 1000 张总时间 |
|---|---|---|
| Baseline 4q 2l RY | 2.15s | 36 min |
| Qubit=6 6q 2l RY | 2.59s | 43 min |
| Depth=4 4q 4l RY | 2.85s | 48 min |
| Encoding RX+RY    | 2.15s | 36 min |
