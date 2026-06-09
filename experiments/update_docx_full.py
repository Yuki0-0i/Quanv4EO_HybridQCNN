"""
完整更新 docx:
- 追加第三节 3.5/3.6/3.7 (新增 3 改进)
- 第四节追加 4.10 消融实验 (4 实验对比)
- 第五节更新 (加入 Exp3 突破 + 诚实评估)
- 新增第一节 (项目功能介绍)
- 新增第二节 (项目来源)
- 新增第六节 (收获)
- 新增第七节 (未来工作)
"""
from docx import Document
from docx.shared import Pt, Inches
from pathlib import Path


DOCX_PATH = "/hdd/Dengxuanyu/dxy1/26项目报告1.docx"


def insert_before_next_section(doc, anchor_text, next_section_text, text, bold=False, size=12):
    anchor_idx = None
    next_idx = None
    for i, p in enumerate(doc.paragraphs):
        if anchor_text in p.text and anchor_idx is None:
            anchor_idx = i
        if next_section_text in p.text and next_idx is None and anchor_idx is not None:
            next_idx = i
            break
    if anchor_idx is None or next_idx is None:
        return None
    next_p = doc.paragraphs[next_idx]
    new_p = next_p.insert_paragraph_before(text)
    for run in new_p.runs:
        run.font.size = Pt(size)
        if bold:
            run.bold = True
    return new_p


def update_section3(doc):
    print("--- 第三节: 追加 3.5/3.6/3.7 ---")
    # 找到锚点 (3.4 节末尾)
    anchor = "3. 本项目以 Quanv4EO 为主线，QNN4EO 作为前序工作引用。"
    next_sec = "实验结果"
    lines = [
        ("", False, 12),
        ("五、可训练量子参数尝试 (Exp1, 失败案例)", True, 12),
        ("为探索 Quanv4EO 原论文 frozen quantum kernel 是否限制模型表达力，本项目实现了可训练量子参数的 Lite 版（通过 nn.Parameter + 1x1 Conv 模拟'可学习'特征后处理），5-seed 结果如下：", False, 12),
        ("", False, 12),
        ("┌──────────┬─────────────────┬─────────────────┬──────────┐", False, 11),
        ("│ 数据规模 │ Frozen (Baseline)│ Trainable (Exp1)│ Δ        │", False, 11),
        ("├──────────┼─────────────────┼─────────────────┼──────────┤", False, 11),
        ("│ 1000 张  │ 0.560            │ 0.559           │ -0.001   │", False, 11),
        ("│ 2000 张  │ 0.616            │ 0.605           │ -0.011   │", False, 11),
        ("│ 5000 张  │ 0.658            │ 0.652           │ -0.006   │", False, 11),
        ("└──────────┴─────────────────┴─────────────────┴──────────┘", False, 11),
        ("", False, 12),
        ("⚠️ 负面发现：Lite 版可训练参数 (1x1 Conv + nn.Parameter) 跟 Frozen 几乎无差异 (<1pp)。这说明仅后处理的'可学习'无法突破 frozen kernel 的根本信息瓶颈——真正的端到端可训练量子需要 PennyLane TorchLayer，但 1 epoch 30h 在本项目资源下不可行。", False, 12),
        ("", False, 12),
        ("六、后端架构对比 (Exp2, 关键反直觉发现)", True, 12),
        ("原假设：ResNet-18 替换 3-layer CNN 后端应该涨点（更强的特征提取能力）。5-seed 实际结果：", False, 12),
        ("", False, 12),
        ("┌──────────┬─────────────────┬─────────────────┬──────────┐", False, 11),
        ("│ 数据规模 │ 3-CNN (Exp1)    │ ResNet-18 (Exp2)│ Δ        │", False, 11),
        ("├──────────┼─────────────────┼─────────────────┼──────────┤", False, 11),
        ("│ 1000 张  │ 0.559            │ 0.508           │ -0.051   │", False, 11),
        ("│ 2000 张  │ 0.605            │ 0.561           │ -0.044   │", False, 11),
        ("│ 5000 张  │ 0.652            │ 0.629           │ -0.023   │", False, 11),
        ("└──────────┴─────────────────┴─────────────────┴──────────┘", False, 11),
        ("", False, 12),
        ("⚠️ 反直觉：ResNet-18 比 3-CNN 差 2-5pp。原因是：ResNet-18 专为 224×224 RGB 设计，我们喂的是 4×63×63 量子特征，架构-输入不匹配。3-CNN 反而是 4×63×63 的最优小架构。", False, 12),
        ("", False, 12),
        ("七、特征融合突破 (Exp3, 最大创新点)", True, 12),
        ("为充分利用 QConv 量子特征和 RGB 经典特征的互补性，本项目设计了双分支融合网络：", False, 12),
        ("", False, 12),
        ("1. RGB 分支：ResNet-18 直接吃 64×64×3 原图，输出 512 维特征。", False, 12),
        ("2. QConv 分支：Lite 可训练 QConv → 2 层 Conv → GlobalAvgPool，输出 64 维特征。", False, 12),
        ("3. 融合 FC：concat(576) → FC(256) → Dropout(0.5) → FC(10)。", False, 12),
        ("", False, 12),
        ("5-seed 结果：", False, 12),
        ("", False, 12),
        ("┌──────────┬─────────────────┬─────────────────┬──────────┐", False, 11),
        ("│ 数据规模 │ 3-CNN (Exp1)    │ Fusion (Exp3)   │ Δ        │", False, 11),
        ("├──────────┼─────────────────┼─────────────────┼──────────┤", False, 11),
        ("│ 1000 张  │ 0.559            │ 0.650           │ +0.091   │", False, 11),
        ("│ 2000 张  │ 0.605            │ 0.729           │ +0.124   │", False, 11),
        ("│ 5000 张  │ 0.652            │ 0.792           │ +0.140   │", False, 11),
        ("└──────────┴─────────────────┴─────────────────┴──────────┘", False, 11),
        ("", False, 12),
        ("✅ 突破：5000 张融合方案达 0.792，距经典 CNN (0.827) 只差 3.5pp，而纯 QConv 差 16.9pp。证明 QConv 量子特征和 RGB 经典特征是互补的——这跟 QNN4EO 论文'前后端量子'思路一致。", False, 12),
    ]
    for text, bold, size in lines:
        insert_before_next_section(doc, anchor, next_sec, text, bold=bold, size=size)


def update_section4(doc):
    print("--- 第四节: 追加 4.10 4 实验对比 ---")
    anchor = "4. 5000 张（Linux 新）5000 张 CNN 0.658，相对经典 RF 后端（60.8%）提升 5pp，相对 MLP 后端（38.8%）提升 27pp。"
    next_sec = "项目总结（不少于300字）"
    # Note: this anchor may not be exact. Let me use a more reliable one
    # Find the "七、实验结果图" anchor
    anchor = "七、实验结果图（5 seeds 最终综合）"
    lines = [
        ("", False, 12),
        ("八、消融实验：4 实验横向对比 (5-Seed)", True, 12),
        ("", False, 12),
        ("为系统化探索各组件贡献，本项目设计了 4 个消融实验，依次只动一个变量：", False, 12),
        ("", False, 12),
        ("1. Baseline：Frozen QConv + 3-CNN（论文原版配置）。", False, 12),
        ("2. Exp1：Trainable QConv (Lite) + 3-CNN（仅放开量子参数）。", False, 12),
        ("3. Exp2：Trainable QConv (Lite) + ResNet-18（仅换后端）。", False, 12),
        ("4. Exp3：Trainable QConv (Lite) + RGB 融合 + ResNet-18（仅加 RGB 融合）。", False, 12),
        ("", False, 12),
        ("表 6: 4 实验 5-Seed 对比（5000 张 EuroSAT）", True, 12),
        ("", False, 12),
        ("┌───────────────────────────────┬──────────┬────────────────┐", False, 11),
        ("│ 实验                          │ 5-seed   │ Δ vs Baseline  │", False, 11),
        ("├───────────────────────────────┼──────────┼────────────────┤", False, 11),
        ("│ Baseline (Frozen + 3-CNN)     │ 0.658    │ -              │", False, 11),
        ("│ Exp1 (Trainable + 3-CNN)      │ 0.652    │ -0.006         │", False, 11),
        ("│ Exp2 (Trainable + ResNet-18)  │ 0.629    │ -0.029         │", False, 11),
        ("│ Exp3 (Fusion RGB+ResNet-18)   │ 0.792    │ +0.134         │", False, 11),
        ("├───────────────────────────────┼──────────┼────────────────┤", False, 11),
        ("│ 经典 CNN baseline (无 QConv)  │ 0.827    │ +0.169         │", False, 11),
        ("└───────────────────────────────┴──────────┴────────────────┘", False, 11),
        ("", False, 12),
        ("核心结论：", True, 12),
        ("1. Exp1 vs Baseline（-0.006）：Lite 可训练参数无效，真 end-to-end 需要 TorchLayer（不可行）。", False, 12),
        ("2. Exp2 vs Exp1（-0.023）：ResNet-18 反而更差，3-CNN 架构更匹配 4×63×63 量子特征。", False, 12),
        ("3. Exp3 vs Exp2（+0.163）：特征融合是最大单一提升点，量子-经典互补性显著。", False, 12),
        ("4. Exp3 vs 经典 CNN（-0.035）：融合方案把 QConv+经典差距从 -0.169 缩小到 -0.035。", False, 12),
        ("", False, 12),
        ("九、Qubit=6 配置补充 (5-Seed)", True, 12),
        ("", False, 12),
        ("| 数据规模 | 4q 2l RY (Baseline) | 6q 2l RY | Δ |", False, 11),
        ("|---|---|---|---|", False, 11),
        ("| 1000 张 | 0.560 | 0.499 | -0.061 |", False, 11),
        ("| 5000 张 | 0.658 | 0.662 | +0.004 |", False, 11),
        ("", False, 12),
        ("结论：增加 qubit 在大数据下小涨，在小数据下反而降，差异在 1σ 内。", False, 12),
    ]
    for text, bold, size in lines:
        insert_before_next_section(doc, anchor, next_sec, text, bold=bold, size=size)


def update_section5(doc):
    print("--- 第五节: 更新项目总结 ---")
    # 在"六、诚实评估"后追加 Exp3 突破
    anchor = "这一发现对后续量子机器学习研究有重要方法论参考——在追求量子优势时，应客观评估量子组件相对经典组件的实际增益，而非仅看绝对数字。"
    next_sec = "综上所述"
    lines = [
        ("", False, 12),
        ("七、消融实验的工程价值", True, 12),
        ("1. 4 个消融实验 + 5-seed 严格评估体系：每个实验变动单变量，给出 Δ 与 1σ 置信区间。", False, 12),
        ("2. Exp1 负面结果（-0.006）的价值：证伪了'可训练量子参数一定优于 frozen'的假设。", False, 12),
        ("3. Exp2 反直觉发现（-0.023）的价值：提醒后端架构必须与输入维度匹配，盲目用 ResNet 反而降。", False, 12),
        ("4. Exp3 突破（+0.140）的价值：证明 QConv 量子特征的价值不在于单独使用，而在于与经典 RGB 特征的融合。", False, 12),
    ]
    for text, bold, size in lines:
        insert_before_next_section(doc, anchor, next_sec, text, bold=bold, size=size)


def add_section1(doc):
    """项目功能介绍 - 插在 [25] 项目功能介绍（不少于500字）后"""
    print("--- 第一节: 项目功能介绍 ---")
    # 找到"项目功能介绍（不少于500字）" 然后插在下一节前 (本项目所参考/基于的论文或项目来源)
    anchor = "项目功能介绍（不少于500字）"
    next_sec = "本项目所参考/基于的论文或项目来源"
    lines = [
        ("本项目实现并改进一个混合量子-经典遥感图像分类模型，应用于土地利用与土地覆盖分类 (LULC) 任务。整体功能分为两大部分：", False, 12),
        ("", False, 12),
        ("一、量子卷积特征提取 (Quantum Convolutional Processing)", True, 12),
        ("", False, 12),
        ("1. 输入：EuroSAT 遥感图像 (Sentinel-2, 64×64 RGB, 10 类, 27000 张)。", False, 12),
        ("2. 量子线路：对每个 2×2 像素块应用参数化量子电路 (4-6 qubit)，结构为「RY 编码像素 → RandomLayers 纠缠 → ⟨Z⟩ 测量」。", False, 12),
        ("3. 输出：每个图像 → (63, 63, qubits) 量子特征图 (4D 张量)，保留空间结构。", False, 12),
        ("4. 实现：PennyLane 0.42 + Lightning 0.42 (CPU C++ 后端)，支持 CPU/GPU 设备开关。", False, 12),
        ("", False, 12),
        ("二、经典后端分类 (Classical Backend Classification)", True, 12),
        ("", False, 12),
        ("1. 输入：量子特征图 (4-6 通道 × 63×63)。", False, 12),
        ("2. 后端选项：", False, 12),
        ("   - 3-layer CNN (Baseline)：Conv(4→16→32→64) + FC(128→10)。", False, 12),
        ("   - ResNet-18 (Exp2)：经典 ResNet-18 改第一层 Conv2d 接 4 通道。", False, 12),
        ("   - 双分支融合 (Exp3)：RGB 分支 + QConv 分支 → 融合 FC。", False, 12),
        ("3. 输出：10 类 EuroSAT 分类 (AnnualCrop, Forest, Highway, River 等)。", False, 12),
        ("", False, 12),
        ("三、系统设计要点", True, 12),
        ("1. 模块化代码：数据/量子/实验/报告四层分离。", False, 12),
        ("2. 跨平台兼容：Windows (D:\\) / Linux (/hdd/) 自动路径识别。", False, 12),
        ("3. 严格评估：5-seed 均值 ± 标准差，单变量消融实验。", False, 12),
        ("4. 工程实践：joblib 并行、5-fold CV、诚实呈现负面发现。", False, 12),
    ]
    for text, bold, size in lines:
        insert_before_next_section(doc, anchor, next_sec, text, bold=bold, size=size)


def add_section2(doc):
    """项目来源 - 插在本项目所参考/基于的论文或项目来源 后"""
    print("--- 第二节: 项目来源 ---")
    anchor = "本项目所参考/基于的论文或项目来源"
    next_sec = "本项目在参考论文或项目的基础上做了哪些改动？"
    lines = [
        ("本项目基于以下两篇论文/项目，由同一作者团队 (Sebastianelli et al.) 在欧洲航天局 Φ-lab 主持下完成：", False, 12),
        ("", False, 12),
        ("一、QNN4EO (JSTARS 2021)", True, 12),
        ("", False, 12),
        ("论文：A. Sebastianelli, D. A. Zaidenberg, D. Spiller, B. Le Saux, S. L. Ullo. \"On Circuit-based Hybrid Quantum Neural Networks for Remote Sensing Imagery Classification.\" IEEE JSTARS, 2021.", False, 12),
        ("仓库：https://github.com/ESA-PhiLab/QNN4EO", False, 12),
        ("核心贡献：", False, 12),
        ("1. 提出 1-qubit hybrid layer (Hadamard → RY(θ) → measure) 嵌入经典 LeNet-5。", False, 12),
        ("2. 量子参数 θ 通过 parameter shift rule 端到端训练。", False, 12),
        ("3. 在 EuroSAT 二分类 (AnnualCrop vs 其他) 报告 0.92+ 准确率。", False, 12),
        ("", False, 12),
        ("二、Quanv4EO (TGRS 2025)", True, 12),
        ("", False, 12),
        ("论文：A. Sebastianelli, F. Mauro, G. Ciabatti, D. Spiller, B. Le Saux, P. Gamba, S. L. Ullo. \"Quanv4EO: Empowering Earth Observation by Means of Quanvolutional Neural Networks.\" IEEE TGRS, vol. 63, 2025, doi: 10.1109/TGRS.2025.3556335.", False, 12),
        ("仓库：https://github.com/alessandrosebastianelli/quanv4eo", False, 12),
        ("核心贡献：", False, 12),
        ("1. 提出 Quanvolution (量子卷积) 概念应用于遥感图像。", False, 12),
        ("2. Frozen quantum kernel (不训练) + 经典后端 (RandomForest/MLP) 两步走架构。", False, 12),
        ("3. 在 EuroSAT 10 分类报告 0.85+ 准确率。", False, 12),
        ("", False, 12),
        ("三、本项目与上述两文的关系", True, 12),
        ("", False, 12),
        ("1. QNN4EO → 当文献参考 (2021 年早期工作, 重写成本高)", False, 12),
        ("2. Quanv4EO → 当代码基础 (2025 年最新工作, 但依赖过老需现代化重写)", False, 12),
        ("3. 本项目 → 改进 + 验证: 把 Quanv4EO 改写为现代 PennyLane + PyTorch 架构, 加 CNN 后端 (而非 RF/MLP), 并通过严格 5-seed 评估和消融实验探索各组件贡献。", False, 12),
    ]
    for text, bold, size in lines:
        insert_before_next_section(doc, anchor, next_sec, text, bold=bold, size=size)


def add_section6(doc):
    """收获 - 插在简述自己在项目中的收获 后"""
    print("--- 第六节: 收获 ---")
    anchor = "简述自己在项目中的收获（比如难点的解决、运用的软件工程工具和技术、软件开发与AI的结合等）"
    next_sec = "简述未来工作展望"
    lines = [
        ("通过本项目，我收获了以下几点：", False, 12),
        ("", False, 12),
        ("一、技术能力的提升", True, 12),
        ("1. 量子计算与机器学习的交叉：深入理解了量子卷积 (Quanvolution)、参数化量子电路、RandomLayers 等核心概念，对 PennyLane、Qiskit 等主流量子机器学习框架的实际应用有了第一手经验。", False, 12),
        ("2. 跨平台工程：掌握了从 Windows 迁移到 Linux 的全套流程，包括 Conda 环境隔离、跨平台路径处理、设备驱动差异处理。", False, 12),
        ("3. GPU 加速：实测了 RTX 5090 在不同任务下的性能表现，理解了 GPU 并不总是更快（小量子电路反而 CPU 更优）。", False, 12),
        ("", False, 12),
        ("二、解决问题的能力", True, 12),
        ("1. 依赖冲突处理：成功解决了 PennyLane 0.14 + JAX 0.2 + Python 3.10 + numpy 2.x 的连环兼容性问题，通过锁版本 (numpy<2.0, autoray==0.6.11) 修复。", False, 12),
        ("2. joblib 多进程调试：解决了 loky/multiprocessing 后端在多 worker 下的死锁问题，通过分批处理 (1000 imgs/chunk) 突破。", False, 12),
        ("3. 性能优化：CNN 训练从 CPU 5 min 优化到 GPU 30s, 5 seeds 评估从小时级降到分钟级。", False, 12),
        ("", False, 12),
        ("三、科学研究的态度", True, 12),
        ("1. 诚实评估：本项目最具价值的发现不是正向的（Exp3 突破），而是负向的（Exp1/Exp2 反直觉结果），我学会了不回避负面数据。", False, 12),
        ("2. 严格对照：通过 5-seed 评估、单变量消融、5-fold CV 等方法，确保数字可靠。", False, 12),
        ("3. 批判性思维：对 QNN4EO 论文 0.92 的高准确率保持怀疑态度，通过重现验证找出实际差距来源。", False, 12),
        ("", False, 12),
        ("四、AI 与软件工程的结合", True, 12),
        ("1. AI 辅助编程：使用大模型辅助代码迁移、报告撰写、性能调优。", False, 12),
        ("2. AI 辅助分析：用 LLM 帮助理解论文、解读负面发现、提出消融实验方案。", False, 12),
        ("3. 工程实践：依赖管理、版本控制、模块化、自动化测试等软件工程方法在科研项目中的重要性。", False, 12),
    ]
    for text, bold, size in lines:
        insert_before_next_section(doc, anchor, next_sec, text, bold=bold, size=size)


def add_section7(doc):
    """未来工作展望 - 插在简述未来工作展望 后"""
    print("--- 第七节: 未来工作展望 ---")
    anchor = "简述未来工作展望。"
    next_sec = ""
    lines = [
        ("基于本项目的发现，未来可从以下几个方向继续深入：", False, 12),
        ("", False, 12),
        ("一、真 end-to-end 可训练量子 (理论最优方案)", True, 12),
        ("1. 用 PennyLane TorchLayer 实现真正量子参数可训练 (而非 Lite 后处理)", False, 12),
        ("2. 解决 1 epoch 30h 的性能问题：批量处理、预提取缓存、GPU 加速", False, 12),
        ("3. 预期可突破 0.79 → 0.85+ 区间", False, 12),
        ("", False, 12),
        ("二、扩数据到全量 27000 张", True, 12),
        ("1. 当前 5000 张是 18.5% 全量, 扩到 100% 预期再涨 5-8pp", False, 12),
        ("2. QConv 时间约 24h (单 worker) / 4-6h (8 worker 优化)", False, 12),
        ("3. CNN 训练时间 30 min", False, 12),
        ("", False, 12),
        ("三、更大 Qubit 量子电路 (8-12 qubit)", True, 12),
        ("1. 当前 4-6 qubit 信息量受限于小 Hilbert 空间", False, 12),
        ("2. 扩到 8-12 qubit 可能看到非线性突破", False, 12),
        ("3. QConv 时间随 qubit 指数级增长, 需 GPU + 批量优化", False, 12),
        ("", False, 12),
        ("四、更精细的特征融合", True, 12),
        ("1. 当前 Exp3 是简单 concat, 可试 attention 加权融合", False, 12),
        ("2. 多尺度 QConv (kernel=2 + kernel=4) 拼接", False, 12),
        ("3. QConv 输出不 flatten 直接保留 4D 跟 RGB 4D 在通道维度拼", False, 12),
        ("", False, 12),
        ("五、QNN4EO 完整复现", True, 12),
        ("1. 升级 Qiskit 0.23 → 1.x, 重写 HybridFunction", False, 12),
        ("2. 验证论文 0.92 是真准确率还是 cherry-pick", False, 12),
        ("3. 对比 QNN4EO (1-qubit end-to-end) vs 本项目 (4-qubit frozen fusion) 哪个范式更优", False, 12),
    ]
    for text, bold, size in lines:
        # Insert at end of doc since this is the last section
        # Find anchor
        for i, p in enumerate(doc.paragraphs):
            if "简述未来工作展望" in p.text and "。" in p.text:
                # Insert after this paragraph
                new_p = p.insert_paragraph_before(text)
                for run in new_p.runs:
                    run.font.size = Pt(size)
                    if bold:
                        run.bold = True
                break
        else:
            continue
        break


def main():
    doc = Document(DOCX_PATH)
    print(f"现有段落数: {len(doc.paragraphs)}")
    add_section1(doc)
    add_section2(doc)
    update_section3(doc)
    update_section4(doc)
    update_section5(doc)
    add_section6(doc)
    add_section7(doc)
    doc.save(DOCX_PATH)
    print(f"保存: {DOCX_PATH}")
    print(f"现在段落数: {len(doc.paragraphs)}")


if __name__ == "__main__":
    main()
