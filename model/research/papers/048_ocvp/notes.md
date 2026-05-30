# OCVP (048) - 论文笔记

> **注意**：此文件夹中的 PDF 实际上是另一篇论文（见下方），代码目录包含的是 Genesis/Genesis-V2 的代码库。原始 notes.md 中描述的 "OCVP: Object-Centric Video Prediction" 与实际 PDF 内容不符。

---

## 基本信息

- **标题**：Antagonising Explanation and Revealing Bias Directly Through Sequencing and Multimodal Inference
- **作者**：Luís Arandas (INESC-TEC, University of Porto), Mick Grierson (CCI, University of the Arts London), Miguel Carvalhais (i2ADS, University of Porto)
- **年份**：2023
- **会议**：XAIxArts Workshop @ ACM Creativity and Cognition (C&C) 2023
- **类型**：短文/Position Paper（3页），非技术方法论文
- **PDF路径**：`papers/048_ocvp/ocvp.pdf`
- **代码路径**：`papers/048_ocvp/code/`（实际为 Genesis/Genesis-V2 代码库，与本文无直接关系）

## 核心贡献

1. **提出"通过生成式模型回溯时间"的观点**：扩散模型等深度生成模型在重建过程中不可避免地依赖训练数据中的文化印记（cultural marks），生成过程可视为"回到过去"
2. **数据集本身具有预测性**：训练数据集不仅是静态集合，其记录（records）本身就蕴含特定时间点的文化和视觉特征，这些特征在生成时被再现
3. **提出"虚拟时间线"（Virtual Timelines）概念**：同步事件调度器（synced event schedulers），用于协调扩散模型的帧间生成过程，揭示模型的重建能力和偏差
4. **将电影/视频生成视为"文献性"与"实验性"的结合**：生成的视频既是对训练数据的客观文献记录（documental），也是通过参数操控产生的实验报告（abstractive）

## 核心论点

### 数据集 → 预测性记录
- 大规模数据集（如 LAION-5B）是物理世界在特定时间点的快照
- 生成模型的采样过程实质上是在重建（reconstruct）这些记录
- 通过文本引导的扩散模型，可以系统性地探索和暴露这些记录中的偏差

### 扩散过程作为"时间回溯"
- 反向扩散过程表面上是从噪声生成图像（向前推进时间）
- 但生成的每一帧都受到训练数据中特定文化印记的约束
- 因此可以理解为"回到"数据被记录的那个时刻

### 虚拟时间线（Virtual Timelines）框架
- 作为跨模型协调的模板，控制以下可调度参数：
  1. **帧跳步和间距**（frame-skip-steps and spacing）
  2. **3D 视场平面**（three-dimensional field of view planes）
  3. **语言引导比例**（language-guidance ratios）
  4. **脚本嵌入和自动关键帧组织**（manuscript embedding and automatic keyframe organisation）
  5. **相机角度变换模板**（camera angles transform point of view shot templates）

## 方法论要素

### 两种实用方法
1. **文本引导帧生成**：用文本提示序列引导每一帧的时间步
2. **数据集索引检索**：索引特定数据集元素（如 Jan Bot 项目从 Eye 博物馆的素材库中检索）

### 生成类型
- **文献性生成**（Documental）：通过索引特定记录影响输出，具有可追溯性
- **抽象性生成**（Abstractive）：通过参数空间的混沌/随机性产生创造性变化

### 模型与工具
- 使用带分类器引导的图像扩散模型（如 OpenCLIP + LAION-5B）
- 对比分类器引导 vs 无分类器引导（DDIM/PLMS）的不确定性差异
- 多模态推理：图像+语言模型的联合建模

## 代码实际情况（Genesis/Genesis-V2）

`code/` 目录实际包含的是 Oxford 大学 Applied AI Lab 的 **Genesis** 和 **Genesis-V2** 代码库：

### Genesis (ICLR 2020)
- **论文**：GENESIS: Generative Scene Inference and Sampling with Object-Centric Latent Representations
- **作者**：Martin Engelcke, Adam R. Kosiorek, Oiwi Parker Jones, Ingmar Posner
- **方法**：基于物体中心潜表示的场景生成，使用 SBP（Stick-Breaking Process）进行注意力分割

### Genesis-V2 (NeurIPS 2021)
- **论文**：GENESIS-V2: Inferring Unordered Object Representations without Iterative Refinement
- **改进**：使用 Instance Colouring SBP 替代迭代精炼，通过颜色聚类实现无序物体表示的推断
- **架构**：UNet 编码器 → Instance Colouring SBP 分割 → 组件 VAE 解码
- **支持数据集**：Multi-dSprites, GQN, ShapeStacks, ObjectsRoom, Sketchy, APC

### 关键代码模块
- `modules/attention.py`：SBP 注意力机制（SimpleSBP, LatentSBP, InstanceColouringSBP）
- `modules/decoders.py`：组件解码器
- `models/genesisv2_config.py`：Genesis-V2 模型配置和前向传播
- `utils/geco.py`：GECO 约束优化目标
- `train.py`：统一训练脚本，支持多种模型和数据集

## 与当前研究的关联

### 本文（Arandas et al.）的启发
- **偏差分析思路**：可通过系统性的参数调度来暴露和分析生成模型的偏差
- **时间线协调**：虚拟时间线的概念可用于视频生成中的多帧协调
- **可解释性**：将生成过程视为对训练数据的"文献式记录"，有助于理解模型行为

### Genesis/Genesis-V2 的启发
- **物体中心表示**：SBP 注意力分割是经典的物体发现方法
- **Instance Colouring**：通过颜色嵌入进行无监督分割，无需迭代精炼
- **GECO 优化**：约束优化目标比固定 β-VAE 更灵活

### 局限性
- PDF 论文为概念性/立场性短文，无实验结果
- PDF 内容与文件夹名称 "ocvp"（暗示 Object-Centric Video Prediction）不匹配
- 代码库（Genesis）与 PDF 论文无直接关联
