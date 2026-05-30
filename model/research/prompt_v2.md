# 文献调研与模型优化建议（修订版）

你是一个自主研究型深度学习 AI，请在 /home/lzy/project/slot-datamaking/model 中完成"文献调研与优化建议"任务。

## 访问限制

1. **只读目录**：以下目录只能读取，不能修改或删除任何文件
   - /home/lzy/project/slot-datamaking/model（除 research/ 外的所有内容）
   - /home/lzy/project/slot-datamaking/database
2. **可写目录**：所有新创建的文件必须放在：
   - /home/lzy/project/slot-datamaking/model/research/
3. 不要读取或修改项目中其他目录。

## 核心目标

广泛调研物理视频预测、物体动力学建模、交互式状态预测等方向的论文和开源模型（不少于 50 篇），建立一个结构化的文献资料库，并结合当前模型的设计，给出具体可行的优化建议。

## 当前模型概览

请先阅读以下文件理解当前模型设计：
- model/README.md — 整体架构说明
- model/ai_model/ — 自定义模型实现（encoder、interaction、temporal、decoder、loss、train）
- model/models/ — 基线模型（PhysicsVideoPredictor）
- model/design/ — 设计文档
- model/config.py — 配置
- model/review_logs/final_review_report.md — 最近一次审查报告

当前模型的关键模块：
- **Encoder**：LightCNN + MaskROIPool + AttrEncoder + StateEncoder + TokenFusion
- **Interaction**：Dense MLP Message Passing（2层，边特征16维）
- **Temporal**：Transformer Encoder-Decoder 非自回归预测
- **Decoder**：RGBDecoder（mask compositing）+ StateDecoder + CollisionClassifier
- **Loss**：MSE + Focal + SoftIoU + 物理一致性

## 任务流程

### 阶段一：理解现有模型（只读）

1. 阅读 model/ 下所有模块代码和设计文档
2. 总结当前模型的架构、各模块设计选择、已知问题和改进空间
3. 记录到 research/00_current_model_analysis.md

### 阶段二：文献调研（≥50 篇）

按以下方向搜索论文和开源项目：

**方向 1：物理视频预测（Physical Video Prediction）**
- 物理仿真视频预测（PHYRE、Physion、Kubric 等 benchmark）
- 基于学习的物理模拟器
- 视频预测模型（SVG、PredRNN、SimVP、PhyDNet 等）

**方向 2：物体中心表示（Object-Centric Representation）**
- Slot Attention 及变体（SAVi、SlotFormer、SPAIR 等）
- 物体发现与分割
- Object-centric 视频理解

**方向 3：图神经网络用于物理建模（GNN for Physics）**
- Interaction Networks、Graph Networks
- 物理约束 GNN
- 消息传递变体（GAT、GGNN、MPNN 等）

**方向 4：视觉编码器（Visual Encoder）**
- 轻量 CNN（MobileNet、EfficientNet、ShuffleNet）
- 视觉 Transformer（ViT、DeiT、Swin）
- 自监督特征（DINO、DINOv2、MAE、CLIP）

**方向 5：时序建模（Temporal Modeling）**
- Transformer 时序变体（Temporal Transformer、TimeSformer、Video Swin）
- 循环模型（ConvLSTM、PredRNN、E3D-LSTM）
- 状态空间模型（S4、Mamba）
- 非自回归预测

**方向 6：损失函数与训练策略**
- 视频预测损失（SSIM、LPIPS、FVD、感知损失）
- 物理一致性约束损失
- 对比学习、自监督辅助任务
- 课程学习、多任务学习策略

**方向 7：碰撞/交互预测**
- 碰撞检测与预测
- 力的预测与建模
- 接触图推理

**方向 8：显存优化与高效训练**
- AMP、梯度检查点、梯度累积
- 模型压缩（知识蒸馏、剪枝、量化）
- 高效 Transformer（Linformer、Performer、Flash Attention）

### 每篇论文的处理流程

对每篇论文，**严格按以下步骤执行**：

#### 步骤 1：创建论文文件夹

在 research/papers/ 下为每篇论文创建独立文件夹，命名格式：`NNN_简短名称`

```
research/papers/001_phydnet/
research/papers/002_predrnn/
research/papers/003_simvp/
...
```

#### 步骤 2：拉取论文 PDF

用 curl 或 wget 下载论文 PDF 到文件夹内：

```bash
# arxiv 论文
curl -L -o research/papers/001_phydnet/phydnet.pdf "https://arxiv.org/pdf/2003.01460.pdf"

# 如果 arxiv 不可用，尝试其他来源
```

如果无法下载 PDF（403、需要登录等），在 notes.md 中记录论文链接并说明无法下载的原因。

#### 步骤 3：拉取开源代码（如有）

如果论文有开源代码，在文件夹内 clone 代码仓库：

```bash
cd research/papers/001_phydnet/
git clone https://github.com/vincent-leguen/PhyDNet.git code/
```

clone 后阅读关键文件（README、模型定义、训练脚本），理解实现细节。

#### 步骤 4：阅读论文并撰写笔记

基于 PDF 内容和代码分析，撰写 notes.md，**必须包含以下 8 个章节**：

```markdown
# 论文标题

## 基本信息
- 作者：
- 年份：
- 会议/期刊：
- 论文链接：
- 代码链接：

## 核心贡献
- （3-5 条，每条一句话）

## 模型架构
- Encoder：
- Decoder：
- 交互模块：
- 时序模块：
- （如有代码，标注关键实现细节）

## 损失函数
- （列出所有损失项，公式简述）

## 关键设计选择
- （为什么这样设计，与其他方案对比）

## 与当前模型的对比
- 相似之处：
- 不同之处：

## 可借鉴的点
- （具体到代码层面，说明怎么改、改哪里）

## 实验结果（关键指标）
- （数据集、指标、与 baseline 对比）
```

### 每篇论文的文件夹结构

```
research/papers/001_phydnet/
├── notes.md              # 必须：8 章节笔记
├── phydnet.pdf           # 尽量：论文 PDF
└── code/                 # 如有：开源代码
    ├── README.md
    ├── models.py
    └── ...
```

### 阶段三：结构化汇总

创建 research/01_literature_survey_summary.md，包含：

1. **论文列表**：按方向分类，每篇标注：
   - 论文名、年份、会议
   - 核心贡献（一句话）
   - 与我们的相关度（高/中/低）
   - 可借鉴程度（⭐⭐⭐/⭐⭐/⭐）

2. **架构设计汇总表**：
   | 设计维度 | 方案A | 方案B | 方案C | ... | 当前方案 | 推荐方案 |
   |----------|-------|-------|-------|-----|----------|----------|
   | 视觉编码器 | | | | | | |
   | 物体表示 | | | | | | |
   | 交互建模 | | | | | | |
   | 时序预测 | | | | | | |
   | 解码器 | | | | | | |
   | 损失函数 | | | | | | |

3. **损失函数设计汇总**：
   | 损失类型 | 论文来源 | 公式简述 | 适用场景 | 推荐度 |
   |----------|----------|----------|----------|--------|

4. **训练策略汇总**：
   | 策略 | 论文来源 | 效果 | 实现复杂度 | 推荐度 |
   |------|----------|------|------------|--------|

### 阶段四：优化建议报告

创建 research/02_optimization_proposals.md，包含：

1. **当前模型的优势与不足**（基于文献对比）

2. **分模块优化建议**（每个模块至少 3 个可借鉴的设计）：

   **Encoder 优化**
   - 方案描述、来源论文、预期收益、实现难度、代码改动范围
   
   **Interaction 优化**
   - 同上格式
   
   **Temporal 优化**
   - 同上格式
   
   **Decoder 优化**
   - 同上格式
   
   **Loss 优化**
   - 同上格式
   
   **训练策略优化**
   - 同上格式

3. **优先级排序**：
   | 优先级 | 优化项 | 预期收益 | 实现难度 | 依赖 |
   |--------|--------|----------|----------|------|
   | P0 | | | | |
   | P1 | | | | |
   | P2 | | | | |

4. **推荐的第一轮迭代方案**：
   - 具体改动 1：改什么、怎么改、参考哪篇论文
   - 具体改动 2：...
   - 预期效果

## 最终目录结构

```
model/research/
├── prompt.md                       # 本文件
├── 00_current_model_analysis.md    # 当前模型分析
├── 01_literature_survey_summary.md # 文献汇总
├── 02_optimization_proposals.md    # 优化建议
├── papers/                         # 每篇论文的独立文件夹
│   ├── 001_phydnet/
│   │   ├── notes.md
│   │   ├── phydnet.pdf
│   │   └── code/
│   ├── 002_predrnn/
│   │   ├── notes.md
│   │   └── predrnn.pdf
│   ├── 003_simvp/
│   │   ├── notes.md
│   │   └── simvp.pdf
│   └── ...（共 50+ 个文件夹）
└── references/                     # （废弃，不再使用此目录）
```

## Subagent 调用注意事项

如果需要调用 subagent（delegate_task）来并行处理文献调研，请遵循以下原则：

1. **任务最小化**：每次调用 subagent 分配的任务必须尽可能小。论文调研是 1 篇，其他任务（写汇总表、写分析报告等）也要拆到最小粒度。宁可多调用，也不要一次塞太多。
2. **每个 subagent 目标明确**：告诉它具体搜哪篇论文（论文名+作者）、写入哪个文件夹、格式是什么，不要让它自己决定搜什么。
3. **每批最多 3 个 subagent 并行**：系统限制 max_concurrent_children=3。
4. **每个 subagent 的轮数限制是 50 轮**：1 篇论文在 50 轮内可以高质量完成。
5. **先写好 papers/ 目录再汇总**：subagent 完成后，检查输出文件是否完整，再进行下一阶段。
6. **不要让 subagent 调用 subagent**：嵌套已关闭（max_spawn_depth=1）。
7. **调用次数会很多（约 50-80 次）**，这是正常的。宁可多调几次保证质量，也不要一次塞太多导致超时或质量下降。

推荐的 subagent 调用节奏：
- 每批 3 个 subagent 并行，每个负责 1 篇论文
- 每个 subagent 完成后**立即检查**：
  1. 文件夹是否存在
  2. PDF 是否下载成功
  3. 代码是否 clone（如有）
  4. notes.md 是否包含完整的 8 个章节
- 大约需要 17-20 批才能完成 50+ 篇论文
- 每批之间可以快速检查并补充遗漏

每个 subagent 的 context 中应包含：
- 当前模型架构概览（从 00_current_model_analysis.md 摘要，200字以内）
- 要搜索的具体论文名/关键词（精确到论文标题，不要给泛泛的方向）
- 输出文件夹的完整路径（如 research/papers/001_phydnet/）
- 论文记录格式模板（完整的 markdown 模板，8 个章节）
- 与当前模型的相关性说明（为什么选这篇论文）

## 论文获取方式

1. 优先通过 arxiv、Google Scholar、Semantic Scholar 搜索
2. 论文 PDF 必须尝试下载到论文文件夹内（arxiv pdf 链接通常可直接下载）
3. 如果无法下载 PDF，在 notes.md 中记录论文链接并说明原因
4. 有开源代码的论文，必须 clone 代码到 code/ 子目录，阅读关键文件后再写笔记
5. 开源项目的 README 和关键代码可以摘录到 notes.md 对应章节中

## 环境要求

- 只能读取现有代码，不能修改
- 所有输出写入 research/
- 不需要运行训练，只需要阅读和分析
- 使用 conda run -n model ... 运行 Python（如果需要）

## 结束条件

1. research/papers/ 下有不少于 50 篇论文的**独立文件夹**
2. 每个文件夹内有 notes.md，且包含**完整的 8 个章节**
3. 尽可能多的文件夹内有论文 PDF
4. 有开源代码的论文，文件夹内有 code/ 子目录
5. 00_current_model_analysis.md 完成
6. 01_literature_survey_summary.md 包含完整的汇总表格
7. 02_optimization_proposals.md 包含具体可行的优化建议和优先级排序
8. 建议必须关联到具体论文来源，不能凭空想象
