# Video Swin Transformer

## 基本信息
- **作者**：Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, Han Hu
- **机构**：Microsoft Research Asia, 中科大, 华中科技大学, 清华大学
- **年份**：2021 (arXiv), 2022 (CVPR 2022)
- **论文链接**：https://arxiv.org/abs/2106.13230
- **代码**：https://github.com/SwinTransformer/Video-Swin-Transformer（基于 mmaction2）

## 核心贡献

1. **时空局部性归纳偏置**：将 2D Swin Transformer 的局部窗口注意力扩展到 3D 时空域，利用视频的时空局部性（相邻像素更可能相关）来高效近似全局自注意力
2. **3D 移位窗口机制**：将 2D shifted window 扩展到 3D，在时间和空间维度上交替移位，实现跨窗口信息交互，同时保持高效计算
3. **2D→3D 权重迁移**：通过 inflate 策略将预训练的 2D Swin Transformer 权重迁移到 3D，仅需少量调整（patch embedding 和 relative position bias）
4. **学习率差异化**：发现 backbone 使用比 head 低 0.1× 的学习率能更好地保留预训练权重，提升泛化性
5. **SOTA 性能**：在 Kinetics-400 (84.9%)、Kinetics-600 (86.1%)、SSv2 (69.6%) 上达到当时最优

## 模型架构

### 整体结构

输入视频 `T×H×W×3` → 3D Patch Partition → Linear Embedding → 4 个 Stage（含 3D Swin Transformer Blocks + Patch Merging）→ 输出特征

```
输入: T×H×W×3
  ↓ 3D Patch Partition (2×4×4)
Token: T/2 × H/4 × W/4, dim=96
  ↓ Linear Embedding
  ↓ Stage 1: [2 blocks] C=96,  window=(8,7,7)
  ↓ PatchMerging (2×2 空间下采样)
  ↓ Stage 2: [2 blocks] C=192
  ↓ PatchMerging
  ↓ Stage 3: [6 blocks] C=384
  ↓ PatchMerging
  ↓ Stage 4: [2 blocks] C=768
输出: T/2 × H/32 × W/32 × 768
```

### 3D Shifted Window 注意力

**核心思想**：将自注意力限制在不重叠的 3D 窗口 `P×M×M` 内计算（P=时间窗口大小, M=空间窗口大小），而非全局注意力。

**3D 窗口划分**：
- 输入 `T'×H'×W'` 个 token，窗口大小 `P×M×M`
- 划分为 `⌈T'/P⌉ × ⌈H'/M⌉ × ⌈W'/M⌉` 个不重叠窗口
- 每个窗口内做标准多头自注意力

**3D 移位窗口**：
- 偶数层：常规窗口划分（window_size = (P, M, M)）
- 奇数层：窗口沿三个轴各偏移 `(P/2, M/2, M/2)` 个 token
- 通过 cyclic shift + attention mask 实现高效批量计算
- 移位后窗口数增多，但通过原始 Swin 的高效批计算技巧，计算量与常规划分相同

**3D 相对位置偏置**：
- 每个注意力头引入 `B ∈ R^{P²×M²×M²}` 的相对位置偏置
- 参数化为更小的 `B̂ ∈ R^{(2P-1)×(2M-1)×(2M-1)}`，通过索引查表
- 相比 2D Swin 额外增加了时间轴的相对位置信息

**计算公式**（两个连续 block）：
```
ẑ^l = 3D W-MSA(LN(z^{l-1})) + z^{l-1}
z^l = FFN(LN(ẑ^l)) + ẑ^l
ẑ^{l+1} = 3D SW-MSA(LN(z^l)) + z^l
z^{l+1} = FFN(LN(ẑ^{l+1})) + ẑ^{l+1}
```

### 时空注意力设计对比

论文比较了三种时空注意力设计（Table 4, Swin-T on K400）：

| 设计 | Top-1 | FLOPs | Param | 说明 |
|------|-------|-------|-------|------|
| **Joint（默认）** | **78.8** | 88G | 28.2M | 3D 窗口内同时做时空注意力 |
| Split | 76.4 | 83G | 42.0M | 先空间 Swin，再顶层加 temporal transformer（类似 ViViT/VTN） |
| Factorized | 78.5 | 95G | 36.5M | 每个空间 MSA 后加一个 temporal MSA（类似 TimeSformer） |

**结论**：Joint 版本效率-精度最优，因为空间局部性降低了联合版本的计算量。Split 版本时序建模效率低。Factorized 版本参数量更大。

### 模型变体

| 变体 | C (Stage1) | 层数 | 参数量 | 相对规模 |
|------|-----------|------|--------|---------|
| Swin-T | 96 | {2,2,6,2} | 28.2M | 0.25× |
| Swin-S | 96 | {2,2,18,2} | 49.8M | 0.5× |
| Swin-B | 128 | {2,2,18,2} | 88.1M | 1× |
| Swin-L | 192 | {2,2,18,2} | 197-200M | 2× |

默认窗口大小：P=8, M=7；每个头的 query 维度 d=32；MLP 扩展比 α=4。

### 2D→3D 权重初始化（Inflate）

代码中 `inflate_weights` 方法实现了自动迁移：

1. **Patch Embedding**：2D Conv 权重 `(C_out, C_in, 4, 4)` → 沿时间轴复制 `patch_size[0]` 次后除以 `patch_size[0]`，得到 `(C_out, C_in, 2, 4, 4)`
2. **Relative Position Bias**：2D 偏置表 `(2M-1)², nH` 先 bicubic 插值到 `(2M'-1)²` 大小，再沿时间轴复制 `2P-1` 次
3. **其他层**：Linear、LayerNorm 等参数形状不变，直接加载

## 训练细节

### Kinetics-400 / Kinetics-600
- **优化器**：AdamW，30 epochs，cosine decay + 2.5 epochs linear warmup
- **Batch size**：64
- **学习率**：backbone 3e-5，head 3e-4（**backbone lr × 0.1**，关键技巧）
- **输入**：32 帧，temporal stride=2，空间 224×224，得到 16×56×56 个 3D token
- **数据增强**：RandomResizedCrop, Flip, Normalize
- **正则化**：Stochastic depth (T=0.1, S=0.2, B=0.3)，Weight decay (T/S=0.02, B=0.05)
- **推理**：4 clips × 3 crops (4×3 views)，最终分数为所有 view 的平均

### Something-Something v2
- **训练**：60 epochs，更强的数据增强（label smoothing, RandAugment, random erasing）
- **初始化**：使用 K400 预训练的模型
- **Stochastic depth**：ratio=0.4
- **时序窗口**：P=16（全局时序注意力）
- **推理**：1×3 views

### Backbone/Head 学习率比例（关键发现）

| 比例 | 预训练 | Top-1 |
|------|--------|-------|
| **0.1×** | ImageNet-1K | **80.6** |
| 1.0× | ImageNet-1K | 80.2 |
| **0.1×** | ImageNet-21K | **82.6** |
| 1.0× | ImageNet-21K | 82.0 |

**解释**：较低的 backbone lr 使模型缓慢遗忘预训练参数，同时适应新视频输入，提升泛化能力。ImageNet-21K 预训练受益更大。

## 实验结果

### Kinetics-400

| 方法 | 预训练 | Top-1 | Top-5 | FLOPs | Param |
|------|--------|-------|-------|-------|-------|
| SlowFast R101+NL | - | 79.8 | 93.9 | 234G | 59.9M |
| TimeSformer-L | IN-21K | 80.7 | 94.7 | 2380G | 121.4M |
| ViViT-L/16x2 | JFT-300M | 82.8 | 95.5 | 1446G | 310.8M |
| ViViT-H/16x2 | JFT-300M | 84.8 | 95.8 | 8316G | 647.5M |
| **Swin-T** | IN-1K | 78.8 | 93.6 | 88G | 28.2M |
| **Swin-B** | IN-21K | 82.7 | 95.5 | 282G | 88.1M |
| **Swin-L (384↑)** | IN-21K | **84.9** | **96.7** | 2107G | 200.0M |

Swin-L 用 **~20× 更少的预训练数据**（IN-21K vs JFT-300M）和 **~3× 更小的模型**（200M vs 647.5M）超越 ViViT-H。

### Kinetics-600

| 方法 | 预训练 | Top-1 |
|------|--------|-------|
| ViViT-H/16x2 | JFT-300M | 85.8 |
| **Swin-L (384↑)** | IN-21K | **86.1** |

### Something-Something v2

| 方法 | 预训练 | Top-1 |
|------|--------|-------|
| MViT-B-24 | K600 | 68.7 |
| **Swin-B** | K400 | **69.6** |

SSv2 强调时序建模能力，Video Swin 的 3D 联合时空注意力在此任务上优势明显。

### 消融实验关键发现

1. **3D 移位窗口**：+0.7% Top-1（w/o shifting → w/ shifting），时间移位单独贡献 +0.3%
2. **时序窗口大小**：P=8 vs P=16 仅差 0.3%，但计算量减少 17%（88G vs 106G）
3. **时序维度**：越大精度越高但计算越重；16 帧 > 8 帧 > 4 帧
4. **线性嵌入层初始化**：inflate 和 center 初始化效果相同（78.8%）
5. **3D 相对位置偏置初始化**：duplicate 和 center 初始化效果相同（78.8%）

## 代码实现细节

### 核心文件
- **Backbone**：`mmaction/models/backbones/swin_transformer.py`
- **Config**：`configs/recognition/swin/` 下各数据集配置
- **框架**：基于 mmaction2，注册为 `@BACKBONES.register_module()`

### 关键类和函数

**`PatchEmbed3D`**：3D Patch 嵌入
- 使用 `nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)` 将视频切分为 3D patch
- 支持 padding 处理非整除情况
- 可选 LayerNorm

**`WindowAttention3D`**：3D 窗口多头自注意力
- 输入 `(B*num_windows, N, C)`，N = P×M×M
- 3D 相对位置偏置表：`(2P-1)×(2M-1)×(2M-1)` → 通过 `relative_position_index` 查表
- 通过 `torch.meshgrid` 构建三维坐标网格，计算 pairwise 相对位置索引
- 支持 attention mask（用于 shifted window 的 cyclic shift）

**`SwinTransformerBlock3D`**：单个 3D Swin Block
- `forward_part1`：LN → padding → cyclic shift → window partition → 3D W-MSA → window reverse → reverse cyclic shift → 裁剪
- `forward_part2`：LN → MLP
- 残差连接 + DropPath
- 支持 `torch.utils.checkpoint` 节省显存

**`BasicLayer`**：一个 Stage
- 包含 `depth` 个 `SwinTransformerBlock3D`
- 偶数层 `shift_size=(0,0,0)`，奇数层 `shift_size=window_size//2`
- 末尾可选 `PatchMerging` 下采样
- 使用 `@lru_cache()` 缓存 attention mask 计算

**`SwinTransformer3D`**：完整 Backbone
- `pretrained2d=True` 时调用 `inflate_weights()` 从 2D Swin 迁移权重
- `pretrained2d=False` 时直接加载 3D checkpoint
- 支持 `frozen_stages` 冻结前 N 个 stage
- Stochastic depth：线性递增 `drop_path_rate`

**`PatchMerging`**：空间下采样层
- 在 H、W 维度上 2×2 合并（时间维度不下采样）
- 4C → 2C 通过 Linear 层
- 实现：`x0=x[:,:,0::2,0::2,:]`, `x1=x[:,:,1::2,0::2,:]`, ... → concat → norm → linear

**`window_partition` / `window_reverse`**：
- `window_partition`: `(B, D, H, W, C)` → reshape/permute → `(B*nW, P*M*M, C)`
- `window_reverse`: 反操作，恢复原始形状

**`compute_mask`**：计算 shifted window 的 attention mask
- 将 3D 空间分为 27 个区域（3×3×3），相同区域的 token 互相可见，不同区域的 mask 为 -100

### 配置文件结构

```python
# model config (swin_tiny.py)
model = dict(
    type='Recognizer3D',
    backbone=dict(
        type='SwinTransformer3D',
        patch_size=(4,4,4),        # 默认，训练时改为 (2,4,4)
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8,7,7),       # P=8, M=7
        mlp_ratio=4.,
        drop_path_rate=0.2,
        patch_norm=True),
    cls_head=dict(
        type='I3DHead',
        in_channels=768,           # 最后 stage 的通道数
        num_classes=400,
        spatial_type='avg',
        dropout_ratio=0.5))

# training config (kinetics400)
# patch_size 改为 (2,4,4) 以获得更细的时间分辨率
# backbone lr_mult=0.1
# 32 帧, frame_interval=2 → 实际 16 帧输入
# AdamW, lr=1e-3, cosine annealing, 30 epochs
```

### 数据流水线
- **训练**：DecordInit → SampleFrames(clip_len=32, frame_interval=2) → DecordDecode → Resize(256) → RandomResizedCrop → Resize(224) → Flip → Normalize → FormatShape(NCTHW)
- **测试**：SampleFrames(num_clips=4) → ThreeCrop(224) → 4×3 views 取平均

## 与当前研究的关联

### 对 Slot-based 视频理解的启发

1. **时空特征编码**：Video Swin 的 3D 联合时空注意力可以为 slot-based 模型提供更强的时空特征表示，替代简单的 2D CNN + GRU 方案
2. **层次化多尺度特征**：4 个 stage 的多尺度输出可用于 slot attention 的多尺度输入
3. **时间局部性**：时序窗口大小 P=8 带来的精度-效率权衡经验，可指导 slot 模型中时序建模的窗口选择
4. **2D→3D 迁移策略**：inflate 权重初始化方法可复用到其他需要将 2D 预训练模型扩展到 3D 的场景

### 局限性与注意事项

1. **计算量**：即使有窗口注意力，Video Swin 在高分辨率/长视频上仍有较大计算开销
2. **固定窗口大小**：P=8 的时序窗口对长程依赖建模有限
3. **不支持因果推理**：作为纯 encoder，不适合自回归生成任务
4. **需要大量预训练**：依赖 ImageNet-21K 预训练才能达到最优性能
