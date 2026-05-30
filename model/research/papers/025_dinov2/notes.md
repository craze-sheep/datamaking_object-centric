# DINOv2: Learning Robust Visual Features without Supervision

## 基本信息
- **作者**: Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jégou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski
- **年份**: 2023 (arXiv 2304.07193, TMLR 2024 接收)
- **机构**: Meta AI Research (FAIR)
- **论文链接**: https://arxiv.org/abs/2304.07193
- **代码**: https://github.com/facebookresearch/dinov2
- **许可证**: Apache 2.0

## 核心贡献
1. **大规模策展数据集 (LVD-142M)**: 从大规模未标注数据中自动策展出 1.42 亿张高质量图像数据集，无需人工标注
2. **无监督视觉基础模型**: 纯自监督训练即可学习到通用视觉特征，在多项下游任务上匹敌甚至超越弱监督模型 (如 CLIP)
3. **知识蒸馏范式**: 将 DINO + iBOT 的自监督方法扩展到 ViT-g 规模，并通过 teacher-student 蒸馏实现高效训练
4. **Register Tokens**: 发现并解决了 ViT 特征图中的伪影问题（artifact tokens），通过引入额外的 register tokens 消除
5. **开源模型族**: 提供 ViT-S/14, ViT-B/14, ViT-L/14, ViT-g/14 四种规模的预训练模型，支持 torch.hub 直接加载

## 模型架构

### 骨干网络 (Backbone)
采用标准 Vision Transformer (ViT)，patch size 为 14×14：

| 模型 | embed_dim | depth | num_heads | 参数量 |
|------|-----------|-------|-----------|--------|
| ViT-S/14 | 384 | 12 | 6 | ~22M |
| ViT-B/14 | 768 | 12 | 12 | ~86M |
| ViT-L/14 | 1024 | 24 | 16 | ~304M |
| ViT-g/14 | 1536 | 40 | 24 | ~1.1B |

**代码位置**: `dinov2/models/vision_transformer.py` → `DinoVisionTransformer` 类

**关键架构特性**:
- **LayerScale**: 使用 `init_values` 参数初始化 layer scale（通常为 0.1 或 1.0）
- **FFN 变体**: 支持 `mlp`, `swiglu`, `swiglufused` 三种 FFN。ViT-g 使用 SwiGLU FFN（`SwiGLUFFNFused`，hidden_dim 按 `int(hidden_features * 2/3)` 对齐到 8 的倍数）
- **DropPath**: 采用 stochastic depth，`drop_path_uniform=False` 时使用线性递增 schedule
- **NestedTensorBlock**: 继承自 `Block`，支持 xFormers 的 nested tensor 处理，将 global crops 和 local crops 的 token 序列拼接后一起过 attention，效率更高
- **BlockChunk**: 将连续的 Transformer blocks 分组成 `BlockChunk`（`nn.ModuleList`），用于 FSDP wrapping 优化

**Register Tokens**（重要创新）:
```python
# vision_transformer.py
self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

# prepare_tokens_with_masks 中，register tokens 插入在 cls_token 之后、patch tokens 之前
x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
```
- 默认 `num_register_tokens=0`，使用 `_reg` 后缀模型时为 4
- **动机**: 原始 DINO 的 ViT 在特征图中会出现"artifact tokens"——某些 patch token 的注意力被全局信息吸收，导致特征图出现高范数异常点。加入 register tokens 后，这些全局信息有了额外的存放位置，patch tokens 的特征变得更加均匀
- **输出时自动跳过**: `forward_features` 输出中 `x_norm_patchtokens` 从 `x_norm[:, num_register_tokens+1:]` 开始，自动排除 cls 和 register tokens

### 训练框架: DINO + iBOT 蒸馏

**代码位置**: `dinov2/train/ssl_meta_arch.py` → `SSLMetaArch` 类

**Student-Teacher 范式**:
- Student 和 Teacher 结构完全相同
- Teacher 通过 EMA (Exponential Moving Average) 更新：`teacher_params = m * teacher_params + (1-m) * student_params`
- Teacher 使用 centering 或 Sinkhorn-Knopp 归一化
- 分布式训练使用 PyTorch FSDP (Fully Sharded Data Parallel)

**损失函数组合**（核心）:
1. **DINO CLS Loss** (`dino_clstoken_loss.py` → `DINOLoss`):
   - 对 CLS token 做 teacher-student 交叉熵蒸馏
   - Student 使用 temperature=0.1 的 log_softmax，Teacher 使用 centering + sharpening
   - 支持两种 centering 策略: `centering`（EMA center）和 `sinkhorn_knopp`
   - Global crops 间: A→B, B→A 交叉匹配；Local crops→Global crops: 每个 local crop 匹配所有 global crops

2. **iBOT Patch Loss** (`ibot_patch_loss.py` → `iBOTPatchLoss`):
   - 对 masked patch tokens 做 teacher-student 蒸馏（类似 masked image modeling）
   - 使用 block-wise masking（矩形区域 mask），mask ratio 从 `[mask_ratio_min, mask_ratio_max]` 均匀采样
   - `mask_sample_probability` 控制有多少比例的图像被 mask
   - 支持与 DINO head 共享或使用独立的 ibot head
   - 使用 xFormers 的 `cross_entropy` 做内存高效的交叉熵计算

3. **KoLeo Regularization** (`koleo_loss.py` → `KoLeoLoss`):
   - Kozachenko-Leonenko 熵估计正则化
   - 目的: 使 student 的 CLS token 在单位超球面上均匀分布
   - 计算方式: L2 归一化 → 找最近邻 → 计算 `-log(distance)` 的均值
   - 只在 global crop 的 CLS token 上施加，且同一图像的两个 crop 之间不计算（`chunk(2)` 分别计算）
   - **代码**:
     ```python
     koleo_loss = koleo_loss_weight * sum(
         self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
     )
     ```

**总损失**: `L = w_dino * L_dino + w_ibot * L_ibot + w_koleo * L_koleo`

### DINO Head
**代码位置**: `dinov2/layers/dino_head.py` → `DINOHead`
- 3 层 MLP: `embed_dim → hidden_dim(2048) → bottleneck_dim(256) → out_dim(65536)`
- L2 归一化后通过 weight_norm 的线性层
- 最后一层使用 weight normalization（`weight_g` 固定为 1）
- Student 和 Teacher 各有独立的 DINO Head

### 数据增强
**代码位置**: `dinov2/data/augmentations.py` → `DataAugmentationDINO`

- **Global crops** (2个): 224×224，scale=(0.32, 1.0)
  - 第一个: RandomResizedCrop + RandomHorizontalFlip + ColorJitter + GaussianBlur + Normalize
  - 第二个: 同上 + RandomSolarize(threshold=128, p=0.2)
- **Local crops** (默认8个): 96×96，scale=(0.05, 0.32)
  - RandomResizedCrop + RandomHorizontalFlip + ColorJitter + GaussianBlur(p=0.5) + Normalize
- ColorJitter: brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
- RandomGrayscale: p=0.2

### Masking 策略
**代码位置**: `dinov2/data/masking.py` → `MaskingGenerator`
- Block-wise masking: 生成矩形区域的 mask（类似 BEiT）
- aspect ratio 在 [0.3, 1/0.3] 之间随机采样
- mask 最多覆盖 50% 的 patch
- collate 时根据 `mask_ratio_tuple` 均匀采样每个样本的 mask 比例

## 训练细节

### 数据集 (LVD-142M)
- 从大规模未标注数据中策展
- 使用自监督方法 (DINOv1 + 检索) 进行数据清洗和去重
- 包含多样化的图像来源（网页爬取、学术数据集等）

### Curriculum Training（课程学习）
论文提出了一种课程训练策略：
- **阶段 1**: 先在较小数据集上训练短时间（如 ImageNet-22k），建立初步特征
- **阶段 2**: 使用阶段 1 的 checkpoint 初始化，扩展到完整的大规模数据集 LVD-142M 训练
- Teacher 模型用 Student 的权重初始化，无需从头训练
- **代码中的体现**: `cfg.student.pretrained_weights` 支持加载预训练权重初始化 student

### 优化器与调度
**代码位置**: `dinov2/train/train.py`

- **优化器**: AdamW
  - β1=0.9, β2=0.999（默认值，通过 config 调整）
  - 梯度裁剪: `cfg.optim.clip_grad`
  - Mixed precision 训练 (fp16/bf16)
- **学习率调度**: Cosine schedule
  - Warmup 阶段: 从 0 线性增长到 base_lr
  - Cosine 衰减到 `min_lr`
  - **Last layer 单独调度**: 冻结前 N 个 epoch（`freeze_last_layer_epochs`），lr=0
- **Weight Decay**: 从 `weight_decay` cosine 衰减到 `weight_decay_end`
- **Teacher Momentum**: 从 `momentum_teacher` cosine 变化到 `final_momentum_teacher`
- **Teacher Temperature**: 前 `warmup_teacher_temp_epochs` 从 `warmup_teacher_temp` 线性增长到 `teacher_temp`
- **Layer-wise LR Decay** (`param_groups.py`):
  - 越深的层 lr 越低（`lr_decay_rate^(num_layers+1-layer_id)`）
  - pos_embed, patch_embed, cls_token, mask_token, register_tokens 归为 layer 0
  - bias, norm, gamma 不加 weight decay（wd_multiplier=0）
  - patch_embed 可设置额外 lr 倍率 (`patch_embed_lr_mult`)

### 分布式训练
- 使用 PyTorch FSDP (Fully Sharded Data Parallel)
- 支持 `FULL_SHARD`, `SHARD_GRAD_OP`, `NO_SHARD` 策略
- Block chunking 优化: 将 Transformer blocks 分块后分别 FSDP wrap
- 使用 `ShardedGradScaler` 做混合精度训练的梯度缩放
- Teacher 和 Student 通过 `foreach` 操作高效更新参数

## 实验结果

### ImageNet 分类 (Linear Probing)
| 模型 | ImageNet Top-1 | 方法 |
|------|---------------|------|
| DINOv2-ViT-S/14 | 81.1% | 自监督 |
| DINOv2-ViT-B/14 | 84.2% | 自监督 |
| DINOv2-ViT-L/14 | 86.3% | 自监督 |
| DINOv2-ViT-g/14 | 86.5% | 自监督 |
| CLIP-ViT-L/14 | ~85.0% | 弱监督 |

### 密集预测任务
- **语义分割 (ADE20K)**: DINOv2-ViT-g 使用 linear probe 达到 ~50+ mIoU
- **深度估计 (NYU Depth V2)**: 在 ViT 特征上接简单 decoder 即可达到 SOTA
- **实例分割 (COCO)**: 作为 Mask R-CNN 的 backbone 也表现优异

### 特征可视化
- DINOv2 的 attention map 能精确勾勒物体边界
- Register tokens 的引入消除了原始 DINO 特征图中的 artifact（高范数异常点）

## 代码实现细节

### 项目结构
```
dinov2/
├── models/
│   ├── __init__.py          # build_model: 构建 student/teacher
│   └── vision_transformer.py # DinoVisionTransformer 主体
├── train/
│   ├── ssl_meta_arch.py      # SSLMetaArch: 训练核心，组合所有 loss
│   └── train.py              # 训练循环、优化器构建、调度器
├── loss/
│   ├── dino_clstoken_loss.py # DINOLoss: CLS token 蒸馏
│   ├── ibot_patch_loss.py    # iBOTPatchLoss: patch token 蒸馏
│   └── koleo_loss.py         # KoLeoLoss: 熵正则化
├── layers/
│   ├── block.py              # Block, NestedTensorBlock (xFormers 加速)
│   ├── dino_head.py          # DINOHead: 3层 MLP + weight norm
│   ├── swiglu_ffn.py         # SwiGLU FFN 实现
│   ├── attention.py          # MemEffAttention (xFormers)
│   ├── patch_embed.py        # PatchEmbed
│   └── layer_scale.py        # LayerScale
├── data/
│   ├── augmentations.py      # DataAugmentationDINO
│   ├── collate.py            # collate_data_and_cast (masking + 类型转换)
│   └── masking.py            # MaskingGenerator (block-wise)
├── fsdp/
│   └── __init__.py           # FSDP wrapper, FSDPCheckpointer
├── hub/
│   ├── backbones.py          # torch.hub 入口
│   ├── classifiers.py        # 线性分类头
│   ├── depthers.py           # 深度估计 decoder
│   └── text/                 # DINOv2 + 文本 (DINOtxt)
└── utils/
    ├── param_groups.py       # Layer-wise LR decay
    ├── utils.py              # CosineScheduler
    └── config.py             # 配置加载
```

### 关键代码流程 (forward_backward)
1. **Teacher 前向** (`get_teacher_output`):
   - 输入 global crops → backbone → CLS tokens + patch tokens
   - DINO head 处理 → centering/SK 归一化 → teacher soft targets
   - iBOT head 处理 masked patch tokens → centering/SK → teacher patch targets
2. **Student 前向**:
   - 输入 global crops + local crops（local crops 无 masking）
   - NestedTensorBlock 将 global/local crops 一起处理
   - CLS tokens: local + global → DINO head（通过 `fmha.BlockDiagonalMask.from_tensor_list` 高效批处理）
   - Patch tokens: masked patches → iBOT head
3. **损失计算**:
   - DINO loss: local→global + global→global
   - iBOT loss: masked student patches → masked teacher patches
   - KoLeo loss: global CLS tokens 的均匀分布正则
4. **Teacher EMA 更新**: `foreach_mul_` + `foreach_add_` 高效更新

### 模型加载 (torch.hub)
```python
# 基础模型
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# 带 register tokens 的模型
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')

# 提取中间层特征
features = model.get_intermediate_layers(x, n=1)  # 取最后 n 层
# 返回: [B, num_patches, embed_dim]（已排除 cls 和 register tokens）
```

## 与当前研究的关联

### 对视觉特征提取的意义
- DINOv2 提供了目前最强的通用视觉特征，无需任何文本监督
- 特征同时包含语义（CLS token）和空间信息（patch tokens），适合密集预测任务
- Register tokens 的发现对理解 ViT 内部机制有重要价值

### 可借鉴的技术点
1. **KoLeo 正则化**: 简单有效的特征均匀分布约束，可用于任何需要特征多样性保证的场景
2. **课程训练**: 大规模自监督预训练的实用策略，先小数据再大数据
3. **Layer-wise LR Decay**: 深层 ViT 的标准训练技巧
4. **NestedTensor 处理**: 多尺度 crop 的高效批处理方法
5. **Block-wise Masking**: iBOT 的矩形 mask 策略比 random patch mask 更有效
