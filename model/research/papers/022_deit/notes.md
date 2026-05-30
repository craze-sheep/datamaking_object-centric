# DeiT: Training data-efficient image transformers & distillation through attention

## 基本信息
- **作者**: Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou
- **机构**: Facebook AI, Sorbonne University
- **年份**: 2020 (arXiv), 2021 (ICML)
- **会议**: ICML 2021
- **论文链接**: https://arxiv.org/abs/2012.12877
- **代码**: https://github.com/facebookresearch/deit

## 核心贡献

1. **无需大规模数据即可训练 ViT**: 证明纯 Transformer 架构（无卷积）仅在 ImageNet-1K 上训练即可达到与 CNN 竞争的性能，无需 JFT-300M 等外部数据集
2. **引入 Distillation Token**: 提出一种 Transformer 专用的蒸馏策略——在输入序列中添加一个可学习的 distillation token，通过 self-attention 与 class token 和 patch tokens 交互，专门用于从 CNN teacher 学习
3. **Hard-label distillation 优于 soft distillation**: 对 Transformer 而言，使用 teacher 的硬标签（argmax）作为监督信号，效果显著优于传统的 soft label KL 散度蒸馏
4. **CNN teacher 优于 Transformer teacher**: 蒸馏时使用 CNN（如 RegNetY）作为 teacher 比使用 Transformer teacher 效果更好，因为蒸馏过程将 CNN 的归纳偏置（inductive bias）软性地转移到了 Transformer 中
5. **提出 DeiT-S 和 DeiT-Ti 轻量模型**: 分别对标 ResNet-50（22M 参数）和 ResNet-18（5M 参数）

## 模型架构

### 整体架构
基于 ViT 架构，**完全不使用卷积层**。输入图像被分割为 16×16 的 patch，通过线性投影得到 patch embeddings，加上位置编码后送入 Transformer blocks。

### 关键组件

#### 1. Distillation Token（蒸馏 token）
- 在 class token 和 patch tokens 之外，**额外添加一个可学习的 distillation token**
- 该 token 与 class token 和 patch tokens 通过 self-attention 交互
- **训练时**: distillation token 的目标是复现 teacher 模型的硬标签预测（argmax）
- **推理时**: class token 和 distillation token 各自通过独立的线性分类器，最终将两个 softmax 输出相加（late fusion）作为最终预测

```python
# models.py 中的关键实现
class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))  # 可学习蒸馏 token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))  # +2: cls + dist
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes)  # 蒸馏分类头

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)  # [cls, dist, patch1, ..., patchN]
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]  # 返回 cls 和 dist 的输出

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist  # 训练时返回两个输出
        else:
            return (x + x_dist) / 2  # 推理时取平均
```

- class token 和 distillation token 学到的表示**趋向不同方向**: 平均余弦相似度仅 0.06，但在最后一层相似度较高（cos=0.93），说明它们捕获互补信息
- 对比实验: 若用两个 class token（相同目标），它们会收敛到几乎相同的向量（cos=0.999），不会带来性能提升

#### 2. 模型变体

| 模型 | 嵌入维度 | 头数 | 层数 | 参数量 | 吞吐量 (im/s) |
|------|---------|------|------|--------|--------------|
| DeiT-Ti | 192 | 3 | 12 | 5M | 2536 |
| DeiT-S | 384 | 6 | 12 | 22M | 940 |
| DeiT-B | 768 | 12 | 12 | 86M | 292 |

每头维度固定为 64，通过改变头数和嵌入维度来调整模型大小。

#### 3. 蒸馏策略

**Soft Distillation**（传统方法）:
$$L_{global} = (1-\lambda) \cdot L_{CE}(\psi(Z_s), y) + \lambda \tau^2 \cdot KL(\psi(Z_s/\tau), \psi(Z_t/\tau))$$

**Hard-label Distillation**（本文提出，效果更好）:
$$L_{global} = \frac{1}{2} \cdot L_{CE}(\psi(Z_s), y) + \frac{1}{2} \cdot L_{CE}(\psi(Z_s), y_t)$$

其中 $y_t = \arg\max_c Z_t(c)$ 是 teacher 的硬标签预测。

```python
# losses.py 中的实现
class DistillationLoss(torch.nn.Module):
    def forward(self, inputs, outputs, labels):
        outputs, outputs_kd = outputs  # cls 输出和 dist 输出
        base_loss = self.base_criterion(outputs, labels)  # CE on true labels
        
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        
        if self.distillation_type == 'soft':
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_outputs / T, dim=1),
                reduction='sum', log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        
        loss = base_loss * (1 - alpha) + distillation_loss * alpha
        return loss
```

#### 4. Data Augmentation（数据增强）
Transformer 缺乏 CNN 的归纳偏置，需要更强的数据增强来弥补数据不足:

| 增强方法 | 参数 | 说明 |
|---------|------|------|
| RandAugment | 9/0.5 | 随机数据增强策略 |
| Mixup | α=0.8 | 样本混合 |
| CutMix | α=1.0 | 区域混合 |
| Random Erasing | p=0.25 | 随机擦除 |
| Repeated Augmentation | 3x | 每个样本重复增强 3 次 |
| Label Smoothing | ε=0.1 | 标签平滑 |
| Stochastic Depth | p=0.1 | 随机深度（正则化） |

**关键发现**: Dropout 对 Transformer 训练有害，不使用。

## 训练细节

### 超参数配置

| 参数 | DeiT 设置 | 原始 ViT-B 设置 |
|------|----------|----------------|
| Epochs | 300 | 300 |
| Batch size | 1024 | 4096 |
| Optimizer | AdamW | AdamW |
| Learning rate | 5e-4 × batch_size/512 | 3e-3 |
| LR decay | cosine | cosine |
| Weight decay | 0.05 | 0.3 |
| Warmup epochs | 5 | 3.4 |
| Label smoothing | 0.1 | 无 |
| Dropout | 无 | 0.1 |
| Stochastic Depth | 0.1 | 无 |
| Repeated Aug | ✓ (3次) | ✗ |

### 训练流程
1. **预训练**: 224×224 分辨率，300 epochs，单节点 8 GPU 约 53 小时
2. **微调**（可选）: 384×384 分辨率，25 epochs，约 20 小时
3. 位置编码通过双三次插值（bicubic interpolation）适配不同分辨率
4. 学习率缩放公式: `lr_scaled = lr_base / 512 × batch_size`
5. 使用 Exponential Moving Average (EMA)，衰减系数 0.99996

### 初始化
- 权重使用截断正态分布初始化（truncated normal），标准差 0.02
- dist_token 初始化为零向量，然后用截断正态分布扰动

## 实验结果

### ImageNet 主要结果

| 模型 | 参数量 | 分辨率 | ImageNet top-1 | Real top-1 | V2 top-1 |
|------|--------|--------|---------------|------------|----------|
| DeiT-Ti | 5M | 224 | 72.2% | 80.1% | 60.4% |
| DeiT-S | 22M | 224 | 79.8% | 85.7% | 68.5% |
| DeiT-B | 86M | 224 | 81.8% | 86.7% | 71.5% |
| DeiT-B↑384 | 86M | 384 | 83.1% | 87.7% | 72.4% |
| **DeiT-B⚗↑384 (1000ep)** | **87M** | **384** | **85.2%** | **89.3%** | **75.2%** |

### 蒸馏效果对比（DeiT-B, 300 epochs）

| 方法 | 监督信号 | 224 top-1 | 384 top-1 |
|------|---------|-----------|-----------|
| 无蒸馏 | label only | 81.8% | 83.1% |
| Soft distillation | soft teacher | 81.8% | 83.2% |
| Hard distillation | hard teacher | 83.0% | 84.0% |
| **DeiT⚗ (class+dist)** | **label + hard teacher** | **83.4%** | **84.5%** |

### Teacher 模型对比

| Teacher | Teacher acc. | DeiT-B⚗ (224) | DeiT-B⚗ (384) |
|---------|-------------|---------------|---------------|
| DeiT-B | 81.8% | 81.9% | 83.1% |
| RegNetY-4GF | 80.0% | 82.7% | 83.6% |
| RegNetY-16GF | 82.9% | 83.1% | 84.2% |

**结论**: CNN teacher（RegNetY）优于 Transformer teacher（DeiT-B），即使 teacher 自身准确率较低也能带来更大提升。

### 迁移学习

在 CIFAR-10/100、Flowers-102、Stanford Cars、iNaturalist 等数据集上微调，DeiT 与 EfficientNet 性能相当:
- DeiT-B⚗↑384: CIFAR-10 99.2%, Flowers 98.9%, Cars 93.9%

### 消融实验关键发现
- **Repeated Augmentation** 是关键成分，去掉后性能下降 ~2%
- **RandAugment** 优于 AutoAugment
- **Stochastic Depth** 对深层 Transformer 收敛至关重要
- **Mixup + CutMix** 显著提升性能
- **Dropout** 有害，应去掉
- **AdamW** 远优于 SGD（74.5% vs 81.8%）
- 蒸馏训练中，**更长的训练（1000 epochs）持续带来提升**，而无蒸馏训练在 400 epochs 后饱和

## 代码实现细节

### 代码结构
```
code/
├── models.py          # DeiT 模型定义（DistilledVisionTransformer）
├── models_v2.py       # DeiT v2 模型
├── main.py            # 训练入口，包含所有超参数和训练循环
├── engine.py          # train_one_epoch 和 evaluate 函数
├── losses.py          # DistillationLoss（soft/hard 蒸馏损失）
├── augment.py         # 3Augment 数据增强策略
├── samplers.py        # RASampler（Repeated Augmentation 采样器）
├── datasets.py        # 数据集构建
├── utils.py           # 工具函数
├── cait_models.py     # CaiT 模型
├── resmlp_models.py   # ResMLP 模型
└── patchconvnet_models.py  # PatchConvNet 模型
```

### 关键实现要点

1. **DistilledVisionTransformer 继承自 timm 的 VisionTransformer**: 仅添加了 `dist_token`、扩展的 `pos_embed`（+2）和 `head_dist` 分类头

2. **训练时返回两个输出** `(cls_logits, dist_logits)`，推理时返回平均值 `(cls_logits + dist_logits) / 2`

3. **DistillationLoss 封装**: 将基础损失（CE/LabelSmoothing/SoftTarget）与蒸馏损失组合，通过 `alpha` 参数平衡

4. **RASampler**: 实现 Repeated Augmentation 的分布式采样器，确保每个 epoch 中每个样本被不同 GPU 上的不同增强看到

5. **Position Embedding 插值**: 微调不同分辨率时，对位置编码进行双三次插值，保留 class token 和 distillation token 的位置编码不变

6. **Mixed Precision 训练**: 使用 `torch.cuda.amp.autocast()` 和 `NativeScaler` 进行混合精度训练

### 典型训练命令
```bash
# 基础训练（无蒸馏）
python main.py --model deit_base_patch16_224 --batch-size 64 --epochs 300 \
    --data-path /path/to/imagenet --output_dir /path/to/output

# 带蒸馏训练
python main.py --model deit_base_distilled_patch16_224 --batch-size 64 --epochs 300 \
    --distillation-type hard --teacher-model regnety_160 \
    --teacher-path /path/to/teacher.pth --distillation-alpha 0.5 \
    --data-path /path/to/imagenet --output_dir /path/to/output

# 微调到 384 分辨率
python main.py --model deit_base_distilled_patch16_384 --batch-size 32 --epochs 25 \
    --finetune /path/to/checkpoint.pth --input-size 384 \
    --data-path /path/to/imagenet --output_dir /path/to/output
```

## 与当前研究的关联

### 对视觉编码器改进的启示

1. **Distillation Token 机制可直接借鉴**: 在视觉编码器中添加蒸馏 token，从预训练的 ViT 或 CNN teacher 学习，可以显著提升特征质量，同时保持计算量可控

2. **训练策略复用**: DeiT 的训练配方（RandAugment + Mixup + CutMix + Stochastic Depth + Repeated Augmentation + AdamW）是训练 ViT 类模型的成熟方案

3. **Hard-label distillation 简单有效**: 相比 soft distillation，hard distillation 无需调温度参数，实现更简单，效果更好

4. **CNN teacher 的归纳偏置迁移**: 通过蒸馏将 CNN 的平移不变性、局部性等归纳偏置软性注入 Transformer，是一个轻量且有效的策略

5. **模型变体的灵活性**: DeiT-Ti（5M）、DeiT-S（22M）提供了不同计算预算下的选择，适合不同场景

### 局限性
- 蒸馏需要额外训练一个 CNN teacher（如 RegNetY-16GF）
- Distillation token 在推理时增加了一个分类头的计算
- 300 epochs 的训练时间仍然较长（~53 小时 on 8 GPUs）
