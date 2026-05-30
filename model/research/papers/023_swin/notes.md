# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 基本信息

- **作者**: Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, Baining Guo
- **机构**: Microsoft Research Asia
- **年份**: 2021
- **会议**: ICCV 2021 (Best Paper)
- **论文链接**: https://arxiv.org/abs/2103.14030
- **代码**: https://github.com/microsoft/Swin-Transformer

## 核心贡献

1. **层级化 Transformer 架构**: 提出通过逐步合并 patch 构建多尺度特征金字塔的 Transformer，类似 CNN 中 VGG/ResNet 的层级结构，可直接用于密集预测任务（检测、分割）
2. **移位窗口注意力 (Shifted Window Attention)**: 通过在连续 Transformer 层之间交替使用常规窗口划分和移位窗口划分，实现跨窗口信息交流，同时保持线性计算复杂度
3. **线性计算复杂度**: 自注意力仅在局部窗口内计算，复杂度从全局 MSA 的 O((hw)²) 降至 O(hw·M²)，对高分辨率图像可扩展
4. **通用视觉骨干网络**: 在分类、检测、分割三大任务上全面超越 CNN 和此前的 ViT/DeiT，成为视觉领域的通用 backbone

## 模型架构

### 整体架构

Swin Transformer 将输入 RGB 图像通过 Patch Partition 分割为不重叠的 4×4 patch（特征维度 48），经线性嵌入层投影到 C 维，然后经过 4 个 Stage 逐步构建层级特征图：

| Stage | 输出分辨率 (224²输入) | 下采样方式 | Swin-T 层数 | Swin-T 维度 | 注意力头数 |
|-------|----------------------|-----------|------------|------------|-----------|
| Stage 1 | 56×56 | Patch Partition 4×4 | 2 | 96 | 3 |
| Stage 2 | 28×28 | Patch Merging 2×2 | 2 | 192 | 6 |
| Stage 3 | 14×14 | Patch Merging 2×2 | 6 | 384 | 12 |
| Stage 4 | 7×7 | Patch Merging 2×2 | 2 | 768 | 24 |

每个 Stage 输出的特征图分辨率与 CNN backbone（ResNet/VGG）一致，可直接替换现有方法中的 CNN backbone。

### Swin Transformer Block

每个 Block 由两个连续子层组成，采用 Pre-Norm 结构：

```
z^l = W-MSA(LN(z^{l-1})) + z^{l-1}        # 窗口注意力（常规）
z^l = MLP(LN(z^l)) + z^l                   # FFN

z^{l+1} = SW-MSA(LN(z^l)) + z^l            # 移位窗口注意力
z^{l+1} = MLP(LN(z^{l+1})) + z^{l+1}       # FFN
```

核心组件：LayerNorm → (Shifted) Window MSA → 残差连接 → LayerNorm → MLP(GELU) → 残差连接。

### Window Attention (W-MSA)

**动机**: 全局自注意力的计算复杂度为 O(4hwC² + 2(hw)²C)，对高分辨率输入不可行。

**方案**: 将特征图划分为不重叠的 M×M 窗口（默认 M=7），仅在每个窗口内计算自注意力：
- 复杂度降为 O(4hwC² + 2M²hwC)，与 token 数 hw 成线性关系
- 窗口划分函数 `window_partition`: 输入 (B, H, W, C) → reshape 为 (B, H/M, M, W/M, M, C) → permute 后合并为 (num_windows*B, M, M, C)

**问题**: 纯窗口注意力缺乏跨窗口连接，限制了建模能力。

### Shifted Window Attention (SW-MSA)

**核心思想**: 在连续层之间交替使用两种窗口配置：

1. **第 l 层**: 常规窗口划分，从左上角开始
2. **第 l+1 层**: 窗口整体平移 (⌊M/2⌋, ⌊M/2⌋) 像素，产生新的窗口划分

移位后，新窗口跨越了上一层窗口的边界，实现了跨窗口信息交流。

**高效批量计算 (Cyclic Shift)**:
- 直接移位会产生更多窗口（如 2×2 → 3×3，增加 2.25 倍计算量）
- 解决方案：将特征图循环移位（cyclic shift）到左上方向，使窗口数量不变
- 使用 attention mask 屏蔽不同子窗口之间的注意力，确保只在有效区域内计算
- 代码实现：`torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))` 移位，计算后 `torch.roll` 回去

**速度优势**: 相比滑动窗口方案，移位窗口快 3.6~4.1 倍（Swin-T/S/B），且精度相当。

### Patch Merging

在 Stage 之间实现 2× 下采样：
1. 选取 2×2 相邻 patch 中的 4 个子位置（偶偶、奇偶、偶奇、奇奇）
2. 在通道维度拼接，得到 4C 维特征
3. 经 LayerNorm + Linear(4C → 2C) 降维

```python
# 代码实现
x0 = x[:, 0::2, 0::2, :]  # 偶行偶列
x1 = x[:, 1::2, 0::2, :]  # 奇行偶列
x2 = x[:, 0::2, 1::2, :]  # 偶行奇列
x3 = x[:, 1::2, 1::2, :]  # 奇行奇列
x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4C
x = self.norm(x)
x = self.reduction(x)  # Linear(4C → 2C)
```

### Relative Position Bias

在注意力计算中加入相对位置偏置：

```
Attention(Q, K, V) = SoftMax(QK^T / √d + B) · V
```

- B ∈ R^{M²×M²} 是相对位置偏置矩阵，每个注意力头独立
- 参数化为较小的 B̂ ∈ R^{(2M-1)×(2M-1)}，通过索引映射获取实际值
- 相对位置沿各轴范围为 [−M+1, M−1]
- 使用 trunc_normal_(std=0.02) 初始化

**关键发现**:
- 相比无位置编码: ImageNet +1.2%, COCO +1.3 AP
- 相比绝对位置编码: ImageNet +0.8%, COCO +1.5 AP（绝对位置编码对检测/分割有害）
- 预训练的相对位置偏置可通过双线性插值迁移到不同窗口大小

## 模型变体

| 变体 | C (Stage1) | 层数 | 参数量 | FLOPs | ImageNet Top-1 |
|------|-----------|------|--------|-------|---------------|
| Swin-T | 96 | [2,2,6,2] | 29M | 4.5G | 81.3% |
| Swin-S | 96 | [2,2,18,2] | 50M | 8.7G | 83.0% |
| Swin-B | 128 | [2,2,18,2] | 88M | 15.4G | 83.5% |
| Swin-L | 192 | [2,2,18,2] | 197M | 103.9G | 87.3%* |

*使用 ImageNet-22K 预训练

默认设置：窗口大小 M=7，每头查询维度 d=32，MLP 扩展比 α=4。

## 训练细节

### ImageNet-1K 分类

- **优化器**: AdamW, batch size 1024, lr 0.001, weight decay 0.05
- **调度器**: 余弦退火 + 20 epoch 线性 warmup
- **训练轮数**: 300 epochs
- **数据增强**: RandAugment, Mixup, Cutmix, Random Erasing, Stochastic Depth
- **Stochastic Depth**: Swin-T 0.2, Swin-S 0.3, Swin-B 0.5
- **不使用**: Repeated Augmentation 和 EMA（对 Swin 无帮助，但对 ViT 训练稳定至关重要）

### ImageNet-22K 预训练

- 在 14.2M 图像 / 22K 类上预训练 90 epochs
- AdamW, batch size 4096, lr 0.001, weight decay 0.01, 线性衰减
- 在 ImageNet-1K 上微调 30 epochs, lr 10⁻⁵, weight decay 10⁻⁸

### 大分辨率微调

- 从 224² 模型微调到 384²，30 epochs, lr 10⁻⁵
- 相对位置偏置通过双线性插值适配新窗口大小

### COCO 目标检测

- 框架: Cascade Mask R-CNN, ATSS, RepPoints v2, Sparse RCNN (mmdetection)
- 多尺度训练: 短边 480~800, 长边 ≤1333
- AdamW, lr 0.0001, weight decay 0.05, batch size 16
- 3x schedule (36 epochs)

### ADE20K 语义分割

- 框架: UperNet (mmsegmentation)
- AdamW, lr 6×10⁻⁵, weight decay 0.01
- 8 GPU × 2 images, 160K iterations
- 输入 512×512（Swin-T/S）/ 640×640（Swin-B/L with 22K 预训练）

## 实验结果

### ImageNet-1K 分类

| 方法 | 输入 | 参数 | FLOPs | Top-1 |
|------|------|------|-------|-------|
| DeiT-S | 224² | 22M | 4.6G | 79.8% |
| DeiT-B | 224² | 86M | 17.5G | 81.8% |
| **Swin-T** | 224² | 29M | 4.5G | **81.3%** |
| **Swin-S** | 224² | 50M | 8.7G | **83.0%** |
| **Swin-B** | 224² | 88M | 15.4G | **83.5%** |
| **Swin-L** (22K) | 384² | 197M | 103.9G | **87.3%** |

### COCO 目标检测 (Cascade Mask R-CNN)

| Backbone | AP^box | AP^mask | 参数 | FLOPs |
|----------|--------|---------|------|-------|
| ResNet-50 | 46.3 | 40.1 | 82M | 739G |
| DeiT-S† | 48.0 | 41.4 | 80M | 889G |
| **Swin-T** | **50.5** | **43.7** | 86M | 745G |
| **Swin-S** | **51.8** | **44.7** | 107M | 838G |
| **Swin-B** | **51.9** | **45.0** | 145M | 982G |

Swin-T 相比 ResNet-50: +4.2 box AP, +3.6 mask AP

### ADE20K 语义分割 (UperNet)

| Backbone | mIoU | 参数 | FLOPs |
|----------|------|------|-------|
| ResNet-101 | 44.9 | 86M | 1029G |
| DeiT-S† | 44.0 | 52M | 1099G |
| **Swin-T** | **46.1** | 60M | 945G |
| **Swin-S** | **49.3** | 81M | 1038G |
| **Swin-L** (22K) | **53.5** | 234M | 3230G |

### 消融实验关键结论

1. **移位窗口 vs 无移位**: ImageNet +1.1%, COCO +2.8 AP, ADE20K +2.8 mIoU
2. **相对位置偏置 vs 无位置**: ImageNet +1.2%, COCO +1.3 AP, ADE20K +2.3 mIoU
3. **相对位置偏置 vs 绝对位置**: COCO +1.5 AP（绝对位置编码对密集预测有害）
4. **Cyclic shift vs Padding**: 速度提升 13~18%，精度相当
5. 移位窗口方法同样适用于 MLP-Mixer 架构（Swin-Mixer）

## 代码实现细节

### 核心类层次结构

```
SwinTransformer (完整模型)
├── PatchEmbed (4×4 Conv2d 做 patch 投影)
└── BasicLayer × 4 (每个 Stage)
    ├── SwinTransformerBlock × depth
    │   ├── WindowAttention (W-MSA 或 SW-MSA)
    │   │   ├── qkv: Linear(dim, 3*dim)
    │   │   ├── relative_position_bias_table: Parameter((2M-1)×(2M-1), nH)
    │   │   └── proj: Linear(dim, dim)
    │   ├── Mlp (Linear → GELU → Linear)
    │   ├── LayerNorm × 2
    │   └── DropPath (Stochastic Depth)
    └── PatchMerging (2×2 下采样, 仅前 3 个 Stage)
```

### 关键实现细节

1. **窗口划分 `window_partition`**: 通过 view + permute 实现高效重排，避免显式循环
2. **移位窗口的 attention mask**: 构造 9 区域标签图，通过 `window_partition` 分割后计算两两差值，非零处设为 -100 阻断注意力
3. **Cyclic Shift**: `torch.roll` 循环移位，计算后 `torch.roll` 反向移位恢复
4. **相对位置索引预计算**: 在 `__init__` 中通过 meshgrid 计算所有 patch 对的相对位置索引，注册为 buffer，forward 时直接查表
5. **Stochastic Depth**: 使用线性递增的 drop rate，浅层小、深层大
6. **Gradient Checkpointing**: 支持 `use_checkpoint=True` 节省显存，以计算换内存
7. **PatchEmbed 使用 Conv2d**: `nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)`，等价于线性投影但更高效

### 分类头设计

使用全局平均池化 + 线性分类器，而非 ViT 的 CLS token 方式，实测精度相当：
```python
x = self.norm(x)           # LayerNorm
x = self.avgpool(x.T)      # AdaptiveAvgPool1d(1)
x = torch.flatten(x, 1)
x = self.head(x)           # Linear → logits
```

## 与当前研究的关联

### 直接应用价值

1. **作为通用视觉 Backbone**: Swin 的层级特征图设计使其可直接替换现有 CNN backbone（如 ResNet），用于目标检测、语义分割、实例分割等密集预测任务，且性能显著优于 CNN
2. **多尺度特征金字塔**: 4 个 Stage 输出不同分辨率的特征，可直接接入 FPN、UperNet 等框架，无需额外适配
3. **高效注意力机制**: 移位窗口方案在保持建模能力的同时大幅降低计算量，适合高分辨率输入

### 技术启发

1. **局部性 + 跨窗口通信的平衡**: 移位窗口是一种优雅的方式来平衡计算效率和全局建模能力，这一思想可迁移到其他需要处理长序列的场景
2. **相对位置偏置**: 相比绝对位置编码，相对位置偏置对密集预测任务更友好，且支持分辨率迁移
3. **层级结构设计**: 通过逐步合并 patch 构建多尺度表示的范式，为后续 ConvNeXt、Focal Transformer 等工作提供了重要参考
4. **与 MLP 架构的通用性**: 移位窗口和层级设计同样适用于 MLP-Mixer，表明这些设计原则具有架构无关的通用价值
