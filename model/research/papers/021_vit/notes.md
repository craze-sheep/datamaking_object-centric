# ViT: An Image is Worth 16x16 Words — Transformers for Image Recognition at Scale

## 基本信息
- **作者**：Alexey Dosovitskiy*, Lucas Beyer*, Alexander Kolesnikov*, Dirk Weissenborn*, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (*共同一作)
- **年份**：2020 (arXiv: 2020.10, ICLR 2021)
- **会议**：ICLR 2021 (International Conference on Learning Representations)
- **论文链接**：https://arxiv.org/abs/2010.11929
- **代码框架**：JAX/Flax（Google 官方实现）

---

## 核心贡献

1. **证明纯 Transformer 可直接处理图像**：无需卷积，仅将图像拆分为 patch 序列即可用标准 Transformer 编码器处理
2. **Patch Embedding 方案**：将 H×W 图像切分为 N=(H/P)×(W/P) 个 P×P patch，每个 patch 通过线性投影映射为 token——这是关键的视觉-语言桥接设计
3. **数据规模决定性能**：在中等规模数据集（ImageNet-1K）上训练时，ViT 不及同规模 CNN；但在大规模数据集（ImageNet-21K、JFT-300M）上预训练后，ViT 全面超越 CNN
4. **开启 Vision Transformer 时代**：催生了 DeiT、Swin Transformer、BEiT、MAE 等大量后续工作

---

## 模型架构

### 整体流程
```
输入图像 (H×W×C)
  → Patch Embedding (切分 + 线性投影)
  → Prepend [CLS] token
  + Position Embedding (可学习)
  → L 层 Transformer Encoder
  → [CLS] token 输出 → MLP Head → 分类 logits
```

### 1. Patch Embedding
- 将 224×224 图像切分为 14×14=196 个 16×16 patch（ViT-B/16）
- 每个 patch 展平为 16×16×3=768 维向量
- 通过**单个线性层**（等价于 Conv2D kernel_size=stride=patch_size）投影为 D 维 embedding
- 代码实现（`models_vit.py`）：
  ```python
  # 等价于一个卷积操作，kernel_size = stride = patch_size
  x = nn.Conv(
      features=self.hidden_size,       # D=768 (ViT-B)
      kernel_size=self.patches.size,   # (16, 16)
      strides=self.patches.size,       # (16, 16)
      padding='VALID',
      name='embedding')(x)
  # 输出: (batch, h, w, D) → reshape → (batch, N, D)  其中 N = h*w
  x = jnp.reshape(x, [n, h * w, c])
  ```

### 2. [CLS] Token
- 在 patch token 序列前拼接一个**可学习的 [CLS] token**
- 初始化为全零向量（代码中 `nn.initializers.zeros`）
- 最终取 [CLS] token 的输出作为图像的全局表示，送入分类头
- 代码实现：
  ```python
  cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
  cls = jnp.tile(cls, [n, 1, 1])           # 扩展到 batch 维度
  x = jnp.concatenate([cls, x], axis=1)     # (batch, N+1, D)
  ```
- 也支持 **GAP（全局平均池化）** 作为替代，代码中 `classifier='gap'`

### 3. Position Embedding
- **可学习的 1D 位置编码**，形状为 (1, N+1, D)
- 初始化标准差 0.02（来自 BERT）
- 直接加到 token embedding 上：`x = x + pos_embedding`
- 实验中发现 2D 感知的位置编码并无显著优势，1D 编码已能学会行/列结构
- 代码实现（`AddPositionEmbs` 类）：
  ```python
  pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])  # (1, N+1, D)
  pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
  return inputs + pe
  ```

### 4. Transformer Encoder
- 标准的 Pre-Norm Transformer Encoder（LayerNorm 在 attention/MLP 之前）
- 每层包含：
  1. **LayerNorm → Multi-Head Self-Attention → Dropout → Residual**
  2. **LayerNorm → MLP (Dense→GELU→Dense) → Dropout → Residual**
- MLP 隐藏层维度通常为 4D（如 ViT-B: D=768, mlp_dim=3072）
- 代码实现（`Encoder1DBlock`）：
  ```python
  # Pre-Norm Self-Attention
  x = nn.LayerNorm()(inputs)
  x = nn.MultiHeadDotProductAttention(
      num_heads=self.num_heads, dropout_rate=self.attention_dropout_rate)(x, x)
  x = nn.Dropout()(x) + inputs   # residual

  # Pre-Norm MLP
  y = nn.LayerNorm()(x)
  y = MlpBlock(mlp_dim=self.mlp_dim)(y)
  output = x + y                  # residual
  ```
- 最终经过 LayerNorm 得到编码输出

### 5. 分类头
- 取 [CLS] token 对应位置的输出：`x = x[:, 0]`
- 经过一个可选的 `pre_logits` 层（Dense → tanh），即 representation_size
- 最后通过 Dense 层映射到 num_classes（bias 初始化为 0）

### 6. Hybrid 模型（R50+ViT）
- 也支持用 ResNet 前几层作为特征提取器，再接 Transformer
- ResNet stem + stage 做 16x 下采样后，用 1×1 patch embedding 送入 Transformer
- 代码中通过 `self.resnet` 配置实现

---

## 模型变体（来自 configs/models.py）

| 模型 | Hidden Dim | MLP Dim | Heads | Layers | Patch | 参数量 |
|------|-----------|---------|-------|--------|-------|--------|
| ViT-Ti/16 | 192 | 768 | 3 | 12 | 16×16 | ~5M |
| ViT-S/16 | 384 | 1536 | 6 | 12 | 16×16 | ~22M |
| ViT-B/16 | 768 | 3072 | 12 | 12 | 16×16 | ~86M |
| ViT-L/16 | 1024 | 4096 | 16 | 24 | 16×16 | ~307M |
| ViT-H/14 | 1280 | 5120 | 16 | 32 | 14×14 | ~632M |
| R50+ViT-B/16 | 768 | 3072 | 12 | 12 | 1×1 (ResNet输出) | ~98M |

---

## 训练细节

### 预训练（Pre-training）
- **数据集**：
  - ImageNet-1K（128 万张，1000 类）
  - ImageNet-21K（1400 万张，21841 类）
  - JFT-300M（3 亿张，18415 类）
- **优化器**：SGD + Momentum=0.9（代码中 `optax.sgd`）
- **学习率调度**：
  - Warmup + Cosine Decay（默认）
  - 基础学习率：0.03（代码默认 `base_lr=0.03`）
  - Warmup 步数：500（代码默认 `warmup_steps=500`）
- **Batch Size**：512（代码默认），8 步梯度累积（`accum_steps=8`）
- **梯度裁剪**：全局范数裁剪为 1.0（`grad_norm_clip=1.0`）
- **权重衰减**：0.1（论文中提到）
- **Dropout**：0.1（代码默认，部分配置为 0.0）
- **训练分辨率**：224（预训练），384（微调时提高分辨率）
- **训练步数**：ImageNet-1K: 20,000 步; CIFAR-10: 10,000 步

### 微调（Fine-tuning）
- 从预训练 checkpoint 加载权重（代码中 `checkpoint.load_pretrained`）
- 学习率降低（如 0.01 on CIFAR-10）
- 使用较小 batch size 和较短训练时间
- 高分辨率微调时，对 position embedding 做双线性插值

### 数据增强
- 训练时：RandomResizedCrop、RandAugment、Mixup、Cutmix
- 测试时：CenterCrop 到 384

### 损失函数
- 标准交叉熵 loss（代码实现）：
  ```python
  def cross_entropy_loss(logits, labels):
      logp = jax.nn.log_softmax(logits)
      return -jnp.mean(jnp.sum(logp * labels, axis=1))
  ```

---

## 实验结果

### 主要发现

1. **数据量不足时，ViT 不如 CNN**：
   - ImageNet-1K（1.3M）上，ViT-B/16 略弱于 ResNet-152

2. **大规模预训练后，ViT 全面超越 CNN**：
   - JFT-300M 预训练后：ViT-H/14 在 ImageNet 达到 **88.55%** top-1 accuracy
   - 同时 ViT-H/14 预训练所需的计算量远少于大尺度 EfficientNet

3. **性能对比**（ImageNet top-1 accuracy）：

   | 模型 | 预训练数据 | ImageNet Acc |
   |------|-----------|-------------|
   | ResNet-152 | ImageNet-1K | 78.3% |
   | ViT-B/16 | ImageNet-21K | 84.0% |
   | ViT-L/16 | ImageNet-21K | 85.2% |
   | ViT-H/14 | JFT-300M | **88.55%** |
   | BiT-L (ResNet152x4) | JFT-300M | 87.54% |
   | EfficientNet-L2 | JFT-300M | 88.4% |

4. **计算效率**：在 JFT-300M 上，ViT-H/14 用 ¼ 的计算量达到比 EfficientNet 更好的性能

5. **注意力距离分析**：浅层 attention 既有局部也有全局，深层以全局为主；ViT 自发学会了关注图像结构

6. **位置编码**：模型学到了 2D 空间结构（行/列相似性），即使只用 1D 编码

---

## 代码实现细节

### 关键文件结构
```
code/vit_jax/
├── models_vit.py      # ViT 核心模型定义（AddPositionEmbs, MlpBlock, Encoder1DBlock, Encoder, VisionTransformer）
├── models.py           # 模型注册/分发入口
├── configs/
│   ├── models.py       # 所有模型变体配置（Ti/S/B/L/H，各 patch size，hybrid）
│   ├── common.py       # 训练超参配置（lr, batch, steps, dataset）
│   ├── vit.py          # 微调入口配置
│   └── augreg.py       # AugReg 论文相关配置
├── train.py            # 训练循环（SGD, pmap 并行）
├── checkpoint.py       # 预训练权重加载
├── input_pipeline.py   # 数据加载
└── preprocess.py       # 图像预处理
```

### 关键实现特点
1. **JAX/Flax 框架**：使用 `flax.linen`（nn.Module 风格），支持 `nn.compact`
2. **Patch Embedding 用 Conv2D 实现**：kernel_size=stride=patch_size，等价于线性投影
3. **Pre-Norm Transformer**：LayerNorm 在 attention 和 MLP 之前（区别于原始 Transformer 的 Post-Norm）
4. **多 GPU 训练**：使用 `jax.pmap` 实现数据并行训练
5. **梯度累积**：`accum_steps=8` 节省显存
6. **SGD 优化器**：非 Adam，使用 momentum=0.9 + gradient clipping
7. **位置编码初始化**：`nn.initializers.normal(stddev=0.02)` 来自 BERT

### 核心代码解析——VisionTransformer.forward：
```python
def __call__(self, inputs, *, train):
    x = inputs
    # [可选] ResNet stem 前处理
    if self.resnet is not None:
        x = models_resnet.StdConv(...)(x)  # ResNet 特征提取
        ...

    # Patch Embedding（卷积实现）
    x = nn.Conv(features=hidden_size, kernel_size=patches.size,
                strides=patches.size, padding='VALID', name='embedding')(x)
    x = jnp.reshape(x, [n, h * w, c])  # (B, N, D)

    # [CLS] token
    cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
    x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)  # (B, N+1, D)

    # Transformer Encoder（含 Position Embedding）
    x = self.encoder(name='Transformer', **self.transformer)(x, train=train)

    # 取 [CLS] token 输出
    x = x[:, 0]

    # 分类头
    if self.representation_size:
        x = nn.Dense(representation_size, name='pre_logits')(x)
        x = nn.tanh(x)
    x = nn.Dense(num_classes, name='head', kernel_init=zeros)(x)
    return x
```

---

## 与当前研究的关联

### 对 Slot-Based 模型的价值
1. **更强的视觉 backbone**：Slot Attention、SLATE 等模型的视觉编码器通常用 CNN（如 4 层 ResNet）。ViT 预训练权重提供了更强、更通用的视觉特征提取能力
2. **全局上下文建模**：CNN 感受野有限，而 ViT 的 self-attention 从第一层就能看到全局信息，更适合需要整体理解的 slot 分解任务
3. **与 Slot Attention 的配合**：
   - ViT 输出的 patch token 序列可以**直接**作为 Slot Attention 的输入（每个 patch token 类似一个"局部特征"）
   - 替代 CNN → flatten 的传统流程

### 潜在应用方向
1. **ViT 作为视觉编码器**：用预训练的 ViT-S/16 替代 CNN，提取 patch-level 物体特征
2. **Patch Token → Slot Attention**：将 ViT 的 N 个 patch token 送入 Slot Attention，初始化 K 个 slot
3. **利用 [CLS] token**：可作为全局上下文信号，辅助 slot 分配
4. **ViT 中间层特征**：类似 FPN 的方式，利用不同层的特征来捕获多尺度物体信息

### 注意事项
- ViT 需要大数据预训练才能发挥优势，小数据集上可能不如 CNN
- 高分辨率输入时，序列长度 N=(H/P)² 增长很快，attention 复杂度 O(N²)
- 推理速度通常慢于同等精度的 CNN
