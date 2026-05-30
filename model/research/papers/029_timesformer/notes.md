# TimeSformer: Is Space-Time Attention All You Need for Video Understanding?

## 基本信息
- **作者**：Gedas Bertasius, Heng Wang, Lorenzo Torresani
- **年份**：2021
- **会议**：ICML 2021
- **论文链接**：https://arxiv.org/abs/2102.05095
- **代码**：https://github.com/facebookresearch/TimeSformer
- **关键词**：视频理解、纯 Transformer、时空注意力、无卷积

## 核心贡献

1. **首个纯 Transformer 视频分类模型**：完全抛弃卷积，仅用 self-attention 进行时空特征学习
2. **Divided Space-Time Attention**：提出分离的时间注意力和空间注意力方案，在效率和精度上取得最佳平衡
3. **系统对比了 5 种时空注意力方案**：joint space-time、divided space-time、sparse local、sparse global、axial attention
4. **SOTA 结果**：在 Kinetics-400（80.7%）、Kinetics-600（82.4%）上达到当时最佳
5. **高效训练与推理**：比 3D CNN 训练更快，可处理超过 1 分钟的长视频

## 模型架构

### 整体架构

基于 ViT (Vision Transformer) 扩展到视频领域：

```
输入: 视频片段 [B, C, T, H, W]
  ↓
Patch Embedding: 每帧独立分成 16×16 patch，线性投影到 D=768 维
  ↓
Positional Embedding: 空间位置编码 (可学习) + 时间位置编码 (可学习)
  ↓
CLS Token: 添加到序列开头
  ↓
L 层 Transformer Block (L=12, Depth=12)
  ↓
LayerNorm + MLP Head → 分类输出
```

### Patch Embedding 实现
```python
# 对每帧独立做 2D patch embedding
# 输入: [B, C, T, H, W] → reshape 为 [B*T, C, H, W]
# Conv2d(kernel=16, stride=16) → [B*T, num_patches, D]
# 输出 token 数量: (H/16) × (W/16) per frame
```

### 五种时空注意力方案

#### 1. Joint Space-Time Attention（联合时空注意力）
- 将所有帧的所有 patch 视为一个长序列，做全局 self-attention
- 每个 token 可以 attend 到所有帧的所有位置
- **复杂度**：O((T·H·W)²)，最高但最灵活
- **实现**：直接使用标准 Multi-Head Attention，无需特殊处理

#### 2. Divided Space-Time Attention（分离时空注意力）⭐ 最佳方案
- **先时间注意力，后空间注意力**，交替进行
- 每个 Transformer Block 内部：
  ```
  ① Temporal Attention: 每个空间位置独立地跨帧做 attention
     → rearrange: [B, H*W*T, D] → [B*H*W, T, D]
     → 对 T 个时间步做 self-attention
  ② Spatial Attention: 每帧独立地做空间 attention
     → rearrange: [B, H*W*T, D] → [B*T, H*W, D]
     → 对 H*W 个空间位置做 self-attention
  ③ MLP: 标准 FFN
  ```
- **复杂度**：O(T·(H·W)²) + O(H·W·T²)，远低于联合注意力
- **关键代码**（来自 `vit.py` Block.forward）：
  ```python
  # Temporal attention
  xt = rearrange(x[:,1:,:], 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)
  res_temporal = self.temporal_attn(self.temporal_norm1(xt))
  res_temporal = self.temporal_fc(res_temporal)  # 线性投影回原空间
  xt = x[:,1:,:] + res_temporal  # 残差连接

  # Spatial attention
  xs = rearrange(xt, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
  xs = torch.cat((cls_token, xs), 1)  # 每帧添加 CLS token
  res_spatial = self.attn(self.norm1(xs))

  # MLP
  x = residual + res_spatial
  x = x + self.mlp(self.norm2(x))
  ```

#### 3. Sparse Local Attention
- 每个 token 只 attend 到同一帧的空间邻域 + 同一位置的相邻帧
- 局部窗口限制，复杂度最低

#### 4. Sparse Global Attention
- 空间注意力限制在局部窗口，时间维度做全局 attention
- 或反过来

#### 5. Axial Attention
- 分别沿空间轴（H 和 W）和时间轴 T 独立做 attention
- 三步：row attention → column attention → temporal attention

### 位置编码设计

```python
# 空间位置编码
self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))  # +1 for CLS

# 时间位置编码（仅 joint 和 divided 模式使用）
self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))

# 前向传播中的拼接方式：
# 1. 先加空间位置编码到每个 patch token
# 2. 再 rearrange 后加时间位置编码
# 支持推理时的分辨率/帧数自适应插值
```

### 时间注意力权重初始化策略

**关键设计**：从预训练的 ViT（图像）权重初始化时间注意力：
```python
# 时间注意力的 QKV 权重直接复制空间注意力的权重
for key in state_dict:
    if 'blocks' in key and 'attn' in key:
        new_key = key.replace('attn', 'temporal_attn')
        new_state_dict[new_key] = state_dict[key]  # 复制

# temporal_fc（时间注意力输出的线性投影）初始化为 0
nn.init.constant_(m.temporal_fc.weight, 0)
nn.init.constant_(m.temporal_fc.bias, 0)
```

**意义**：初始化时时间注意力不产生任何输出，模型等价于纯空间 ViT，然后逐步学习时间依赖。这保证了从图像预训练权重到视频模型的平滑迁移。

## 训练细节

### 超参数配置（Kinetics-400）

| 参数 | 值 |
|------|------|
| 模型 | ViT-Base (patch16, 768-dim, 12 heads, 12 layers) |
| 输入分辨率 | 224×224 |
| Patch 大小 | 16×16 |
| 帧数 T | 8（默认）、96（长视频） |
| 采样率 | 32（8帧时）、4（96帧时） |
| Batch Size | 8 per GPU × 8 GPUs = 64 |
| 优化器 | SGD |
| 基础学习率 | 0.005 |
| 学习率策略 | steps_with_relative_lrs: [0, 11, 14] → [1, 0.1, 0.01] |
| 总 Epoch | 15 |
| Weight Decay | 1e-4 |
| Momentum | 0.9 |
| Dropout | 0.5（分类头前） |
| Drop Path | 0.1（stochastic depth） |
| Mixup | 可选，label_smoothing=0.1 |
| 预训练 | ImageNet ViT 权重 |

### 数据增强
- 训练：随机裁剪 224×224，jitter scale [256, 320]
- 测试：3 spatial crops × 1 temporal view
- SSv2 数据集：无随机翻转，反向输入通道

### 训练流程
1. 从 ImageNet 预训练的 ViT-Base 加载权重
2. 复制空间注意力权重到时间注意力，temporal_fc 初始化为 0
3. 插值空间位置编码以匹配目标分辨率
4. 插值时间位置编码以匹配目标帧数
5. 15 个 epoch 微调（学习率在第 11、14 epoch 衰减）

## 实验结果

### Kinetics-400 主要结果

| 方法 | 类型 | Top-1 Acc | Top-5 Acc | 训练时间/epoch |
|------|------|-----------|-----------|----------------|
| SlowFast (R101+NL) | 3D CNN | 79.8% | 93.9% | 基准 |
| ViT (space-only) | Transformer | ~78% | - | 最快 |
| **TimeSformer (divided)** | **Transformer** | **80.7%** | **94.7%** | 快 |
| TimeSformer (joint) | Transformer | 80.0% | 94.4% | 最慢 |

### 不同注意力方案对比

| 注意力方案 | Top-1 Acc | 相对训练速度 | 内存需求 |
|------------|-----------|-------------|----------|
| Joint Space-Time | 80.0% | 1×（最慢） | 最高 |
| **Divided Space-Time** | **80.7%** | **~5×** | **适中** |
| Sparse Local | 79.3% | ~6× | 低 |
| Sparse Global | 79.6% | ~5× | 适中 |
| Axial (TimeSformer-A) | 79.5% | ~5× | 适中 |

### 长视频处理
- 96 帧（采样率 4）：可处理约 1 分钟视频
- 384 分辨率 + 96 帧：K400 达到 80.7%
- 相比 3D CNN 受限于显存的短片段，TimeSformer 可扩展到更长视频

### 其他数据集

| 数据集 | TimeSformer | 前 SOTA |
|--------|-------------|---------|
| Kinetics-400 | 80.7% | 79.8% (SlowFast) |
| Kinetics-600 | 82.4% | 81.1% (SlowFast) |
| Something-Something V2 | ~62% | ~65% (需更大帧数) |
| Diving-48 | 81.0% | - |
| Epic-Kitchens | 有竞争力 | - |

### SSv2 结果注意
- Something-Something V2 对时间建模要求极高
- Divided attention 在该数据集上用 64 帧效果更好
- 说明分离注意力对短时运动模式建模仍有提升空间

## 代码实现细节

### 项目结构
```
code/
├── configs/                    # 配置文件
│   ├── Kinetics/
│   │   ├── TimeSformer_divST_8x32_224.yaml    # 分离注意力，8帧
│   │   ├── TimeSformer_jointST_8x32_224.yaml  # 联合注意力
│   │   ├── TimeSformer_spaceOnly_8x32_224.yaml # 纯空间（基线）
│   │   └── TimeSformer_divST_96x4_224.yaml    # 96帧长视频
│   └── SSv2/
│       ├── TimeSformer_divST_8_224.yaml
│       └── TimeSformer_divST_64_224.yaml
├── timesformer/
│   ├── models/
│   │   ├── vit.py              # ⭐ 核心：TimeSformer 模型定义
│   │   ├── vit_utils.py        # DropPath, trunc_normal_ 等工具
│   │   ├── helpers.py          # 权重加载，预训练适配
│   │   ├── video_model_builder.py  # SlowFast/ResNet 等基线模型
│   │   └── optimizer.py        # 优化器构建
│   ├── datasets/               # 数据加载
│   └── utils/                  # 工具函数
└── tools/
    ├── train_net.py            # 训练入口
    └── test_net.py             # 测试入口
```

### 核心类层次

```
VisionTransformer (vit.py)
├── PatchEmbed: 2D Conv2d 做 patch 投影
├── pos_embed: 空间位置编码 [1, HW+1, D]
├── time_embed: 时间位置编码 [1, T, D]
├── cls_token: [1, 1, D]
├── blocks × 12:
│   └── Block (每个包含):
│       ├── [temporal] temporal_norm1 + temporal_attn + temporal_fc  (divided 模式)
│       ├── [spatial]  norm1 + attn (标准 self-attention)
│       ├── DropPath (stochastic depth)
│       ├── norm2 + MLP
│       └── 残差连接
├── norm: LayerNorm
└── head: Linear(D, num_classes)
```

### 关键实现细节

**1. einops 重排操作**：
```python
# 时空张量的核心变换
# 时间注意力：[B, H*W*T, D] → [B*H*W, T, D]
xt = rearrange(xt, 'b (h w t) m -> (b h w) t m', b=B, h=H, w=W, t=T)

# 空间注意力：[B, H*W*T, D] → [B*T, H*W, D]
xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m', b=B, h=H, w=W, t=T)
```

**2. CLS Token 处理**：
- 时间注意力阶段：不包含 CLS token（仅对 patch token 做）
- 空间注意力阶段：每帧复制 CLS token，空间注意力后取平均
```python
cls_token = init_cls_token.repeat(1, T, 1)  # 复制 T 份
# 空间注意力后
cls_token = torch.mean(cls_token, 1, True)  # 对 T 帧取平均
```

**3. temporal_fc 的作用**：
- 时间注意力输出后经过一个线性层 `temporal_fc`
- 初始化为 0，使得训练初期时间注意力不干扰空间特征
- 这是平滑迁移的关键设计

**4. 位置编码自适应**：
```python
# 推理时分辨率/帧数不匹配时自动插值
if x.size(1) != self.pos_embed.size(1):
    # 双线性插值空间位置编码
    pos_embed = F.interpolate(pos_embed, size=(H, W), mode='nearest')

if T != self.time_embed.size(1):
    # 插值时间位置编码
    new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
```

**5. 预训练权重适配**（helpers.py）：
- 将 ViT 图像预训练权重扩展到视频
- 空间注意力权重直接使用
- 时间注意力权重从空间注意力复制
- temporal_fc 初始化为零
- 位置编码支持插值

### 配置驱动的模型选择
```yaml
# 通过 TIMESFORMER.ATTENTION_TYPE 切换不同注意力方案
TIMESFORMER:
  ATTENTION_TYPE: 'divided_space_time'  # 或 'joint_space_time', 'space_only'
```

## 与当前研究的关联

### 与 Video Understanding 的关系
- TimeSformer 开创了纯 Transformer 视频理解的范式
- 后续工作如 ViViT、MViT、VideoMAE 等都受其启发
- Divided attention 成为视频 Transformer 的经典设计

### 与 Slot-Based 模型的关系
- **空间注意力**可以类比为 slot attention 中的 object-centric 学习
- 每帧独立的空间 attention 类似于逐帧的 slot 分配
- 时间 attention 类似于跨帧的 slot 跟踪/关联
- **可借鉴点**：将 divided attention 的思想用于 slot 模型的时序建模

### 对后续工作的启示
1. **分离注意力的普适性**：不仅适用于 Transformer，也可用于其他架构
2. **预训练迁移策略**：从图像模型到视频模型的零初始化时间分支
3. **长视频处理**：通过分离注意力降低复杂度，可处理更长视频
4. **效率与精度的权衡**：divided attention 在精度最高的同时效率也很好

### 局限性
1. Patch 大小固定为 16×16，细粒度时间建模能力有限
2. 时间注意力只在固定空间位置间进行，无法建模运动位移
3. 对 Something-Something 等需要精细运动理解的数据集效果不如专用方法
4. 计算量仍然随帧数线性增长（时间注意力部分）
