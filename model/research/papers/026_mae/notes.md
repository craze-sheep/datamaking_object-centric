# MAE: Masked Autoencoders Are Scalable Vision Learners

## 基本信息
- **作者**：Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick
- **机构**：Meta AI (FAIR)
- **年份**：2021 (arXiv) / 2022 (CVPR)
- **会议**：CVPR 2022
- **论文链接**：https://arxiv.org/abs/2111.06377
- **代码**：https://github.com/facebookresearch/mae

## 核心贡献

1. **提出 Masked Autoencoder (MAE)**：一种简单高效的自监督视觉预训练方法，借鉴 NLP 中 BERT 的 masked prediction 思想
2. **非对称编码器-解码器设计**：编码器只处理可见 patch（25%），解码器轻量且处理全部 patch，大幅提升训练效率
3. **高掩码比例（75%）**：证明在视觉中需要远高于 NLP（15%）的掩码比例，因为图像信息密度远低于自然语言
4. **像素级回归重建**：直接以归一化像素值作为重建目标，无需额外 tokenizer 或预训练网络
5. **极高的训练效率**：ViT-Huge 在 256 个 GPU 上仅需约 6.3 小时即可完成 1600 epoch 预训练，相比此前方法快 3-4 倍
6. **SOTA 性能**：在 ImageNet 上 ViT-H 达到 87.8% fine-tuning 准确率，线性探测达到 75.2%

## 模型架构

### 整体流程
```
输入图像 → Patch Embedding → 随机 Mask (75%) → Encoder (仅处理 25% 可见 patch)
    → 补回 Mask Token → Decoder (处理全部 patch) → 像素重建 → 仅计算 masked patch 的损失
```

### 1. Patch Embedding
- 将 224×224 图像切分为 16×16 的 patch，得到 196 个 patch token
- 使用 `nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)` 实现（来自 timm 的 PatchEmbed）
- 添加 2D sinusoidal 位置编码（固定，不可学习）+ CLS token

### 2. 随机掩码策略
- 对 196 个 patch 生成随机噪声，按噪声排序后保留前 25%（约 49 个 patch）
- 通过 `torch.argsort` 实现高效的 per-sample shuffling
- 返回：masked 序列、二值 mask（0=保留, 1=移除）、恢复索引

### 3. Encoder（非对称设计的核心）
- **只处理可见 patch**：输入仅 49 个 token（+ 1 个 CLS token），而非全部 196 个
- 架构：标准 ViT（Base: 12 层/768 维, Large: 24 层/1024 维, Huge: 32 层/1280 维）
- MLP ratio = 4，多头自注意力
- 输出维度：Base 为 768，Large 为 1024，Huge 为 1280

### 4. Decoder（轻量设计）
- **线性投影**：将 encoder 输出维度（如 1024）映射到 decoder 维度（512）
- **补回 mask token**：将 encoder 输出的可见 token 与 mask token 拼接，恢复到原始序列长度（196）
- **Unshuffle**：用 `ids_restore` 恢复原始 patch 顺序
- 添加 decoder 专用的位置编码（2D sinusoidal）
- 架构：8 层 Transformer，512 维，16 头
- 最后通过线性层投影到 `patch_size² × 3`（即 768 维）输出像素值

### 5. 重建目标与损失函数
- **重建目标**：被 mask 的 patch 的原始像素值
- **patchify**：将图像 reshape 为 `(N, H*W, patch_size²*3)` 作为 target
- **norm_pix_loss**（可选）：对每个 patch 内的像素做归一化（减均值除标准差），使模型关注纹理结构而非绝对亮度
- **损失**：MSE，仅在 masked patch 上计算
  ```
  loss = ((pred - target)²).mean(dim=-1)  # 每个 patch 的 MSE
  loss = (loss * mask).sum() / mask.sum()  # 仅对 masked patch 取平均
  ```

## 训练细节

### 预训练配置
| 参数 | 值 |
|------|-----|
| 数据集 | ImageNet-1K (128 万张图像) |
| 输入大小 | 224×224 |
| Patch 大小 | 16×16 (Base/Large), 14×14 (Huge) |
| Mask 比例 | 75% |
| Epochs | 800 (默认), 1600 (Huge 最优) |
| 优化器 | AdamW (β₁=0.9, β₂=0.95) |
| 基础学习率 | 1.5e-4 (blr), 实际 lr = blr × batch_size / 256 |
| 权重衰减 | 0.05 |
| Batch size | 4096 (64 × 8 nodes × 8 GPUs) |
| Warmup epochs | 40 |
| 学习率调度 | Warmup + 余弦衰减 |
| 数据增强 | RandomResizedCrop (scale 0.2-1.0) + 水平翻转 |
| 归一化 | ImageNet 标准归一化 |
| 精度 | FP16 混合精度 (torch.cuda.amp) |
| 训练时间 | ViT-L: ~42h (64 V100), ViT-H: ~6.3h (256 GPU, 1600ep) |

### 微调配置
| 参数 | 值 |
|------|-----|
| Epochs | 100 (ViT-B), 50 (ViT-L/H) |
| 学习率 | 1e-3 (blr), layer-wise lr decay = 0.65/0.75 |
| Drop path | 0.1 (ViT-B), 0.2 (ViT-L), 0.3 (ViT-H) |
| 数据增强 | RandAugment + Mixup + Cutmix + Random Erase + Label Smoothing |
| 位置编码插值 | 支持从 224 微调到更高分辨率 |

### 线性探测配置
| 参数 | 值 |
|------|-----|
| 优化器 | LARS |
| Batch size | 4096 (16 GPUs) |
| Epochs | 90 |
| 学习率 | 0.1 (blr) |
| Weight decay | 0 |
| 特征提取 | 全局平均池化（不含 CLS token） |

## 实验结果

### ImageNet-1K 分类性能

| 方法 | Backbone | 预训练 Epoch | Fine-tune Acc | 线性探测 Acc |
|------|----------|-------------|---------------|-------------|
| MAE | ViT-B/16 | 800 | 83.6% | 68.0% |
| MAE | ViT-L/16 | 800 | 85.5% | 73.5% |
| **MAE** | **ViT-H/14** | **1600** | **87.8%** | **75.2%** |
| DINO | ViT-B/16 | - | 84.2% | 74.5% |
| BEiT | ViT-L/16 | 300 | 85.2% | 56.7% |
| Supervised | ViT-B/16 | 300 | 81.8% | - |
| Supervised | ViT-L/16 | 300 | 85.1% | - |

### 关键发现
1. **掩码比例**：75% 为最优，远高于 NLP 的 15%。图像信息密度低，相邻 patch 高度冗余
2. **非对称设计的效率**：encoder 只处理 25% patch，训练速度提升约 3×
3. **norm_pix_loss**：归一化像素目标在微调时带来轻微提升，线性探测时提升更明显
4. **decoder 深度**：8 层 decoder 即可，更深 decoder 收益递减；decoder 在预训练后可丢弃
5. **可扩展性**：随着模型增大（B→L→H），MAE 性能持续提升，验证了良好的可扩展性
6. **数据效率**：仅用 ImageNet-1K 预训练即可达到优秀性能
7. **迁移学习**：在 COCO 检测/分割、ADE20K 语义分割等下游任务上均表现优异

## 代码实现细节

### 核心文件结构
```
code/
├── models_mae.py          # MAE 模型定义（预训练）
├── models_vit.py          # ViT 模型定义（微调/线性探测）
├── main_pretrain.py       # 预训练入口
├── main_finetune.py       # 微调入口
├── main_linprobe.py       # 线性探测入口
├── engine_pretrain.py     # 预训练训练循环
├── engine_finetune.py     # 微调训练循环
├── util/
│   ├── pos_embed.py       # 2D sinusoidal 位置编码 + 插值
│   ├── lr_sched.py        # 余弦学习率调度
│   ├── lr_decay.py        # Layer-wise 学习率衰减
│   ├── misc.py            # 分布式训练工具
│   ├── lars.py            # LARS 优化器
│   └── datasets.py        # 数据集构建
├── PRETRAIN.md            # 预训练说明
├── FINETUNE.md            # 微调说明
└── demo/
    └── mae_visualize.ipynb # 可视化 demo
```

### 关键实现细节

#### 1. 随机掩码 (`random_masking`)
```python
# 通过随机噪声排序实现高效 per-sample 随机掩码
noise = torch.rand(N, L, device=x.device)
ids_shuffle = torch.argsort(noise, dim=1)       # 升序：小的保留，大的移除
ids_restore = torch.argsort(ids_shuffle, dim=1)  # 恢复索引
ids_keep = ids_shuffle[:, :len_keep]              # 保留前 25%
x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
```

#### 2. Encoder 前向传播
```python
def forward_encoder(self, x, mask_ratio):
    x = self.patch_embed(x)           # Patch Embedding
    x = x + self.pos_embed[:, 1:, :]  # 加位置编码（不含 CLS）
    x, mask, ids_restore = self.random_masking(x, mask_ratio)  # 掩码
    cls_token = self.cls_token + self.pos_embed[:, :1, :]       # CLS + 位置
    x = torch.cat((cls_tokens, x), dim=1)                       # 拼接
    for blk in self.blocks: x = blk(x)                          # Transformer
    x = self.norm(x)
    return x, mask, ids_restore
```

#### 3. Decoder 前向传播
```python
def forward_decoder(self, x, ids_restore):
    x = self.decoder_embed(x)  # 线性投影到 decoder 维度
    # 补回 mask token
    mask_tokens = self.mask_token.repeat(...)
    x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # 去掉 CLS，拼接 mask
    x_ = torch.gather(x_, dim=1, index=...)              # unshuffle 恢复顺序
    x = torch.cat([x[:, :1, :], x_], dim=1)              # 加回 CLS
    x = x + self.decoder_pos_embed                        # 加 decoder 位置编码
    for blk in self.decoder_blocks: x = blk(x)            # Transformer
    x = self.decoder_pred(x)                               # 线性投影到像素
    x = x[:, 1:, :]                                        # 去掉 CLS
    return x
```

#### 4. 位置编码
- 使用固定的 2D sinusoidal 位置编码（不可学习）
- encoder 和 decoder 各有独立的位置编码
- 微调时支持位置编码插值（`interpolate_pos_embed`），用于适配不同输入分辨率

#### 5. 模型变体
| 变体 | Encoder 深度 | Encoder 维度 | Decoder 深度 | Decoder 维度 | Patch 大小 | 参数量 |
|------|-------------|-------------|-------------|-------------|-----------|--------|
| ViT-B | 12 | 768 | 8 | 512 | 16 | ~86M (encoder) |
| ViT-L | 24 | 1024 | 8 | 512 | 16 | ~304M (encoder) |
| ViT-H | 32 | 1280 | 8 | 512 | 14 | ~632M (encoder) |

所有变体的 decoder 都是 8 层、512 维、16 头（统一的轻量设计）。

#### 6. 训练技巧
- **梯度累积**：通过 `accum_iter` 支持大 batch size
- **混合精度**：`torch.cuda.amp.autocast()` 加速训练
- **分布式训练**：基于 `torch.distributed` 的 DDP
- **Layer-wise lr decay**：微调时使用，靠近输入的层学习率更低（decay=0.65/0.75）

## 与当前研究的关联

### 对视觉表征学习的影响
1. **证明了 masked image modeling 的有效性**：启发了后续 SimMIM、ConvMAE、VideoMAE 等大量工作
2. **高掩码比例的洞察**：揭示了视觉信号的信息密度远低于语言，需要更激进的掩码策略
3. **像素级重建 vs 语义 token**：与 BEiT（使用 dVAE tokenizer）形成对比，证明简单的像素回归就能获得优秀表征
4. **非对称设计范式**：encoder 重、decoder 轻的架构成为后续工作的标准模式

### 对下游任务的启发
- MAE 预训练的 ViT 可作为通用视觉 backbone，适用于分类、检测、分割等任务
- 像素级重建预训练天然适合密集预测任务
- 预训练后 decoder 可丢弃，不增加推理开销

### 局限性
- 主要在 ImageNet 上验证，更大规模数据集的实验较少
- 对于小模型（ViT-S），MAE 的优势不如大模型明显
- 像素级重建目标与高级语义任务存在 gap，后续工作（如 MAE v2）尝试用特征级重建弥补
