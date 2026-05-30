# DETR: End-to-End Object Detection with Transformers

## 基本信息
- **作者**: Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko
- **机构**: Facebook AI
- **年份**: 2020
- **会议**: ECCV 2020
- **论文链接**: https://arxiv.org/abs/2005.12872
- **代码**: https://github.com/facebookresearch/detr

## 核心贡献

1. **将目标检测重新定义为直接的集合预测问题**，去除了传统检测器中的手工组件（anchor 生成、NMS 后处理、proposal 等）
2. **提出 DETR (DEtection TRansformer)**：结合 CNN backbone + Transformer encoder-decoder 架构，通过一组学习到的 object queries 并行输出最终的检测结果集合
3. **基于二部图匹配 (bipartite matching) 的集合损失**：通过匈牙利算法将预测框与 GT 框一一匹配，保证端到端训练且无需 NMS
4. **可自然扩展到全景分割 (Panoptic Segmentation)**，在 stuff 和 thing 类别上用统一方式处理
5. 在 COCO 数据集上达到与高度优化的 Faster R-CNN 相当的性能，且在大物体上显著优于后者（+7.8 APL）

## 模型架构

### 整体流程

```
输入图像 (3×H₀×W₀)
    ↓
CNN Backbone (ResNet-50/101)
    ↓
特征图 f (C×H×W), 其中 H=W=H₀/32
    ↓
1×1 Conv 降维 → z₀ (d×H×W), d=256
    ↓
展平为序列 + 位置编码
    ↓
Transformer Encoder (6层, self-attention)
    ↓
Transformer Decoder (6层) ← Object Queries (N=100个可学习嵌入)
    ↓
FFN 预测头 → N个预测 (类别 + 边界框)
    ↓
匈牙利匹配 + 损失计算
```

### CNN Backbone

- 使用 ImageNet 预训练的 ResNet-50 或 ResNet-101
- **冻结 BatchNorm** 层（使用 `FrozenBatchNorm2d`）
- 输出特征图分辨率 H×W = H₀/32 × W₀/32，通道数 C=2048
- 可选 DC5 (dilated C5) 变体：将最后一个 stage 的 stride 替换为 dilation，分辨率提高 2 倍，有利于小物体检测，但计算量增加约 2 倍
- 位置编码：使用正弦位置编码 (sine) 或学习的位置编码 (learned)，加到每个 attention 层的输入上

### Transformer Encoder

- 标准 Transformer encoder 架构：每层包含 **multi-head self-attention** + **FFN (前馈网络)**
- 输入：展平的特征图序列 (HW×N×d)，加上正弦位置编码
- Self-attention 在全局范围内推理，能分离不同实例，为 decoder 提供全局上下文
- 消融实验表明：encoder 层数从 0→6，AP 从 36.7 提升到 40.6，大物体 AP 提升尤为显著（+6.0）

### Transformer Decoder

- 标准 Transformer decoder，包含三层 attention：
  1. **Self-attention**：在 N 个 object queries 之间建模关系，抑制重复预测
  2. **Cross-attention (encoder-decoder attention)**：queries 关注 encoder 输出的图像特征
  3. **FFN**：逐位置的前馈网络
- **关键区别**：与原始 Transformer 的自回归解码不同，DETR **并行解码** N 个对象
- **Object Queries**：N 个可学习的位置编码嵌入（默认 N=100），作为 decoder 的输入
  - 每个 query 通过 decoder 后独立预测一个 (类别, 边界框) 或 "无物体" (∅)
  - 可视化显示每个 slot 学会了专注于特定的空间区域和框尺寸
- **辅助解码损失 (Auxiliary Decoding Losses)**：在每个 decoder 层后都添加预测 FFN 和匈牙利损失（所有 FFN 共享参数），帮助模型输出正确数量的物体

### Object Queries 的本质

Object queries 是 **可学习的位置编码嵌入**，形状为 `[N, d]`（N=100, d=256）。在 decoder 中：
- 作为 `query_pos` 加到 decoder 输入 `tgt`（初始化为全零）上
- 在 self-attention 和 cross-attention 的 query/key 中都加入
- 每个 query 学会关注图像中特定的空间区域和物体尺度

### 预测头 (FFN)

- **分类头**：`nn.Linear(d, num_classes + 1)` → softmax，包含 "无物体" 类别
- **回归头**：3 层 MLP（d→d→4）+ sigmoid，输出归一化的 (center_x, center_y, width, height)

### 二部图匹配 (Bipartite Matching)

**核心思想**：将预测集合与 GT 集合通过匈牙利算法做最优一对一匹配。

**匹配代价**：
```
L_match(y_i, ŷ_σ(i)) = -1_{c_i≠∅} · p̂_σ(i)(c_i) + 1_{c_i≠∅} · L_box(b_i, b̂_σ(i))
```
- 分类代价：`-p̂(c_i)` （类别概率，非 log 概率，使与框损失量级可比）
- 框代价：`L_box = λ_L1 · ||b - b̂||₁ + λ_iou · L_iou(b, b̂)`
  - 默认权重：`set_cost_class=1, set_cost_bbox=5, set_cost_giou=2`

**匈牙利损失**：
```
L_Hungarian(y, ŷ) = Σ_i [-log p̂_σ̂(i)(c_i) + 1_{c_i≠∅} · L_box(b_i, b̂_σ̂(i))]
```
- 对 "无物体" 类别的 log 概率项降权（`eos_coef=0.1`），处理类别不平衡
- 框损失用 L1 + GIoU 的线性组合，按 batch 内物体数归一化

### 全景分割扩展

- 在 decoder 输出上添加 mask head
- 使用多头 attention 将 decoder 嵌入投影到 encoder 特征图上，生成低分辨率 attention 热图
- 通过 FPN 风格的 CNN 上采样到 stride-4 分辨率
- 每个 mask 独立监督，使用 DICE loss + Focal loss
- 最终通过 per-pixel argmax 合并 mask，无需重叠处理启发式

## 训练细节

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| Transformer 学习率 | 10⁻⁴ |
| Backbone 学习率 | 10⁻⁵ |
| 权重衰减 | 10⁻⁴ |
| 权重初始化 | Xavier init（Transformer），ImageNet 预训练（Backbone） |
| BatchNorm | 冻结 |
| 默认训练轮数 | 300 epochs（消融），500 epochs（与 Faster R-CNN 对比） |
| 学习率衰减 | 在 200/400 epoch 时除以 10 |
| 数据增强 | 短边 480-800，长边 ≤1333；随机裁剪（概率 0.5）+1 AP |
| Dropout | 0.1 |
| Object queries 数量 | 100 |
| Encoder/Decoder 层数 | 6/6 |
| 隐藏维度 d | 256 |
| Attention heads | 8 |
| FFN 中间维度 | 2048 |
| 梯度裁剪 | max_norm=0.1 |
| 硬件 | 16×V100，4 images/GPU，batch_size=64 |
| 训练时间 | 300 epochs 约 3 天 |

**推理技巧**：当 slot 预测为 "无物体" 时，用第二高分类别的分数替换，AP 提升约 2 点。

## 实验结果

### COCO 目标检测 (val set)

| 模型 | Backbone | GFLOPS | FPS | 参数量 | AP | AP₅₀ | AP₇₅ | APS | APM | APL |
|------|----------|--------|-----|--------|-----|------|------|-----|-----|-----|
| Faster R-CNN-FPN | R50 | 180 | 26 | 42M | 40.2 | 61.0 | 43.8 | 24.2 | 43.5 | 52.0 |
| Faster R-CNN-FPN+ | R50 | 180 | 26 | 42M | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4 |
| **DETR** | **R50** | **86** | **28** | **41M** | **42.0** | **62.4** | **44.2** | **20.5** | **45.8** | **61.1** |
| DETR-DC5 | R50 | 187 | 12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 |
| DETR-R101 | R101 | 152 | 20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | 61.8 |
| DETR-DC5-R101 | R101 | 253 | 10 | 60M | 44.9 | 64.7 | 47.7 | 23.7 | 49.5 | 62.3 |

**关键发现**：
- DETR 与 Faster R-CNN 总体 AP 相当，**大物体 APL 显著领先**（61.1 vs 53.4，+7.7），得益于 Transformer 的全局非局部计算
- **小物体 APS 落后**（20.5 vs 26.6，-6.1），缺少类似 FPN 的多尺度特征融合
- DETR 计算效率更高（86 vs 180 GFLOPS），但小物体变体 DC5 计算量增大

### COCO 全景分割 (val set)

| 模型 | Backbone | PQ | PQ_th | PQ_st |
|------|----------|-----|-------|-------|
| PanopticFPN++ | R50 | 42.4 | 49.2 | 32.3 |
| UPSNet | R50 | 42.5 | 48.6 | 33.4 |
| **DETR** | **R50** | **43.4** | **48.2** | **36.3** |
| DETR-DC5 | R50 | 44.6 | 49.4 | 37.3 |
| DETR-R101 | R101 | 45.1 | 50.5 | 37.0 |

- DETR 在 stuff 类上显著领先，归因于 encoder 的全局推理能力
- test-dev 上达到 46 PQ

### 消融实验摘要

| 实验 | 变量 | AP 变化 |
|------|------|---------|
| Encoder 层数 | 0→6 层 | 36.7→40.6 (+3.9) |
| Decoder 层 | 第1层→第6层 | 32.4→40.6 (+8.2) |
| FFN | 去除 FFN | -2.3 AP |
| 位置编码 | 无空间编码 | -7.8 AP |
| 损失组合 | 仅 GIoU（无 L1） | 39.9 (-0.7) |
| 损失组合 | 仅 L1（无 GIoU） | 35.8 (-4.8) |
| 训练时长 | 300→500 epochs | +1.5 AP |

## 代码实现细节

### 目录结构
```
code/
├── main.py              # 训练入口
├── engine.py            # 训练/评估循环
├── models/
│   ├── detr.py          # DETR 模型 + SetCriterion 损失 + PostProcess
│   ├── transformer.py   # Transformer encoder/decoder 实现
│   ├── matcher.py       # HungarianMatcher 匈牙利匹配
│   ├── backbone.py      # ResNet backbone + 位置编码包装
│   ├── position_encoding.py  # Sine/Learned 位置编码
│   └── segmentation.py  # 全景分割 mask head
├── datasets/            # COCO 数据加载
└── util/                # 工具函数 (misc, box_ops, plot_utils)
```

### 关键实现要点

**1. DETR 模型 (`models/detr.py`)**
```python
class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss):
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)       # 3层MLP
        self.query_embed = nn.Embedding(num_queries, hidden_dim)   # 可学习object queries
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, 1)  # 1×1 conv降维

    def forward(self, samples):
        features, pos = self.backbone(samples)           # 提取特征+位置编码
        src, mask = features[-1].decompose()              # 取最后一层特征
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])
        outputs_class = self.class_embed(hs)              # [layers, B, N, num_classes+1]
        outputs_coord = self.bbox_embed(hs).sigmoid()     # [layers, B, N, 4], 归一化坐标
        # 返回最后一层结果 + 辅助损失
```

**2. 匈牙利匹配 (`models/matcher.py`)**
```python
class HungarianMatcher(nn.Module):
    def forward(self, outputs, targets):
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        
        cost_class = -out_prob[:, tgt_ids]              # 分类代价：负概率
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # L1 距离
        cost_giou = -generalized_box_iou(...)             # 负 GIoU
        
        C = cost_bbox * 5 + cost_class * 1 + cost_giou * 2  # 加权求和
        # 按 batch 拆分，对每个样本独立求解匈牙利算法
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
```

**3. 损失计算 (`SetCriterion`)**
```python
class SetCriterion(nn.Module):
    def forward(self, outputs, targets):
        # 1) 对最后一层输出做匈牙利匹配
        indices = self.matcher(outputs_without_aux, targets)
        # 2) 计算分类损失 (cross entropy) + 框损失 (L1 + GIoU)
        # 3) 对辅助层 (aux_outputs) 重复此过程
        for i, aux_outputs in enumerate(outputs['aux_outputs']):
            indices = self.matcher(aux_outputs, targets)  # 每层独立匹配
            # 计算损失...
```

**4. Transformer (`models/transformer.py`)**
```python
class Transformer(nn.Module):
    def forward(self, src, mask, query_embed, pos_embed):
        # src: [B, C, H, W] → flatten → [HW, B, C]
        # query_embed: [N, C] → repeat to [N, B, C]
        tgt = torch.zeros_like(query_embed)  # decoder 输入初始化为零
        memory = self.encoder(src, pos=pos_embed)
        hs = self.decoder(tgt, memory, pos=pos_embed, query_pos=query_embed)
        # hs: [layers, B, N, C]（return_intermediate_dec=True）
```

**5. 位置编码 (`models/position_encoding.py`)**
- Sine 编码：2D 图像上累积非 mask 区域，归一化后用 sin/cos 编码，类似原始 Transformer 推广到 2D
- Learned 编码：使用 `nn.Embedding(50, num_pos_feats)` 分别编码行列

**6. Backbone (`models/backbone.py`)**
- 使用 torchvision 的 ResNet，通过 `IntermediateLayerGetter` 提取指定层
- `FrozenBatchNorm2d`：冻结 BN 统计量，避免小 batch 统计不准
- `Joiner`：将 backbone 与 position encoding 组合，返回特征和对应的位置编码

### 训练超参数 (main.py 中的默认值)
```
--lr 1e-4          # Transformer 学习率
--lr_backbone 1e-5 # Backbone 学习率 (10x smaller)
--weight_decay 1e-4
--epochs 300
--lr_drop 200      # 学习率衰减点
--hidden_dim 256
--nheads 8
--enc_layers 6
--dec_layers 6
--dim_feedforward 2048
--num_queries 100
--dropout 0.1
--set_cost_class 1   # 匹配代价：分类权重
--set_cost_bbox 5    # 匹配代价：L1 权重
--set_cost_giou 2    # 匹配代价：GIoU 权重
--bbox_loss_coef 5   # 损失：L1 权重
--giou_loss_coef 2   # 损失：GIoU 权重
--eos_coef 0.1       # 无物体类别的分类权重（降权）
```

## 与当前研究的关联

### DETR 的优势与局限

**优势**：
- 端到端，无需 NMS 和 anchor 设计，概念简洁
- 全局推理能力强，大物体检测优秀
- 可自然扩展到全景分割、实例分割等任务
- Object queries 的概念影响深远（后续 Deformable DETR、DAB-DETR、DINO 等）

**局限**：
- 训练收敛慢（需 300-500 epochs，而 Faster R-CNN 约 36 epochs）
- 小物体检测性能较差（缺乏多尺度特征融合）
- Object queries 数量固定（N=100），无法自适应
- 每个 decoder 层的匈牙利匹配独立计算，未充分利用层间信息

### 对后续研究的影响

DETR 开创了 Transformer 在目标检测中的应用，直接催生了一系列改进工作：
- **Deformable DETR**：用可变形 attention 替代全局 attention，加速收敛并提升小物体性能
- **DAB-DETR**：将 object queries 动态化为 anchor box
- **DN-DETR**：引入去噪训练，稳定匈牙利匹配
- **DINO**：综合以上改进，达到 SOTA
- **Co-DETR**：进一步优化 encoder 设计

### 对 Slot-based 模型的启发

DETR 的 object queries 与 Slot Attention 等概念有深刻联系：
- Object queries 本质上是一种 **可学习的 slot 表示**，每个 slot 负责检测一个物体
- 匈牙利匹配确保 slot 间的分工不重复
- 这种 "固定数量 slot + 匹配损失" 的范式可以推广到无监督物体发现、场景理解等任务
