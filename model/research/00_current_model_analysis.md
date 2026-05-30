# 当前模型分析：PhysicsObjectGraphPredictor

## 1. 整体架构

```
Input Batch
    │
    ▼
┌─────────────────────────────────────┐
│  PhysicsObjectEncoder (双流编码器)    │
│  ├─ VisualEncoder (CNN)             │
│  ├─ MaskedROIPooler (ROI池化)       │
│  ├─ PhysicsEncoder (属性+状态)       │
│  └─ FusionProj (融合投影)            │
└─────────────────────────────────────┘
    │ tokens: [B, T, N, 128]
    ▼
┌─────────────────────────────────────┐
│  ForceAwareGNN (力感知图神经网络)     │
│  ├─ ForceAwareEdgeNetwork (边特征)   │
│  └─ GNNSingleLayer x 2 (消息传递)    │
└─────────────────────────────────────┘
    │ interacted: [B, T, N, 128]
    ▼
┌─────────────────────────────────────┐
│  TemporalGRU (时序预测)              │
│  ├─ encode_history (编码历史)        │
│  └─ decode_future (自回归解码)       │
└─────────────────────────────────────┘
    │ future_tokens: [B, Tp, N, 128]
    ▼
┌─────────────────────────────────────┐
│  MultiHeadDecoder (多头解码器)       │
│  ├─ StateHead → state_pred [16]     │
│  ├─ CollisionHead → logits [N,N]    │
│  ├─ MaskHead → mask_logits [H,W]    │
│  └─ RGBDecoder → rgb_pred [3,H,W]   │
└─────────────────────────────────────┘
```

## 2. 各模块详细分析

### 2.1 VisualEncoder

- **结构**：4层 Conv2d(stride=2) + GroupNorm + GELU
- **通道**：3→32→64→128→128
- **下采样**：16x (128→8)
- **输入**：[B, T, 3, 128, 128]
- **输出**：[B, T, 128, 8, 8]

**优势**：
- 轻量，参数少
- GroupNorm 对 batch size 不敏感

**不足**：
- 无预训练权重，特征质量受限
- 固定感受野（3x3），大物体整体信息可能遗漏
- 无残差连接，深层特征可能退化
- 无注意力机制，所有空间位置同等对待

### 2.2 MaskedROIPooler

- **方式**：Masked Average Pooling（非 bbox ROI）
- **流程**：mask 下采样到特征图分辨率 → 加权平均 → 线性投影
- **输入**：feat_map [BT, C, 8, 8], masks [BT, N, 128, 128]
- **输出**：[BT, N, 128]

**优势**：
- 利用 GT mask 做精确的物体级特征提取
- 比 bbox ROI pooling 更适合不规则形状

**不足**：
- 依赖 GT mask，推理时如果没有 mask 就无法使用
- nearest 插值 resize mask 可能有量化误差
- 无 soft attention 机制

### 2.3 PhysicsEncoder

- **属性编码**：Linear(14→64) + LN + GELU + Linear(64→64) + LN + GELU
- **状态编码**：Linear(16→64) + LN + GELU + Linear(64→64) + LN + GELU
- **融合**：Linear(128→128) + LN + GELU
- **输出**：[B, T, N, 128]

**优势**：
- 显式编码物理属性（质量、摩擦、几何）
- 静态属性跨时间步共享，动态状态逐帧编码

**不足**：
- 简单 MLP，无物理先验约束
- 位置/速度归一化是固定的（pos_scale=5, vel_scale=10），可能不适合所有场景
- 无旋转（四元数）的专门编码

### 2.4 ForceAwareGNN

- **边特征**：14维（force_ij[3] + force_ji[3] + rel_pos[3] + rel_vel[3] + dist[1] + force_mag[1]）→ 投影到 64 维
- **消息函数**：MLP(node_i, node_j, edge_ij) → message
- **聚合**：mean（带 valid mask）
- **更新**：GRUCell(old_state, aggregated_messages)
- **层数**：2层

**优势**：
- 显式利用力矩阵作为边特征（物理先验强）
- GRU 门控更新，梯度流稳定
- N<=7 时 dense 计算高效

**不足**：
- mean 聚合对所有邻居等权，无法区分重要性
- 边特征固定 14 维，无法学习未显式编码的关系
- 边特征不随层更新，每层复用相同 edge_feat
- 2层 MP 对应 2-hop 邻域，远距离物体信息需间接传递
- 无注意力加权机制

### 2.5 TemporalGRU

- **结构**：2层 GRU（input=128, hidden=128）
- **编码**：[B, Th, N, D] → [B*N, Th, D] → GRU → h_last
- **解码**：自回归，每步输出作为下一步输入
- **输出**：[B, Tp, N, 128]

**优势**：
- 比 Transformer 更轻量，显存友好
- 显式递归，天然适合物理序列
- 自回归解码捕捉时序依赖

**不足**：
- 无注意力机制，可能遗漏长距离依赖
- 自回归解码误差累积（exploding/banana problem）
- 无 teacher forcing 训练策略
- 无时序注意力（cross-attention to history）

### 2.6 MultiHeadDecoder

**StateHead**：
- 3层 MLP：128→256→256→16
- 简单直接，无物理约束

**CollisionHead**：
- 输入：concat(node_i, node_j, |node_i - node_j|) = 3*128 = 384
- MLP：384→64→1
- 无时序信息（只用当前帧 token）

**MaskHead**：
- Linear → reshape → 4层 ConvTranspose2d（上采样 16x）
- token→8x8→16x32→32x64→64x128

**RGBDecoder**：
- 每物体 appearance_head → 3维 RGB 颜色
- 背景：mean token → ConvTranspose2d 解码
- 合成：sum(mask * appearance) + bg * (1 - total_mask)

**优势**：
- 多头并行，各任务独立
- RGB 用 mask compositing 而非全像素生成，更高效

**不足**：
- StateHead 无物理约束（如能量守恒）
- CollisionHead 无时序信息
- RGBDecoder 的 appearance 是静态的（每物体固定颜色），无法表达纹理变化
- MaskHead 无 skip connection

### 2.7 PhysicsLoss

- **RGB**：L1 + 0.5*MSE
- **State**：Smooth L1，带 per-component 权重
- **Collision**：Focal BCE（gamma=2.0）
- **Mask**：BCE + Dice
- **总权重**：rgb=1.0, state=0.5, collision=0.5, mask=0.3

**优势**：
- 多任务学习，各损失互补
- Focal loss 处理正负样本不平衡
- Dice loss 处理 mask 类别不平衡
- Per-component state 权重（位置>速度>角速度>力）

**不足**：
- 无感知损失（LPIPS）
- 无时序一致性损失
- 无物理一致性正则（如牛顿第三定律）
- 无 SSIM/FVD 等视频质量指标
- 各损失权重手动设定，无自适应机制

## 3. 训练策略

| 项目 | 当前实现 |
|------|----------|
| 优化器 | AdamW (lr=1e-3, wd=1e-4) |
| 学习率调度 | CosineAnnealingLR |
| 梯度裁剪 | max_norm=1.0 |
| 混合精度 | AMP (可选) |
| OOM 处理 | 自动减半 batch size，再缩小模型 |
| Checkpoint | 保存 best.pt + 每 epoch |
| Resume | 支持 --resume |

**缺失**：
- 无课程学习（curriculum learning）
- 无梯度累积
- 无 EMA（Exponential Moving Average）
- 无 early stopping
- 无数据增强

## 4. 数据使用情况

| 数据字段 | 使用方式 |
|----------|----------|
| RGB frames | VisualEncoder 输入 + RGB loss target |
| Object masks | ROI pooling + Mask loss target |
| Static attrs | PhysicsEncoder 输入 |
| Dynamic state | PhysicsEncoder 输入 + GNN 边特征 |
| Force matrix | GNN 边特征 + Collision labels |
| Video metadata | 仅数据加载 |

**未使用**：深度图、相机参数、四元数（编码但未专门建模）

## 5. 模型规模

| 配置 | fused_dim | 参数量 | batch_size |
|------|-----------|--------|------------|
| tiny (smoke) | 32 | 132,032 | 2 |
| small_8gb | 96 | ~363,904 | 4 |
| 默认 | 128 | ~500K+ | 4 |

## 6. 已知问题（from review）

1. mask_loss 原始值 ~40K-60K，被 weight=0.001 压制
2. attr_dim=14 但只有 13 个值被填充
3. 无 DINOv2 预训练视觉特征
4. 训练仅验证了 50 步，未达到收敛

## 7. 与基线 PhysicsVideoPredictor 的对比

| 维度 | 基线 | 当前模型 |
|------|------|----------|
| 视觉编码器 | LightCNN | LightCNN（类似） |
| 物体特征 | MaskROIPool | MaskedROIPool（类似） |
| 交互模块 | Dense MLP MP | ForceAwareGNN（改进） |
| 时序模块 | Transformer Enc-Dec | GRU（不同选择） |
| 解码器 | 与当前类似 | MultiHeadDecoder |
| 损失函数 | 类似 | PhysicsLoss |

**关键区别**：基线用 Transformer 做时序，当前模型用 GRU；基线用 Dense MLP 做交互，当前模型用 GNN。
