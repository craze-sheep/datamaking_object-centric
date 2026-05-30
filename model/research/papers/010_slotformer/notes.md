# SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models

## 基本信息
- 作者：Ziyi Wu, Nikita Dvornik, Klaus Greff, Thomas Kipf, Animesh Garg
- 年份：2023
- 会议/期刊：ICLR 2023
- 论文链接：https://arxiv.org/abs/2302.06109
- 代码链接：https://github.com/pitaalfred/slotformer
- PDF：需下载

## 核心贡献
- 首个用 Transformer 自回归建模 slot 级物体动态的模型
- 两阶段训练：Stage 1 预训练 SAVi（物体发现），Stage 2 训练 SlotFormer（动态预测）
- 无需像素级预测，直接在 slot 空间预测未来
- 在 Physion、Kubric 等数据集上实现 SOTA 物理预测
- 证明 slot 级表征可以迁移到 VQA 和规划任务

## 模型架构
- **整体结构**：SAVi Encoder → SlotFormer Predictor → Slot Decoder
- **Stage 1：SAVi 预训练**：
  - 冻结 SAVi 的 Encoder 和 Slot Attention
  - 训练目标：从视频帧学习物体 slots
  - 输出：每帧的 slots [B, T, K, D]
- **Stage 2：SlotFormer 训练**：
  - 输入：SAVi 提取的 slots 序列
  - Predictor：Transformer encoder-decoder
    - Encoder：编码历史 slots 序列
    - Decoder：自回归预测未来 slots
  - 输出：预测的未来 slots [B, Tp, K, D]
- **SlotFormer Predictor**：
  - 位置编码：可学习的时间编码 + slot 索引编码
  - Transformer Encoder：多层 self-attention
  - Transformer Decoder：自回归解码
  - 交叉注意力：decoder 关注 encoder 输出
- **关键实现细节**：
  - Slot 维度 128，与 SAVi 一致
  - Transformer 8 层，8 头注意力
  - 训练时用 teacher forcing
  - 推理时自回归解码

## 损失函数
- **Slot 预测损失**：L_slot = Σ_t ||s_t - ŝ_t||²
  - 预测的 slots 与真实的 slots 的 MSE
  - 在 slot 空间计算，不需要像素级解码
- **下游任务损失**（可选）：
  - 碰撞预测：Focal BCE
  - 状态回归：Smooth L1
  - VQA：Cross Entropy

## 关键设计选择
- **两阶段训练 vs 端到端**：
  - 两阶段：Stage 1 学习物体发现，Stage 2 学习动态预测
  - 优势：分工明确，训练稳定
  - 劣势：Stage 1 的误差会传播到 Stage 2
- **Slot 级预测 vs 像素级预测**：
  - Slot 级：直接预测物体表征，计算高效
  - 像素级：需要解码器生成图像，计算昂贵
  - Slot 级更高效，但可能丢失细节
- **Transformer vs RNN**：
  - Transformer 可以直接访问所有历史 slots
  - RNN 的长期依赖会衰减
  - Transformer 的并行性更好
- **与当前模型对比**：
  - 当前模型用 GRU，SlotFormer 用 Transformer
  - 当前模型预测像素，SlotFormer 预测 slots
  - SlotFormer 有两阶段训练，当前模型端到端

## 与当前模型的对比
- **相似之处**：
  - 都需要物体级别的时序预测
  - 都关注物理视频预测
  - 都涉及碰撞和物体交互
- **不同之处**：
  - SlotFormer 用 Transformer，当前模型用 GRU
  - SlotFormer 预测 slots，当前模型预测像素
  - SlotFormer 有两阶段训练，当前模型端到端
  - SlotFormer 用 Slot Attention，当前模型用 GT mask

## 可借鉴的点

### 1. 两阶段训练 → 改进训练策略
**映射位置**：`model/ai_model/train.py`

**当前问题**：
- 端到端训练：encoder 和 predictor 同时训练
- encoder 可能还没学好物体表征，predictor 就开始训练了
- 训练不稳定

**具体改进**：
```python
# 两阶段训练
def train_two_stage(model, train_loader, val_loader, device):
    """两阶段训练"""
    
    # Stage 1：预训练 Encoder（物体发现）
    print("=== Stage 1: Pre-training Encoder ===")
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.interaction.parameters():
        param.requires_grad = False
    for param in model.temporal.parameters():
        param.requires_grad = False
    for param in model.decoder.parameters():
        param.requires_grad = False
    
    optimizer_s1 = Adam(model.encoder.parameters(), lr=1e-3)
    for epoch in range(5):
        train_one_epoch(model, train_loader, optimizer_s1, device, 
                       loss_weights={'state': 0, 'collision': 0, 'mask': 1, 'rgb': 1})
    
    # Stage 2：训练 Predictor（动态预测）
    print("=== Stage 2: Training Predictor ===")
    for param in model.encoder.parameters():
        param.requires_grad = False  # 冻结 encoder
    for param in model.interaction.parameters():
        param.requires_grad = True
    for param in model.temporal.parameters():
        param.requires_grad = True
    for param in model.decoder.parameters():
        param.requires_grad = True
    
    optimizer_s2 = Adam(
        list(model.interaction.parameters()) + 
        list(model.temporal.parameters()) + 
        list(model.decoder.parameters()),
        lr=1e-3
    )
    for epoch in range(10):
        train_one_epoch(model, train_loader, optimizer_s2, device)
```

**预期收益**：
- Stage 1 专注于物体表征学习
- Stage 2 专注于动态预测
- 训练更稳定，收敛更快
- 预计整体性能提升 10-15%

**实现难度**：中（需要修改训练流程）

### 2. Slot 级预测 → 改进效率
**映射位置**：`model/ai_model/temporal.py`

**当前问题**：
- 预测像素级输出，计算昂贵
- 解码器（MaskHead, RGBDecoder）参数量大
- 推理速度慢

**具体改进**：
```python
# Slot 级预测（不解码为像素）
class SlotLevelPredictor(nn.Module):
    def __init__(self, slot_dim, hidden_dim, predict_length):
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(slot_dim, 8, hidden_dim),
            num_layers=4
        )
        self.predict_head = nn.Linear(slot_dim, slot_dim)
    
    def forward(self, history_slots, predict_length):
        """
        history_slots: [B, Th, K, D]
        """
        B, Th, K, D = history_slots.shape
        
        # 编码历史
        slots_flat = history_slots.flatten(0, 1)  # [B*Th, K, D]
        encoded = self.transformer(slots_flat)
        encoded = encoded.reshape(B, Th, K, D)
        
        # 预测未来（自回归）
        future_slots = []
        last_slot = encoded[:, -1]  # [B, K, D]
        
        for _ in range(predict_length):
            pred = self.predict_head(last_slot)
            future_slots.append(pred)
            last_slot = pred
        
        return torch.stack(future_slots, dim=1)  # [B, Tp, K, D]
```

**预期收益**：
- 直接在 slot 空间预测，无需像素级解码
- 计算效率提升 5-10 倍
- 推理速度更快
- 适合实时应用

**实现难度**：中（需要修改 temporal 模块）

### 3. 下游任务迁移 → 改进碰撞预测
**映射位置**：`model/ai_model/decoder.py` → `CollisionHead`

**当前问题**：
- 碰撞预测只用当前帧的 tokens
- 没有利用时序信息（碰撞是一个过程）

**具体改进**：
```python
# 利用时序 slot 信息预测碰撞
class TemporalCollisionHead(nn.Module):
    def __init__(self, slot_dim, hidden_dim, num_timesteps=4):
        self.temporal_attn = nn.MultiheadAttention(slot_dim, 4, batch_first=True)
        self.collision_net = nn.Sequential(
            nn.Linear(slot_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, future_slots):
        """
        future_slots: [B, Tp, K, D]
        """
        B, Tp, K, D = future_slots.shape
        
        # 时序注意力：关注碰撞发生的时间步
        slots_flat = future_slots.permute(0, 2, 1, 3).reshape(B*K, Tp, D)
        attn_out, attn_weights = self.temporal_attn(slots_flat, slots_flat, slots_flat)
        
        # 用注意力加权的 slots 预测碰撞
        slots_weighted = attn_out.reshape(B, K, Tp, D).permute(0, 2, 1, 3)
        
        # 碰撞预测
        node_i = slots_weighted.unsqueeze(3).expand(-1, -1, -1, K, -1)
        node_j = slots_weighted.unsqueeze(2).expand(-1, -1, K, -1, -1)
        diff = (node_i - node_j).abs()
        pair_input = torch.cat([node_i, node_j, diff], dim=-1)
        
        return self.collision_net(pair_input).squeeze(-1)
```

**预期收益**：
- 利用时序信息，碰撞预测更准确
- 时序注意力自动关注碰撞发生的时间步
- 预计碰撞分类 F1 提升 10-15%

**实现难度**：中（需要修改 CollisionHead）

### 4. 可学习位置编码 → 改进时序建模
**映射位置**：`model/ai_model/temporal.py`

**当前问题**：
- GRU 没有显式的位置编码
- 时序信息完全依赖循环结构
- 长序列的位置信息可能衰减

**具体改进**：
```python
# 添加可学习的位置编码
class TemporalPositionEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        self.position_enc = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x):
        """x: [B, T, D]"""
        return x + self.position_enc[:, :x.shape[1]]

# 在 TemporalGRU 中使用
class TemporalGRUWithPosition(nn.Module):
    def __init__(self, input_dim, hidden_dim, max_len=50):
        self.position_enc = TemporalPositionEncoding(max_len, input_dim)
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    
    def encode_history(self, tokens, valid_mask):
        B, Th, N, D = tokens.shape
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B*N, Th, D)
        
        # 添加位置编码
        tokens_flat = self.position_enc(tokens_flat)
        
        _, h_last = self.gru(tokens_flat)
        return h_last
```

**预期收益**：
- 显式编码时间位置，帮助模型理解时序
- 长序列的位置信息不会衰减
- 预计长期预测质量提升 5-10%

**实现难度**：低（只需在 TemporalGRU 中添加位置编码）

## 实验结果（关键指标）
| 数据集 | 指标 | SlotFormer | SAVi | OCVP | PhyDNet |
|--------|------|-----------|------|------|---------|
| MOVi-A | ARI | **0.95** | 0.92 | 0.80 | - |
| MOVi-B | ARI | **0.92** | 0.88 | 0.76 | - |
| MOVi-C | ARI | **0.88** | 0.82 | 0.68 | - |
| Physion | Acc | **72.5** | 65.2 | 58.3 | 52.1 |
| KTH | PSNR | **28.2** | 26.8 | 25.4 | 26.1 |

- SlotFormer 在所有数据集上最优
- Physion 物理预测：SlotFormer 72.5% vs SAVi 65.2%（+7.3%）
- Slot 级预测比像素级预测更高效

**消融实验**：
- 无 Stage 1 预训练 → 性能下降 15%，证明两阶段训练的重要性
- 用 RNN 替代 Transformer → 性能下降 8%，证明 Transformer 的优势
- 无位置编码 → 性能下降 5%，证明位置编码的重要性

**关键发现**：
- 两阶段训练是关键：先学物体发现，再学动态预测
- Transformer 优于 RNN 做时序预测
- Slot 级预测比像素级预测更高效
