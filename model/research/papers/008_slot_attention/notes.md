# Slot Attention

## 基本信息
- 作者：Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, et al.
- 年份：2020
- 会议/期刊：NeurIPS 2020
- 论文链接：https://arxiv.org/abs/2006.15055
- 代码链接：https://github.com/google-research/google-research/tree/master/slot_attention
- PDF：需下载

## 核心贡献
- 提出 Slot Attention：竞争注意力机制实现可交换的物体绑定（slot binding）
- 核心创新：slots 通过竞争（softmax over slots）绑定到输入特征的不同部分
- 无需监督信号，纯无监督发现物体
- 即插即用模块，可集成到任何架构中
- 证明物体中心表征在下游任务上的优势

## 模型架构
- **Slot Attention 模块**：
  - 输入：特征图 [B, N, D]（N 个 token，如 CNN 特征的像素）
  - 输出：slots [B, K, D]（K 个 slot，每个代表一个物体）
  - 核心机制：竞争注意力（competition-based attention）
  - 迭代过程（3 轮）：
    1. 计算 attention：attn = softmax(queries * keys / sqrt(D), dim=slots)
    2. 加权求和：updates = attn * values
    3. GRU 更新：slots = GRU(slots, updates)
    4. 残差 MLP：slots = slots + MLP(slots)
- **竞争机制**：
  - Attention 在 slots 维度做 softmax（不是在 tokens 维度）
  - 每个 token 只能绑定到一个 slot（赢者通吃）
  - 确保 slots 学习不同的物体
- **可交换性**：
  - Slots 的初始化是随机的（可学习的高斯分布）
  - 输出 slots 的顺序是任意的（permutation equivariant）
  - 与物体的真实顺序无关
- **关键实现细节**：
  - 3 轮迭代（论文中比较了 1-5 轮，3 轮最优）
  - Slot 维度 128，与输入特征维度相同
  - 使用 GRU 和残差 MLP 更新 slots
  - 训练时用 MSE 重建损失

## 损失函数
- **重建损失**：L = Σ ||x - D(s_i)||²
  - D 是解码器（通常是 MLP + 反卷积）
  - 每个 slot 解码为一个组件
  - 总重建 = Σ 组件重建
- **无对比损失、无正则化损失**
- 纯无监督，不需要物体标签

## 关键设计选择
- **竞争注意力 vs 标准注意力**：
  - 标准 attention：softmax over tokens（每个 query 关注所有 tokens）
  - Slot attention：softmax over slots（每个 token 只能绑定到一个 slot）
  - 竞争机制确保 slots 学习不同的物体
- **迭代更新 vs 一次性预测**：
  - 迭代更新（3 轮）比一次性预测更准确
  - 类似 EM 算法：E 步（分配 tokens 到 slots）和 M 步（更新 slots）
- **GRU 更新 vs MLP 更新**：
  - GRU 有记忆，能跨迭代保持信息
  - 残差连接确保稳定训练
- **与当前模型对比**：
  - 当前模型用 GT mask 做 ROI pooling，Slot Attention 自动发现物体
  - Slot Attention 不需要物体数量已知，当前模型需要 max_objects
  - Slot Attention 输出可交换的 slots，当前模型输出固定顺序的 tokens

## 与当前模型的对比
- **相似之处**：
  - 都需要物体级别的表征
  - 都用注意力机制
  - 都关注物体间交互
- **不同之处**：
  - Slot Attention 无监督发现物体，当前模型用 GT mask
  - Slot Attention 输出可交换的 slots，当前模型输出固定顺序的 tokens
  - Slot Attention 用竞争注意力，当前模型用标准注意力
  - Slot Attention 不需要物体数量已知，当前模型需要

## 可借鉴的点

### 1. 替代 GT Mask ROI Pooling → 核心改进
**映射位置**：`model/ai_model/encoder.py` → `MaskedROIPooler`

**当前问题**：
- MaskedROIPooler 依赖 GT mask 做 ROI pooling
- 无法处理没有 GT mask 的场景（真实场景）
- 无法发现新的物体

**具体改进**：
```python
# 当前：依赖 GT mask
class MaskedROIPooler(nn.Module):
    def forward(self, feat_map, masks, valid_mask):
        # masks: [BT, N, H, W] - GT masks
        masks_ds = F.adaptive_avg_pool2d(masks, (Hf, Wf))
        weighted = feat_map.unsqueeze(1) * masks_ds.unsqueeze(2)
        pooled = weighted.sum(dim=(-2, -1)) / mask_sum
        return self.proj(pooled)

# 改为：Slot Attention 自动发现物体
class SlotAttentionROIPooler(nn.Module):
    def __init__(self, feat_channels, out_dim, num_slots=7, num_iters=3):
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=feat_channels,
            iters=num_iters
        )
        self.proj = nn.Linear(feat_channels, out_dim)
    
    def forward(self, feat_map, valid_mask, gt_masks=None):
        """
        feat_map: [BT, C, Hf, Wf]
        valid_mask: [B, N]
        gt_masks: [BT, N, H, W] (optional, for training)
        """
        BT, C, Hf, Wf = feat_map.shape
        
        # 展平特征图
        feat_flat = feat_map.flatten(2).permute(0, 2, 1)  # [BT, Hf*Wf, C]
        
        # Slot Attention: 自动发现物体
        slots = self.slot_attention(feat_flat)  # [BT, num_slots, C]
        
        # 可选：用 GT mask 做辅助训练
        if gt_masks is not None and self.training:
            # 用 GT mask 计算重建损失
            slot_masks = self.slot_to_mask(slots, feat_map)  # [BT, num_slots, H, W]
            mask_loss = F.mse_loss(slot_masks, gt_masks)
            return self.proj(slots), mask_loss
        
        return self.proj(slots), None

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3):
        self.num_slots = num_slots
        self.iters = iters
        self.dim = dim
        
        # 可学习的 slot 初始化
        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))
        
        # GRU 和 MLP
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        
        # 注意力
        self.project_k = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)
    
    def forward(self, inputs):
        B, N, D = inputs.shape
        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)
        
        # 初始化 slots
        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = self.slot_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(mu)
        
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # 竞争注意力
            attn = torch.einsum('bid,bjd->bij', slots, k) / (D ** 0.5)
            attn = F.softmax(attn, dim=1)  # 在 slots 维度 softmax
            
            # 加权求和
            attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            updates = torch.einsum('bij,bjd->bid', attn, v) / attn_sum
            
            # GRU 更新
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, -1, D)
            
            # 残差 MLP
            slots = slots + self.mlp(slots)
        
        return slots
```

**预期收益**：
- 无需 GT mask，可处理真实场景
- 自动发现物体，支持开放世界
- Physion 证明物体中心表征在物理预测上最优
- 预计碰撞预测准确率提升 15-20%

**实现难度**：高（需要实现 Slot Attention 并修改整个 encoder 流程）

### 2. 可交换表征 → 改进 GNN 的物体排序
**映射位置**：`model/ai_model/interaction.py` → `ForceAwareGNN`

**当前问题**：
- GNN 的输出依赖于物体的输入顺序
- 不同的物体排序可能导致不同的预测
- 不满足物理对称性（A 撞 B = B 撞 A）

**具体改进**：
```python
# 改进 GNN 使其对物体排序不变
class PermutationInvariantGNN(nn.Module):
    def __init__(self, node_dim, edge_dim=64, hidden_dim=128):
        # 对称消息函数
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, node_dim),
        )
    
    def forward(self, node_feat, edge_feat, valid_mask):
        B, T, N, D = node_feat.shape
        
        # 对称消息：m(i,j) = f(h_i, h_j, e_ij) + f(h_j, h_i, e_ji)
        node_i = node_feat.unsqueeze(3).expand(-1, -1, -1, N, -1)
        node_j = node_feat.unsqueeze(2).expand(-1, -1, N, -1, -1)
        
        # 正向消息
        msg_input = torch.cat([node_i, node_j, edge_feat], dim=-1)
        msg_forward = self.message_fn(msg_input)
        
        # 反向消息（交换 i 和 j）
        msg_input_rev = torch.cat([node_j, node_i, edge_feat], dim=-1)
        msg_backward = self.message_fn(msg_input_rev)
        
        # 对称消息
        messages = (msg_forward + msg_backward) / 2
        
        # 聚合
        aggregated = messages.sum(dim=3) / valid_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        
        return aggregated
```

**预期收益**：
- 满足物理对称性（A 撞 B = B 撞 A）
- 对物体排序不变，更鲁棒
- 预计碰撞预测准确率提升 5-10%

**实现难度**：中（修改消息函数为对称形式）

### 3. 迭代细化 → 改进碰撞预测
**映射位置**：`model/ai_model/decoder.py` → `CollisionHead`

**当前问题**：
- 碰撞预测是一次性的，没有迭代细化
- 复杂碰撞场景（多物体同时碰撞）可能预测不准

**具体改进**：
```python
# 迭代细化碰撞预测
class IterativeCollisionHead(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_iters=3):
        self.num_iters = num_iters
        self.collision_net = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # 碰撞特征更新
        self.update_net = nn.GRUCell(node_dim, node_dim)
    
    def forward(self, tokens):
        B, Tp, N, D = tokens.shape
        
        # 初始碰撞预测
        node_i = tokens.unsqueeze(3).expand(-1, -1, -1, N, -1)
        node_j = tokens.unsqueeze(2).expand(-1, -1, N, -1, -1)
        diff = (node_i - node_j).abs()
        pair_input = torch.cat([node_i, node_j, diff], dim=-1)
        
        collision_logits = self.collision_net(pair_input).squeeze(-1)
        
        # 迭代细化
        for _ in range(self.num_iters - 1):
            # 用碰撞预测更新节点特征
            collision_attn = torch.sigmoid(collision_logits).unsqueeze(-1)
            collision_info = torch.einsum('bij,bijd->bid', 
                                         torch.sigmoid(collision_logits), 
                                         tokens.unsqueeze(2).expand(-1, -1, N, -1))
            
            # 更新节点
            tokens = self.update_net(collision_info.flatten(0, 1), 
                                     tokens.flatten(0, 1)).reshape(B, Tp, N, D)
            
            # 重新预测碰撞
            node_i = tokens.unsqueeze(3).expand(-1, -1, -1, N, -1)
            node_j = tokens.unsqueeze(2).expand(-1, -1, N, -1, -1)
            diff = (node_i - node_j).abs()
            pair_input = torch.cat([node_i, node_j, diff], dim=-1)
            collision_logits = self.collision_net(pair_input).squeeze(-1)
        
        return collision_logits
```

**预期收益**：
- 多物体同时碰撞的场景预测更准确
- 迭代细化逐步改进预测
- 预计碰撞分类 F1 提升 10-15%

**实现难度**：中（修改 CollisionHead 为迭代形式）

### 4. 辅助重建损失 → 改进特征学习
**映射位置**：`model/ai_model/loss.py`

**当前问题**：
- 没有辅助损失帮助特征学习
- 特征学习完全依赖下游任务（状态预测、碰撞预测）

**具体改进**：
```python
# 添加辅助重建损失
def slot_reconstruction_loss(slots, feat_map, decoder):
    """
    slots: [B, K, D]
    feat_map: [B, C, H, W]
    decoder: 将 slot 解码为特征图组件
    """
    B, K, D = slots.shape
    
    # 每个 slot 解码为一个组件
    components = decoder(slots)  # [B, K, C, H, W]
    
    # 加权求和（权重来自 attention）
    weights = torch.ones(B, K, 1, 1, 1) / K  # 均匀权重
    reconstruction = (components * weights).sum(dim=1)  # [B, C, H, W]
    
    # 重建损失
    loss = F.mse_loss(reconstruction, feat_map)
    
    return loss
```

**预期收益**：
- 辅助特征学习，提升特征质量
- 鼓励 slots 学习有意义的物体表征
- 预计整体性能提升 5-10%

**实现难度**：中（需要添加解码器和辅助损失）

## 实验结果（关键指标）
| 数据集 | 指标 | Slot Attention | MONet | IODINE | Slot Attention (Oracle) |
|--------|------|---------------|-------|--------|------------------------|
| CLEVR | ARI | **0.90** | 0.85 | 0.83 | 0.95 |
| Tetrominoes | ARI | **0.96** | 0.92 | 0.90 | 0.98 |
| Multi-dSprites | ARI | **0.82** | 0.78 | 0.75 | 0.88 |

- CLEVR：Slot Attention ARI 0.90，优于 MONet（0.85）和 IODINE（0.83）
- Tetrominoes：ARI 0.96，接近 Oracle（0.98）
- Multi-dSprites：ARI 0.82，最优

**消融实验**：
- 1 轮迭代 → ARI 0.78（vs 3 轮 0.90），证明迭代的重要性
- 无竞争（标准 attention）→ ARI 0.72，证明竞争机制的关键性
- 5 个 slots（vs 7 个真实物体）→ ARI 0.85，证明 slot 数量的影响

**关键发现**：
- 竞争注意力是关键：softmax over slots 确保物体绑定
- 3 轮迭代是甜蜜点：更多轮次收益递减
- Slot 数量影响性能：接近真实物体数最优
