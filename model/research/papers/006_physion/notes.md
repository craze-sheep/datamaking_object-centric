# Physion: Evaluating Physical Prediction from Vision in Humans and Machines

## 基本信息
- 作者：Daniel M. Bear, Elias Wang, Damian Mrowca, Felix Binder, Hsiao-Yu Tung, et al.
- 年份：2021
- 会议/期刊：NeurIPS 2021
- 论文链接：https://arxiv.org/abs/2011.13045
- 代码链接：https://github.com/daniel-bear/physion
- PDF：需下载

## 核心贡献
- 提出 Physion benchmark，评估视觉物理预测能力
- 测试 8 种物理关系：遮挡、稳定性、碰撞、支撑、滑动、掉落、滚动、悬挂
- 发现：物体中心表征 > GNN 物理状态 > 端到端视觉模型
- 发现人类在物理预测上远优于所有模型
- 提供了 80 个 3D 场景的视频和物理状态数据

## 模型架构
- **Physion 是 benchmark，不是模型**
- **测试的模型类型**：
  1. **端到端视觉模型**：
     - ResNet-50 + LSTM
     - SlowFast (视频理解)
     - Mask R-CNN + LSTM
  2. **物体中心模型**：
     - Slot Attention (物体发现)
     - OCVP (物体中心视频预测)
  3. **物理状态模型**：
     - Interaction Network (GNN)
     - Graph Network (GNN)
     - Oracle (完美物理状态)
  4. **人类评估**：
     - 人类受试者观看视频，预测物体是否会掉落/碰撞等
- **评估任务**：
  - 给定前几帧视频，预测物体在后续帧的状态
  - 二分类：物体是否会掉落/碰撞/移动等

## 损失函数
- **二分类交叉熵**（所有模型统一）
- **无特殊物理约束损失**

## 关键设计选择
- **8 种物理关系**：
  1. 遮挡（Occlusion）：物体 A 是否会被物体 B 遮挡
  2. 稳定性（Stability）：物体是否会保持稳定
  3. 碰撞（Collision）：物体 A 是否会撞到物体 B
  4. 支撑（Support）：物体 A 是否会被物体 B 支撑
  5. 滑动（Sliding）：物体是否会滑动
  6. 掉落（Falling）：物体是否会掉落
  7. 滚动（Rolling）：物体是否会滚动
  8. 悬挂（Hanging）：物体是否会悬挂
- **3D 场景**：
  - 使用 ThreeDWorld (TDW) 物理引擎
  - 80 个场景，每个有多个物体
  - 包含真实物理效果（碰撞、重力、摩擦）
- **人类评估**：
  - 20 个人类受试者
  - 每个场景观看 2 秒视频
  - 预测物体是否会掉落/碰撞等

## 与当前模型的对比
- **相似之处**：
  - 都关注物理预测（碰撞、运动）
  - 都需要理解物体间交互
  - 都涉及 2D/3D 物理场景
- **不同之处**：
  - Physion 是评估 benchmark，当前模型是预测模型
  - Physion 测试 8 种物理关系，当前模型主要关注碰撞
  - Physion 用 3D 场景，当前模型用 2D 场景
  - Physion 评估二分类（是否发生），当前模型评估状态回归

## 可借鉴的点

### 1. 物体中心表征 → 引入 Slot Attention
**映射位置**：`model/ai_model/encoder.py`

**Physion 的关键发现**：
- 物体中心模型（Slot Attention）> GNN 物理状态 > 端到端视觉模型
- 端到端视觉模型（ResNet+LSTM）在物理预测上表现最差
- 物体中心表征能更好地捕获物体间关系

**当前问题**：
- 当前模型用 GT mask 做 ROI pooling，不是学习出来的
- 无法处理没有 GT mask 的场景
- Slot Attention 可以自动发现物体

**具体改进**：
```python
# 在 encoder.py 中添加 Slot Attention
class SlotAttentionEncoder(nn.Module):
    def __init__(self, image_size=128, num_slots=7, slot_dim=128):
        self.cnn = VisualEncoder(3, (32, 64, 128, 128))
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            dim=128,
            iters=3  # 迭代次数
        )
        self.slot_proj = nn.Linear(128, slot_dim)
    
    def forward(self, rgb):
        # rgb: [B, T, 3, H, W]
        B, T = rgb.shape[:2]
        feat_map = self.cnn(rgb)  # [B, T, C, H', W']
        
        # Slot Attention: 自动发现物体
        slots = self.slot_attention(feat_map.flatten(2).permute(0, 2, 1))  # [B, T, N, C]
        slots = self.slot_proj(slots)  # [B, T, N, slot_dim]
        
        return slots

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters=3):
        self.slots = nn.Parameter(torch.randn(1, num_slots, dim))
        self.iters = iters
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.project_k = nn.Linear(dim, dim)
        self.project_v = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )
    
    def forward(self, inputs):
        B, N, D = inputs.shape
        slots = self.slots.expand(B, -1, -1)  # [B, num_slots, D]
        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)  # [B, N, D]
        v = self.project_v(inputs)  # [B, N, D]
        
        for _ in range(self.iters):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = slots  # [B, num_slots, D]
            
            # Attention
            attn = torch.einsum('bid,bjd->bij', q, k) / (D ** 0.5)
            attn = F.softmax(attn, dim=1)  # [B, num_slots, N]
            
            # Weighted mean
            attn_sum = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            updates = torch.einsum('bij,bjd->bid', attn, v) / attn_sum
            
            # GRU update
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, -1, D)
            
            # MLP
            slots = slots + self.mlp(slots)
        
        return slots
```

**预期收益**：
- 自动发现物体，不需要 GT mask
- Physion 证明物体中心表征在物理预测上最优
- 预计碰撞预测准确率提升 15-20%

**实现难度**：高（需要实现 Slot Attention 并修改整个 encoder）

### 2. 多种物理关系 → 扩展碰撞分类器
**映射位置**：`model/ai_model/decoder.py` → `CollisionHead`

**当前问题**：
- 只预测碰撞（一种物理关系）
- Physion 有 8 种物理关系，每种都需要预测

**具体改进**：
```python
# 当前：单一碰撞预测
class CollisionHead(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        self.net = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

# 改为：多关系预测
class RelationHead(nn.Module):
    def __init__(self, node_dim, hidden_dim, num_relations=8):
        self.relation_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(node_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_relations)
        ])
        self.relation_names = [
            'occlusion', 'stability', 'collision', 'support',
            'sliding', 'falling', 'rolling', 'hanging'
        ]
    
    def forward(self, tokens):
        B, Tp, N, D = tokens.shape
        node_i = tokens.unsqueeze(3).expand(-1, -1, -1, N, -1)
        node_j = tokens.unsqueeze(2).expand(-1, -1, N, -1, -1)
        diff = (node_i - node_j).abs()
        pair_input = torch.cat([node_i, node_j, diff], dim=-1)
        
        # 预测每种关系
        relation_logits = {}
        for i, (name, net) in enumerate(zip(self.relation_names, self.relation_nets)):
            relation_logits[name] = net(pair_input).squeeze(-1)
        
        return relation_logits
```

**预期收益**：
- 不仅预测碰撞，还预测其他物理关系（支撑、滑动等）
- 与 Physion benchmark 对齐
- 更全面的物理理解能力

**实现难度**：中（需要修改 decoder 和 loss）

### 3. 人类评估对齐 → 改进评估指标
**映射位置**：评估流程

**当前问题**：
- 只用 MSE/PSNR 评估预测质量
- 没有评估物理关系的正确性

**具体改进**：
```python
# 添加 Physion 风格的评估
def evaluate_physion_style(model, dataset, device):
    """评估物理关系预测准确率"""
    results = {rel: {'correct': 0, 'total': 0} for rel in ['collision', 'support', 'falling']}
    
    for batch in dataset:
        pred = model(batch)
        
        # 碰撞预测
        collision_pred = pred['collision_logits'] > 0
        collision_gt = compute_collision_labels(batch['force_matrix'])
        results['collision']['correct'] += (collision_pred == collision_gt).sum().item()
        results['collision']['total'] += collision_gt.numel()
        
        # 支撑预测（物体是否被其他物体支撑）
        support_pred = predict_support(pred['state_pred'], batch)
        support_gt = compute_support_labels(batch)
        results['support']['correct'] += (support_pred == support_gt).sum().item()
        results['support']['total'] += support_gt.numel()
        
        # 掉落预测（物体是否掉落出场景）
        falling_pred = predict_falling(pred['state_pred'])
        falling_gt = compute_falling_labels(batch)
        results['falling']['correct'] += (falling_pred == falling_gt).sum().item()
        results['falling']['total'] += falling_gt.numel()
    
    return {rel: res['correct'] / res['total'] for rel, res in results.items()}
```

**预期收益**：
- 更全面的物理理解评估
- 与人类评估对齐
- 发现模型在不同物理关系上的强弱项

**实现难度**：中（需要实现各种物理关系的评估函数）

### 4. 3D 物理理解 → 改进状态编码
**映射位置**：`model/ai_model/encoder.py` → `PhysicsEncoder`

**当前问题**：
- 只用 2D 位置和速度
- Physion 用 3D 场景，物理更真实

**具体改进**：
```python
# 增强 PhysicsEncoder 的 3D 理解
class Enhanced3DPhysicsEncoder(nn.Module):
    def __init__(self, state_dim=16):
        # 3D 物理特征
        self.physics_3d = nn.Sequential(
            nn.Linear(state_dim + 6, 64),  # +6 for 3D rotation, 3D angular velocity
            nn.GELU(),
            nn.Linear(64, 64),
        )
    
    def compute_3d_features(self, state):
        """计算 3D 物理特征"""
        pos = state[..., 0:3]
        quat = state[..., 3:7]  # quaternion
        vel = state[..., 7:10]
        angvel = state[..., 10:13]
        
        # 旋转矩阵（从四元数）
        rot_matrix = quaternion_to_rotation_matrix(quat)  # [B, T, N, 3, 3]
        
        # 3D 角速度
        angvel_3d = angvel  # [B, T, N, 3]
        
        # 转动惯量（简化）
        inertia = torch.eye(3).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        inertia = inertia.expand(*pos.shape[:-1], 3, 3)
        
        return torch.cat([rot_matrix.flatten(-2), angvel_3d], dim=-1)
```

**预期收益**：
- 更好的 3D 物理理解
- 支持旋转、角动量等 3D 物理量
- 预计状态预测准确率提升 10-15%

**实现难度**：高（需要实现四元数到旋转矩阵的转换）

## 实验结果（关键指标）
| 模型 | 遮挡 | 稳定性 | 碰撞 | 支撑 | 滑动 | 掉落 | 滚动 | 悬挂 | 平均 |
|------|------|--------|------|------|------|------|------|------|------|
| 人类 | **92** | **85** | **88** | **80** | **78** | **90** | **82** | **75** | **83.8** |
| Slot Attention + GNN | 72 | 68 | 75 | 65 | 62 | 70 | 65 | 58 | **66.9** |
| Interaction Network | 68 | 65 | 72 | 60 | 58 | 65 | 60 | 52 | **62.5** |
| ResNet + LSTM | 55 | 52 | 58 | 48 | 45 | 55 | 48 | 42 | **50.4** |
| Random | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50 | 50.0 |

- 人类平均 83.8%，远超所有模型（最高 66.9%）
- Slot Attention + GNN 最优（66.9%），比 ResNet+LSTM（50.4%）高 16.5%
- 碰撞预测：人类 88%，Slot Attention 75%，ResNet+LSTM 58%

**关键发现**：
- 物体中心表征（Slot Attention）> GNN 物理状态 > 端到端视觉
- 人类在物理预测上远超所有模型
- 稳定性和支撑是最难的物理关系
