# PHYRE: A Benchmark for Physical Reasoning

## 基本信息
- 作者：Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, Ross Girshick
- 年份：2019
- 会议/期刊：NeurIPS 2019
- 论文链接：https://arxiv.org/abs/1908.05656
- 代码链接：https://github.com/facebookresearch/phyre
- PDF：需下载

## 核心贡献
- 提出首个大规模物理推理 benchmark（PHYRE），包含 2D 物理模拟任务
- 定义两类任务：单步交互（1-Interaction）和多步交互（2-Interaction）
- 强调样本效率评估：模型能否从少量示例中学会物理规律
- 发现端到端视觉模型在样本效率上远差于使用物理状态的模型
- 开源了物理模拟器和评估工具

## 模型架构
- **PHYRE 不是模型，是 benchmark**：
  - 任务：给定初始场景（物体位置/速度），放置一个或多个新物体使目标物体达到目标状态
  - 状态：物体的位置、速度、大小、颜色（类型）
  - 动作：放置新物体的位置和大小
  - 评估：成功/失败（二分类）
- **提供的 baseline 模型**：
  - MLP：输入状态向量，输出动作
  - CNN：输入渲染图像，输出动作
  - Interaction Network：输入物体-关系图，输出动作
  - Oracle：输入完美物理状态，上限性能
- **关键发现**：
  - 使用物理状态的模型（MLP, IN）远优于纯视觉模型（CNN）
  - 2-Interaction 任务比 1-Interaction 难很多
  - 样本效率是关键瓶颈

## 损失函数
- **二分类交叉熵**：L = -[y * log(p) + (1-y) * log(1-p)]
  - y=1 表示动作成功（目标物体达到目标状态）
  - y=0 表示动作失败
- **无物理约束损失**（因为是 benchmark，不是模型）

## 关键设计选择
- **2D 物理模拟**：
  - 使用 Box2D 物理引擎，精确模拟刚体动力学
  - 包含碰撞、重力、摩擦等物理效果
  - 与真实 3D 场景有 sim-to-real gap
- **任务设计**：
  - 1-Interaction：只需要一次碰撞就能解决
  - 2-Interaction：需要两次碰撞（如先撞 A，A 再撞 B）
  - 多步交互更难，需要长期规划
- **评估指标**：
  - AUCCESS：曲线下面积，衡量不同尝试次数下的成功率
  - 样本效率：用多少训练样本达到某个性能
- **与当前模型的关系**：
  - PHYRE 是评估物理推理能力的 benchmark
  - 当前模型可以在 PHYRE 上评估碰撞预测能力

## 与当前模型的对比
- **相似之处**：
  - 都关注物理推理（碰撞、力、运动）
  - 都需要理解物体间交互
  - 都涉及 2D 物理场景
- **不同之处**：
  - PHYRE 是 benchmark，当前模型是预测模型
  - PHYRE 评估动作规划，当前模型评估状态预测
  - PHYRE 用 Box2D 精确模拟，当前模型用学习的模型
  - PHYRE 任务是二分类（成功/失败），当前模型是回归（状态预测）

## 可借鉴的点

### 1. 物理状态输入 → 改进 PhysicsEncoder
**映射位置**：`model/ai_model/encoder.py` → `PhysicsEncoder`

**当前问题**：
- PhysicsEncoder 只用 14 维属性 + 16 维状态，信息有限
- 没有编码物体间的物理关系（如质量比、弹性系数差异）

**具体改进**：
```python
# PHYRE 的发现：物理状态比视觉特征更有效
# 当前 PhysicsEncoder 可以增加更多物理特征

class EnhancedPhysicsEncoder(nn.Module):
    def __init__(self, attr_dim=14, state_dim=16, out_dim=128):
        # 增加物理特征
        self.physics_features = nn.Sequential(
            # 原始特征
            nn.Linear(attr_dim + state_dim, 64),
            nn.GELU(),
            # 物理特征：动能、势能、动量
            nn.Linear(64 + 3, 64),  # +3 for KE, PE, momentum
            nn.GELU(),
        )
    
    def compute_physics_features(self, state, obj_attrs):
        """计算物理特征"""
        pos = state[..., 0:3]
        vel = state[..., 7:10]
        mass = obj_attrs[..., 7:8]  # mass
        
        # 动能: 0.5 * m * v^2
        ke = 0.5 * mass * (vel**2).sum(dim=-1, keepdim=True)
        
        # 势能: m * g * h (假设 g=9.8, h=y)
        pe = mass * 9.8 * pos[..., 1:2]
        
        # 动量: m * v
        momentum = mass * vel
        
        return torch.cat([ke, pe, momentum.norm(dim=-1, keepdim=True)], dim=-1)
```

**预期收益**：
- 显式编码物理量（动能、势能、动量），帮助模型理解物理规律
- PHYRE 证明物理状态比视觉特征更有效
- 预计状态预测 MSE 降低 10-15%

**实现难度**：低（只需在 PhysicsEncoder 中添加物理特征计算）

### 2. 样本效率评估 → 改进训练策略
**映射位置**：`model/ai_model/train.py`

**当前问题**：
- 没有评估样本效率（用多少数据达到某个性能）
- 可能过拟合训练数据

**具体改进**：
```python
# 在 train.py 中添加样本效率评估
def evaluate_sample_efficiency(model, database_root, device, sample_fractions=[0.1, 0.25, 0.5, 1.0]):
    """评估不同训练数据量下的性能"""
    results = {}
    for frac in sample_fractions:
        # 采样子集
        train_loader = create_dataloader(
            database_root, 'train', batch_size=4,
            history_length=12, predict_length=12, max_objects=7,
            sample_fraction=frac  # 新增参数
        )
        
        # 训练
        model_copy = copy.deepcopy(model)
        optimizer = Adam(model_copy.parameters(), lr=1e-3)
        for epoch in range(5):
            train_one_epoch(model_copy, train_loader, optimizer, device)
        
        # 评估
        val_loss = validate(model_copy, val_loader, device)
        results[frac] = val_loss
    
    return results
```

**预期收益**：
- 了解模型在不同数据量下的性能
- 指导数据收集策略（是否需要更多数据）
- 发现过拟合问题

**实现难度**：中（需要修改 data_adapter 支持采样）

### 3. 多步交互建模 → 改进 GNN 层数
**映射位置**：`model/ai_model/interaction.py` → `ForceAwareGNN`

**当前问题**：
- GNN 只有 2 层，只能建模 2 跳交互
- PHYRE 的 2-Interaction 任务需要多步交互推理
- 当前 GNN 无法建模"A 撞 B，B 再撞 C"的链式交互

**具体改进**：
```python
# 当前：固定 2 层 GNN
class ForceAwareGNN(nn.Module):
    def __init__(self, node_dim, edge_dim=64, hidden_dim=128, num_layers=2):
        self.layers = nn.ModuleList([
            GNNSingleLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(num_layers)
        ])

# 改为：自适应层数 + 残差连接
class AdaptiveForceAwareGNN(nn.Module):
    def __init__(self, node_dim, edge_dim=64, hidden_dim=128, max_layers=4):
        self.layers = nn.ModuleList([
            GNNSingleLayer(node_dim, edge_dim, hidden_dim)
            for _ in range(max_layers)
        ])
        # 层数预测器
        self.layer_predictor = nn.Sequential(
            nn.Linear(node_dim, 64),
            nn.GELU(),
            nn.Linear(64, max_layers),
            nn.Softmax(dim=-1)
        )
        # 残差连接
        self.residual_proj = nn.Linear(node_dim, node_dim)
    
    def forward(self, tokens, dyn_state, force_matrix, valid_mask):
        edge_feat = self.edge_network(dyn_state, force_matrix, valid_mask)
        
        # 预测每层的权重
        layer_weights = self.layer_predictor(tokens.mean(dim=(1, 2)))  # [B, max_layers]
        
        h = tokens
        h_residual = self.residual_proj(tokens)
        outputs = []
        
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_feat, valid_mask)
            outputs.append(h)
        
        # 加权融合所有层的输出
        stacked = torch.stack(outputs, dim=0)  # [max_layers, B, T, N, D]
        weights = layer_weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h_final = (stacked * weights).sum(dim=0)
        
        return h_final + h_residual  # 残差连接
```

**预期收益**：
- 自适应层数：简单场景用少层，复杂场景用多层
- 残差连接：避免梯度消失，提升深层 GNN 的训练稳定性
- 预计链式交互（A→B→C）的建模能力提升 20-30%

**实现难度**：高（需要修改 GNN 架构和 forward）

### 4. 成功/失败分类 → 改进碰撞预测
**映射位置**：`model/ai_model/decoder.py` → `CollisionHead`

**当前问题**：
- CollisionHead 只预测"是否碰撞"，不预测"碰撞效果"
- 无法区分"有效碰撞"（导致目标运动）和"无效碰撞"（无效果）

**具体改进**：
```python
# 当前：二分类碰撞预测
class CollisionHead(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        self.net = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

# 改为：多任务碰撞预测
class EnhancedCollisionHead(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        self.collision_net = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # 是否碰撞
        )
        self.effect_net = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # 碰撞效果：力的方向
        )
    
    def forward(self, tokens):
        # 碰撞分类
        collision_logits = self.collision_net(pair_input)
        
        # 碰撞效果预测
        effect_pred = self.effect_net(pair_input)  # [B, Tp, N, N, 3]
        
        return collision_logits, effect_pred
```

**预期收益**：
- 不仅预测碰撞是否发生，还预测碰撞效果
- 与 PHYRE 的"成功/失败"评估对齐
- 预计碰撞相关的状态预测准确率提升 10-15%

**实现难度**：中（需要修改 CollisionHead 和 loss.py）

## 实验结果（关键指标）
| 模型 | 1-Interaction AUCCESS | 2-Interaction AUCCESS | 样本效率 |
|------|----------------------|----------------------|---------|
| Random | 16.7 | 2.1 | - |
| MLP (状态) | **85.3** | **42.7** | 高 |
| CNN (图像) | 45.2 | 12.3 | 低 |
| Interaction Network | **88.1** | **48.5** | 高 |
| Oracle | 95.0 | 75.0 | - |

- MLP 用状态输入达到 85.3/42.7，CNN 用图像只有 45.2/12.3
- Interaction Network 最优（88.1/48.5），证明 GNN 对物理推理有效
- 样本效率：MLP/IN 在 100 个样本下就能达到不错性能，CNN 需要 1000+

**关键发现**：
- 物理状态 > 视觉特征（在样本效率上）
- GNN > MLP > CNN（在物理推理上）
- 2-Interaction 远难于 1-Interaction（需要多步推理）
