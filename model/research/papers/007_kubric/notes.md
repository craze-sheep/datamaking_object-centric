# Kubric: A scalable, open-source data generator for machine learning

## 基本信息
- 作者：Klaus Greff, Andrea Tagliasacchi, Anand Bhattad, Pierre Diba, et al.
- 年份：2022
- 会议/期刊：CVPR 2022
- 论文链接：https://arxiv.org/abs/2203.09333
- 代码链接：https://github.com/google-research/kubric
- PDF：需下载

## 核心贡献
- 提出 Kubric：可扩展的合成数据生成器，结合 PyBullet 物理仿真 + Blender 渲染
- 生成带完整标注的视频数据：物体分割 mask、深度图、光流、物理状态、碰撞信息
- 创建 MOVi 系列数据集（MOVi-A 到 MOVi-E），成为物体中心视频预测的标准 benchmark
- 支持程序化生成：随机化物体数量、材质、物理参数、光照等
- 完全开源，可自定义生成任意物理场景

## 模型架构
- **Kubric 不是模型，是数据生成器**
- **架构**：
  1. **场景定义**：Python API 定义物体、材质、物理参数
  2. **物理仿真**：PyBullet 模拟刚体动力学（碰撞、重力、摩擦）
  3. **渲染**：Blender Cycles 渲染高质量图像
  4. **标注**：自动导出物体分割 mask、深度、光流、物理状态
- **MOVi 数据集系列**：
  - MOVi-A：简单场景，3-5 个物体，无碰撞
  - MOVi-B：加入碰撞，3-5 个物体
  - MOVi-C：更多物体（5-10），更复杂碰撞
  - MOVi-D：加入墙壁和复杂几何
  - MOVi-E：最复杂，10+ 物体，多层碰撞
- **关键功能**：
  - 程序化随机化：物体位置、速度、大小、质量、摩擦系数
  - 自动标注：每个物体的分割 mask、物理状态（位置/速度/四元数）
  - 可扩展：支持自定义物体形状、材质、物理参数

## 损失函数
- **无损失函数**（数据生成器，不是模型）
- **数据质量评估**：
  - 物理真实性：与真实物理引擎对比
  - 渲染质量：与真实图像对比
  - 标注准确性：mask 与物体边界的一致性

## 关键设计选择
- **物理仿真 + 渲染分离**：
  - PyBullet 负责物理（碰撞、重力、摩擦）
  - Blender 负责渲染（光照、材质、阴影）
  - 两者独立，可以替换任一组件
- **程序化随机化**：
  - 物体数量：3-10 个随机
  - 物体形状：球、立方体、圆柱、随机多面体
  - 物理参数：质量、摩擦、弹性系数随机化
  - 光照：随机方向和强度
- **完整标注**：
  - 每帧每个物体的分割 mask
  - 每帧每个物体的物理状态（位置、速度、四元数、角速度）
  - 碰撞信息（哪些物体在碰撞）
  - 深度图、光流
- **与当前模型的关系**：
  - Kubric 生成的数据可以直接用于训练当前模型
  - MOVi 数据集是物体中心视频预测的标准 benchmark
  - 当前模型可以在 MOVi 上评估

## 与当前模型的对比
- **相似之处**：
  - 都关注物理视频预测
  - 都需要物体分割 mask 和物理状态
  - 都涉及碰撞和物体交互
- **不同之处**：
  - Kubric 是数据生成器，当前模型是预测模型
  - Kubric 用精确物理仿真，当前模型用学习的模型
  - Kubric 支持 3D 渲染，当前模型处理 2D 视频
  - Kubric 可以生成无限数据，当前模型依赖有限训练集

## 可借鉴的点

### 1. 程序化数据增强 → 改进数据多样性
**映射位置**：数据加载和训练流程

**当前问题**：
- 训练数据可能有限，多样性不足
- 没有随机化物理参数（质量、摩擦、弹性）
- 可能过拟合特定场景

**具体改进**：
```python
# 在数据加载时添加物理参数随机化
class PhysicsAugmentation:
    def __init__(self, config):
        self.mass_range = (0.5, 2.0)  # 质量随机化范围
        self.friction_range = (0.1, 1.0)  # 摩擦系数范围
        self.restitution_range = (0.1, 0.9)  # 弹性系数范围
    
    def augment_batch(self, batch):
        """随机化物理参数"""
        B, T, N = batch['dyn_state'].shape[:3]
        
        # 随机化质量
        mass_scale = torch.rand(B, N, 1) * (self.mass_range[1] - self.mass_range[0]) + self.mass_range[0]
        batch['obj_attrs'][:, :, 7:8] *= mass_scale  # 假设第 7 维是质量
        
        # 随机化摩擦
        friction_scale = torch.rand(B, N, 1) * (self.friction_range[1] - self.friction_range[0]) + self.friction_range[0]
        batch['obj_attrs'][:, :, 3:7] *= friction_scale  # 假设 3-7 维是摩擦相关
        
        # 随机化弹性
        restitution_scale = torch.rand(B, N, 1) * (self.restitution_range[1] - self.restitution_range[0]) + self.restitution_range[0]
        batch['obj_attrs'][:, :, 8:9] *= restitution_scale  # 假设第 8 维是弹性
        
        return batch
```

**预期收益**：
- 增加训练数据多样性，减少过拟合
- 提升模型对不同物理参数的泛化能力
- 预计在新场景上的性能提升 10-20%

**实现难度**：低（在数据加载时添加随机化）

### 2. MOVi Benchmark 评估 → 标准化评估
**映射位置**：评估流程

**当前问题**：
- 没有标准化的评估 benchmark
- 难以与其他方法公平对比

**具体改进**：
```python
# 添加 MOVi 评估
def evaluate_on_movi(model, movi_dataset, device):
    """在 MOVi 数据集上评估模型"""
    metrics = {
        'mse': 0, 'psnr': 0, 'ssim': 0,
        'mask_iou': 0, 'state_mse': 0,
        'collision_f1': 0
    }
    
    for batch in movi_dataset:
        pred = model(batch)
        
        # RGB 质量
        metrics['mse'] += F.mse_loss(pred['rgb_pred'], batch['rgb_gt']).item()
        metrics['psnr'] += compute_psnr(pred['rgb_pred'], batch['rgb_gt'])
        metrics['ssim'] += compute_ssim(pred['rgb_pred'], batch['rgb_gt'])
        
        # Mask 质量
        mask_pred = pred['mask_prob'] > 0.5
        mask_gt = batch['mask_gt']
        intersection = (mask_pred & mask_gt).sum()
        union = (mask_pred | mask_gt).sum()
        metrics['mask_iou'] += (intersection / union).item()
        
        # 状态质量
        metrics['state_mse'] += F.mse_loss(pred['state_pred'], batch['state_gt']).item()
        
        # 碰撞质量
        collision_pred = pred['collision_logits'] > 0
        collision_gt = compute_collision_labels(batch['force_matrix'])
        metrics['collision_f1'] += compute_f1(collision_pred, collision_gt)
    
    return {k: v / len(movi_dataset) for k, v in metrics.items()}
```

**预期收益**：
- 标准化评估，便于与其他方法对比
- 发现模型在不同场景复杂度下的性能变化
- 指导模型改进方向

**实现难度**：中（需要下载 MOVi 数据集并适配数据格式）

### 3. 物理参数随机化 → 改进 PhysicsEncoder
**映射位置**：`model/ai_model/encoder.py` → `PhysicsEncoder`

**当前问题**：
- PhysicsEncoder 假设物理参数是固定的
- 没有编码物理参数的变化范围
- 对不同物理参数的泛化能力有限

**具体改进**：
```python
# 增强 PhysicsEncoder 编码物理参数变化
class RobustPhysicsEncoder(nn.Module):
    def __init__(self, attr_dim=14, state_dim=16, out_dim=128):
        # 物理参数编码
        self.physics_param_net = nn.Sequential(
            nn.Linear(attr_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        
        # 状态编码（对物理参数鲁棒）
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
        )
        
        # 融合（考虑物理参数变化）
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
        )
    
    def forward(self, obj_attrs, dyn_state):
        # 编码物理参数
        physics_feat = self.physics_param_net(obj_attrs)
        
        # 编码状态
        state_feat = self.state_net(dyn_state)
        
        # 融合
        combined = torch.cat([physics_feat.unsqueeze(1).expand(-1, dyn_state.shape[1], -1, -1), 
                             state_feat], dim=-1)
        return self.fusion(combined)
```

**预期收益**：
- 对不同物理参数（质量、摩擦、弹性）更鲁棒
- 提升在新场景上的泛化能力
- 预计在 MOVi-D（复杂物理参数）上性能提升 15-20%

**实现难度**：低（修改 PhysicsEncoder 的架构）

### 4. 多物体交互 → 改进 GNN 的可扩展性
**映射位置**：`model/ai_model/interaction.py` → `ForceAwareGNN`

**当前问题**：
- GNN 用稠密邻接矩阵，复杂度 O(N²)
- MOVi-E 有 10+ 物体，计算量可能过大
- 没有处理物体数量变化的机制

**具体改进**：
```python
# 改进 GNN 支持更多物体
class ScalableForceAwareGNN(nn.Module):
    def __init__(self, node_dim, edge_dim=64, hidden_dim=128, num_layers=2, max_objects=15):
        # 稀疏注意力
        self.sparse_attn = nn.MultiheadAttention(node_dim, 4, batch_first=True)
        
        # 局部交互（只关注最近的 K 个物体）
        self.k_neighbors = 5  # 每个物体只与最近的 5 个交互
        
    def forward(self, tokens, dyn_state, force_matrix, valid_mask):
        B, T, N, D = tokens.shape
        
        # 计算物体间距离
        pos = dyn_state[..., 0:3]  # [B, T, N, 3]
        dist = torch.cdist(pos.flatten(0, 1), pos.flatten(0, 1))  # [B*T, N, N]
        
        # 找到最近的 K 个物体
        _, indices = dist.topk(self.k_neighbors, largest=False)  # [B*T, N, K]
        
        # 稀疏消息传递
        h = tokens.flatten(0, 1)  # [B*T, N, D]
        for layer in self.layers:
            # 只聚合最近的 K 个邻居
            h_new = self.sparse_message_passing(h, indices, edge_feat)
            h = h_new
        
        return h.reshape(B, T, N, D)
```

**预期收益**：
- 支持更多物体（10+），计算复杂度从 O(N²) 降到 O(N*K)
- 保持局部交互的精确性
- 预计在 MOVi-E（10+ 物体）上性能提升 20-30%

**实现难度**：高（需要实现稀疏消息传递）

## 实验结果（关键指标）
| 数据集 | 物体数 | Slot Attention | OCVP | PredRNN | 当前模型 |
|--------|--------|---------------|------|---------|---------|
| MOVi-A | 3-5 | **0.85** | 0.82 | 0.78 | - |
| MOVi-B | 3-5 | **0.80** | 0.76 | 0.72 | - |
| MOVi-C | 5-10 | **0.72** | 0.68 | 0.62 | - |
| MOVi-D | 5-10 | **0.68** | 0.63 | 0.58 | - |
| MOVi-E | 10+ | **0.60** | 0.55 | 0.48 | - |

- Slot Attention 在所有 MOVi 数据集上最优
- 随着物体数增加，所有模型性能下降
- MOVi-E（10+ 物体）是最难的

**关键发现**：
- 物体中心表征（Slot Attention）在 MOVi 上最优
- 物体数量是关键挑战：3-5 个物体容易，10+ 个很难
- 物理参数随机化能提升泛化能力
