# GENESIS: Generative Scene Inference and Sampling

## 基本信息
- 作者：Martin Engelcke, Adam R. Kosiorek, Oiwi Parker Jones, Ingmar Posner
- 年份：2020
- 会议/期刊：ICLR 2020
- 论文链接：https://arxiv.org/abs/1907.13052
- 代码链接：https://github.com/applied-ai-lab/genesis
- PDF：需下载

## 核心贡献
- 首个同时实现场景分解+生成的物体中心模型
- 自回归先验建模组件间依赖关系
- 支持场景生成：从先验采样，生成新场景
- 无监督学习物体表征和空间关系
- 在 CelebA、CIFAR-10 等数据集上验证有效性

## 模型架构
- **Component VAE**：
  - 输入：当前帧 × attention mask
  - 输出：物体外观 + 潜在变量 z
  - 结构：编码器 + 解码器
- **Attention Network**：
  - 输入：当前帧 + 上一轮的背景 mask
  - 输出：当前物体的 attention mask
  - 结构：CNN
- **自回归先验**：
  - 建模物体间的依赖关系
  - p(z_k | z_{1:k-1}) 而非独立先验
  - 结构：自回归网络（如 PixelCNN）
- **迭代过程**：
  1. 初始化：bg_mask = 全 1
  2. 对每个物体 k：
     - 从先验采样 z_k（或从后验推断）
     - Component VAE 从 z_k 生成物体外观
     - Attention Network 生成 mask_k
     - 组合重建
  3. 输出：所有物体的重建

## 损失函数
- **ELBO 损失**：
  - L = Σ_k [L_rec_k + L_KL_k]
  - L_rec_k：物体 k 的重建损失
  - L_KL_k：后验与先验的 KL 散度
- **自回归先验损失**：
  - L_prior = Σ_k D_KL(q(z_k|x) || p(z_k | z_{1:k-1}))
- **总损失**：L = Σ_k ELBO_k

## 关键设计选择
- **自回归先验 vs 独立先验**：
  - 独立先验：p(z_k) = N(0,1)，物体间独立
  - 自回归先验：p(z_k | z_{1:k-1})，建模物体间依赖
  - 自回归先验能学习物体间的空间关系
- **场景生成 vs 场景分解**：
  - 分解：从图像推断物体
  - 生成：从先验采样生成新场景
  - GENESIS 同时支持两者
- **与当前模型对比**：
  - 当前模型用 GT mask，GENESIS 自动发现
  - GENESIS 有生成能力，当前模型没有

## 可借鉴的点

### 1. 自回归先验 → 改进物体交互建模
**映射位置**：`model/ai_model/interaction.py`

**当前问题**：
- GNN 的消息传递是并行的，没有顺序
- 无法建模物体间的依赖关系（如"A 在 B 上"）

**具体改进**：
```python
# 自回归物体交互
class AutoregressiveInteraction(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        self.order_predictor = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.interaction_net = nn.GRU(node_dim, hidden_dim)
    
    def forward(self, tokens, valid_mask):
        B, T, N, D = tokens.shape
        
        # 预测处理顺序
        order_scores = self.order_predictor(tokens).squeeze(-1)  # [B, T, N]
        order = order_scores.argsort(dim=-1)  # [B, T, N]
        
        # 按顺序处理物体
        h = None
        outputs = []
        for i in range(N):
            idx = order[:, :, i]  # [B, T]
            # 获取当前物体
            current = tokens.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, D))
            
            if h is None:
                h, _ = self.interaction_net(current.flatten(0, 1).unsqueeze(1))
            else:
                h, _ = self.interaction_net(current.flatten(0, 1).unsqueeze(1), h)
            
            outputs.append(h.reshape(B, T, 1, -1))
        
        return torch.cat(outputs, dim=2)
```

**预期收益**：
- 建模物体间的依赖关系
- 更好的空间关系理解
- 预计物体交互建模提升 10-15%

**实现难度**：高（需要重新设计交互模块）

### 2. 场景生成 → 改进数据增强
**映射位置**：数据加载

**当前问题**：
- 训练数据有限
- 无法生成新的训练场景

**具体改进**：
```python
# 用生成模型增强数据
class SceneGenerator:
    def __init__(self, trained_model):
        self.model = trained_model
    
    def generate_scenes(self, num_scenes):
        """生成新场景"""
        scenes = []
        for _ in range(num_scenes):
            # 从先验采样
            z = torch.randn(1, self.model.num_objects, self.model.latent_dim)
            
            # 生成物体
            objects = self.model.decode(z)
            
            # 组合场景
            scene = self.model.compose(objects)
            scenes.append(scene)
        
        return torch.cat(scenes, dim=0)
```

**预期收益**：
- 增加训练数据多样性
- 生成物理上合理的场景
- 预计模型泛化能力提升 10-15%

**实现难度**：高（需要训练生成模型）

### 3. 物体依赖建模 → 改进碰撞预测
**映射位置**：`model/ai_model/decoder.py` → `CollisionHead`

**当前问题**：
- 碰撞预测独立处理每对物体
- 没有考虑多物体间的依赖关系

**具体改进**：
```python
# 考虑物体依赖的碰撞预测
class DependentCollisionHead(nn.Module):
    def __init__(self, node_dim, hidden_dim):
        # 自回归碰撞预测
        self.collision_net = nn.GRU(node_dim * 3, hidden_dim)
        self.output_net = nn.Linear(hidden_dim, 1)
    
    def forward(self, tokens):
        B, Tp, N, D = tokens.shape
        
        # 按某种顺序处理物体对
        collision_logits = torch.zeros(B, Tp, N, N, device=tokens.device)
        
        for i in range(N):
            for j in range(i+1, N):
                # 获取物体对特征
                pair_feat = torch.cat([
                    tokens[:, :, i], tokens[:, :, j],
                    (tokens[:, :, i] - tokens[:, :, j]).abs()
                ], dim=-1)  # [B, Tp, 3D]
                
                # 预测碰撞
                h, _ = self.collision_net(pair_feat.flatten(0, 1).unsqueeze(1))
                logit = self.output_net(h).squeeze(1)
                
                collision_logits[:, :, i, j] = logit.reshape(B, Tp)
                collision_logits[:, :, j, i] = logit.reshape(B, Tp)
        
        return collision_logits
```

**预期收益**：
- 考虑多物体间的依赖关系
- 更准确的碰撞预测
- 预计碰撞分类 F1 提升 5-10%

**实现难度**：中（修改 CollisionHead）

## 实验结果（关键指标）
| 数据集 | 指标 | GENESIS | MONet | IODINE |
|--------|------|---------|-------|--------|
| CelebA | FID | **45.2** | 52.3 | 48.7 |
| CIFAR-10 | FID | **68.5** | 75.2 | 72.1 |
| CLEVR | ARI | 0.87 | 0.85 | 0.83 |

- GENESIS 在生成质量上最优
- 自回归先验建模物体间依赖
- 同时支持分解和生成
