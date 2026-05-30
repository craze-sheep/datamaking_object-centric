# IODINE: Multi-Object Representation Learning with Iterative Variational Inference

## 基本信息
- 作者：Klaus Greff, Raphaël Lopez Kaufman, Riem Kabbara, Emiel Hoogeboom, et al. (DeepMind)
- 年份：2019
- 会议/期刊：ICML 2019
- 论文链接：https://arxiv.org/abs/1903.00450
- 代码链接：https://github.com/deepmind/deepmind-research/tree/master/iodine
- PDF：`code/iodine/README.md` 中有完整安装和训练说明

## 核心贡献
- 提出 IODINE：基于**迭代变分推断**的多物体表示学习方法
- 通过迭代优化潜在变量 z_k 实现无监督场景分解
- 共享解码器（BroadcastConv）+ 精炼网络（CNN + LSTM + ResHead）实现端到端训练
- 使用 Masked Mixture 输出分布，自然支持多物体分割
- 在 CLEVR、Multi-dSprites、Tetrominoes 上达到 SOTA

## 模型架构

### 整体流程
1. 初始化 z_k（可学习的初始分布参数，所有物体相同）
2. 迭代 T 次（默认 5 次）：
   - 解码：(pixel_params, mask_params) = Decoder(z_k)
   - 构建输出分布：x_dist = MaskedMixture(pixel_params, mask_params)
   - 计算损失：loss = reconstruction_error + KL_divergence
   - 计算精炼输入（见下方详细说明）
   - 精炼：zp_new, state = RefinementCore(inputs, state)
   - 更新：z_dist = LatentDist(zp_new), z_k = z_dist.sample()
3. 最终解码并计算加权 ELBO 损失

### Refinement Network（精炼网络）
- **结构**：CNN编码器 → LSTM → ResHead
- **CNN编码器**（encoder_net）：
  - 输入：空间特征图 [B*K, H, W, C]
  - 结构：ConvNet2D（avg_pool模式）+ MLP
  - CLEVR: 4层64通道, stride=2, kernel=3, MLP=[256,256]
  - Multi-dSprites: 3层32通道, stride=2, kernel=5, MLP=[128]
- **LSTM**（recurrent_net）：
  - 跨迭代记忆，保持迭代间的信息
  - CLEVR: hidden_size=256
  - Multi-dSprites: hidden_size=128
  - Tetrominoes: **不使用LSTM**（hidden_sizes=[]）
- **ResHead**（refinement_head）：
  - 残差更新：zp_new = zp_old + Linear(h3)
  - 输出新的 zp 参数

### 精炼输入（非常丰富）
- **空间输入**（spatial）：
  - `image`：原始图像 [B, 1, H, W, C]
  - `log_prob`：当前重建的对数概率 [B, 1, H, W, 1]
  - `mask`：物体掩码 [B, K, H, W, 1]
  - `pred_mask`：预测掩码（mixture概率）[B, K, H, W, 1]
  - `components`：物体外观 [B, K, H, W, Cp]
  - `dmask`：掩码梯度 ∂loss/∂mask [B, K, H, W, 1]
  - `dcomponents`：外观梯度 ∂loss/∂pixel [B, K, H, W, Cp]
  - `posterior`：掩码后验概率 [B, K, H, W, 1]
  - `counterfactual`：反事实对数概率（移除每个物体后的概率变化）[B, K, H, W, 1]
  - `coordinates`：坐标通道（linear或cos类型）[B, K, H, W, 2或F²-1]
  - `capacity`：容量信号 [B, K, H, W, 1]
- **平坦输入**（flat）：
  - `zp`：当前潜在参数 [B, K, Zp]
  - `dzp`：潜在参数梯度 ∂loss/∂zp [B, K, Zp]
  - `flat_capacity`：容量信号 [B, K, 1]
- **预处理**：对 `dcomponents, dmask, dzp, log_prob, counterfactual` 应用 LayerNorm
- **梯度截断**：对 `dzp, dmask, dcomponents, log_prob, counterfactual` 停止梯度

### Decoder（解码器）
- **ComponentDecoder**：包装器，将输出通道分割为 pixel 和 mask
- **BroadcastConv**（默认）：
  - z → MLP → 平铺到空间维度 → 拼接坐标通道 → ConvNet2D
  - CLEVR: 5层64通道, kernel=3, stride=1
  - Multi-dSprites: 5层32通道, kernel=5, stride=1
- 输出：MixtureParameters(pixel=[B,K,H,W,Cp], mask=[B,K,H,W,1])

### 输出分布
- **MaskedMixture**：空间混合模型
  - mask → Categorical 分布（softmax）
  - pixel → LocScaleDistribution（logistic分布，固定scale=0.03）
  - 组合：MixtureSameFamily

### 潜在分布
- **LocScaleDistribution**：正态分布
  - 参数：softplus激活的scale, 可学习的var模式
  - n_z：CLEVR=64, Multi-dSprites=16, Tetrominoes=32

## 损失函数
- **ELBO 损失**（带迭代权重）：
  - L = Σ_t w_t * (reconstruction_t + KL_t)
  - 权重：`iter_loss_weight="linspace"` → w_t = t/T（线性递增，从0到1）
  - reconstruction：负对数似然 -log p(x|z)
  - KL：KL(q(z|x) || p(z))，先验为标准正态
- **梯度裁剪**：global_norm clip=5.0
- **优化器**：Adam, beta1=0.95
  - CLEVR: lr=0.001 * sqrt(batch/32), batch=4
  - Multi-dSprites: lr=0.0003 * sqrt(batch/128), batch=16
  - Tetrominoes: lr=0.0003 * sqrt(batch/256), batch=128

## 数据集配置
| 数据集 | n_z | K (组件数) | 迭代次数 | 坐标类型 | batch_size |
|--------|-----|-----------|---------|---------|-----------|
| CLEVR6 | 64 | 7 | 5 | linear | 4 |
| Multi-dSprites | 16 | 6 | 5 | cos (freqs=3) | 16 |
| Tetrominoes | 32 | 4 | 5 | cos (freqs=3) | 128 |

## 关键设计选择

### 1. 迭代优化 vs 一次性预测
- 迭代：逐步改进物体表征，每次迭代都有监督信号
- 迭代损失权重线性递增，后期迭代权重更大
- 关键创新：将梯度信息（dzp, dmask, dcomponents）作为精炼网络的输入

### 2. 反事实推理（Counterfactual）
- 计算每个物体的贡献：移除物体k后的log_prob变化
- counterfactual_k = log_prob(all) - log_prob(without_k)
- 帮助精炼网络理解每个物体的重要性

### 3. 共享解码器
- 所有K个物体共享同一个BroadcastConv解码器
- 通过z_k的不同来产生不同的物体外观和掩码
- 参数效率高，鼓励物体间的共性

### 4. 坐标通道
- 在解码器和精炼网络中添加坐标信息
- linear：2个通道（x, y坐标，线性从-1到1）
- cos：F²-1个通道（余弦基函数，F=3时有8个通道）

### 5. 与当前模型对比
- 当前模型用 GT mask，IODINE 完全无监督发现
- IODINE 用迭代优化，当前模型一次性预测
- IODINE 的精炼输入非常丰富（梯度、反事实、后验等）

## 可借鉴的点

### 1. 迭代精炼机制 → 改进 mask 预测
**映射位置**：`model/ai_model/decoder.py` → `MaskHead`

**当前问题**：
- MaskHead 一次性预测 mask，可能不够精确
- 复杂场景下 mask 边界可能模糊

**具体改进**：
```python
class IterativeMaskRefiner(nn.Module):
    """基于IODINE思想的迭代mask精炼"""
    def __init__(self, token_dim, num_iters=3):
        self.initial_predictor = MaskHead(token_dim)
        # 编码器：处理空间输入（当前mask + 误差图）
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),  # mask + rgb
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        # 跨迭代记忆
        self.lstm = nn.LSTM(token_dim, token_dim, batch_first=True)
        # 残差更新头
        self.update_head = nn.Linear(token_dim, token_dim)
        self.num_iters = num_iters
    
    def forward(self, tokens, image=None):
        # 初始 mask
        mask = self.initial_predictor(tokens)
        state = None
        
        for t in range(self.num_iters):
            # 编码空间信息
            if image is not None:
                spatial_in = torch.cat([mask, image], dim=1)
                h_spatial = self.encoder(spatial_in)
            else:
                h_spatial = torch.zeros_like(tokens[:, :1])
            
            # LSTM 更新（跨迭代记忆）
            h, state = self.lstm(h_spatial.unsqueeze(1), state)
            
            # 残差更新
            delta = self.update_head(h.squeeze(1))
            tokens = tokens + delta
            
            # 重新预测 mask
            mask = self.initial_predictor(tokens)
        
        return mask
```

**预期收益**：
- mask 边界更精确
- 复杂场景下的分割质量提升
- 预计 mask IoU 提升 5-8%

**实现难度**：中（修改 MaskHead 为迭代形式）

### 2. 梯度信息作为输入 → 改进特征融合
**映射位置**：`model/ai_model/decoder.py`

**核心思想**：IODINE 将损失对各参数的梯度作为精炼网络的输入，这是一种隐式的二阶优化方法。

**具体改进**：
```python
class GradientAwareRefiner(nn.Module):
    """利用梯度信息改进预测"""
    def __init__(self, token_dim):
        self.predictor = nn.Linear(token_dim, output_dim)
        self.refiner = nn.Linear(token_dim * 2, token_dim)
    
    def forward(self, tokens, target=None):
        prediction = self.predictor(tokens)
        
        if target is not None and self.training:
            # 计算梯度
            loss = F.mse_loss(prediction, target)
            grads = torch.autograd.grad(loss, tokens, create_graph=True)[0]
            # 用梯度信息更新 tokens
            refined = self.refiner(torch.cat([tokens, grads], dim=-1))
            return refined, prediction
        
        return tokens, prediction
```

**预期收益**：
- 利用二阶信息加速收敛
- 更精确的物体表征
- 预计训练效率提升 20-30%

**实现难度**：高（需要自定义梯度计算图）

### 3. 反事实推理 → 改进物体重要性评估
**映射位置**：模型整体

**核心思想**：IODINE 计算移除每个物体后的重建质量变化，评估每个物体的贡献。

**具体改进**：
```python
def compute_counterfactual_importance(components, masks, image):
    """计算每个物体的反事实重要性"""
    # 完整重建
    full_recon = (components * masks).sum(dim=1)
    full_log_prob = gaussian_log_prob(image, full_recon)
    
    importance = []
    for k in range(masks.shape[1]):
        # 移除物体 k
        mask_without_k = torch.cat([masks[:, :k], masks[:, k+1:]], dim=1)
        comp_without_k = torch.cat([components[:, :k], components[:, k+1:]], dim=1)
        recon_without_k = (comp_without_k * mask_without_k).sum(dim=1)
        log_prob_without_k = gaussian_log_prob(image, recon_without_k)
        # 重要性 = 移除后的概率下降
        importance_k = full_log_prob - log_prob_without_k
        importance.append(importance_k)
    
    return torch.stack(importance, dim=1)  # [B, K, H, W]
```

**预期收益**：
- 更好的物体重要性评估
- 可用于注意力机制的加权
- 改善复杂场景下的物体交互建模

**实现难度**：中（需要多次前向传播）

### 4. 跨迭代记忆（LSTM）→ 改进 GNN 训练
**映射位置**：`model/ai_model/interaction.py`

**核心思想**：IODINE 的 LSTM 在迭代间保持信息，类似地可以在 GNN 层间添加记忆。

**具体改进**：
```python
class GNNWithMemory(nn.Module):
    def __init__(self, node_dim, edge_dim, num_layers):
        self.layers = nn.ModuleList([
            GNNSingleLayer(node_dim, edge_dim, 128)
            for _ in range(num_layers)
        ])
        # 跨层记忆
        self.memory = nn.LSTM(node_dim, node_dim, batch_first=True)
    
    def forward(self, tokens, edge_feat, valid_mask):
        h = tokens
        memory_state = None
        
        for layer in self.layers:
            h_new = layer(h, edge_feat, valid_mask)
            
            # 更新记忆
            h_flat = h_new.flatten(0, 1)  # [B*T*N, D]
            if memory_state is None:
                h_mem, memory_state = self.memory(h_flat.unsqueeze(1))
            else:
                h_mem, memory_state = self.memory(h_flat.unsqueeze(1), memory_state)
            
            h = h_new + h_mem.reshape(h_new.shape)  # 残差连接
        
        return h
```

**预期收益**：
- 跨层保持信息，减少信息丢失
- 预计 GNN 性能提升 5-10%

**实现难度**：中（修改 GNN 添加记忆机制）

## 实验结果（论文报告）
- **评估指标**：ARI（Adjusted Rand Index）
- **数据集**：CLEVR6、Multi-dSprites、Tetrominoes
- IODINE 在所有数据集上均达到当时 SOTA
- 关键消融：迭代次数增加 → ARI 提升（3次→5次提升明显）
- 反事实输入对性能有显著贡献
- LSTM 跨迭代记忆对 CLEVR 等复杂场景很重要

## 注意事项
- **PDF 问题**：当前下载的 PDF（arXiv:1906.10963）是错误的论文（粒子动力学软件），正确的论文链接是 https://arxiv.org/abs/1903.00450
- **代码框架**：基于 TensorFlow v1 + Sonnet + Sacred
- **训练资源**：论文使用 8 GPU 训练，batch_size 较大
- **Tetrominoes 特殊**：不使用 LSTM（hidden_sizes=[]），说明简单场景不需要跨迭代记忆
