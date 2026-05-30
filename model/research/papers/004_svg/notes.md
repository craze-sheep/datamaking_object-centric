# SVG: Stochastic Video Generation with a Learned Prior

## 基本信息
- 作者：Emily Denton, Rob Fergus
- 年份：2018
- 会议/期刊：ICML 2018
- 论文链接：https://arxiv.org/abs/1802.07687
- 代码链接：https://github.com/edenton/svg
- PDF：svg.pdf（需下载）

## 核心贡献
- 提出 SVG（Stochastic Video Generation），首个将 VAE 引入视频预测的模型
- 设计确定性+随机性双通路：确定性通路捕获可预测运动，随机性通路建模不确定性
- 学习条件先验（conditioned prior），而非固定高斯先验，提升生成质量
- 支持多样化预测：同一输入可生成多个合理的未来，适用于物理场景的多模态性
- 在 Human3.6M、KTH Actions 上超越确定性方法

## 模型架构
- **Encoder**：CNN 编码器，将输入帧映射到特征空间
  - 结构：多层 Conv + ReLU + MaxPool
  - 输入：历史帧 [B, Th, C, H, W]
  - 输出：特征向量 [B, D]
- **确定性通路**：LSTM 编码历史序列
  - 输入：CNN 特征序列
  - 输出：隐藏状态 h_det [B, hidden_dim]
  - 捕获可预测的运动（如匀速直线运动）
- **随机性通路**：VAE 结构
  - **后验（Posterior）**：q(z|x_{1:T}) = N(μ, σ²)，从未来帧推断潜在变量
  - **先验（Prior）**：p(z|x_{1:T_h}) = N(μ_prior, σ_prior²)，从历史帧推断
  - 关键创新：先验是条件的（conditioned on history），不是固定 N(0,1)
  - 重参数化技巧：z = μ + σ * ε，ε ~ N(0,1)
- **Decoder**：CNN 解码器，从 (h_det, z) 生成预测帧
  - 输入：h_det（确定性状态）+ z（随机变量）
  - 输出：预测帧 [B, C, H, W]
  - 结构：反卷积堆叠
- **生成过程**：
  - 训练：用后验 q(z|x_{1:T}) 采样 z
  - 推理：用先验 p(z|x_{1:T_h}) 采样 z
  - 多次采样可生成多个不同的未来

## 损失函数
- **重建损失**：L_rec = Σ ||x_t - x̂_t||²，MSE 或 L1
- **KL 散度**：L_KL = D_KL(q(z|x_{1:T}) || p(z|x_{1:T_h}))
  - 后验应接近先验，但不能坍缩到先验（否则丢失信息）
  - 使用 KL annealing：前期 KL 权重小，后期逐渐增大
- **总损失**：L = L_rec + β * L_KL
  - β 控制 KL 权重，通常从 0 逐渐增大到 1

## 关键设计选择
- **确定性+随机性双通路**：
  - 确定性通路：捕获可预测的运动（物理规律）
  - 随机性通路：建模不确定性（物体外观变化、多模态未来）
  - 两者互补：确定性提供基础预测，随机性提供多样性
- **条件先验**：
  - 传统 VAE 用 N(0,1) 先验，SVG 用历史帧条件化的先验
  - 优势：先验更接近后验，KL 更容易优化
  - 实现：prior_net MLP 从 h_det 预测 μ_prior, σ_prior
- **KL Annealing**：
  - 前期 β=0（纯重建），后期 β=1（标准 VAE）
  - 防止 KL 过早约束导致的后验坍缩
- **与当前模型对比**：
  - 当前模型是确定性预测，SVG 是随机性预测
  - 当前模型无法生成多个可能的未来，SVG 可以
  - 当前模型用 GRU，SVG 用 LSTM + VAE

## 与当前模型的对比
- **相似之处**：
  - 都用 CNN 做视觉编码
  - 都用循环结构做时序建模
  - 都需要预测多帧视频
- **不同之处**：
  - SVG 有随机性通路（VAE），当前模型纯确定性
  - SVG 可以生成多样化预测，当前模型只能生成单一预测
  - SVG 没有物理先验，当前模型有 force-aware GNN
  - SVG 是像素级预测，当前模型是物体级预测
  - SVG 用 LSTM，当前模型用 GRU

## 可借鉴的点

### 1. 随机性通路 → 改进碰撞预测的多模态性
**映射位置**：`model/ai_model/temporal.py`, `model/ai_model/loss.py`

**当前问题**：
- 碰撞后的运动有多种可能（向左弹、向右弹、反弹角度等）
- 确定性模型只能预测"平均"结果，导致模糊
- 碰撞分类器只预测"是否碰撞"，不预测"碰撞后的结果"

**具体改进**：
```python
# 在 TemporalGRU 中添加随机性通路
class StochasticTemporalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=32):
        self.gru = nn.GRU(input_dim, hidden_dim)
        # 后验网络：从未来帧推断 z
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # μ, log σ
        )
        # 先验网络：从历史帧推断 z
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # μ, log σ
        )
        # 解码器：从 (h, z) 生成预测
        self.decoder = nn.Linear(hidden_dim + latent_dim, input_dim)
    
    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, tokens, valid_mask, history_length, predict_length,
                future_tokens_gt=None):
        # 编码历史
        h_last = self.encode_history(tokens[:, :history_length], valid_mask)
        
        if self.training and future_tokens_gt is not None:
            # 训练：用后验
            h_future = self.encode_history(future_tokens_gt, valid_mask)
            posterior_input = torch.cat([h_last, h_future], dim=-1)
            mu_post, log_sigma_post = self.posterior_net(posterior_input).chunk(2, -1)
            z = self.reparameterize(mu_post, log_sigma_post)
        else:
            # 推理：用先验
            mu_post, log_sigma_post = None, None
            mu_prior, log_sigma_prior = self.prior_net(h_last).chunk(2, -1)
            z = self.reparameterize(mu_prior, log_sigma_prior)
        
        # 解码
        h_z = torch.cat([h_last, z], dim=-1)
        future = self.decoder(h_z)  # [B, N, Tp, D]
        
        return future, mu_post, log_sigma_post, mu_prior, log_sigma_prior

# 在 loss.py 中添加 KL 散度
def kl_divergence(mu_post, log_sigma_post, mu_prior, log_sigma_prior):
    """KL(q || p)"""
    kl = 0.5 * (
        (log_sigma_prior - log_sigma_post) +
        (torch.exp(log_sigma_post) + (mu_post - mu_prior)**2) / torch.exp(log_sigma_prior) - 1
    ).sum(dim=-1)
    return kl.mean()
```

**预期收益**：
- 碰撞后的运动可以有多个合理预测（向左弹、向右弹）
- 生成更清晰的预测帧（避免平均模糊）
- 可以采样多次获得多样化结果，用于下游决策

**实现难度**：高（需要修改 TemporalGRU、loss.py、train.py）

### 2. KL Annealing → 改进碰撞分类器训练
**映射位置**：`model/ai_model/loss.py`, `model/ai_model/train.py`

**当前问题**：
- 碰撞分类器用 Focal BCE，但碰撞样本很少（严重不平衡）
- 一开始就用全部 loss 权重可能导致分类器过早收敛到"无碰撞"

**具体改进**：
```python
# 在 train.py 中添加 KL annealing
def collision_weight_annealing(epoch, total_epochs, warmup_epochs=5):
    """逐步增加碰撞 loss 的权重"""
    if epoch < warmup_epochs:
        return 0.1  # 前期低权重，让其他模块先学好
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return 0.1 + 0.4 * progress  # 从 0.1 逐渐增大到 0.5
```

**预期收益**：
- 前期让 encoder 和 temporal 先学好特征，后期再重点训练碰撞分类器
- 避免碰撞分类器在特征不成熟时就过早收敛

**实现难度**：低（只需修改 train.py 的 loss 权重计算）

### 3. 条件先验 → 改进 Edge Network
**映射位置**：`model/ai_model/interaction.py` → `ForceAwareEdgeNetwork`

**当前问题**：
- Edge Network 用固定的 14 维特征（force + rel_pos + rel_vel + dist）
- 不同物体对之间的交互模式不同（球-球 vs 球-墙），但 edge network 是共享的

**具体改进**：
```python
# 当前：固定的 edge feature 投影
class ForceAwareEdgeNetwork(nn.Module):
    def __init__(self, state_dim, edge_dim):
        self.edge_net = nn.Sequential(
            nn.Linear(14, edge_dim),
            ...
        )

# 改为：条件 edge network
class ConditionalEdgeNetwork(nn.Module):
    def __init__(self, state_dim, edge_dim, obj_type_dim=4):
        # 物体类型条件
        self.type_cond = nn.Linear(obj_type_dim * 2, edge_dim)
        # Edge feature 投影
        self.edge_net = nn.Sequential(
            nn.Linear(14, edge_dim),
            nn.LayerNorm(edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
        )
        # 条件门控
        self.gate = nn.Linear(edge_dim * 2, edge_dim)
    
    def forward(self, dyn_state, force_matrix, valid_mask, obj_attrs):
        # 基础 edge feature
        edge_feat = self.edge_net(edge_input)  # [B, T, N, N, edge_dim]
        
        # 物体类型条件
        obj_type = obj_attrs[..., -4:]  # 假设最后 4 维是类型
        type_i = obj_type.unsqueeze(2).expand(-1, -1, N, -1)
        type_j = obj_type.unsqueeze(1).expand(-1, N, -1, -1)
        type_cond = self.type_cond(torch.cat([type_i, type_j], dim=-1))
        
        # 条件门控
        gate = torch.sigmoid(self.gate(torch.cat([edge_feat, type_cond], dim=-1)))
        edge_feat = gate * edge_feat + (1 - gate) * type_cond
        
        return edge_feat
```

**预期收益**：
- 不同物体对（球-球、球-墙）有不同的交互模式
- 提升碰撞预测的准确性
- 预计碰撞分类 F1 提升 5-10%

**实现难度**：中（需要修改 ForceAwareEdgeNetwork 和 GNN 的 forward）

### 4. 多样性采样 → 改进 RGB 预测
**映射位置**：`model/ai_model/decoder.py` → `RGBDecoder`

**当前问题**：
- RGBDecoder 只输出单一 RGB 预测，无法表达不确定性
- 碰撞后的 RGB 可能有多种合理结果

**具体改进**：
```python
# 在 RGBDecoder 中添加多样性采样
class StochasticRGBDecoder(nn.Module):
    def __init__(self, token_dim, base_channels, image_size=128, num_samples=4):
        self.num_samples = num_samples
        self.appearance_head = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3 * num_samples),  # 多个颜色假设
            nn.Sigmoid(),
        )
    
    def forward(self, tokens, mask_prob, valid_mask, sample_idx=None):
        # 生成多个颜色假设
        appearance = self.appearance_head(tokens)  # [B, Tp, N, 3*K]
        appearance = appearance.reshape(*tokens.shape[:3], self.num_samples, 3)
        
        if sample_idx is not None:
            # 选择特定样本
            appearance = appearance[:, :, :, sample_idx]
        else:
            # 取平均（确定性预测）
            appearance = appearance.mean(dim=3)
        
        # 后续合成逻辑不变
        ...
```

**预期收益**：
- 可以生成多个可能的 RGB 预测，用于不确定性估计
- 与 SVG 的随机性通路结合，实现真正的多样化视频预测

**实现难度**：中（需要修改 RGBDecoder 的输出维度）

## 实验结果（关键指标）
| 数据集 | 指标 | SVG | ConvLSTM | MCNET | SV2P |
|--------|------|-----|----------|-------|------|
| Human3.6M | PSNR (dB) | **27.2** | 25.8 | 25.1 | 25.4 |
| KTH Actions | PSNR (dB) | **25.7** | 24.9 | 24.1 | 24.3 |
| KTH Actions | FVD | **262** | 312 | 351 | 338 |

- Human3.6M：SVG PSNR 27.2dB，比 ConvLSTM（25.8）提升 1.4dB
- KTH Actions：PSNR 25.7dB，FVD 262（最优）
- 多样性：SVG 生成 8 个样本，LPIPS 多样性 0.15（ConvLSTM 为 0.02）

**消融实验**：
- 无随机性通路 → PSNR 从 25.7 降到 24.8（-0.9dB），证明随机性的重要性
- 固定先验 N(0,1) → PSNR 从 25.7 降到 25.1（-0.6dB），证明条件先验的优势
- 无 KL annealing → PSNR 从 25.7 降到 25.3（-0.4dB），证明 KL annealing 的有效性
