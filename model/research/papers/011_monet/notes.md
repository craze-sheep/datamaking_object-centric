# MONet: Multi-Object Network

## 基本信息
- 作者：Christopher P. Burgess, Loic Matthey, Nicholas Watters, Rishabh Kabra, et al.
- 年份：2019
- 会议/期刊：CVPR 2019
- 论文链接：https://arxiv.org/abs/1901.11370
- 代码链接：https://github.com/deepmind/multi-object-networks
- PDF：需下载

## 核心贡献
- 提出 MONet：迭代注意力机制实现场景分解
- 设计 Attention Network + Component VAE 的组合架构
- 无监督学习物体分割和表征
- 组合式 mask 生成：每个物体一个 mask，加权求和重建原图
- 物体中心表示学习的奠基工作

## 模型架构
- **Attention Network**：
  - 输入：当前帧 + 上一轮的背景 mask
  - 输出：当前物体的 attention mask
  - 结构：U-Net 架构
  - 迭代：每次关注一个物体，逐步发现所有物体
- **Component VAE**：
  - 输入：当前帧 × attention mask
  - 输出：重建的物体外观 + 潜在变量 z
  - 结构：编码器 + 解码器
  - 每个物体独立编码/解码
- **迭代过程**：
  1. 初始化：背景 mask = 全 1
  2. 对每个物体 k：
     - Attention Network 从 (image, bg_mask) 生成 mask_k
     - Component VAE 从 (image × mask_k) 生成重建_k
     - 更新 bg_mask = bg_mask - mask_k
  3. 背景：bg_mask 剩余部分作为背景
- **组合重建**：
  - 每个物体的重建 × 对应 mask
  - 背景 × 背景 mask
  - 求和得到总重建
- **关键实现细节**：
  - 物体数量固定（预设 K 个物体）
  - Attention Network 和 Component VAE 联合训练
  - 潜在变量 z 用于生成多样性

## 损失函数
- **重建损失**：L_rec = Σ ||x - Σ_k (m_k * x̂_k)||²
- **KL 散度**：L_KL = Σ_k D_KL(q(z_k|x) || p(z_k))
  - 每个物体的潜在变量都应接近先验
- **注意力正则化**：L_attn = Σ_k ||m_k||₁
  - 鼓励 mask 稀疏（每个物体关注区域尽量小）
- **总损失**：L = L_rec + β * L_KL + λ_attn * L_attn

## 关键设计选择
- **迭代注意力 vs 并行发现**：
  - 迭代：每次发现一个物体，逐步减少未发现区域
  - 并行：同时发现所有物体，可能重叠
  - 迭代更稳定，但需要预设物体数量
- **组合式重建 vs 整体重建**：
  - 组合式：每个物体独立重建，加权求和
  - 整体：直接重建整张图
  - 组合式更可解释，但计算量更大
- **与当前模型对比**：
  - 当前模型用 GT mask，MONet 自动发现
  - MONet 用 VAE 建模不确定性，当前模型纯确定性

## 可借鉴的点

### 1. 迭代注意力 → 改进物体发现
**映射位置**：`model/ai_model/encoder.py`

**当前问题**：
- MaskedROIPooler 依赖 GT mask
- 无法处理没有 GT mask 的场景

**具体改进**：
```python
# 迭代注意力物体发现
class IterativeAttentionEncoder(nn.Module):
    def __init__(self, feat_channels, num_objects=7):
        self.attention_net = UNet(feat_channels, 1)  # 输出单个 mask
        self.num_objects = num_objects
    
    def forward(self, feat_map):
        B, C, H, W = feat_map.shape
        
        bg_mask = torch.ones(B, 1, H, W, device=feat_map.device)
        masks = []
        
        for k in range(self.num_objects):
            # Attention: 从 (feat, bg_mask) 生成 mask_k
            attn_input = torch.cat([feat_map, bg_mask], dim=1)
            mask_k = torch.sigmoid(self.attention_net(attn_input))
            
            masks.append(mask_k)
            bg_mask = bg_mask - mask_k
        
        return torch.cat(masks, dim=1)  # [B, K, H, W]
```

**预期收益**：
- 无需 GT mask，自动发现物体
- 迭代方式避免重复发现
- 预计物体发现准确率提升 15-20%

**实现难度**：高（需要实现 UNet 和迭代流程）

### 2. 组合式重建 → 改进 RGB 解码
**映射位置**：`model/ai_model/decoder.py` → `RGBDecoder`

**当前问题**：
- RGBDecoder 用简单的颜色+mask 合成
- 没有独立的物体重建

**具体改进**：
```python
# 组合式 RGB 解码
class CompositionalRGBDecoder(nn.Module):
    def __init__(self, token_dim, base_channels, image_size=128):
        # 每个物体独立解码
        self.object_decoder = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3 * 8 * 8),
            nn.GELU(),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
        # 背景解码
        self.bg_decoder = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3 * 8 * 8),
            nn.GELU(),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.ConvTranspose2d(3, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, tokens, mask_prob, valid_mask):
        B, Tp, N, D = tokens.shape
        
        # 每个物体独立重建
        object_rgbs = []
        for i in range(N):
            obj_rgb = self.object_decoder(tokens[:, :, i])  # [B, Tp, 3, H, W]
            object_rgbs.append(obj_rgb)
        object_rgbs = torch.stack(object_rgbs, dim=2)  # [B, Tp, N, 3, H, W]
        
        # 背景重建
        scene_token = tokens.mean(dim=2)  # [B, Tp, D]
        bg_rgb = self.bg_decoder(scene_token)  # [B, Tp, 3, H, W]
        
        # 组合
        total_mask = mask_prob.sum(dim=2).clamp(0, 1)  # [B, Tp, H, W]
        rgb_pred = (object_rgbs * mask_prob.unsqueeze(3)).sum(dim=2) + bg_rgb * (1 - total_mask.unsqueeze(2))
        
        return rgb_pred
```

**预期收益**：
- 每个物体独立重建，更可解释
- 背景单独建模，减少干扰
- 预计 RGB 质量提升 5-10%

**实现难度**：高（需要修改 RGBDecoder）

### 3. KL 正则化 → 改进特征学习
**映射位置**：`model/ai_model/loss.py`

**当前问题**：
- 特征学习没有正则化
- 可能过拟合训练数据

**具体改进**：
```python
# 添加 KL 正则化
def feature_kl_loss(tokens, valid_mask):
    """
    tokens: [B, T, N, D]
    """
    B, T, N, D = tokens.shape
    
    # 假设 tokens 应该接近标准正态分布
    mu = tokens.mean(dim=(0, 1))  # [N, D]
    sigma = tokens.std(dim=(0, 1))  # [N, D]
    
    # KL 散度
    kl = 0.5 * (mu**2 + sigma**2 - torch.log(sigma**2 + 1e-8) - 1).sum(dim=-1)
    
    # 只考虑有效物体
    kl = (kl * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)
    
    return kl
```

**预期收益**：
- 正则化特征学习，减少过拟合
- 特征分布更规整，提升泛化能力
- 预计在新场景上的性能提升 5-10%

**实现难度**：低（只需在 loss.py 中添加）

## 实验结果（关键指标）
| 数据集 | 指标 | MONet | IODINE | Slot Attention |
|--------|------|-------|--------|---------------|
| CLEVR | ARI | 0.85 | 0.83 | **0.90** |
| Multi-dSprites | ARI | 0.78 | 0.75 | **0.82** |
| Tetrominoes | ARI | 0.92 | 0.90 | **0.96** |

- MONet ARI 0.85（CLEVR），Slot Attention 0.90
- MONet 是物体中心表示的奠基工作
- 迭代注意力机制影响了后续 SAVi、SlotFormer 等工作
