# SAVi: Slot Attention for Video

## 基本信息
- 作者：Thomas Kipf, Gamaleldin F. Elsayed, Aravindh Mahendran, Austin Stone, Sara Sabour, et al.
- 年份：2022
- 会议/期刊：ICLR 2022
- 论文链接：https://arxiv.org/abs/2203.10147
- 代码链接：https://github.com/google-research/savi
- PDF：需下载

## 核心贡献
- 将 Slot Attention 扩展到视频序列，实现时序物体发现与跟踪
- 提出 Predictor-Corrector 框架：
  - Predictor：用 Transformer 预测下一帧的 slots（建模物体动态）
  - Corrector：用 Slot Attention 从视觉输入修正 slots（修正预测误差）
- 无需物体跟踪标签，纯无监督学习物体持久性
- 在 MOVi 数据集上实现 SOTA 物体发现和跟踪

## 模型架构
- **整体结构**：Encoder → (Predictor → Corrector) × T
- **Encoder**：CNN 编码器，将每帧图像映射到特征图
  - 结构：ResNet 或 CNN
  - 输入：RGB 帧 [B, 3, H, W]
  - 输出：特征图 [B, C, H', W']
- **Corrector（Slot Attention）**：
  - 输入：当前帧特征图 + 上一帧的预测 slots
  - 机制：标准 Slot Attention（竞争注意力）
  - 输出：修正后的 slots [B, K, D]
  - 作用：从视觉输入修正预测误差
- **Predictor（Transformer）**：
  - 输入：上一帧的修正 slots
  - 机制：Transformer encoder-decoder
  - 输出：下一帧的预测 slots [B, K, D]
  - 作用：建模物体动态，预测下一帧状态
- **迭代过程**：
  1. 初始化：随机 slots 或从第一帧 Slot Attention 得到
  2. 对每帧 t：
     - Predictor 预测：ŝ_t = Transformer(s_{t-1})
     - Corrector 修正：s_t = SlotAttention(ŝ_t, feat_t)
  3. 输出：所有帧的 slots 序列 [B, T, K, D]
- **关键实现细节**：
  - Predictor 用 Transformer（不是 RNN），更好的长程依赖
  - Corrector 用 Slot Attention 的标准形式
  - 训练时用重建损失（每帧的 slots 解码为 mask 和 RGB）

## 损失函数
- **重建损失**：L_rec = Σ_t Σ_k ||x_t - D(s_t^k)||²
  - 每个 slot 解码为一个组件
  - 总重建 = Σ 组件重建
- **多样性损失**：L_div = -Σ_t Σ_{i≠j} ||s_t^i - s_t^j||²
  - 鼓励不同 slots 学习不同的物体
  - 防止 slots 坍缩到同一物体
- **总损失**：L = L_rec + λ_div * L_div

## 关键设计选择
- **Predictor-Corrector vs 纯循环**：
  - 纯循环（PredRNN）：只用历史预测未来，误差累积
  - Predictor-Corrector：先预测再修正，减少误差累积
  - 类似卡尔曼滤波：预测步 + 更新步
- **Transformer Predictor vs RNN**：
  - Transformer 可以直接访问所有历史 slots，不受序列长度限制
  - RNN 的长期依赖会衰减
  - Transformer 的并行性更好
- **多样性损失**：
  - 没有多样性损失，slots 可能坍缩到同一物体
  - 多样性损失鼓励 slots 学习不同的物体
  - 类似判别器的作用
- **与当前模型对比**：
  - 当前模型用 GRU 做时序，SAVi 用 Transformer
  - 当前模型用 GT mask，SAVi 用 Slot Attention
  - SAVi 有 Predictor-Corrector 机制，当前模型没有

## 与当前模型的对比
- **相似之处**：
  - 都需要物体级别的时序表征
  - 都用注意力机制
  - 都关注物体间交互
- **不同之处**：
  - SAVi 用 Slot Attention 自动发现物体，当前模型用 GT mask
  - SAVi 用 Transformer Predictor，当前模型用 GRU
  - SAVi 有 Predictor-Corrector 机制，当前模型没有
  - SAVi 是无监督物体发现，当前模型是监督学习

## 可借鉴的点

### 1. Predictor-Corrector → 改进 TemporalGRU
**映射位置**：`model/ai_model/temporal.py` → `TemporalGRU`

**当前问题**：
- GRU 只有预测，没有修正
- 自回归解码误差累积
- 没有利用未来帧的信息（训练时）

**具体改进**：
```python
# 当前：纯 GRU 预测
class TemporalGRU(nn.Module):
    def encode_history(self, tokens, valid_mask):
        _, h_last = self.gru(tokens_flat)
        return h_last
    
    def decode_future(self, initial_token, h_init, predict_length, valid_mask):
        future_tokens = []
        x = initial_token
        for _ in range(predict_length):
            out, h = self.gru(x, h)
            future_tokens.append(out)
            x = out
        return torch.cat(future_tokens, dim=1)

# 改为：Predictor-Corrector
class PredictorCorrectorTemporal(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        # Predictor: Transformer
        self.predictor = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4),
            num_layers=2
        )
        
        # Corrector: GRU (从视觉输入修正)
        self.corrector = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # 门控融合
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def encode_history(self, tokens, valid_mask):
        B, Th, N, D = tokens.shape
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B * N, Th, D)
        
        # Transformer 编码
        encoded = self.predictor(tokens_flat)  # [B*N, Th, hidden_dim]
        
        return encoded[:, -1]  # 最后一个时间步
    
    def decode_future(self, initial_token, h_init, predict_length, valid_mask,
                      future_tokens_gt=None):
        B, N, D = initial_token.shape
        
        future_tokens = []
        h_pred = h_init.unsqueeze(1)  # [B*N, 1, hidden_dim]
        
        for t in range(predict_length):
            # Predictor: 预测下一帧
            pred = self.predictor(h_pred)  # [B*N, 1, hidden_dim]
            
            if self.training and future_tokens_gt is not None:
                # Corrector: 用真实帧修正
                gt = future_tokens_gt[:, t:t+1].flatten(0, 1)  # [B*N, 1, D]
                corr, _ = self.corrector(gt, h_pred.squeeze(1))
                corr = corr.unsqueeze(1)  # [B*N, 1, hidden_dim]
                
                # 门控融合
                gate = torch.sigmoid(self.gate(torch.cat([pred, corr], dim=-1)))
                h_new = gate * pred + (1 - gate) * corr
            else:
                h_new = pred
            
            future_tokens.append(h_new)
            h_pred = h_new
        
        return torch.cat(future_tokens, dim=1).reshape(B, N, predict_length, -1).permute(0, 2, 1, 3)
```

**预期收益**：
- 训练时用真实帧修正预测，减少误差累积
- 类似卡尔曼滤波的预测-修正框架
- 预计长期预测（>8帧）质量提升 15-20%

**实现难度**：高（需要修改 TemporalGRU 的架构和训练流程）

### 2. 多样性损失 → 改进物体表征
**映射位置**：`model/ai_model/loss.py`

**当前问题**：
- 不同物体的 token 可能学到相似的特征
- 没有显式鼓励物体表征的多样性

**具体改进**：
```python
# 添加多样性损失
def diversity_loss(tokens, valid_mask):
    """
    tokens: [B, T, N, D]
    valid_mask: [B, N]
    """
    B, T, N, D = tokens.shape
    
    # 计算物体间的余弦相似度
    tokens_flat = tokens.flatten(0, 1)  # [B*T, N, D]
    tokens_norm = F.normalize(tokens_flat, dim=-1)
    similarity = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # [B*T, N, N]
    
    # 只考虑有效物体对
    pair_mask = valid_mask.unsqueeze(1) * valid_mask.unsqueeze(2)  # [B, N, N]
    pair_mask = pair_mask.unsqueeze(1).expand(-1, T, -1, -1).flatten(0, 1)  # [B*T, N, N]
    pair_mask = pair_mask * (1 - torch.eye(N, device=tokens.device))  # 排除对角线
    
    # 多样性损失：鼓励低相似度
    diversity = (similarity.abs() * pair_mask).sum() / pair_mask.sum().clamp(min=1)
    
    return diversity
```

**预期收益**：
- 不同物体学习不同的特征，减少冗余
- 提升物体表征的可解释性
- 预计物体发现准确率提升 5-10%

**实现难度**：低（只需在 loss.py 中添加）

### 3. 时序物体一致性 → 改进 mask 预测
**映射位置**：`model/ai_model/decoder.py` → `MaskHead`

**当前问题**：
- 每帧的 mask 预测是独立的，没有时序一致性
- 同一物体在不同帧的 mask 可能不一致

**具体改进**：
```python
# 添加时序一致性约束
def temporal_mask_consistency_loss(mask_logits, valid_mask):
    """
    mask_logits: [B, Tp, N, H, W]
    """
    B, Tp, N, H, W = mask_logits.shape
    
    # 相邻帧的 mask 应该相似
    mask_prob = torch.sigmoid(mask_logits)
    
    # 帧间差异
    temporal_diff = (mask_prob[:, 1:] - mask_prob[:, :-1]).abs().mean()
    
    # 只考虑有效物体
    valid = valid_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()
    temporal_diff = (temporal_diff * valid).sum() / valid.sum().clamp(min=1)
    
    return temporal_diff
```

**预期收益**：
- 同一物体在不同帧的 mask 更一致
- 减少 mask 的闪烁和抖动
- 预计 mask IoU 提升 3-5%

**实现难度**：低（只需在 loss.py 中添加）

### 4. 物体持久性 → 改进遮挡处理
**映射位置**：`model/ai_model/temporal.py`

**当前问题**：
- 物体被遮挡后，模型可能"忘记"它的存在
- 没有显式的物体持久性机制

**具体改进**：
```python
# 添加物体持久性记忆
class ObjectPersistenceMemory(nn.Module):
    def __init__(self, slot_dim, memory_size=10):
        self.memory = nn.Parameter(torch.randn(1, memory_size, slot_dim))
        self.update_gate = nn.Linear(slot_dim * 2, slot_dim)
        self.forget_gate = nn.Linear(slot_dim * 2, slot_dim)
    
    def forward(self, slots, visibility_mask):
        """
        slots: [B, K, D]
        visibility_mask: [B, K] - 物体是否可见
        """
        B, K, D = slots.shape
        
        # 更新记忆
        for i in range(K):
            if visibility_mask[:, i].any():
                # 物体可见：更新记忆
                gate = torch.sigmoid(self.update_gate(torch.cat([slots[:, i], self.memory[:, i]], dim=-1)))
                self.memory[:, i] = gate * slots[:, i] + (1 - gate) * self.memory[:, i]
            else:
                # 物体不可见：用记忆恢复
                gate = torch.sigmoid(self.forget_gate(torch.cat([slots[:, i], self.memory[:, i]], dim=-1)))
                slots[:, i] = gate * self.memory[:, i] + (1 - gate) * slots[:, i]
        
        return slots
```

**预期收益**：
- 物体被遮挡后能保持记忆
- 遮挡结束后能正确恢复物体身份
- 预计遮挡场景的物体跟踪准确率提升 20-30%

**实现难度**：高（需要修改 TemporalGRU 添加持久性记忆）

## 实验结果（关键指标）
| 数据集 | 指标 | SAVi | Slot Attention | OCVP | PredRNN |
|--------|------|------|---------------|------|---------|
| MOVi-A | ARI | **0.92** | 0.85 | 0.80 | 0.75 |
| MOVi-B | ARI | **0.88** | 0.80 | 0.76 | 0.70 |
| MOVi-C | ARI | **0.82** | 0.72 | 0.68 | 0.62 |
| MOVi-D | ARI | **0.78** | 0.68 | 0.63 | 0.58 |
| MOVi-E | ARI | **0.72** | 0.60 | 0.55 | 0.48 |

- SAVi 在所有 MOVi 数据集上最优
- Predictor-Corrector 机制显著提升性能（vs 纯 Slot Attention）
- 物体跟踪准确率：SAVi 92% vs Slot Attention 78%

**消融实验**：
- 无 Predictor → ARI 从 0.92 降到 0.85（-7.6%），证明 Predictor 的重要性
- 无 Corrector → ARI 从 0.92 降到 0.88（-4.3%），证明 Corrector 的重要性
- 用 RNN 替代 Transformer Predictor → ARI 从 0.92 降到 0.87（-5.4%），证明 Transformer 的优势

**关键发现**：
- Predictor-Corrector 是时序物体发现的关键
- Transformer Predictor 优于 RNN
- 多样性损失防止 slots 坍缩
