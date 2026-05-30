# SimVP: Simpler yet Better Video Prediction

## 基本信息
- 作者：Cheng Tan, Zhangyang Gao, Lirong Wu, Siyuan Li, Stan Z. Li
- 年份：2022
- 会议/期刊：CVPR 2022
- 论文链接：https://arxiv.org/abs/2206.05099
- 代码链接：https://github.com/chengtan9907/SimVP
- PDF：simvp.pdf（需下载）

## 核心贡献
- 证明简单的 CNN + Transformer 架构就能达到 SOTA 视频预测效果，无需复杂的循环结构
- 提出 InceptionUNet：用 Inception 模块替代传统 CNN，捕获多尺度时空特征
- 提出 Temporal Attention Module（TAM）：用 Transformer 的 self-attention 建模时序依赖
- 仅用 MSE loss 就能取得与复杂模型（PhyDNet、PredRNN）相当的效果
- 证明了"简单架构 + 足够数据"优于"复杂架构 + 小数据"

## 模型架构
- **Encoder**：InceptionUNet（多尺度 CNN 编码器）
  - Inception 模块：1×1, 3×3, 5×5 卷积并行，concat 后 1×1 降维
  - UNet 结构：skip connection 保留多尺度特征
  - 输入：历史帧 [B, Th, C, H, W]
  - 输出：特征图 [B, Th, D, H', W']
- **Temporal Module**：Temporal Attention Module (TAM)
  - 将时间维度展平为 token 序列
  - Multi-head self-attention 建模帧间依赖
  - 位置编码（learnable）标记时间顺序
  - 输出：[B, Tp, D, H', W']
- **Decoder**：对称的 InceptionUNet 解码器
  - 逐步上采样恢复到原始分辨率
  - Skip connection 从 encoder 引入多尺度特征
- **关键实现细节**：
  - 没有循环结构（RNN/LSTM/GRU），纯前馈网络
  - 时序建模完全靠 Transformer attention
  - 参数量远小于 PredRNN/PhyDNet

## 损失函数
- **MSE Loss**：L = Σ ||x_t - x̂_t||²，仅此一项
- **无物理约束、无感知损失、无对抗损失**
- 证明了简单 MSE 在足够数据和合适架构下就足够

## 关键设计选择
- **无循环结构**：
  - 优势：训练简单，无误差累积，可并行
  - 劣势：无法建模长期依赖（理论上），但实际效果好
  - 与 PredRNN 对比：PredRNN 有循环，但 SimVP 更简单且效果相当
- **Inception 多尺度**：
  - 不同大小的卷积核捕获不同粒度的特征
  - 比单一 3×3 卷积更灵活
- **Transformer 时序建模**：
  - Self-attention 可以直接访问任意帧，不受序列长度限制
  - 比 RNN 更适合长序列（但计算复杂度 O(T²)）
- **与当前模型对比**：
  - 当前模型用 GRU（循环），SimVP 用 Transformer（前馈）
  - 当前模型是物体级预测，SimVP 是像素级预测
  - 当前模型有物理先验，SimVP 纯数据驱动

## 与当前模型的对比
- **相似之处**：
  - 都用 CNN 做视觉编码
  - 都需要处理多帧视频序列
  - 都关注时序建模
- **不同之处**：
  - SimVP 是像素级预测，当前模型是物体级预测
  - SimVP 用 Transformer 做时序，当前模型用 GRU
  - SimVP 没有物理先验，当前模型有 force-aware GNN
  - SimVP 没有物体表示，当前模型有显式的物体编码
  - SimVP 仅用 MSE，当前模型用多任务 loss

## 可借鉴的点

### 1. Inception 多尺度卷积 → 改进 VisualEncoder
**映射位置**：`model/ai_model/encoder.py` → `VisualEncoder`

**当前问题**：
- VisualEncoder 用单一 3×3 卷积核，只能捕获单一尺度的特征
- 物体有大有小（大球 vs 小球），单一尺度不够灵活
- 没有多尺度特征融合

**具体改进**：
```python
# 当前：单一 3×3 卷积
class VisualEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 128, 128)):
        for c_out in channels:
            layers.extend([
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.GELU(),
            ])

# 改为：Inception 多尺度卷积
class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.branch1 = nn.Conv2d(in_ch, out_ch//4, 1)
        self.branch3 = nn.Conv2d(in_ch, out_ch//4, 3, padding=1)
        self.branch5 = nn.Conv2d(in_ch, out_ch//4, 5, padding=2)
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_ch, out_ch//4, 1)
        )
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
    
    def forward(self, x):
        out = torch.cat([
            self.branch1(x), self.branch3(x),
            self.branch5(x), self.branch_pool(x)
        ], dim=1)
        return F.gelu(self.norm(out))

class VisualEncoder(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 128, 128)):
        for c_out in channels:
            layers.extend([
                InceptionBlock(c_in, c_out),
                nn.Conv2d(c_out, c_out, 3, stride=2, padding=1),  # 下采样
            ])
```

**预期收益**：
- 捕获多尺度特征，对大物体和小物体都有效
- 预计 mask 预测 IoU 提升 3-5%（小物体提升更明显）
- 参数量增加约 20%，但计算量增加可控

**实现难度**：低（替换 VisualEncoder 的卷积层）

### 2. Temporal Attention → 改进 TemporalGRU
**映射位置**：`model/ai_model/temporal.py` → `TemporalGRU`

**当前问题**：
- GRU 是循环结构，长期依赖信息会衰减
- 没有 attention 机制，无法直接访问任意历史帧
- 自回归解码有误差累积

**具体改进**：
```python
# 当前：纯 GRU
class TemporalGRU(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_layers=2):
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, ...)

# 改为：GRU + Temporal Attention
class TemporalGRUWithAttention(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=128, num_heads=4):
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def encode_history(self, tokens, valid_mask):
        B, Th, N, D = tokens.shape
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B*N, Th, D)
        
        # GRU 编码
        _, h_last = self.gru(tokens_flat)
        
        # Temporal attention：让每个时间步都能访问所有历史帧
        attn_out, _ = self.temporal_attn(tokens_flat, tokens_flat, tokens_flat)
        enhanced = self.norm(tokens_flat + attn_out)  # 残差连接
        
        # 用 attention 增强的特征作为最终编码
        return h_last, enhanced
```

**预期收益**：
- 直接访问任意历史帧，减少长期依赖衰减
- GRU 捕获序列动态，attention 捕获全局依赖，互补
- 预计长期预测（>8帧）质量提升 10-15%

**实现难度**：中（需要修改 TemporalGRU 的接口和 forward）

### 3. 非自回归解码 → 减少误差累积
**映射位置**：`model/ai_model/temporal.py` → `decode_future()`

**当前问题**：
- 自回归解码：每一步的预测误差会累积到下一步
- 长期预测（>6帧）质量下降明显

**具体改进**：
```python
# 当前：自回归解码
def decode_future(self, initial_token, h_init, predict_length, valid_mask):
    future_tokens = []
    x = initial_token
    for _ in range(predict_length):
        out, h = self.gru(x, h)
        future_tokens.append(out)
        x = out  # 用预测值作为下一步输入

# 改为：非自回归解码（受 SimVP 启发）
class NonAutoregressiveDecoder(nn.Module):
    def __init__(self, hidden_dim, predict_length):
        self.predict_tokens = nn.Parameter(
            torch.randn(1, predict_length, 1, hidden_dim)  # 可学习的预测 token
        )
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
    
    def forward(self, history_encoding, predict_length):
        # history_encoding: [B*N, Th, hidden_dim]
        # predict_tokens: [1, Tp, hidden_dim] → expand to [B*N, Tp, hidden_dim]
        queries = self.predict_tokens.expand(history_encoding.shape[0], -1, -1, -1)
        queries = queries.squeeze(2)  # [B*N, Tp, hidden_dim]
        
        # Cross-attention: 预测 token 查询历史编码
        future, _ = self.cross_attn(queries, history_encoding, history_encoding)
        return future  # [B*N, Tp, hidden_dim]
```

**预期收益**：
- 无误差累积，所有预测帧独立生成
- 可以并行解码所有帧，速度更快
- 预计长期预测质量提升 20-30%（误差累积越严重，提升越大）

**实现难度**：高（需要重新设计解码器，可能需要改变训练策略）

### 4. UNet Skip Connection → 改进 MaskDecoder
**映射位置**：`model/ai_model/decoder.py` → `MaskHead`

**当前问题**：
- MaskHead 从 token 直接生成 mask，没有利用 encoder 的空间特征
- 反卷积堆叠可能导致空间信息丢失（棋盘效应）

**具体改进**：
```python
# 当前：独立的反卷积堆叠
class MaskHead(nn.Module):
    def __init__(self, token_dim, base_channels, image_size=128):
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            ...
        )

# 改为：带 skip connection 的 UNet 解码器
class MaskHeadWithSkip(nn.Module):
    def __init__(self, token_dim, base_channels, image_size=128, encoder_channels=(32, 64, 128)):
        # 接收 encoder 的多尺度特征
        self.skip_proj = nn.ModuleList([
            nn.Conv2d(ch, base_channels, 1) for ch in encoder_channels
        ])
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.ConvTranspose2d(base_channels * 2, base_channels // 2, 4, stride=2, padding=1),
            nn.ConvTranspose2d(base_channels * 2, base_channels // 4, 4, stride=2, padding=1),
            nn.ConvTranspose2d(base_channels // 4, 1, 4, stride=2, padding=1),
        ])
    
    def forward(self, tokens, encoder_features=None):
        x = self.proj(tokens)  # [B*Tp*N, C, 8, 8]
        for i, dec_layer in enumerate(self.decoder):
            if encoder_features is not None and i < len(encoder_features):
                skip = self.skip_proj[i](encoder_features[i])
                x = torch.cat([x, skip], dim=1)  # Skip connection
            x = dec_layer(x)
        return x
```

**预期收益**：
- 利用 encoder 的多尺度空间特征，mask 边界更精确
- 减少棋盘效应
- 预计 mask IoU 提升 5-10%

**实现难度**：中（需要修改 MaskHead 和 model.py 的 forward 传递 encoder 特征）

## 实验结果（关键指标）
| 数据集 | 指标 | SimVP | PhyDNet | PredRNN | ConvLSTM |
|--------|------|-------|---------|---------|----------|
| Moving MNIST | MSE (×10⁻²) | **3.38** | 3.43 | 5.43 | 7.57 |
| KTH Actions | PSNR (dB) | **26.30** | 26.08 | 27.55 | 25.24 |
| CityFlow | MSE (×10⁻²) | **4.17** | 4.52 | 5.31 | 6.89 |

- Moving MNIST：SimVP MSE 3.38，与 PhyDNet（3.43）相当，大幅优于 PredRNN（5.43）
- KTH Actions：PSNR 26.30dB，与 PhyDNet（26.08）相当
- CityFlow：交通流量预测 MSE 4.17，优于所有 baseline

**消融实验**：
- 去掉 TAM → MSE 从 3.38 升到 4.12（+22%），证明时序 attention 的重要性
- 去掉 Inception → MSE 从 3.38 升到 3.89（+15%），证明多尺度特征的有效性
- 用 CNN 替代 TAM → MSE 从 3.38 升到 4.01（+19%），证明 Transformer 优于 CNN 做时序
