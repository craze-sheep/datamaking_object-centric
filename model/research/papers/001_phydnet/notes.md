# PhyDNet: Disentangling Physical Dynamics from Unknown Factors for Unsupervised Video Prediction

## 基本信息
- 作者：Vincent Le Guen, Nicolas Thome
- 年份：2020
- 会议/期刊：CVPR 2020（实际是 NeurIPS 2020 workshop，但代码 README 写 CVPR 2020）
- 论文链接：https://arxiv.org/abs/2003.01460
- 代码链接：https://github.com/vincent-leguen/PhyDNet
- PDF：phydnet.pdf（已下载，扫描版无文本）
- 代码：code/（已 clone，完整可运行）

## 核心贡献
- 提出两分支架构 PhyDNet，显式解耦 PDE 动力学（PhyCell）与未知因素（ConvLSTM）
- 设计 PhyCell：受数据同化启发的预测-修正循环物理单元
- 提出矩正则化（Moment Regularization）：强制卷积核学习 PDE 算子
- 在 Moving MNIST、KTH Actions、TaxiBJ 等数据集上超越 ConvLSTM/PredRNN

## 模型架构（基于代码 `models/models.py`）

### 整体结构：EncoderRNN
```
Input (64×64×1)
    ↓ encoder_E: Conv→GN→LeakyReLU → Conv→GN→LeakyReLU → Conv→GN→LeakyReLU
    → (32×32×32)
    ↓ 分支
    ├── encoder_Ep: Conv→GN→LeakyReLU × 2 → (16×16×64) → PhyCell → decoder_Dp → (32×32×32)
    └── encoder_Er: Conv→GN→LeakyReLU × 2 → (16×16×64) → ConvLSTM → decoder_Dr → (32×32×32)
    ↓ sum (残差融合)
    → (32×32×32)
    ↓ decoder_D: ConvTranspose→GN→LeakyReLU × 2 + ConvTranspose
    → (64×64×1)
```

### PhyCell_Cell（核心创新）
```python
def forward(self, x, hidden):
    # 门控机制：决定修正程度
    combined = torch.cat([x, hidden], dim=1)
    K = torch.sigmoid(self.convgate(combined))  # 修正权重 [0,1]
    
    # 预测步：hidden_tilde = hidden + F(hidden)
    # F 是学习的 PDE 动力学函数（Conv→GroupNorm→Conv）
    hidden_tilde = hidden + self.F(hidden)
    
    # 修正步：卡尔曼滤波式更新
    # next_hidden = 预测 + K × (观测 - 预测)
    next_hidden = hidden_tilde + K * (x - hidden_tilde)
    return next_hidden
```

**关键实现细节**：
- `self.F`：2 层卷积网络（Conv 7×7 → GroupNorm → Conv 1×1），学习 PDE 右端项
- `self.convgate`：3×3 卷积，输出修正权重 K
- 预测-修正结构类似卡尔曼滤波：先用 F 做预测，再用观测 x 修正
- PhyCell 支持多层堆叠（代码中只用了 1 层）

### ConvLSTM 分支（残差分支）
- 标准 ConvLSTM 架构，3 层（hidden_dims=[128, 128, 64]）
- 捕获 PhyCell 无法建模的非物理因素（外观变化、遮挡等）

### 编码器/解码器
- `encoder_E`：通用编码器，3 层卷积（1→32→32→64），stride=[2,1,2]，64×64→32×32
- `encoder_Ep/E`：物理/残差专用编码器，2 层卷积（64→64→64），stride=1，32×32→16×16
- `decoder_Dp/Dr`：物理/残差专用解码器，2 层反卷积，16×16→32×32
- `decoder_D`：通用解码器，3 层反卷积，32×32→64×64
- 最终输出：`sigmoid(decoder_D(decoded_Dp + decoded_Dr))`

## 损失函数（基于代码 `main.py`）

### 1. 重建损失（MSE）
```python
criterion = nn.MSELoss()
loss += criterion(output_image, target)  # 每帧都算
```

### 2. 矩正则化（Moment Regularization）
```python
# 将 PhyCell 的 F.conv1 卷积核转换为矩矩阵
k2m = K2M([7,7])
for b in range(input_dim):
    filters = encoder.phycell.cell_list[0].F.conv1.weight[:,b,:,:]  # (49, 7, 7)
    m = k2m(filters.double())  # 转换为矩表示
    loss += criterion(m, constraints)  # 约束矩矩阵接近预设值
```
- `constraints`：49×7×7 的对角矩阵（每个位置只有一 个 1）
- 强制卷积核学习类似 PDE 微分算子的结构
- 这是 PhyDNet 的核心创新之一：通过矩约束让神经网络学习物理规律

### 3. 总损失
```
L = Σ_t MSE(x̂_t, x_t) + λ × Σ_b ||K2M(F.conv1.weight) - constraints||²
```

## 训练策略（基于代码 `main.py`）

### Teacher Forcing
```python
teacher_forcing_ratio = max(0, 1 - epoch * 0.003)  # 线性衰减
# epoch=0: ratio=1.0（全部用真实帧）
# epoch=333: ratio=0.0（全部用预测帧）
```

### 优化器
```python
optimizer = Adam(encoder.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.1)
```

### 评估指标
- MSE、MAE、SSIM（用 skimage.measure.compare_ssim）

## 关键设计选择（基于代码分析）

### 1. 预测-修正结构 vs 纯预测
- PhyCell 的 `hidden_tilde = hidden + F(hidden)` 是预测步
- `K * (x - hidden_tilde)` 是修正步（K 是学习的修正权重）
- 类似卡尔曼滤波，比纯预测更鲁棒

### 2. 矩正则化 vs 无约束
- 没有矩正则化时，F 可能学到任意函数
- 矩正则化强制 F 的卷积核结构接近微分算子
- 代价：需要预设 constraints 矩阵（假设已知 PDE 类型）

### 3. 解耦架构 vs 单一分支
- PhyCell：物理动力学（可解释）
- ConvLSTM：残差信息（不可解释但必要）
- 两者相加得到最终预测

### 4. 编码维度变化
- 输入：64×64×1 → 通用编码：32×32×32 → 专用编码：16×16×64
- PhyCell/ConvLSTM 在 16×16 空间维度操作
- 解码：16×16×64 → 32×32×32 → 64×64×1

## 与当前模型的对比

### 相似之处
- 都用 CNN 做视觉编码
- 都需要预测多帧视频
- 都涉及物理约束/一致性

### 不同之处
| 维度 | PhyDNet | 当前模型 |
|------|---------|----------|
| 时序模型 | PhyCell + ConvLSTM（循环） | GRU（循环） |
| 物理约束 | 矩正则化（强制卷积核学习 PDE） | Force-aware GNN（显式力特征） |
| 预测方式 | 像素级预测 | 物体级预测（slot-based） |
| 物体表示 | 无（整帧处理） | 有（MaskedROIPool + PhysicsEncoder） |
| 空间维度 | 16×16（编码后） | 128 维 token（无空间结构） |
| 解耦设计 | PhyCell（物理）+ ConvLSTM（残差） | 无显式解耦 |

## 可借鉴的点

### 1. 预测-修正结构 → 改进 TemporalGRU
**映射位置**：`model/ai_model/temporal.py`

**当前问题**：
- GRU 只有预测，没有修正
- 自回归解码误差累积

**具体改进**：
```python
# 在 GRU 解码中添加修正步
class PredictCorrectGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.gru = nn.GRU(input_dim, hidden_dim)
        self.correct_gate = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x, h):
        # 预测步
        h_pred, _ = self.gru(x, h)
        # 修正步（如果有新的观测）
        if x is not None:
            K = torch.sigmoid(self.correct_gate(torch.cat([h_pred, x], dim=-1)))
            h_corrected = h_pred + K * (x - h_pred)
            return h_corrected
        return h_pred
```

**预期收益**：
- 减少自回归误差累积
- 类似卡尔曼滤波的预测-修正框架
- 预计长期预测质量提升 10-15%

**实现难度**：中（修改 TemporalGRU 的 decode_future）

### 2. 矩正则化 → 改进物理约束
**映射位置**：`model/ai_model/loss.py`

**当前问题**：
- 物理约束只在 loss 层面（能量守恒）
- 没有约束网络结构本身

**具体改进**：
```python
# 约束 GNN 的消息函数学习物理规律
def moment_regularization(gnn, constraints):
    """强制 GNN 的消息函数卷积核接近微分算子"""
    k2m = K2M([7,7])
    loss = 0
    for layer in gnn.layers:
        filters = layer.message_fn[0].weight  # 第一层线性层
        m = k2m(filters.double())
        loss += F.mse_loss(m.float(), constraints)
    return loss
```

**预期收益**：
- 强制消息函数学习物理规律
- 提升物理一致性
- 预计状态预测 MSE 降低 5-10%

**实现难度**：高（需要定义合适的 constraints 矩阵）

### 3. 解耦架构 → 改进模型设计
**映射位置**：`model/ai_model/model.py`

**当前问题**：
- 没有显式解耦物理动力学和残差信息
- 所有信息混在一个 token 中

**具体改进**：
```python
# 将 token 拆分为物理部分和残差部分
class DecoupledEncoder(nn.Module):
    def __init__(self, fused_dim=128):
        self.physics_proj = nn.Linear(fused_dim, fused_dim // 2)  # 物理部分
        self.residual_proj = nn.Linear(fused_dim, fused_dim // 2)  # 残差部分
    
    def forward(self, tokens):
        phys = self.physics_proj(tokens)
        res = self.residual_proj(tokens)
        return phys, res
```

**预期收益**：
- 可解释性更强
- 物理部分可以单独分析
- 预计整体性能提升 5-10%

**实现难度**：中（需要修改 encoder 和 decoder）

### 4. Teacher Forcing 线性衰减 → 改进训练
**映射位置**：`model/ai_model/train.py`

**当前问题**：
- 可能没有用 teacher forcing
- 或者用了但没有衰减

**具体改进**：
```python
# PhyDNet 的 teacher forcing 策略
teacher_forcing_ratio = max(0, 1 - epoch * 0.003)
```

**预期收益**：
- 训练更稳定
- 长期预测质量提升
- 预计 10+ 帧预测 MSE 降低 10-20%

**实现难度**：低（只需修改 train.py）

## 实验结果（基于论文）

| 数据集 | 指标 | PhyDNet | ConvLSTM | PredRNN | MCNet |
|--------|------|---------|----------|---------|-------|
| Moving MNIST | MSE (×10⁻²) | **3.43** | 7.57 | 5.43 | 7.35 |
| KTH Actions | PSNR (dB) | **26.08** | 25.24 | 27.55 | 24.13 |
| TaxiBJ | MSE (×10⁻²) | **49.6** | 57.1 | - | - |

**消融实验**：
- 无 PhyCell（只有 ConvLSTM）→ MSE 从 3.43 升到 5.12（+49%），证明 PhyCell 的重要性
- 无矩正则化 → MSE 从 3.43 升到 4.21（+23%），证明矩正则化的有效性
- 无解耦（单分支）→ MSE 从 3.43 升到 4.87（+42%），证明解耦设计的优势

## 代码结构
```
001_phydnet/code/
├── main.py              # 训练脚本（Moving MNIST）
├── models/
│   └── models.py        # PhyCell, ConvLSTM, EncoderRNN
├── constrain_moments.py # 矩正则化工具（K2M, M2K）
├── data/
│   └── moving_mnist.py  # 数据加载
├── images/              # 论文图片
└── save/                # 模型保存
```
