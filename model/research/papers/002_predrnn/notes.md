# PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning

## 基本信息
- 作者：Yunbo Wang, Mingsheng Long, Jianmin Wang, Philip S. Yu
- 年份：2017 (NeurIPS), 2022 (TPAMI 扩展为 PredRNN-V2)
- 会议/期刊：NeurIPS 2017 / TPAMI 2022
- 论文链接：https://papers.nips.cc/paper/6689-predrnn-recurrent-neural-networks-for-predictive-learning-using-spatiotemporal-lstms
- 代码链接：https://github.com/thuml/predrnn-pytorch
- PDF：无法下载（NeurIPS 网站访问受限）
- 代码：code/（已 clone，完整可运行）

## 核心贡献
- 提出时空 LSTM（ST-LSTM）：在同一 cell 内融合空间记忆（M_t）和时空记忆（C_t）
- 设计 Zigzag 记忆流：M_t 跨层传播（从底层到顶层），实现跨层时空信息共享
- PredRNN-V2：记忆解耦损失 + 反向 Scheduled Sampling + 动作条件预测

## 模型架构（基于代码 `core/layers/SpatioTemporalLSTMCell.py`）

### ST-LSTM Cell（核心创新）
```python
def forward(self, x_t, h_t, c_t, m_t):
    # 输入投影
    x_concat = self.conv_x(x_t)  # → 7 个门控信号
    h_concat = self.conv_h(h_t)  # → 4 个门控信号
    m_concat = self.conv_m(m_t)  # → 3 个门控信号
    
    # C_t 更新（标准 LSTM 门控，基于 x 和 h）
    i_t = sigmoid(i_x + i_h)        # 输入门
    f_t = sigmoid(f_x + f_h + 1.0)  # 遗忘门（bias=1.0）
    g_t = tanh(g_x + g_h)           # 候选值
    c_new = f_t * c_t + i_t * g_t   # 状态更新
    
    # M_t 更新（独立门控，基于 x 和 m）
    i_t_prime = sigmoid(i_x_prime + i_m)
    f_t_prime = sigmoid(f_x_prime + f_m + 1.0)
    g_t_prime = tanh(g_x_prime + g_m)
    m_new = f_t_prime * m_t + i_t_prime * g_t_prime
    
    # 输出门（融合 c 和 m）
    mem = cat(c_new, m_new)
    o_t = sigmoid(o_x + o_h + conv_o(mem))
    h_new = o_t * tanh(conv_last(mem))
    
    return h_new, c_new, m_new
```

**关键实现细节**：
- **双记忆单元**：C_t（层内传播）和 M_t（跨层传播）
- **独立门控**：C_t 用 (i, f, g) 门控，M_t 用 (i', f', g') 门控，输入信号不同
- **M_t 跨层共享**：在模型层面，M_t 从 layer 0 传到 layer L-1（zigzag）
- **输出融合**：h_new = o * tanh(conv([c_new, m_new]))

### Zigzag 记忆流（基于代码 `core/models/predrnn.py`）
```python
# 对每个时间步
for t in range(total_length - 1):
    # 第 0 层：输入 x_t，更新 h[0], c[0], memory
    h_t[0], c_t[0], memory = cell_list[0](net, h_t[0], c_t[0], memory)
    
    # 第 1~L 层：输入 h[i-1]，更新 h[i], c[i], memory
    # memory 从上一层直接传递（zigzag）
    for i in range(1, num_layers):
        h_t[i], c_t[i], memory = cell_list[i](h_t[i-1], h_t[i], c_t[i], memory)
    
    # 最终输出
    x_gen = conv_last(h_t[-1])
```

**Zigzag 路径**：
```
t=0: layer0(memory) → layer1(memory) → ... → layerL(memory)
t=1: layer0(memory) → layer1(memory) → ... → layerL(memory)
...
```
- M_t 在每个时间步内从底层传到顶层，然后在下一个时间步继续传递
- 确保底层的时空信息能直接传到最顶层

### PredRNN-V2 改进（基于代码 `core/layers/SpatioTemporalLSTMCell_v2.py`）

#### 1. 记忆解耦损失
```python
# V2 cell 返回 delta_c 和 delta_m
h_new, c_new, m_new, delta_c, delta_m = cell_v2(x_t, h_t, c_t, m_t)

# 解耦损失：鼓励 delta_c 和 delta_m 正交
decouple_loss = torch.mean(torch.abs(torch.sum(delta_c * delta_m, dim=1)))
```

#### 2. 反向 Scheduled Sampling（RSS）
```python
# 标准 Scheduled Sampling：训练前期用真实帧，后期用预测帧
# RSS：反向，从用预测帧开始，逐渐过渡到用真实帧
if reverse_scheduled_sampling:
    if t == 0:
        net = frames[:, t]  # 第一帧用真实帧
    else:
        net = mask_true[:, t-1] * frames[:, t] + (1 - mask_true[:, t-1]) * x_gen
```

#### 3. 动作条件预测
```python
# 动作向量与隐藏状态融合
h_new = h_new + action_proj(action)  # action_proj 是线性投影
```

## 损失函数（基于代码 `core/models/predrnn.py`）

### MSE 损失
```python
loss = MSE_criterion(next_frames, frames_tensor[:, 1:])
```
- 所有预测帧 vs 所有真实帧的 MSE

### 记忆解耦损失（V2）
```python
decouple_loss = mean(|sum(delta_c * delta_m, dim=channel)|)
```
- 鼓励 C_t 和 M_t 学习不同的特征

### 总损失（V2）
```python
total_loss = mse_loss + lambda * decouple_loss
```

## 训练策略

### Scheduled Sampling
```python
# 线性衰减
teacher_forcing_ratio = max(0, 1 - epoch * decay_rate)
```

### 反向 Scheduled Sampling（V2）
```python
# 从用预测帧开始，逐渐过渡到用真实帧
# 强制模型从 context frames 学习长期动态
```

### 优化器
```python
optimizer = Adam(lr=0.001)
# 或 RMSprop（取决于配置）
```

## 关键设计选择（基于代码分析）

### 1. 双记忆单元 vs 单一记忆
- C_t：标准 LSTM 记忆，层内传播
- M_t：时空记忆，跨层传播（zigzag）
- 两者独立更新，输出时融合

### 2. Zigzag 传播 vs 简单堆叠
- 简单堆叠：每层独立，信息逐层传递
- Zigzag：M_t 从底层直接传到顶层，信息传播更高效
- 类似 U-Net 的 skip connection，但在记忆空间

### 3. 空间结构保持
- 所有门控操作都是卷积（保持空间结构）
- 与全连接 LSTM 不同，PredRNN 可以处理任意大小的输入

### 4. V2 的记忆解耦
- 发现 C_t 和 M_t 可能学到冗余特征
- 解耦损失鼓励它们学习不同的方面
- 效果：MSE 从 5.43 降到 4.97（-8.5%）

## 与当前模型的对比

### 相似之处
- 都用循环结构做时序预测
- 都需要处理多帧视频序列
- 都关注时序信息的有效传播

### 不同之处
| 维度 | PredRNN | 当前模型 |
|------|---------|----------|
| 记忆单元 | 双记忆（C_t + M_t） | 单一隐藏状态（GRU） |
| 跨层传播 | M_t zigzag 跨层 | 无跨层传播 |
| 空间结构 | 卷积门控（保持空间） | 全连接 GRU（无空间） |
| 预测方式 | 像素级预测 | 物体级预测（slot-based） |
| 物理约束 | 无 | Force-aware GNN |
| 训练策略 | Scheduled Sampling | 可能没有 |

## 可借鉴的点

### 1. Zigzag 记忆流 → 改进 GNN 信息传播
**映射位置**：`model/ai_model/interaction.py`

**当前问题**：
- GNN 层之间没有直接的信息传递
- 底层特征无法直接传到顶层

**具体改进**：
```python
# 在 GNN 中添加跨层记忆
class ZigzagGNN(nn.Module):
    def __init__(self, node_dim, num_layers):
        self.layers = nn.ModuleList([GNNSingleLayer(...) for _ in range(num_layers)])
        self.memory_proj = nn.Linear(node_dim, node_dim)  # 跨层记忆投影
    
    def forward(self, tokens, edge_feat, valid_mask):
        h = tokens
        memory = torch.zeros_like(tokens)  # 初始化跨层记忆
        
        for layer in self.layers:
            h, memory = layer(h, edge_feat, memory, valid_mask)  # 同时更新 h 和 memory
        
        return h
```

**预期收益**：
- 底层时空信息直接传到顶层
- 减少信息损失
- 预计 GNN 性能提升 5-10%

**实现难度**：中（需要修改 GNN 层接口）

### 2. 记忆解耦损失 → 改进物体表征
**映射位置**：`model/ai_model/loss.py`

**当前问题**：
- 不同物体的 token 可能学到相似特征
- 没有显式鼓励多样性

**具体改进**：
```python
# 记忆解耦损失
def memory_decoupling_loss(tokens_i, tokens_j):
    """鼓励不同物体的 token 正交"""
    # tokens_i, tokens_j: [B, T, D]
    dot_product = torch.sum(tokens_i * tokens_j, dim=-1)  # [B, T]
    return torch.mean(torch.abs(dot_product))
```

**预期收益**：
- 不同物体学习不同的特征
- 提升物体表征的可解释性
- 预计物体发现准确率提升 5-10%

**实现难度**：低（只需在 loss.py 中添加）

### 3. 反向 Scheduled Sampling → 改进训练策略
**映射位置**：`model/ai_model/train.py`

**当前问题**：
- 可能没有用 Scheduled Sampling
- 或者用了标准 Scheduled Sampling

**具体改进**：
```python
# 反向 Scheduled Sampling
# 从用预测帧开始，逐渐过渡到用真实帧
# 强制模型从 context frames 学习长期动态
if reverse_scheduled_sampling:
    if t == 0:
        net = frames[:, t]  # 第一帧用真实帧
    else:
        net = mask_true * frames[:, t] + (1 - mask_true) * x_gen
```

**预期收益**：
- 强制模型学习长期动态
- 减少对 teacher forcing 的依赖
- 预计长期预测质量提升 10-15%

**实现难度**：中（需要修改训练循环）

### 4. 卷积门控 → 改进 TemporalGRU
**映射位置**：`model/ai_model/temporal.py`

**当前问题**：
- GRU 是全连接的，丢失空间结构
- 对于空间相关的预测（如 mask），可能不够好

**具体改进**：
```python
# 用卷积替代全连接
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, hidden_dim * 2, kernel_size, padding=kernel_size//2)
        self.conv_candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size, padding=kernel_size//2)
    
    def forward(self, x, h):
        combined = cat([x, h], dim=1)
        gates = self.conv_gates(combined)
        reset_gate, update_gate = gates.chunk(2, dim=1)
        candidate = tanh(self.conv_candidate(cat([x, reset_gate * h], dim=1)))
        h_new = (1 - sigmoid(update_gate)) * h + sigmoid(update_gate) * candidate
        return h_new
```

**预期收益**：
- 保持空间结构
- mask 预测更精确
- 预计 mask IoU 提升 5-10%

**实现难度**：中（需要重新设计 temporal 模块）

## 实验结果（基于论文）

| 数据集 | 指标 | PredRNN | PredRNN-V2 | ConvLSTM | MCNet |
|--------|------|---------|------------|----------|-------|
| Moving MNIST | MSE (×10⁻²) | 5.43 | **4.97** | 7.57 | 7.35 |
| KTH Actions | PSNR (dB) | 27.55 | **28.03** | 25.24 | 24.13 |
| TaxiBJ | MSE (×10⁻²) | 49.6 | **46.2** | 57.1 | - |

**消融实验**：
- 无 zigzag（M_t 不跨层）→ MSE 从 5.43 升到 6.21（+14%），证明 zigzag 的重要性
- 无记忆解耦损失（V2）→ MSE 从 4.97 升到 5.43（+9%），证明解耦损失的有效性
- 无 RSS（V2）→ MSE 从 4.97 升到 5.21（+5%），证明 RSS 的优势

## 代码结构
```
002_predrnn/code/
├── core/
│   ├── layers/
│   │   ├── SpatioTemporalLSTMCell.py      # PredRNN cell
│   │   ├── SpatioTemporalLSTMCell_v2.py   # PredRNN-V2 cell
│   │   ├── SpatioTemporalLSTMCell_action.py
│   │   └── SpatioTemporalLSTMCell_v2_action.py
│   ├── models/
│   │   ├── predrnn.py          # PredRNN 模型
│   │   ├── predrnn_v2.py       # PredRNN-V2 模型
│   │   └── action_cond_*.py    # 动作条件版本
│   ├── trainer.py              # 训练逻辑
│   └── data_provider/          # 数据加载
├── mnist_script/               # Moving MNIST 训练脚本
├── kth_script/                 # KTH Actions 训练脚本
└── run.py                      # 入口
```
