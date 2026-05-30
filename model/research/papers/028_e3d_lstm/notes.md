# E3D-LSTM: Eidetic 3D LSTM for Video Prediction

## 基本信息
- **作者**: Yunbo Wang, Lu Jiang, Ming-Hsuan Yang, Li-Jia Li, Mingsheng Long, Li Fei-Fei
- **年份**: 2019
- **会议**: ICLR 2019
- **论文链接**: https://openreview.net/forum?id=B1lKS2AqtX
- **代码**: TensorFlow 实现（Google），作者离开 Google 后复现，非官方支持

## 核心贡献
1. **将 3D 卷积集成到 LSTM 门控中**: 用 3D Conv 替代传统 ConvLSTM 中的 2D Conv，使 RNN 的局部感知器具备运动感知能力（motion-aware），记忆单元能存储更好的短期特征
2. **门控自注意力记忆机制（Eidetic Memory）**: 当前记忆状态通过门控自注意力模块与历史记录交互，能有效回忆跨多个时间步的存储记忆，即使经历长时间干扰后仍可召回——作者将此称为"eidetic"（遗觉记忆）
3. **时空预测 SOTA**: 在 Moving MNIST、KTH Actions 等视频预测数据集上达到当时最优
4. **超越视频预测**: 模型在早期活动识别（early activity recognition）任务上也表现良好，仅观察有限帧即可推断正在发生或即将发生的事件

## 模型架构

### 整体框架
输入视频序列 → 多层 E3D-LSTM → 3D Conv 输出层 → 预测帧序列

### Eidetic LSTM Cell 详解

Cell 维护三组状态:
- **hidden**: 短期隐状态
- **cell**: LSTM 单元状态（短期记忆）
- **global_memory**: 全局长期记忆（额外引入）

#### 门控计算（Entangled 3D Convolution）

对 hidden 做 3D Conv 生成 4 份门控信号:
```
[i_h, g_h, r_h, o_h] = Conv3D(hidden, 4*C, kernel=[2,5,5])   # 可选 LayerNorm
```

对 inputs 做 3D Conv 生成 7 份门控信号:
```
[i_x, g_x, r_x, o_x, temp_i_x, temp_g_x, temp_f_x] = Conv3D(inputs, 7*C, kernel=[2,5,5])
```

标准 LSTM 门控:
```
i_t = sigmoid(i_x + i_h)       # input gate
r_t = sigmoid(r_x + r_h)       # recall gate (用于注意力查询)
g_t = tanh(g_x + g_h)          # candidate
```

#### 自注意力记忆召回（Eidetic Memory Recall）
```python
# 用 r_t 作为 query，eidetic_cell（历史 cell 状态拼接）作为 key/value
attn_out = self_attention(query=r_t, keys=eidetic_cell, values=eidetic_cell)
# query/keys/values 都 reshape 到 [B, spatial_flat, C] 做矩阵乘法
attn = softmax(Q @ K^T) @ V

# 更新 cell: 注意力召回结果 + 标准 LSTM 更新
new_cell = LayerNorm(cell + attn_out) + i_t * g_t
```

关键: `eidetic_cell` 是所有历史 cell 状态在时间维度上的拼接（`tf.concat([c_history, cell[i]], dim=1)`），实现跨时间步的全局记忆访问。

#### 全局记忆更新（Global Memory）
对 global_memory 做 3D Conv 生成 4 份信号:
```
[i_m, f_m, g_m, m_m] = Conv3D(global_memory, 4*C, kernel=[2,5,5])

temp_i_t = sigmoid(temp_i_x + i_m)            # temporal input gate
temp_f_t = sigmoid(temp_f_x + f_m + forget_bias)  # temporal forget gate
temp_g_t = tanh(temp_g_x + g_m)               # temporal candidate

new_global_memory = temp_f_t * tanh(m_m) + temp_i_t * temp_g_t
```

#### 输出融合
```python
o_c = Conv3D(new_cell, C, kernel)
o_m = Conv3D(new_global_memory, C, kernel)
output_gate = tanh(o_x + o_h + o_c + o_m)

# 融合 cell 和 global_memory
memory = Conv1x1(concat(new_cell, new_global_memory), C)
output = tanh(memory) * sigmoid(output_gate)
```

### 网络配置
- **层数**: 4 层 E3D-LSTM
- **每层隐藏维度**: 64
- **3D Conv 核**: [2, 5, 5]（时间维度 2，空间 5×5）
- **滑动窗口**: window_length=2, window_stride=1（每步取 2 帧作为 3D Conv 的时间输入）
- **输出层**: Conv3D → squeeze → 生成预测帧

### 数据预处理
- 使用 patch 操作将图像分块: `reshape_patch(img, patch_size)`
- Moving MNIST: patch_size=4, 64×64 图像 → 16×16×16 的 patch tensor
- KTH: patch_size=8, 128×128 图像 → 16×16×64 的 patch tensor
- 测试时 `reshape_patch_back` 恢复原始分辨率

## 训练细节
- **优化器**: 自定义 Adam（mom1=0.95, mom2=0.9995），配合 EMA（decay=0.9995）
- **学习率**: 0.001
- **损失函数**: L2 Loss + L1 Loss（`l2_loss + reduce_sum(abs)`, 未取均值）
- **Scheduled Sampling**:
  - 训练初期使用 teacher forcing（概率 eta=1.0）
  - 每次迭代 eta -= 0.00002，逐步降低到 0（完全使用模型自身预测）
  - stop_iter: 50000（MM）/ 100000（KTH）
- **Batch Size**: 4（MM）/ 2（KTH）
- **输入/输出长度**:
  - Moving MNIST: input_length=10, total_length=20（预测 10 帧）
  - KTH Actions: input_length=10, total_length=30（预测 20 帧）
- **LayerNorm**: 对所有 Conv 输出应用 tensor layer normalization
- **训练步数**: 80000（MM）/ 200000（KTH）
- **GPU**: Nvidia V100，支持多 GPU 分布式训练

## 实验结果

### Moving MNIST
- 输入 10 帧，预测 10 帧
- 64×64 图像，两个弹跳数字
- E3D-LSTM 在 MSE 和视觉质量上优于 ConvLSTM、PredRNN 等

### KTH Actions
- 输入 10 帧，预测 20 帧
- 128×128 灰度图像，6 种人体动作
- 评估指标: MSE, PSNR, SSIM

### 早期活动识别
- 仅观察有限帧即可推断活动类别和趋势
- 证明模型学到的时空表示具有语义意义

## 代码实现细节

### 核心文件结构
```
code/
├── run.py                          # 主入口，FLAGS 配置，训练/测试循环
├── src/
│   ├── models/
│   │   ├── model_factory.py        # Model 类，构建计算图，Adam 优化器，EMA
│   │   └── eidetic_3d_lstm_net.py  # RNN 网络构建，多层 E3D-LSTM 堆叠
│   ├── layers/
│   │   └── rnn_cell.py             # EideticLSTMCell 核心实现
│   ├── data_provider/
│   │   ├── datasets_factory.py     # 数据集工厂
│   │   ├── mnist.py                # Moving MNIST 数据加载
│   │   └── kth_action.py           # KTH Actions 数据加载
│   ├── trainer.py                  # train/test 函数，PSNR 计算
│   └── utils/
│       └── preprocess.py           # patch reshape/reshape_back
└── scripts/
    ├── e3d_lstm_mm_train.sh        # MM 训练脚本
    └── e3d_lstm_kth_train.sh       # KTH 训练脚本
```

### 关键实现要点

**1. 滑动窗口机制**（`eidetic_3d_lstm_net.py`）
- 不是逐帧输入，而是每 1 步取连续 2 帧组成 `[B, 2, H, W, C]` 的 3D tensor
- 这样 3D Conv 的时间维度 kernel=2 才有意义
- 窗口滑动覆盖整个序列

**2. 历史记忆拼接**（`eidetic_3d_lstm_net.py`）
```python
# 每个时间步，将当前 cell 状态拼接到历史
if time_step == 0:
    c_history[i] = cell[i]
else:
    c_history[i] = tf.concat([c_history[i], cell[i]], 1)
# 传入 EideticLSTMCell 的 eidetic_cell 参数
```
- 注意: 随着时间步增加，eidetic_cell 在时间维度不断增长，显存开销大

**3. 自注意力实现**（`rnn_cell.py` `_attn` 方法）
- 将空间维度 flatten: `[B, T, W, H, C]` → `[B, T*W*H, C]`
- 标准 Q·K^T attention: `attn = softmax(Q @ K^T) @ V`
- 没有缩放因子（无 `/sqrt(d_k)`），也没有多头

**4. Scheduled Sampling**（`run.py`）
- 以概率 eta 使用真实帧 vs 模型预测帧作为下一步输入
- eta 线性衰减到 0，实现从 teacher forcing 到 free-running 的平滑过渡

**5. 已知 Bug**
- README 指出 `global_memory` 相关代码有 bug，导致 KTH 预训练模型效果不匹配
- 此 bug 在作者原始实验中不存在，是代码复现时引入的

## 与当前研究的关联

### 与 PredRNN/PredRNN++ 的关系
- E3D-LSTM 是 PredRNN 系列的后续工作（同一作者 Yunbo Wang）
- PredRNN 引入时空记忆流（ST-LSTM），E3D-LSTM 进一步加入 3D Conv 和全局记忆
- 核心改进: 从 2D Conv → 3D Conv，从纯 LSTM 记忆 → LSTM + 自注意力全局记忆

### 与 ConvLSTM 的关系
- ConvLSTM 用 2D Conv 替代全连接，处理空间信息
- E3D-LSTM 用 3D Conv 同时处理时空信息，且门控计算跨时间步
- E3D-LSTM 的"entangled"体现在: 3D Conv 核同时作用于时间和空间维度，门控信号融合了时空信息

### 对当前研究的启发
1. **3D Conv 在 RNN 门控中的应用**: 如果当前模型使用 GRU 做时序，可以考虑用 3D Conv 替代 2D Conv 来增强时空建模
2. **全局记忆 + 自注意力**: 对于需要长期依赖的任务（如长序列预测），引入额外的全局记忆单元并通过注意力机制访问历史
3. **Scheduled Sampling**: 训练策略上的通用技巧，可直接应用于任何自回归预测模型
4. **Patch 化处理**: 将图像分块后再输入网络，减少空间分辨率、增加通道数，是一种有效的降维策略
5. **3D Conv 核尺寸选择**: 时间维度 kernel=2 是最小有效值，配合 window_length=2 使用；空间维度 kernel=5 提供足够的感受野
