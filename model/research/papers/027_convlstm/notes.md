# ConvLSTM: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

## 基本信息
- **作者**: Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung (HKUST), Wai-kin Wong, Wang-chun Woo (Hong Kong Observatory)
- **年份**: 2015
- **会议**: NeurIPS 2015
- **论文链接**: https://arxiv.org/abs/1506.04214
- **代码**: https://github.com/ndrplz/ConvLSTM_pytorch (PyTorch 第三方实现)

## 核心贡献

1. **提出 ConvLSTM Cell**: 将标准 FC-LSTM 中的全连接矩阵乘法替换为卷积操作，使输入到状态 (input-to-state) 和状态到状态 (state-to-state) 的转换都具有卷积结构，从而同时捕捉时空相关性
2. **编码-预测框架 (Encoding-Forecasting)**: 构建端到端可训练的时空序列预测网络，编码器压缩输入序列为隐藏状态，预测器展开隐藏状态生成未来序列
3. **降水临近预报应用**: 首次将深度学习应用于降水临近预报 (precipitation nowcasting)，超越传统光流法 (ROVER)
4. **开创新范式**: 成为后续时空序列预测 (视频预测、天气预报等) 的基础架构

## 模型架构

### ConvLSTM Cell

标准 FC-LSTM 的所有输入 $X_t$、隐状态 $H_t$、cell 状态 $C_t$、门控信号 $i_t, f_t, o_t$ 都是一维向量。ConvLSTM 的关键创新在于将它们全部改为 **3D 张量** (通道 × 行 × 列)，最后两个维度保持空间结构。所有门控操作中的矩阵乘法替换为 **卷积操作** (用 `*` 表示)：

$$i_t = \sigma(W_{xi} * X_t + W_{hi} * H_{t-1} + W_{ci} \circ C_{t-1} + b_i)$$
$$f_t = \sigma(W_{xf} * X_t + W_{hf} * H_{t-1} + W_{cf} \circ C_{t-1} + b_f)$$
$$C_t = f_t \circ C_{t-1} + i_t \circ \tanh(W_{xc} * X_t + W_{hc} * H_{t-1} + b_c)$$
$$o_t = \sigma(W_{xo} * X_t + W_{ho} * H_{t-1} + W_{co} \circ C_t + b_o)$$
$$H_t = o_t \circ \tanh(C_t)$$

其中 `*` 为卷积算子，`∘` 为 Hadamard 逐元素乘积。

**关键设计要点**:
- **输入门 $i_t$**: 控制多少新信息写入 cell 状态，基于当前输入和前一时刻隐藏状态做卷积后经 sigmoid
- **遗忘门 $f_t$**: 控制保留多少历史 cell 状态，同样基于卷积 + sigmoid
- **候选状态 $\tilde{C}_t$**: 通过卷积 + tanh 生成候选值
- **输出门 $o_t$**: 控制 cell 状态中多少信息输出到隐藏状态
- **cell 状态更新**: $C_t = f_t \circ C_{t-1} + i_t \circ \tilde{C}_t$，梯度在 cell 内可长期传播（constant error carousel）
- **peephole 连接**: $W_{ci}, W_{cf}, W_{co}$ 是逐元素的 (Hadamard)，门控可以直接"窥视" cell 状态

**空间结构保持**: FC-LSTM 可以看作 ConvLSTM 的特例——当最后两个空间维度都为 1 时，卷积退化为全连接。

**边界填充**: 对隐藏状态进行零填充 (zero-padding)，边界点被视为"外部世界"，状态初始化为零（对外部一无所知）。这种设计帮助模型处理边界条件（如云团从边界涌入）。

**卷积核大小的影响**: 较大的 state-to-state 卷积核能捕捉更快的运动，较小的核适合捕捉慢速运动。实验表明 **state-to-state 卷积核 > 1 对捕捉时空运动模式至关重要**。

### Encoding-Forecasting 网络结构

整体采用编码器-预测器 (encoder-forecaster) 结构：

```
编码网络:                    预测网络:
  ConvLSTM Layer 2  ----复制状态---->  ConvLSTM Layer 4
  ConvLSTM Layer 1  ----复制状态---->  ConvLSTM Layer 3
       |                                    |
     输入序列                            1×1 Conv → 预测输出
```

- **编码网络**: 由多层 ConvLSTM 堆叠，逐帧处理输入序列，将整个输入序列压缩到最后一层的隐藏状态张量
- **状态复制**: 编码网络最后时刻的隐藏状态 $H$ 和 cell 状态 $C$ 直接复制到预测网络的初始状态
- **预测网络**: 将压缩的隐藏状态展开为未来序列
- **最终预测**: 将预测网络所有时间步的隐藏状态拼接，通过 1×1 卷积层生成最终预测
- **输入输出同维度**: 预测目标与输入具有相同的维度

## 训练细节

### 数据集

**Moving-MNIST 合成数据集**:
- 每个序列 20 帧（10 帧输入 + 10 帧预测）
- 64×64 画面上两个手写数字弹跳
- 10000 训练 / 2000 验证 / 3000 测试
- 数字从 MNIST 子集 500 个样本中随机选取
- 速度幅度 [3, 5)，起始位置和方向均匀随机
- 使用 patch size 4×4，64×64 帧转为 16×16×16 张量

**香港雷达回波数据集**:
- 2011-2013 年香港气象雷达三年数据
- 选取 97 个最高降雨日
- 预处理：强度值归一化到灰度，裁剪中心 330×330 区域，disk filter (radius=10)，缩放到 100×100
- K-means 去噪去除仪器噪声
- 每 6 分钟采集一帧，每天 240 帧
- 每天切分 40 个不重叠帧块：4 块训练 / 1 块测试 / 1 块验证
- 20 帧滑动窗口切片（5 帧输入 + 15 帧预测）
- 共 8148 训练 / 2037 测试 / 2037 验证序列
- Patch size = 2

### 训练设置
- **损失函数**: 交叉熵损失 (cross-entropy loss)
- **优化器**: RMSProp，学习率 10⁻³，衰减率 0.9
- **正则化**: 早停 (early stopping) 基于验证集
- **训练方式**: BPTT (back-propagation through time)
- **实现**: Python + Theano，单块 NVIDIA K20 GPU

## 实验结果

### Moving-MNIST 数据集

| 模型 | 参数量 | 交叉熵损失 |
|------|--------|-----------|
| FC-LSTM-2048-2048 | 142,667,776 | 4832.49 |
| ConvLSTM(5×5)-5×5-256 (1层) | 13,524,496 | 3887.94 |
| ConvLSTM(5×5)-5×5-128-5×5-128 (2层) | 10,042,896 | 3733.56 |
| ConvLSTM(5×5)-5×5-128-5×5-64-5×5-64 (3层) | 7,585,296 | **3670.85** |
| ConvLSTM(9×9)-1×1-128-1×1-128 (2层) | 11,550,224 | 4782.84 |
| ConvLSTM(9×9)-1×1-128-1×1-64-1×1-64 (3层) | 8,830,480 | 4231.50 |

**关键发现**:
- ConvLSTM 参数量仅为 FC-LSTM 的 ~5%，但性能显著优于后者
- 更深的模型效果更好（但 2 层和 3 层之间差异不大）
- **state-to-state 卷积核大小至关重要**: 1×1 核的结果远差于 5×5 核，即使参数量接近。1×1 核的感受野不会随时间增长
- 大的 input-to-state 核 (9×9) 配合小的 state-to-state 核 (1×1) 效果也不好

**域外泛化**: 在未见过的三个数字弹跳场景中，3 层模型平均交叉熵 6379.42，能成功分离重叠数字并预测整体运动。

### 雷达回波降水预报

| 模型 | Rainfall-MSE↓ | CSI↑ | FAR↓ | POD↑ | Correlation↑ |
|------|--------------|------|------|------|--------------|
| ConvLSTM (2层, 64, 3×3) | **1.420** | **0.577** | **0.195** | **0.660** | **0.908** |
| ROVER1 | 1.712 | 0.516 | 0.308 | 0.636 | 0.843 |
| ROVER2 | 1.684 | 0.522 | 0.301 | 0.642 | 0.850 |
| ROVER3 | 1.685 | 0.522 | 0.301 | 0.642 | 0.849 |
| FC-LSTM-2000-2000 | 1.865 | 0.286 | 0.335 | 0.351 | 0.774 |

**关键发现**:
- ConvLSTM 在所有指标上全面超越 ROVER 和 FC-LSTM
- FC-LSTM 性能很差，因为全连接结构有太多冗余连接，无法捕捉雷达图中云团运动的局部空间一致性
- ConvLSTM 优势：(1) 能处理边界条件（云团从边界涌入的突然变化）；(2) 端到端训练能学习复杂的时空模式
- ConvLSTM 预测较模糊，但更准确；ROVER 预测更锐利但虚警率更高
- 模糊预测可能是对任务固有不确定性的合理应对

## 代码实现细节

代码位于 `code/convlstm.py`，为 PyTorch 第三方实现 (github.com/ndrplz/ConvLSTM_pytorch)。

### ConvLSTMCell
```python
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        # 单次卷积同时计算四个门控，高效实现
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,   # 输入 + 隐状态拼接
            out_channels=4 * hidden_dim,          # 四个门: i, f, o, g
            kernel_size=kernel_size,
            padding=kernel_size[0] // 2           # same padding
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # 通道维度拼接
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)       # 输入门
        f = torch.sigmoid(cc_f)       # 遗忘门
        o = torch.sigmoid(cc_o)       # 输出门
        g = torch.tanh(cc_g)          # 候选状态
        c_next = f * c_cur + i * g    # cell 状态更新
        h_next = o * torch.tanh(c_next)  # 隐藏状态
        return h_next, c_next
```

**实现要点**:
- **合并卷积**: 将四个门控的卷积合并为一次 `nn.Conv2d`（输入+隐状态 → 4×hidden_dim），然后 split，效率更高
- **通道拼接**: 沿 channel 维度拼接输入和隐状态 `[input_tensor, h_cur]`
- **same padding**: 通过 `padding = kernel_size // 2` 保证输出空间尺寸与输入一致
- **零初始化**: `init_hidden()` 返回全零的 h 和 c，对应"对未来的完全无知"

**与论文的差异**:
- 代码实现省略了 peephole 连接 ($W_{ci}, W_{cf}, W_{co} \circ C$)，这是简化版本
- 论文原始公式包含 peephole，但代码实现是更常见的简化版（无 peephole）

### ConvLSTM 多层封装
```python
class ConvLSTM(nn.Module):
    # 支持任意层数堆叠
    # 输入: (B, T, C, H, W) 或 (T, B, C, H, W)
    # 输出: layer_output_list, last_state_list
    # 每层可独立设置 hidden_dim 和 kernel_size
```

- 按时间步逐步处理，逐层传递输出
- 支持 `batch_first` 参数控制维度顺序
- `return_all_layers` 控制返回所有层还是仅最后一层的输出

## 与当前研究的关联

### 可借鉴的点

1. **ConvGRU 替代 GRU**: 当前模型使用全连接 GRU 做时序建模，丢失空间结构。可将 ConvLSTM 思想应用于 GRU，用卷积替代全连接，保持空间结构：
   - `model/ai_model/temporal.py` 中的 TemporalGRU 可升级为 ConvGRU
   - 预期对 mask 预测等空间相关任务有提升

2. **编码-预测结构**: 将输入序列编码到隐藏状态再展开预测的范式，适用于需要序列到序列预测的任务

3. **边界填充策略**: 零填充隐状态可帮助模型学习边界条件，适用于有边界效应的空间预测任务

4. **更深层更高效**: 实验表明更深的 ConvLSTM 用更少参数取得更好效果，可作为设计参考

5. **时空特征提取**: ConvLSTM 可叠加在 CNN 特征图之上，用于视频理解、动作识别等任务（论文提到的未来工作方向）

### 局限性
- 预测结果倾向于模糊，尤其长期预测
- 计算开销随空间分辨率增大而增大
- 未引入注意力机制，对长距离空间依赖建模能力有限
- 代码实现省略了 peephole 连接
