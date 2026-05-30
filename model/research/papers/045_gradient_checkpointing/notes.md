# Gradient Checkpointing：以计算换显存的深度网络训练方法

## 基本信息
- **标题**：Training Deep Nets with Sublinear Memory Cost
- **作者**：Tianqi Chen, Bing Xu, Chiyuan Zhang, Carlos Guestrin
- **机构**：University of Washington, Dato Inc, MIT
- **年份**：2016
- **论文链接**：https://arxiv.org/abs/1604.06174
- **代码**：基于 MXNet 实现，附带 PyTorch 版本（pytorch_memonger）

## 核心贡献

1. **提出亚线性显存训练算法**：训练 n 层网络仅需 O(√n) 显存，代价仅为额外一次前向传播
2. **系统化的计算图优化方法**：结合 in-place 操作、内存共享和选择性重计算三种策略
3. **递归式极致压缩**：通过递归应用 checkpoint 策略，理论上可将显存降至 O(log n)，代价为 O(n log n) 前向计算
4. **自动规划算法**：自动搜索最优的 checkpoint 切分点，无需人工手动划分
5. **实用性强**：1000 层 ResNet 在 ImageNet 上训练显存从 48GB 降至 7GB；已成为大模型训练的标准技术

## 模型架构

### 计算图内存优化（基础层）

论文首先对计算图进行系统化分析，提出两种基础优化：

- **In-place 操作**：将操作的输出直接写入输入的内存空间，适用于输入不再被其他操作依赖的场景
- **内存共享**：不再需要的中间结果的内存可被回收复用给后续节点

通过活跃性分析（liveness analysis），以 O(n) 复杂度确定各节点的生命周期，实现安全的内存复用。这些优化对 n 层网络可将预测阶段的显存从 O(n) 降至近 O(1)，训练阶段也可获得常数级提升。

### Activation Checkpointing（核心方法）

核心思想：将网络分为 k 个 segment，仅保存每个 segment 的输入/输出，丢弃 segment 内部的中间特征图。反向传播时，从最近的 checkpoint 点重新前向计算以恢复被丢弃的中间结果。

**算法流程**（Algorithm 1 — 线性链网络）：
1. 前向传播：将网络分为若干 segment，每个 segment 内正常前向但只保存 segment 输出
2. 反向传播：从最近的 checkpoint 点重新执行 segment 内的前向，计算梯度

**通用化**（Algorithm 2 — 任意计算图）：
- 引入 **mirror count 函数** m(v)：指定每个节点被复制（重计算）的次数
- m(v)=0 表示保留该节点输出（作为 checkpoint）；m(v)=1 表示丢弃该节点输出（反向时重算）
- 自动构建内存优化后的梯度计算图，支持控制流依赖

### Selective Recomputation（选择性重计算）

作为特殊应用，**优先丢弃计算代价低的操作的中间结果**，保留计算代价高的：
- **丢弃**：Batch Normalization、ReLU、Pooling 的输出（计算廉价）
- **保留**：卷积层的输出（计算昂贵）

在 Conv-BN-ReLU 流水线中，这种策略以极小的额外计算开销换取显著的显存节省。

### Memory-Compute Tradeoff（显存-计算权衡）

**定量分析**：

| 策略 | 显存复杂度 | 额外计算 |
|------|-----------|---------|
| 标准训练 | O(n) | 无 |
| 均匀分 k 段 | O(n/k) + O(k) | k-1 次额外前向 |
| **最优分段**（k=√n） | **O(2√n)** | **~1 次额外前向** |
| 递归应用（k=1） | O(log₂n) | O(n log₂n) |

最优策略将 n 层网络均分为 √n 段，每段 √n 层。显存 = 段内反向显存 O(n/√n) + checkpoint 存储 O(√n) = O(2√n)。

**自动预算搜索**（Algorithm 3）：
- 给定显存预算 B，贪心地决定在哪些节点设置 checkpoint
- 通过网格搜索 B 值（在 [B/√2, 2B] 范围内取 6 个点）找到最优分配方案
- 经验表明 B = √(xy)（x 为段间特征图存储开销，y 为单段计算开销）是好的初始估计

## 训练细节

- **实验平台**：MXNet 框架，Titan X GPU
- **显存统计**：MXNet 静态分配所有中间特征图，可精确报告 feature map 的显存开销（不含参数和临时显存）
- **图像分类**：Deep ResNet，batch size 32，输入 (3, 224, 224)，深度从 128 到 1024 层
- **语音识别**：4 层 LSTM，hidden=1024，unrolling 步数从 64 到 2048，batch size 64，输入 50 维连续向量，输出 5000 类 softmax
- **对比策略**：no optimization → inplace → sharing → drop bn-relu → sublinear plan
- **梯度正确性**：所有优化策略产生完全相同的权重梯度，可安全使用

## 实验结果

### Deep ResNet（图像分类）
- 系统优化（inplace + sharing）可减少 2-3 倍显存，但仍呈线性趋势
- **Sublinear plan**：1000 层 ResNet 显存从 ~48GB 降至 ~7GB（feature map 部分）
- 呈亚线性增长趋势，可在单 GPU 上训练超深网络

### LSTM（长序列）
- Inplace 优化在 LSTM 上效果显著（可直接在单个 memory cell 上累加权重梯度）
- Sublinear plan 比优化后的 sharing 策略再减少 4 倍以上显存
- 2048 步 unrolling 仍可训练

### 训练速度
- Sublinear plan 额外运行时间约 **30%**（接近理论上的 ~33% 额外一次前向）
- 速度与工作量仍保持线性关系
- 以较小的时间代价，解锁了原本无法训练的大模型

## 代码实现细节

代码仓库提供了 PyTorch 版本的实现（pytorch_memonger），展示如何在多种模型上应用 `torch.utils.checkpoint`：

### PyTorch API
```python
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
```

### ResNet 实现（优化版）
- **checkpoint_sequential**：将 `nn.Sequential` 的模块列表分成若干 chunk，每个 chunk 作为一个 checkpoint 段
- `PreActResNet.forward` 中：`checkpoint_sequential(modules, chunks, input_var)` 直接将整个 features 模块做 checkpoint
- chunks 参数控制分段数，越多越省显存但越慢

### LSTM 实现（优化版）
- 将长序列按时间步分成 `chunks` 段（默认 4 段）
- 每段通过 `checkpoint.checkpoint(self.custom(start, end), emb, hidden[0], hidden[1])` 执行
- `custom()` 方法返回一个闭包，封装了对 RNN 子序列的前向计算
- 梯度通过 hook 机制传递，decoder 部分分块计算 loss 并逐步反向传播以进一步节省显存

### 关键注意事项
- Batch Normalization 和 Dropout 等特殊层需谨慎处理（训练/推理模式切换）
- checkpoint 内的操作会在反向传播时重新执行，需确保随机性行为一致
- 与模型并行、数据并行等正交技术可组合使用

## 与当前研究的关联

1. **大模型训练的基础设施**：Gradient checkpointing 已成为训练 LLM、大规模视觉模型的标配技术，几乎所有主流框架（PyTorch、DeepSpeed、Megatron-LM）都内置支持
2. **与混合精度训练互补**：FP16/BF16 减少单个激活值大小，gradient checkpointing 减少激活值数量，两者可叠加
3. **与序列并行结合**：Megatron-LM 的序列并行（Sequence Parallelism）与 gradient checkpointing 结合，进一步优化长序列训练
4. **选择性 checkpointing 的演进**：现代框架（如 PyTorch 2.x）支持更细粒度的选择性重计算，可对 attention、MLP 等子模块分别配置
5. **显存优化的持续需求**：随着模型规模从十亿级向万亿级发展，显存优化技术仍然是核心研究方向
