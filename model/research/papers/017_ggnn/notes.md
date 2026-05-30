# Gated Graph Sequence Neural Networks (GGNN)

## 基本信息

- **标题**：Gated Graph Sequence Neural Networks
- **作者**：Yujia Li, Daniel Tarlow, Marc Brockschmidt, Richard Zemel
- **机构**：University of Toronto + Microsoft Research
- **年份**：2015（arXiv），2016（ICLR 发表）
- **会议**：ICLR 2016
- **论文链接**：https://arxiv.org/abs/1511.05493
- **代码语言**：Lua/Torch（作者官方实现）
- **引用量**：极高，是图神经网络领域的奠基性工作之一

---

## 核心贡献

1. **用 GRU 门控机制替代原始 GNN 的固定点迭代**：原始 GNN（Scarselli et al., 2009）依赖 Almeida-Pineda 算法迭代至收敛，需要参数满足收缩映射约束，限制了模型表达能力。GGNN 用 GRU 更新替代，固定步展开（如 T=5），通过 BPTT 训练梯度，无需收缩映射约束。

2. **扩展到序列输出（GGS-NN）**：将单步图级别/节点级别输出扩展为序列输出，多个 GGNN 按序列协同工作，支持路径预测、逻辑公式生成等序列任务。

3. **节点标注（Node Annotations）机制**：将任务相关信息（如问题节点标识）编码为节点初始状态的一部分，使模型能处理不同的查询。

4. **证明 GNN 是广泛有用的神经网络变体**：在 bAbI 任务、图算法学习、程序验证等多个任务上验证了有效性。

---

## 模型架构

### 1. 基础 GNN 回顾

原始 GNN 的传播模型为迭代更新直到收敛：
```
h_v^(t) = f*(l_v, l_CO(v), l_NBR(v), h_NBR(v)^(t-1))
```
其中 `f*` 是对每条边的分解求和。参数需满足收缩映射约束以确保收敛，但**限制了信息的远距离传播**（附录证明了收缩映射中信息随距离指数衰减）。

### 2. GG-NN 传播模型（核心公式）

**初始化**（将节点标注复制到隐状态前 annotation_dim 维，其余补零）：
```
h_v^(1) = [x_v; 0]
```

**边消息聚合**（通过邻接矩阵 A 传递信息，A 的稀疏结构对应图的边）：
```
a_v^(t) = A^T [h_1^(t-1); h_2^(t-1); ...; h_|V|^(t-1)] + b
```

其中 A ∈ R^(D|V| × 2D|V|)，矩阵的稀疏结构和参数绑定如图：
- 每种边类型（如 A, B, C）和方向（正向/反向）有独立的参数子矩阵
- A 的子矩阵 A_{v:} 对应节点 v 的行，只在邻居位置非零

**GRU 更新**（替代原始 GNN 的简单线性/MLP 更新）：
```
z_v^(t) = σ(W_z · a_v^(t) + U_z · h_v^(t-1))        # 更新门
r_v^(t) = σ(W_r · a_v^(t) + U_r · h_v^(t-1))        # 重置门
h̃_v^(t) = tanh(W_a · a_v^(t) + U · (r_v^(t) ⊙ h_v^(t-1)))  # 候选隐状态
h_v^(t) = (1 - z_v^(t)) ⊙ h_v^(t-1) + z_v^(t) ⊙ h̃_v^(t)   # 最终更新（插值形式）
```

注意：这里 GRU 的"输入"是聚合后的消息 a_v^(t)，而非传统序列模型中的外部输入。

### 3. 输出模型

**节点选择输出（Node Selection）**：对每个节点 v 输出一个标量分数：
```
o_v = g(h_v^(T), x_v)
```
再对所有节点分数应用 softmax 选择目标节点。代码中还有门控机制（gating），用 sigmoid 门来调节输出分数。

**图级别输出（Graph Level）**：定义图的表示向量：
```
h_G = tanh(Σ_v σ(i(h_v^(T), x_v)) ⊙ tanh(h_v^(T)))
```
其中 σ(i(·)) 是软注意力机制，决定哪些节点与当前图级别任务相关。然后将 h_G 送入分类网络。

### 4. GGS-NN 序列输出架构

对于第 k 个输出步：
- 用一个 GG-NN `F_o` 从 X^(k) 预测输出 o^(k)
- 用另一个 GG-NN `F_X` 从 X^(k) 预测 X^(k+1)（下一步的节点标注）
- X^(k+1) 是从步 k 到步 k+1 传递的隐状态

**节点标注输出**：每个节点独立预测新的标注：
```
x_v^(k+1) = σ(j(h_v^(k,T), x_v^(k)))
```

**两种训练模式**：
- **有观察标注（Observed Annotations）**：训练时提供中间标注 X^(k)，各步 GG-NN 可独立训练，测试时用预测的标注
- **隐标注（Latent Annotations）**：X^(k) 作为隐变量，端到端联合训练

---

## 训练细节

| 参数 | 值 |
|------|-----|
| 传播步数 T | 5（所有说明性任务） |
| 节点向量维度 D | Task 15/16: D=4; Task 4/18/19: D=6; 最短路径/欧拉回路: D=20 |
| 优化器 | Adam |
| 网络参数量 | 说明性任务 < 600 参数 |
| 训练数据 | 每个任务 1000 训练样本（50 用于验证）+ 1000 测试样本 |
| 梯度计算 | BPTT（反向传播通过时间） |

**与原始 GNN 的关键区别**：
- 原始 GNN 用 Almeida-Pineda 算法（无需存储中间状态，但需收缩映射约束）
- GG-NN 用 BPTT（需更多内存，但移除了收缩映射约束，更强大的表达能力）

---

## 实验结果

### 1. 单步输出任务（bAbI）

| 任务 | RNN | LSTM | GG-NN |
|------|-----|------|-------|
| Task 4 (Two Arg Relations) | 99.2% (250样本) | 98.7% (250样本) | **100.0% (50样本)** |
| Task 15 (Basic Deduction) | 46.0% (950样本) | 49.5% (950样本) | **100.0% (50样本)** |
| Task 16 (Basic Induction) | 33.6% (950样本) | 36.9% (950样本) | **100.0% (50样本)** |
| Task 18 (Size Reasoning) | 100.0% (50样本) | 100.0% (50样本) | **100.0% (50样本)** |

关键发现：GG-NN 在仅 50 个训练样本时即可达到完美准确率，RNN/LSTM 在 Task 15/16 上即使 950 样本也无法解决。

### 2. 序列输出任务（GGS-NN）

| 任务 | RNN | LSTM | GGS-NN |
|------|-----|------|--------|
| bAbI Task 19 (Path Finding) | 24.5% (950样本) | 29.4% (950样本) | **99.6% (250样本)** |
| Shortest Path | 11.1% (950样本) | 10.7% (950样本) | **100.0% (50样本)** |
| Eulerian Circuit | 0.5% (950样本) | 0.4% (950样本) | **100.0% (50样本)** |

RNN/LSTM 在序列图任务上完全失败，GGS-NN 仅需 50 个样本即可完美解决。

### 3. 程序验证（真实应用）

- 任务：从堆内存图推断分离逻辑公式（描述数据结构形状）
- 数据集：327 个公式 × 498 个图 = ~160,000 个组合
- 划分：6:2:2（按公式划分，测试集公式不在训练集中）

| 方法 | 准确率 |
|------|--------|
| 手工特征工程（Brockschmidt et al., 2015） | 89.11% |
| **GGS-NN（无特征工程）** | **89.96%** |

无需复杂特征工程即可达到甚至超越手工特征方法的性能。

---

## 代码实现细节

代码使用 Lua/Torch，作者官方实现。结构清晰，分为核心库和实验脚本。

### 核心模块层次结构

```
ggnn/
├── BaseGGNN.lua                        # 基础 GGNN 传播模型（GRU 版本）
├── BaseGGNNSimple.lua                  # 基础 GGNN（vanilla RNN 版本，无门控）
├── BaseGGNNLinear.lua                  # 基础 GGNN（线性版本）
├── BaseGGNNSimpleLinear.lua            # 基础 GGNN（简单线性版本）
├── BaseOutputNet.lua                   # 输出网络基类
├── NodeSelectionGGNN.lua              # 节点选择 GGNN
├── NodeSelectionOutputNet.lua         # 节点选择输出网络
├── NodeSelectionSequenceGGNN.lua      # 节点选择序列 GGNN
├── NodeSelectionSequenceSharePropagationGGNN.lua  # 共享传播网络的节点选择序列 GGNN
├── GraphLevelGGNN.lua                 # 图级别分类 GGNN
├── GraphLevelOutputNet.lua            # 图级别输出网络
├── GraphLevelSequenceGGNN.lua         # 图级别序列 GGNN
├── GraphLevelSequenceSharePropagationGGNN.lua  # 共享传播网络的图级别序列 GGNN
├── PerNodeGGNN.lua                    # 逐节点 GGNN
├── PerNodeOutputNet.lua               # 逐节点输出网络
└── ggnn_util.lua                      # 工具函数
```

### 关键实现细节

**1. 邻接矩阵构建**（ggnn_util.lua）
- 每种边类型有正向和反向两个邻接矩阵
- 邻接矩阵是稀疏的，大小为 D × (2D|V|)，对应论文中的矩阵 A
- 通过 `create_adjacency_matrix_cat` 函数从边列表构建

**2. 传播网络**（BaseGGNN.lua）
- `create_propagation_net_modules()`：为每种边类型创建正向和反向的线性传播网络
- `create_node_update_net_modules()`：创建 GRU 更新网络，包含 `transform` 和 `gate` 两个线性模块
- 传播过程：每步对每种边类型独立传播，然后通过 `add_net`（实际是 GRU 更新网络）聚合

**3. GRU 更新的具体实现**（BaseGGNN.lua 的 `create_one_node_update_net`）
```lua
-- 拼接：正向消息 + 反向消息 + 当前状态
joined_input = JoinTable(forward_input, reverse_input, current_state)

-- 门控计算
gates = Sigmoid(gate_linear(joined_input))
update_gate = gates[1:D]        # 更新门 z
reset_gate = gates[D+1:2D]      # 重置门 r

-- 候选状态
new_joined = JoinTable(forward_input, reverse_input, reset_gate ⊙ current_state)
transformed_output = Tanh(transform_linear(new_joined))

-- 最终更新：h = h + z ⊙ (h̃ - h)  等价于 h = (1-z) ⊙ h + z ⊙ h̃
output = current_state + update_gate ⊙ (transformed_output - current_state)
```

**4. 节点选择输出**（NodeSelectionGGNN.lua）
- 输出网络：Tanh(linear(state + annotation)) → linear → 标量分数
- 门控输出：sigmoid 门乘以分数，基于 annotation 调节
- 预测：取分数最高的节点

**5. 图级别聚合**（GraphLevelGGNN.lua）
- 软注意力聚合：对每个节点计算 gate 和 transformed，逐元素相乘后求和
- 输入：拼接最终节点表示和初始标注
- 输出：经过 Tanh 后送入分类网络

**6. 序列输出**（NodeSelectionSequenceSharePropagationGGNN.lua）
- 多个时间步共享传播网络参数（`create_share_param_copy`）
- 每步：传播网络 → 节点选择输出 + 标注输出 → 下一步输入
- 反向传播：从最后一个时间步向前传播梯度

### 实验脚本

```
babi/
├── babi_train.lua          # bAbI 任务 GGNN 训练
├── babi_eval.lua           # bAbI 任务 GGNN 评估
├── babi_rnn_train.lua      # bAbI RNN 基线训练
├── babi_rnn_eval.lua       # bAbI RNN 基线评估
├── seq_train.lua           # 序列任务 GGNN 训练
├── seq_eval.lua            # 序列任务 GGNN 评估
├── seq_data.lua            # 序列数据加载
├── run_experiments.py      # 实验运行脚本
├── run_rnn_baselines.py    # RNN 基线运行脚本
└── data/
    ├── get_10_fold_data.sh  # 获取 bAbI 10 折数据
    ├── symbolic_preprocess.py
    ├── rnn_preprocess.py
    └── extra_seq_tasks/     # 额外序列任务（最短路径、欧拉回路）
```

---

## 与当前研究的关联

### 1. 与当前模型的对比

| 方面 | GGNN | 当前模型 |
|------|------|---------|
| 更新机制 | GRU（含更新门、重置门） | GRU（类似） |
| 消息计算 | 矩阵乘法（按边类型独立参数） | MLP |
| 传播步数 | 固定 T=5 | 固定 2 层 |
| 消息聚合 | 邻接矩阵 × 节点状态 | 求和/平均 |
| 序列输出 | 支持（GGS-NN） | 视任务而定 |

### 2. 可借鉴的设计

**GRU 更新机制已采用**：当前模型的 GRU 更新与 GGNN 一致，说明设计合理。

**固定步展开策略**：GGNN 的 T=5 展开优于迭代至收敛，当前模型仅 2 层可考虑增加。

**按边类型的独立参数矩阵**：GGNN 对每种边类型用独立的参数矩阵，可考虑在当前模型中也按边类型区分消息计算。

**节点标注作为初始状态**：将任务相关信息编码为节点初始状态，而非仅作为输入特征。

**软注意力图聚合**：图级别任务使用 σ(i(h,x)) ⊙ tanh(h) 的软注意力聚合，可替代简单的全局池化。

### 3. 潜在改进方向

1. **增加传播层数**：从 2 层增至 5 层，参考 GGNN 的 T=5 设置
2. **门控消息聚合**：用 GRU 风格的门控来替代简单的消息求和
3. **边类型特定参数**：为不同边类型学习独立的消息传递参数矩阵
4. **序列输出框架**：对于需要多步输出的任务，参考 GGS-NN 的多步序列预测架构

---

## 关键要点总结

1. **GGNN = GNN + GRU + 固定步展开 + BPTT 训练**，解决了原始 GNN 收缩映射约束限制表达能力的问题
2. **门控机制是核心**：GRU 的更新门/重置门使信息能有选择地传播和保留
3. **少量样本即可学习**：在结构化图任务上，50 个样本即可达到完美准确率
4. **序列输出是重要扩展**：GGS-NN 使图模型能输出序列（路径、公式等），极大拓展了应用场景
5. **代码实现用邻接矩阵**：通过稀疏邻接矩阵高效实现消息传递，每种边类型独立参数
