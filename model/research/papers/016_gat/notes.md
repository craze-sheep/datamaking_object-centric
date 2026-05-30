# GAT: Graph Attention Networks

## 基本信息
- **标题**: Graph Attention Networks
- **作者**: Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
- **年份**: 2018
- **会议**: ICLR 2018
- **论文链接**: https://arxiv.org/abs/1710.10903
- **官方代码**: https://github.com/PetarV-/GAT
- **本地PDF**: `gat.pdf`

## 核心贡献

1. **提出图注意力网络（GAT）**: 将自注意力（self-attention）机制引入图神经网络，用可学习的注意力权重替代GCN中的固定归一化邻接矩阵权重
2. **掩码注意力（Masked Attention）**: 只对一阶邻居计算注意力，将图结构注入注意力机制，而非全图注意力
3. **多头注意力（Multi-Head Attention）**: 借鉴Transformer，使用多个独立注意力头稳定训练、增强表达能力
4. **高效且通用**: 时间复杂度与GCN相当，无需特征分解等昂贵矩阵运算；同时适用于转导（transductive）和归纳（inductive）学习

## 模型架构

### 图注意力层（Graph Attentional Layer）

**输入**: 节点特征集合 $\mathbf{h} = \{\vec{h}_1, \vec{h}_2, \ldots, \vec{h}_N\}$, $\vec{h}_i \in \mathbb{R}^F$

**Step 1 - 线性变换**: 对每个节点施加共享的线性变换 $\mathbf{W} \in \mathbb{R}^{F' \times F}$

**Step 2 - 计算注意力系数**:

$$e_{ij} = \text{LeakyReLU}(\vec{a}^T [\mathbf{W}\vec{h}_i \| \mathbf{W}\vec{h}_j])$$

其中 $\vec{a} \in \mathbb{R}^{2F'}$ 是可学习的注意力权重向量，$\|$ 表示拼接操作。

**Step 3 - Softmax归一化**（masked attention，仅在邻居 $j \in \mathcal{N}_i$ 上）:

$$\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}_i} \exp(e_{ik})}$$

**Step 4 - 加权聚合**:

$$\vec{h}'_i = \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij} \mathbf{W}\vec{h}_j\right)$$

### 多头注意力（Multi-Head Attention）

使用 $K$ 个独立的注意力头，中间层采用**拼接**（concatenation）:

$$\vec{h}'_i = \overset{K}{\Big\|}{}_{k=1} \sigma\left(\sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \vec{h}_j\right)$$

输出层（预测层）采用**平均**:

$$\vec{h}'_i = \sigma\left(\frac{1}{K} \sum_{k=1}^{K} \sum_{j \in \mathcal{N}_i} \alpha_{ij}^k \mathbf{W}^k \vec{h}_j\right)$$

### 注意力机制实现细节（代码中的分解）

实际代码将注意力计算分解为更高效的形式。设 $\mathbf{W}\vec{h}_i$ 经线性变换后为 $f_i$，则：

$$e_{ij} = \text{LeakyReLU}(f_1(f_i) + f_2(f_j))$$

其中 $f_1$ 和 $f_2$ 是两个独立的 $1\times1$ 卷积（线性变换，输出维度为1）。这等价于:

$$e_{ij} = \text{LeakyReLU}(\vec{a}_1^T f_i + \vec{a}_2^T f_j)$$

是原论文公式 $\vec{a}^T[f_i \| f_j]$ 的一种分解实现（先 leaky_relu 再加 bias mask，再 softmax）。

### 与GCN/相关工作的关系

- **vs GCN**: GCN用固定的 $\frac{1}{\sqrt{d_i d_j}}$ 归一化邻接矩阵作为聚合权重；GAT用动态学习的注意力权重，可区分不同邻居的重要性
- **vs GraphSAGE**: GraphSAGE采样固定大小邻居；GAT利用完整邻居集，无需采样
- **vs MoNet**: GAT可视为MoNet的特例，伪坐标函数为 $u(x,y)=f(x)\|f(y)$，权重函数为 $\text{softmax}(\text{MLP}(u))$
- **适用有向图**: 只需在存在边 $j \to i$ 时计算 $\alpha_{ij}$，无需对称邻接矩阵

## 训练细节

### 数据集

| 数据集 | 任务类型 | 节点数 | 边数 | 特征维度 | 类别数 | 训练/验证/测试 |
|--------|---------|--------|------|----------|--------|---------------|
| Cora | 转导 | 2708 | 5429 | 1433 | 7 | 140/500/1000 |
| Citeseer | 转导 | 3327 | 4732 | 3703 | 6 | 120/500/1000 |
| Pubmed | 转导 | 19717 | 44338 | 500 | 3 | 60/500/1000 |
| PPI | 归纳 | 56944(24图) | 818716 | 50 | 121(多标签) | 20图/2图/2图 |

### 转导学习（Cora/Citeseer/Pubmed）

- **架构**: 2层GAT
- **第1层**: K=8个注意力头，每头F'=8个特征（共64维），ELU激活
- **第2层（输出）**: 1个注意力头，C个特征（类别数），Softmax激活
- **Pubmed调整**: 输出层K=8个注意力头（取平均），L2正则从0.0005增至0.001（因训练集仅60个样本）
- **优化器**: Adam, lr=0.005（Pubmed为0.01）
- **正则化**: L2 weight decay λ=0.0005, Dropout p=0.6（同时用于输入和注意力系数）
- **早停**: patience=100 epochs
- **权重初始化**: Glorot初始化
- **运行次数**: 100次取平均

### 归纳学习（PPI）

- **架构**: 3层GAT
- **第1-2层**: K=4个注意力头，每头F'=256个特征（共1024维），ELU激活
- **第3层（输出）**: K=6个注意力头，每头121个特征，取平均，Logistic Sigmoid激活
- **无L2正则和Dropout**（训练集足够大）
- **使用残差连接（Skip Connections）**
- **Batch size**: 2个图
- **对比基线**: Const-GAT（恒定注意力 $a(x,y)=1$，等价于GCN式聚合）

## 实验结果

### 转导学习（分类准确率，%）

| 方法 | Cora | Citeseer | Pubmed |
|------|------|----------|--------|
| MLP | 55.1 | 46.5 | 71.4 |
| ManiReg | 59.5 | 60.1 | 70.7 |
| SemiEmb | 59.0 | 59.6 | 71.7 |
| LP | 68.0 | 45.3 | 63.0 |
| DeepWalk | 67.2 | 43.2 | 65.3 |
| ICA | 75.1 | 69.1 | 73.9 |
| Planetoid | 75.7 | 64.7 | 77.2 |
| Chebyshev | 81.2 | 69.8 | 74.4 |
| GCN | 81.5 | 70.3 | 79.0 |
| MoNet | 81.7±0.5 | — | 78.8±0.3 |
| GCN-64* | 81.4±0.5 | 70.9±0.5 | 79.0±0.3 |
| **GAT** | **83.0±0.7** | **72.5±0.7** | **79.0±0.3** |

### 归纳学习（PPI, Micro-F1）

| 方法 | PPI |
|------|-----|
| Random | 0.396 |
| MLP | 0.422 |
| GraphSAGE-GCN | 0.500 |
| GraphSAGE-mean | 0.598 |
| GraphSAGE-LSTM | 0.612 |
| GraphSAGE-pool | 0.600 |
| GraphSAGE* | 0.768 |
| Const-GAT | 0.934±0.006 |
| **GAT** | **0.973±0.002** |

### 关键发现

1. **注意力机制的有效性**: GAT在Cora上比GCN提升1.5%，Citeseer提升1.6%，证明为不同邻居分配不同权重是有益的
2. **归纳学习的巨大提升**: PPI上GAT比最佳GraphSAGE提升20.5%，比Const-GAT提升3.9%，证明注意力机制和完整邻居观察的价值
3. **t-SNE可视化**: 第一层特征表示在2D空间中呈现出清晰的类别聚类
4. **注意力系数可视化**: 注意力系数的相对强度可用于解释模型决策

### 消融实验

- **Const-GAT vs GAT**: PPI上Const-GAT（GCN式恒定注意力）为0.934，GAT为0.973，直接证明动态注意力权重的重要性
- **多头注意力**: 稳定训练过程，增强模型表达能力
- **Dropout on attention coefficients**: 关键正则化手段，每次训练迭代每个节点看到随机采样的邻居子集

## 代码实现细节

### 项目结构

```
code/
├── models/
│   ├── base_gattn.py    # 基类，包含损失函数、训练方法、评估指标
│   ├── gat.py           # GAT模型（密集矩阵版本）
│   └── sp_gat.py        # SpGAT模型（稀疏矩阵版本，适用于大数据集如Pubmed）
├── utils/
│   ├── layers.py        # 核心注意力层实现（attn_head, sp_attn_head）
│   ├── process.py       # 数据加载与预处理
│   └── process_ppi.py   # PPI数据集处理
├── execute_cora.py      # Cora训练脚本（密集版）
├── execute_cora_sparse.py # Cora训练脚本（稀疏版）
├── data/                # Cora数据文件
└── pre_trained/         # 预训练模型权重
```

### 核心注意力实现（layers.py）

**密集版 attn_head**:
```python
def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):
    # 输入dropout
    seq = tf.nn.dropout(seq, 1.0 - in_drop)
    # 线性变换: 1x1卷积实现 W*h
    seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)
    # 注意力计算分解: a^T[Wh_i || Wh_j] = a1^T*Wh_i + a2^T*Wh_j
    f_1 = tf.layers.conv1d(seq_fts, 1, 1)  # [B, N, 1]
    f_2 = tf.layers.conv1d(seq_fts, 1, 1)  # [B, N, 1]
    logits = f_1 + tf.transpose(f_2, [0, 2, 1])  # [B, N, N] 广播相加
    # LeakyReLU + bias_mask（非邻居设为-1e9） + softmax
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)
    # 注意力系数dropout
    coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
    # 加权聚合
    vals = tf.matmul(coefs, seq_fts)  # [B, N, out_sz]
    ret = tf.contrib.layers.bias_add(vals)
    # 残差连接（可选）
    if residual:
        ret = ret + (conv1d(seq, ret.shape[-1], 1) if seq.shape[-1] != ret.shape[-1] else seq)
    return activation(ret)
```

**稀疏版 sp_attn_head**:
- 使用 `tf.sparse_tensor_dense_matmul` 进行稀疏矩阵乘法
- 存储复杂度为 $O(|V|+|E|)$（线性于节点和边数）
- 限制：仅支持batch_size=1（TF稀疏矩阵乘法限制）
- 使用 `tf.sparse_softmax` 替代dense softmax

### 关键实现要点

1. **注意力分解**: 原论文的 $\vec{a}^T[Wh_i \| Wh_j]$ 被分解为两个独立卷积 $f_1(Wh_i) + f_2(Wh_j)$，通过广播机制高效计算 $N \times N$ 注意力矩阵
2. **Bias Mask**: `adj_to_bias()` 函数将邻接矩阵转换为bias矩阵：非邻居位置设为 $-10^9$，邻居位置为0，加入自环。这样softmax后非邻居注意力为接近0
3. **预处理**: 特征矩阵行归一化（`preprocess_features`），邻接矩阵加自环
4. **损失函数**: Masked softmax cross-entropy（仅计算训练/验证/测试节点的损失）
5. **早停策略**: 同时监控验证集loss和accuracy，两者同时满足时保存模型，连续100个epoch无改善则停止
6. **框架**: TensorFlow 1.x（使用placeholder, Session等老式API）

### 训练超参数汇总

| 超参数 | Cora/Citeseer | Pubmed | PPI |
|--------|--------------|--------|-----|
| 学习率 | 0.005 | 0.01 | 0.005 |
| L2正则 | 0.0005 | 0.001 | 0 |
| Dropout | 0.6 | 0.6 | 0 |
| 隐藏单元 | [8] | [8] | [256, 256] |
| 注意力头数 | [8, 1] | [8, 8] | [4, 4, 6] |
| 残差连接 | False | False | True |
| 激活函数 | ELU | ELU | ELU |
| 早停patience | 100 | 100 | 100 |

## 与当前研究的关联

### 可借鉴的点

1. **注意力聚合替代均值聚合**: 当前GNN的mean聚合对所有邻居赋予相同权重，可替换为GAT式注意力聚合以区分重要邻居
2. **多头注意力增强表达**: 多个独立注意力头可捕获不同方面的邻居关系
3. **稀疏矩阵实现**: 对于大规模图，可参考SpGAT的稀疏实现降低内存消耗
4. **注意力Dropout**: 作为正则化手段，使模型在训练时看到随机邻居子集，防止过拟合

### 对Slot-based模型的启示

- **时序交互建模**: 在slot之间的消息传递中引入注意力机制，让每个slot能区分重要和不重要的邻居slot
- **多粒度注意力**: 可对不同类型的消息（如视觉特征、位置信息）使用不同的注意力头
- **可解释性**: 注意力权重可直接可视化，帮助理解模型关注哪些交互关系
