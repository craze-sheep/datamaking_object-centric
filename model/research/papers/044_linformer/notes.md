# Linformer: Self-Attention with Linear Complexity

## 基本信息
- **作者**：Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
- **机构**：Facebook AI, Seattle
- **年份**：2020
- **论文链接**：https://arxiv.org/abs/2006.04768
- **关键词**：线性注意力、低秩近似、高效Transformer

## 核心贡献
1. **理论证明self-attention是低秩的**：通过对预训练Transformer模型（RoBERTa-base/large）的注意力矩阵P进行奇异值分析，发现其呈现明显的长尾谱分布——大部分信息集中在前几个最大奇异值上，且越高层的注意力矩阵秩越低。
2. **提出线性复杂度的self-attention机制**：通过将K和V投影到低维空间（维度k<<n），将self-attention的时间和空间复杂度从O(n²)降至O(n)。
3. **性能与标准Transformer相当**：在GLUE基准任务和IMDB情感分析上，Linformer与RoBERTa-base性能持平甚至略优。
4. **显著的推理加速和内存节省**：序列长度n=8192、k=128时，推理速度提升5.5倍，内存节省28倍。

## 模型架构

### 整体思路
标准Transformer的瓶颈在于计算上下文映射矩阵 P = softmax(QK^T/√d) ∈ R^{n×n}，其时间和空间复杂度均为O(n²)。Linformer的核心思想是：既然P是低秩的，那么可以用低秩矩阵来近似它。

### 低秩投影（Low-Rank Projection）
引入两个投影矩阵 E, F ∈ R^{k×n}，将原始的n维Key和Value投影到k维空间：

```
K' = E · K    (k × n) · (n × d) → (k × d)
V' = F · V    (k × n) · (n × d) → (k × d)
```

投影后的注意力计算：
```
head_i = softmax(Q·W_Q · (E·K·W_K)^T / √d_k) · (F·V·W_V)
         |_______________ P̄ ∈ R^{n×k} ______________|  |_{k×d}|
```

复杂度从O(n²·d)降为O(n·k·d)，当k<<n时显著降低。

### 理论保证
- **Theorem 1（self-attention低秩性）**：对于任意Q,K,V ∈ R^{n×d}，存在低秩矩阵P̃使得 ||P̃w^T - Pw^T|| ≤ ε||Pw^T|| 以高概率成立，且rank(P̃) = Θ(log n)。
- **Theorem 2（线性self-attention）**：当 k = min{Θ(9d·log(d)/ε²), 5Θ(log(n)/ε²)} 时，存在E,F使得线性注意力近似误差不超过ε。关键结论：k可以独立于序列长度n，仅依赖于嵌入维度d。

### 投影矩阵的参数共享策略
三种共享级别（以12层12头模型为例）：
1. **Headwise Sharing**：每层共享E和F，共24个投影矩阵
2. **Key-Value Sharing**：每层共享一个E（即E=F），共12个投影矩阵
3. **Layerwise Sharing**：所有层所有头共享一个E，仅1个投影矩阵

实验表明，Layerwise Sharing性能几乎不下降，且是效果最好的策略。

### 其他效率优化
- **非均匀投影维度**：不同层/头可使用不同的k值（高层可用更小的k）
- **多种投影方式**：除了线性投影，还可以用mean/max pooling或卷积（stride=n/k）

## 训练细节
- **预训练设置**：
  - 语料：BookCorpus + English Wikipedia（3300M词）
  - 目标：Masked Language Modeling (MLM)
  - 硬件：64张Tesla V100 GPU
  - 更新步数：250k
  - 默认使用混合精度训练
- **微调任务**：
  - SST-2（情感分类）、IMDB（情感分类）
  - QNLI（自然语言推理）、QQP（文本相似度）
- **投影维度k的选择**：
  - n=512时，k=128即可接近标准Transformer性能
  - n=1024时，k=256即可
  - 固定k=256，序列长度从512增到4096，最终perplexity基本不变

## 实验结果

### 预训练Perplexity
- k=128 (n=512)时，Linformer的验证perplexity已接近标准Transformer
- 三种参数共享策略中，Layerwise Sharing表现最佳
- 序列长度增加时，固定k=256的最终perplexity基本不变，验证了线性复杂度特性

### 下游任务（微调结果）

| 模型 | n | k | SST-2 | IMDB | QNLI | QQP | 平均 |
|------|---|---|-------|------|------|-----|------|
| RoBERTa-base | 512 | - | 93.1 | 94.1 | 90.9 | 90.9 | 92.25 |
| Linformer | 512 | 128 | 92.4 | 94.0 | 90.4 | 90.2 | 91.75 |
| Linformer (shared kv+layer) | 512 | 256 | 93.1 | 94.1 | 91.2 | 90.8 | 92.30 |
| BERT-base | 512 | - | 92.7 | 93.5 | 91.8 | 89.6 | 91.90 |
| DistilBERT | 512 | - | 91.3 | 92.8 | 89.2 | 88.5 | 90.45 |
| Linformer (shared kv+layer) | 1024 | 256 | 93.2 | 94.2 | 90.8 | 90.5 | 92.18 |

### 推理效率（Tesla V100 16GB）

**时间加速比**（Linformer vs Transformer，Layerwise Sharing）：

| 序列长度n | k=128 | k=256 | k=512 | k=1024 | k=2048 |
|-----------|-------|-------|-------|--------|--------|
| 512 | 1.5× | 1.3× | - | - | - |
| 1024 | 1.7× | 1.6× | 1.3× | - | - |
| 2048 | 2.6× | 2.4× | 2.1× | 1.3× | - |
| 4096 | 3.4× | 3.2× | 2.8× | 2.2× | 1.3× |
| 8192 | 5.5× | 5.0× | 4.4× | 3.5× | 2.1× |
| 16384 | 8.6× | 7.8× | 7.0× | 5.6× | 3.3× |

**内存节省比**：

| 序列长度n | k=128 | k=256 | k=512 |
|-----------|-------|-------|-------|
| 2048 | 6.1× | 5.6× | 3.6× |
| 4096 | 14× | 13× | 8.3× |
| 8192 | 28× | 26× | 17× |
| 16384 | 56× | 48× | 32× |
| 65536 | 60× | 52× | 40× |

## 代码实现细节

代码位于 `code/linformer/linformer.py`，核心类为 `LinformerSelfAttention`：

### 关键实现要点

1. **投影矩阵的实现**：
   - E和F不是nn.Linear层，而是nn.Parameter，形状为 `(seq_len, k)`
   - 初始化方式：`uniform_(-1/√dim, 1/√dim)`（通过`init_`函数）
   - 投影操作通过`torch.einsum('bnd,nk->bkd', ...)`实现，即沿序列维度做矩阵乘法

2. **序列长度自适应**：
   - 当实际序列长度 < 最大seq_len时，通过切片投影矩阵 `t[:kv_len]` 来适配
   ```python
   if kv_len < self.seq_len:
       kv_projs = map(lambda t: t[:kv_len], kv_projs)
   ```

3. **share_kv选项**：
   - 当`share_kv=True`时，Key和Value共享同一个投影矩阵`proj_k`
   - 同时Value直接复用Key的线性变换结果，不单独计算`to_v`

4. **one_kv_head选项**：
   - 当`one_kv_head=True`时，K和V只用一个头的维度（dim_head），然后通过expand广播到所有头
   - 这进一步减少了参数量

5. **注意力计算流程**：
   ```
   x → to_q → queries: [B, N, H*D_h]
   x → to_k → keys:    [B, N, D_h或H*D_h]
   keys × proj_k → projected_keys: [B, K, D_h]  (沿序列维度降维)
   queries × projected_keys^T → dots: [B, H, N, K]
   softmax → attn → × projected_values → out
   ```

6. **可逆网络支持**（reversible.py）：
   - `ReversibleBlock`：将输入分为x1,x2两半，通过y1=x1+f(x2), y2=x2+g(y1)实现可逆
   - 反向传播时不需要存储中间激活值，通过逆运算恢复，节省内存
   - `SequentialSequence`：标准的顺序执行，支持layer_dropout
   - `ReversibleSequence`：可逆执行，通过拼接[x,x]实现，最终对两半求和

### 与论文的差异
- 论文中的E,F是独立于每个头和层的（可共享），代码中默认每层独立但支持share_kv
- 代码中投影矩阵是可学习参数（非固定的随机矩阵），与论文理论分析中的JL投影不同
- 代码支持`one_kv_head`模式（论文未提及），进一步压缩参数

## 与其他高效Transformer的对比

| 方法 | 复杂度 | 顺序操作数 | 特点 |
|------|--------|-----------|------|
| Recurrent | O(n) | O(n) | 顺序处理，无法并行 |
| Transformer | O(n²) | O(1) | 基线 |
| Sparse Transformer | O(n√n) | O(1) | 稀疏注意力，性能下降~2% |
| Reformer | O(n·log n) | O(log n) | LSH哈希，仅在n>2048时有效 |
| **Linformer** | **O(n)** | **O(1)** | **低秩投影，性能几乎无损** |

## 与当前研究的关联

Linformer的核心思想——通过低秩投影降低注意力复杂度——对处理长序列任务有重要参考价值：
- 对于slot attention等需要处理较长序列的场景，Linformer的投影思想可直接借鉴
- Layerwise Sharing策略表明单个投影矩阵即可胜任，实现简单且效果好
- 代码中的einsum实现方式简洁高效，可直接复用
