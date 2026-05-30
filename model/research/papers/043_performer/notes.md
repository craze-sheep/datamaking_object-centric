# Performer: Rethinking Attention with Performers

## 基本信息
- **作者**：Krzysztof Choromanski*, Valerii Likhosherstov*, David Dohan*, Xingyou Song* 等
- **机构**：Google, University of Cambridge, DeepMind, Alan Turing Institute
- **年份**：2020（arXiv），ICLR 2021 正式发表
- **论文链接**：https://arxiv.org/abs/2009.14794
- **代码**：github.com/google-research/google-research/tree/master/performer（官方 Jax）；github.com/lucidrains/performer-pytorch（PyTorch 第三方实现）

## 核心贡献

1. **首个可证明精确估计标准 softmax attention 的线性 Transformer**：时间和空间复杂度从 O(L²d) 降至 O(Lrd)，其中 r 为随机特征数，通常 r ≪ L。
2. **FAVOR+ 机制**（Fast Attention Via positive Orthogonal Random features）：
   - 提出**正交随机特征（Orthogonal Random Features, ORF）**：通过 Gram-Schmidt 正交化将随机投影向量 ω₁,...,ωₘ 约束为正交，降低估计方差。
   - 提出**正随机特征（Positive Random Features, PRF）**：用 cosh / exp 替代 sin/cos 三角函数来近似 softmax 核，保证特征值非负，避免训练不稳定（三角特征会导致负对角归一化器 D⁻¹，甚至 NaN）。
3. **理论保证**：无偏/近无偏估计、一致收敛（uniform convergence）、大偏差概率指数级小的界。
4. **通用性**：FAVOR+ 可用于任意可核化注意力（不仅是 softmax），包括 ReLU 等广义注意力。
5. **向后兼容**：可直接加载预训练 Transformer 权重，少量微调即可恢复性能。

## 模型架构

### 标准 Attention 的问题
标准 softmax attention：`Att(Q,K,V) = D⁻¹ exp(QK^T/√d) V`
- 需显式存储 L×L 注意力矩阵 A = exp(QK^T/√d)
- 时间复杂度 O(L²d)，空间复杂度 O(L² + Ld)

### FAVOR+ 核心思想
将 softmax 核 SM(x,y) = exp(x^T y) 用随机特征映射 φ 近似：
```
SM(x,y) ≈ φ(x)^T φ(y)
```
从而将 attention 改写为：
```
Att(Q,K,V) ≈ D̃⁻¹ (Q' ((K')^T V))
```
其中 Q' = φ(Q), K' = φ(K)。关键：**先算 (K')^T V 得到 r×d 矩阵，再乘 Q'**，避免存储 L×L 矩阵。

### 正交随机特征 (ORF)
- 生成随机矩阵 W ∈ R^{m×d}，将 m 行分为若干块（每块 d×d），对每块做 QR 分解取正交行。
- 正交化保证不同随机样本 ωᵢ 之间精确正交，降低方差。
- 要求 m ≤ d（实验中总是满足，因为 m ≈ d log d）。

### 正随机特征 (PRF) — 关键创新
传统方法用三角函数（sin/cos）近似 softmax 核：
```
SM_trig_m(x,y) = h(x)h(y) * (1/m) Σ [sin(ωᵢ^T x)sin(ωᵢ^T y) + cos(ωᵢ^T x)cos(ωᵢ^T y)]
```
**问题**：sin/cos 产生负值 → kernel 估计值可为负 → 归一化器 D 对角线可为负 → 训练崩溃。

**解决方案**：利用恒等式 `exp(x^T y) = Λ · E[cosh(ω^T(x+y))]`，其中 `Λ = exp(-(‖x‖²+‖y‖²)/2)`：
- `SM+_m(x,y)`：使用 `exp(ω^T x - ‖x‖²/2)` 形式的特征，值恒非负
- `SM^hyp+_m(x,y)`：使用 `exp(ω^T z)` 和 `exp(-ω^T z)`（z = x+y），进一步降低方差

**理论保证**（Lemma 2）：当 SM(x,y) → 0 时：
- 三角特征 MSE → ∞（灾难性）
- 正特征 MSE → 0（理想）

### 正则化 Softmax 核
将 ω 替换为 √d · ω/‖ω‖（均匀球面分布），得到正则化 softmax 核 SM_REG。
Theorem 1 证明 SM_REG 是 SM 的全局下界，可用其近似标准 softmax。

### 两种工作模式
1. **Softmax 近似模式**（Performer-SOFTMAX）：直接近似标准 softmax attention
2. **广义注意力模式**（默认推荐）：使用 ReLU 等非 softmax 核函数（`generalized_kernel`），`kernel_fn=ReLU`, `kernel_epsilon=1e-3`

## 训练细节

- **优化器**：Adam（β₁=0.9, β₂=0.98, ε=10⁻⁹）
- **学习率**：固定 10⁻³
- **梯度裁剪**：0.5
- **权重衰减**：0.1
- **Dropout**：0.1
- **默认超参**：
  - Softmax 模式：`num_features=256`, `ortho_features=True`, `ortho_scaling=0.0`
  - 广义注意力模式：`num_features=256`, `kernel=ReLU`, `kernel_epsilon=1e-3`
- **随机特征重绘**：建议每隔一定步数（如 1000 步）重新生成随机投影矩阵，改善训练稳定性
- **硬件**：16×16 TPU-v2（蛋白质实验），V100 GPU（速度对比）
- **无需额外调参**：Performer 使用与标准 Transformer 完全相同的训练超参

## 实验结果

### 计算效率对比
- Performer 在 L 维度上达到近线性时间和亚二次内存（不存储 L² 注意力矩阵）
- 效率接近"注意力直接返回 V"的理论最优（OPT 线）
- V100 GPU 上，L=2048 时 Performer 比 Transformer 快约 2×

### 近似误差验证（L=4096, d=16）
- 正交特征 < IID 特征（MSE 更低）
- 正特征 < 三角特征（MSE 更低，尤其在 kernel 值小的区域）

### 蛋白质序列建模（TrEMBL，36 层，L 可变）
- Reformer 和 Linformer 在蛋白质数据上准确率显著下降
- Performer-RELU（广义注意力）在单向和双向设置下均取得最高准确率
- Performer-SOFTMAX 与标准 Transformer 准确率一致，验证理论

### 长序列任务
- **ImageNet64**（L=12288）：标准 Transformer 无法运行；Performer/6层 ≈ Reformer/12层；Performer/12层 ≈ Reformer/24层；Performer 比 Reformer 快 2×
- **蛋白质交互预测**（L=8192）：标准 Transformer OOM；Performer 可用标准架构训练，达到 ≈24% 准确率 vs 小 Transformer 的 ≈19%

### 与预训练模型的向后兼容
- 在 LM1B 上：将预训练 Transformer 权重加载到 Performer（初始准确率 0.07），少量微调即可恢复
- 在 PG-19 上：三角特征不稳定；正特征+重绘+SM_REG 才能匹配 Transformer

## 代码实现细节

代码位于 `code/performer_pytorch/performer_pytorch.py`（PyTorch 实现，lucidrains 维护）：

### 核心组件
1. **`softmax_kernel(data, projection_matrix, is_query)`**：
   - 对 Q/K 施加随机投影：`data_dash = (data_normalizer * data) @ projection_matrix`
   - 减去 `‖data‖²/2` 项（对应 PRF 的 exp(-‖x‖²/2)）
   - 减去 `amax` 做数值稳定化
   - 加 eps=1e-4 防止零值

2. **`generalized_kernel(data, projection_matrix, kernel_fn)`**：
   - 先投影到随机特征空间，再施加 kernel_fn（默认 ReLU）
   - 加 kernel_epsilon=0.001
   - 若 `projection_matrix=None`，直接对归一化数据施加 kernel_fn

3. **`gaussian_orthogonal_random_matrix(nb_rows, nb_columns)`**：
   - 生成正交随机矩阵：分块 QR 分解
   - `scaling=0`：每行用随机高斯向量的范数缩放
   - `scaling=1`：用 √d 固定缩放

4. **`FastAttention`**：
   - 默认 `nb_features = dim_heads * log(dim_heads)`
   - 支持 softmax 模式和 generalized 模式
   - 非因果：`linear_attention(q,k,v)` — 先算 k^T v 再乘 q
   - 因果：`causal_linear_attention(q,k,v)` — 使用 CUDA 加速的前缀和（来自 fast_transformers）
   - 支持 `redraw_projection_matrix()` 定期重绘随机投影

5. **`ProjectionUpdater`**：
   - 跟踪调用次数，每隔 `feature_redraw_interval`（默认 1000）步自动重绘所有 FastAttention 层的投影矩阵

6. **`Performer` 类**：
   - 标准 Transformer 块：Attention + FeedForward，支持 PreLayerNorm / PreScaleNorm / ReZero
   - 支持 local attention heads 与 global fast attention 混合
   - 支持 reversible layers 节省内存
   - 支持 cross-attention

7. **`PerformerLM`**：
   - 语言模型封装：token embedding + positional embedding + Performer backbone + LayerNorm + 输出投影
   - 支持 rotary positional embedding、axial positional embedding、absolute positional embedding
   - 支持 weight tying（`tie_embed`）

### 关键实现细节
- **数值稳定**：softmax_kernel 中对 query 和 key 分别减去 amax（query 按最后一维，key 按最后两维）
- **数据归一化**：`data_normalizer = d^{-0.25}`，将 Q/K 乘以该因子
- **比率因子**：`ratio = m^{-0.5}`（m 为随机特征数）
- **因果 attention**：非因果用 einsum 直接计算；因果用 CUDA kernel 的前缀和（`CausalDotProduct`），不支持 CUDA 时回退到分块循环实现

## 与当前研究的关联

1. **线性 Attention 范式的奠基工作**：Performer 是首个严格证明可以无偏近似 softmax attention 的线性方法，后续大量线性 attention 工作（如 Linear Transformer、CosFormer 等）都受其启发。

2. **核方法与 Attention 的统一视角**：将 attention 视为核函数 `K(q,k) = E[φ(q)^T φ(k)]`，为后续 kernel-based attention 研究提供了理论框架。

3. **随机特征方法的扩展**：FAVOR+ 中的正交随机特征和正随机特征可独立用于其他需要核近似的场景（如高斯过程、Wasserstein 距离估计等）。

4. **长序列建模**：Performer 的线性复杂度使其能处理 L=8192~12288 级别的长序列，在蛋白质序列、图像生成等任务上展现出优势，与当前长上下文模型（如 Mamba、RWKV 等）的目标一致。

5. **与 FlashAttention 等方法的互补**：FlashAttention 通过 IO 优化加速标准 attention（仍是 O(L²)），Performer 从根本上改变复杂度（O(Lr)）。两者可结合使用。

6. **广义注意力的探索**：论文发现 ReLU 替代 softmax 在蛋白质任务上效果更好，提示 softmax 并非唯一最优选择，启发了后续对不同注意力核函数的探索。
