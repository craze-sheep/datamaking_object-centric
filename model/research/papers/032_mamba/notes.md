# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

## 基本信息
- **作者**：Albert Gu (CMU), Tri Dao (Princeton)
- **年份**：2023 (arXiv:2312.00752v2, 更新于2024年5月)
- **会议/期刊**：arXiv preprint
- **论文链接**：https://arxiv.org/abs/2312.00752
- **代码**：https://github.com/state-spaces/mamba
- **关键词**：选择性状态空间模型 (Selective SSM)、线性时间序列建模、硬件感知算法、内容感知推理

## 核心贡献

1. **选择性机制 (Selection Mechanism)**：提出将 SSM 参数 (Δ, B, C) 设计为输入的函数，使模型能够根据内容选择性地传播或遗忘信息，解决了传统 LTI (线性时不变) SSM 无法进行内容感知推理的根本缺陷
2. **硬件感知算法 (Hardware-Aware Algorithm)**：设计了融合的并行扫描算法，利用 GPU 内存层级 (HBM → SRAM) 避免显式展开状态矩阵，实现高效计算
3. **简化架构 (Simplified Architecture)**：将 SSM 块与 MLP 块合并为统一的 Mamba 块，无需注意力机制或独立的 MLP 块
4. **首个达到 Transformer 质量的线性时间模型**：在语言、音频、基因组学等多个模态上达到 SOTA

## 模型架构

### 3.1 状态空间模型 (SSM) 基础

SSM 是一类受经典状态空间理论启发的序列模型，核心是连续到离散的映射：

**连续形式**：
```
h'(t) = A·h(t) + B·x(t)
y(t) = C·h(t)
```

**离散形式 (通过离散化)**：
```
h_t = Ā·h_{t-1} + B̄·x_t
y_t = C·h_t
```

**卷积形式** (仅当参数时不变时可用)：
```
K = (CB, CAB, ..., CA^k B, ...)
y = x * K
```

**零阶保持 (ZOH) 离散化**：
```
Ā = exp(ΔA)
B̄ = (ΔA)^{-1}·(exp(ΔA) - I)·ΔB
```

**结构化 SSM**：A 矩阵采用对角结构，将 N×N 矩阵简化为 N 个参数。每个通道独立应用 SSM，总隐藏状态维度为 D×N。

### 3.2 选择性 SSM (Selective SSM / S6)

**核心思想**：让 SSM 参数依赖输入，从而实现内容感知推理。

**与传统 SSM 的对比**：
| 参数 | S4 (LTI) | S6 (Selective) |
|------|----------|----------------|
| A | 固定 (D, N) | 固定 (D, N) |
| B | 固定 (D, N) | **输入依赖** (B, L, N) |
| C | 固定 (D, N) | **输入依赖** (B, L, N) |
| Δ | 固定 (D) | **输入依赖** (B, L, D) |

**参数化方式**：
- `s_B(x) = Linear_N(x)` — B 是输入的线性投影
- `s_C(x) = Linear_N(x)` — C 是输入的线性投影
- `s_Δ(x) = Broadcast_D(Linear_1(x))` — Δ 先投影到1维再广播到D维
- `τ_Δ = softplus` — 确保 Δ 为正

**Δ 的关键作用**：
- **大 Δ**：重置状态 h，聚焦当前输入 → "选择"当前 token
- **小 Δ**：保持状态，忽略当前输入 → 记忆历史信息
- Δ 投影到1维再广播的动机：当某个输入 x_t 应被完全忽略时，所有 D 个通道应同时忽略

**与 RNN 门控机制的关系** (定理1)：
当 N=1, A=-1, B=1 时，选择性 SSM 退化为：
```
g_t = σ(Linear(x_t))
h_t = (1-g_t)·h_{t-1} + g_t·x_t
```
这正是经典的 RNN 门控机制，Δ 是门控的广义形式。

### 3.3 硬件感知算法 (Hardware-Aware Algorithm)

**问题**：选择性 SSM 破坏了时不变性，无法使用卷积模式高效计算。朴素递推需要将 (B, L, D, N) 的状态展开到 GPU HBM，IO 开销巨大。

**解决方案 — 三种经典技术**：

1. **核融合 (Kernel Fusion)**：
   - 不在 HBM 中准备扫描输入 (A, B) 的完整 (B, L, D, N) 张量
   - 直接从 HBM 加载参数 (Δ, A, B, C) 到快速 SRAM
   - 在 SRAM 中执行离散化和递推
   - 仅将最终输出 (B, L, D) 写回 HBM

2. **并行扫描 (Parallel Scan)**：
   - 虽然选择性 SSM 不是线性的，但仍可用 work-efficient 并行扫描算法 (Blelloch 1990) 并行化
   - 避免顺序递推的瓶颈

3. **重计算 (Recomputation)**：
   - 不保存中间状态（反向传播需要）
   - 反向传播时从 HBM 重新加载输入并在 SRAM 中重新计算
   - 最终内存需求与 FlashAttention 优化的 Transformer 相同

**效率优势**：
- 训练：高效扫描比标准实现快 40×，比 FlashAttention-2 在序列长度 >2K 时更快
- 推理：由于无 KV 缓存，Mamba 实现 5× 吞吐量提升（可使用更大 batch size）

### 3.4 Mamba 架构

**设计哲学**：将 H3 块（SSM + 门控）和 MLP 块合并为单一的 Mamba 块。

**Mamba 块结构**：
```
输入 x (B, L, D)
  ├─ Linear → xz (B, L, 2·E·D)     # 扩展因子 E=2
  ├─ split → x, z
  ├─ Conv1d(x) → SiLU               # 深度可分离卷积, kernel_size=4
  ├─ Linear → dt, B, C              # 选择性参数
  ├─ Selective SSM(x)               # 核心 SSM 递推
  ├─ y · SiLU(z)                    # 门控乘法
  └─ Linear → output (B, L, D)
```

**关键设计选择**：
- 扩展因子 E=2，两层堆叠匹配 Transformer 的 12D² 参数量
- 使用 SiLU/Swish 激活函数 → Gated MLP 变成 SwiGLU
- 可选的 LayerNorm（借鉴 RetNet）
- 残差连接 + 标准归一化

### 3.5 选择机制的性质

1. **可变间距 (Variable Spacing)**：过滤无关噪声 token（如语言填充词 "um"）
2. **上下文过滤 (Filtering Context)**：LTI 模型随上下文增长性能可能下降；选择性模型可随时重置状态
3. **边界重置 (Boundary Reset)**：独立序列拼接时可重置状态（Δ→∞ 或 g_t→1）

### 3.6 额外模型细节

- **实数 vs 复数**：默认使用实数值（对离散数据如文本、DNA 更好；复数对连续数据如音频更好）
- **初始化**：A_n = -(n+1)（S4D-Real），随机初始化也工作良好
- **Δ 参数化**：投影维度 R 设为 D 的一小部分（R=d_model/16），使用低秩投影

## 训练细节

### 语言建模
- **数据集**：Pile
- **模型规格**：镜像 GPT-3 规格（125M 到 1.3B 参数）
- **训练长度**：300B tokens，上下文长度 2048
- **Tokenizer**：NeoX（与 Pythia 相同）
- **优化器**：遵循 Brown et al. (2020) 的训练配方

### DNA 预训练
- **数据集**：HG38（人类基因组，~4.5B tokens）
- **训练**：因果语言建模（下一个 token 预测）
- **上下文长度**：从 1024 到 1M
- **序列长度预热**：类似 HyenaDNA

### 音频建模
- **数据集**：YouTubeMix（4小时钢琴音乐，16kHz 采样）
- **架构**：替换 SaShiMi 中的 S4+MLP 为 Mamba 块
- **唯一使用复数参数化的实验**

## 实验结果

### 合成任务

**选择性复制 (Selective Copying)**：
- S4 仅 18.3% 准确率，S6 达到 97.0%
- Mamba 架构 + S6：99.8%
- 证明选择机制是关键，而非架构门控

**归纳头 (Induction Heads)**：
- Mamba 在训练长度 256 上训练，外推到 1M tokens（4000×）仍完美
- 其他方法最多外推 2×

### 语言建模

**Scaling Laws** (125M → 1.3B)：
- Mamba 是首个匹配强 Transformer++ 配方（PaLM/LLaMa 风格）的无注意力模型
- 随序列长度增长优势更明显

**零样本评估** (Table 3)：

| 模型 | 参数量 | LAMBADA ppl↓ | HellaSwag acc↑ | 平均 acc↑ |
|------|--------|-------------|----------------|-----------|
| Pythia-160M | 160M | 38.10 | 30.2 | 40.6 |
| **Mamba-130M** | 130M | **16.07** | **35.3** | **44.7** |
| Pythia-1.4B | 1.4B | 6.08 | 52.1 | 55.2 |
| RWKV-1.5B | 1.5B | 7.04 | 52.5 | 54.3 |
| **Mamba-1.4B** | 1.4B | **5.04** | **59.1** | **59.7** |
| Pythia-2.8B | 2.8B | 5.04 | 59.3 | 59.1 |
| RWKV-3B | 3B | 5.24 | 59.6 | 59.6 |
| **Mamba-2.8B** | 2.8B | **4.23** | **66.1** | **63.3** |

- Mamba-3B 质量匹配 2× 大小的 Transformer
- 每个尺寸的每个评估指标均为同类最佳

### DNA 建模
- **模型规模**：Mamba 用 3-4× 更少参数匹配 Transformer++ 和 HyenaDNA
- **上下文长度**：Mamba 随上下文增长持续改善直到 1M；HyenaDNA 随上下文增长反而变差
- **大猿 DNA 分类**：在 99% DNA 相似的 5 个物种分类中，Mamba 准确率随序列长度从 49.7% 提升到 78.4%

### 音频建模
- **自回归预训练**：Mamba 在所有上下文长度上优于 SaShiMi (S4+MLP)
- **语音生成 (SC09)**：
  - Mamba (6.1M) FID=0.94，超越 GAN 和扩散模型
  - Mamba (24.3M) FID=0.67，接近训练数据质量

### 效率基准
- **训练**：高效扫描比 PyTorch 标准实现快 40×
- **推理**：5× 吞吐量提升（Mamba-6.9B 推理吞吐 > Transformer-1.3B）

### 消融实验
- **选择性参数**：Δ 最重要（连接 RNN 门控），但 Δ+B+C 协同效果最好
- **A 初始化**：S4D-Real (A_n=-(n+1)) 优于 S4D-Lin（复数）
- **状态维度 N**：当 B、C 也是选择性时，增大 N（1→16）带来 >1.0 困惑度下降，参数增加仅 1%

## 代码实现细节

### Mamba 1 (mamba_simple.py)

**核心类 `Mamba`**：
```python
class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_rank="auto", ...):
        d_inner = expand * d_model  # 默认 2× 扩展
        dt_rank = ceil(d_model / 16)  # Δ 投影的低秩维度

        # 输入投影：将 d_model 投影到 2*d_inner (x 和 z 各一半)
        in_proj = Linear(d_model, d_inner * 2)

        # 深度可分离因果卷积
        conv1d = Conv1d(d_inner, d_inner, kernel_size=d_conv, groups=d_inner, padding=d_conv-1)

        # 选择性参数投影
        x_proj = Linear(d_inner, dt_rank + 2*d_state)  # 输出 dt_rank + d_state(B) + d_state(C)
        dt_proj = Linear(dt_rank, d_inner)  # Δ 从低秩投影到 d_inner

        # A 矩阵：S4D-Real 初始化，存储 log(A)
        A = arange(1, d_state+1)  # [1, 2, ..., N]
        A_log = log(A)  # nn.Parameter，保持 fp32

        # D 跳跃参数
        D = ones(d_inner)  # nn.Parameter，保持 fp32
```

**前向传播流程**：
1. `in_proj` 将输入投影为 xz (B, L, 2D)
2. split 为 x 和 z
3. `conv1d` 对 x 做因果卷积 + SiLU 激活
4. `x_proj` 从卷积输出提取 Δ, B, C
5. `dt_proj` 将 Δ 从低秩空间投影到 d_inner
6. 调用 `selective_scan_fn` (CUDA 融合核) 执行选择性扫描
7. 输出乘以 SiLU(z) 门控
8. `out_proj` 投影回 d_model

**推理模式 (step 函数)**：
- 维护 conv_state (B, D, W) 和 ssm_state (B, D, N)
- 逐步更新：卷积状态滚动 → SSM 状态递推
- 每步常数时间，无需 KV 缓存

### 选择性扫描核 (selective_scan_interface.py)

**`SelectiveScanFn`** (torch.autograd.Function)：
- 前向：调用 `selective_scan_cuda.fwd`，融合的 CUDA 核
- 反向：调用 `selective_scan_cuda.bwd`，支持重计算节省内存
- 输入布局：u(B,D,L), delta(B,D,L), A(D,N), B(B,N,L 或 B,1,N,L), C(B,N,L 或 B,1,N,L)

**参考实现 `selective_scan_ref`**：
```python
deltaA = exp(einsum('bdl,dn->bdln', delta, A))       # 离散化 A
deltaB_u = einsum('bdl,bnl,bdl->bdln', delta, B, u)  # 离散化 B 并乘以输入
for i in range(L):
    x = deltaA[:,:,i] * x + deltaB_u[:,:,i]  # 递推
    y = einsum('bdn,bn->bd', x, C[:,:,i])    # 输出
out = y + u * D  # 加跳跃连接
out = out * silu(z)  # 门控
```

**`MambaInnerFn`**：完全融合的前向+反向核，将卷积→参数提取→扫描→输出投影全部融合，减少中间张量的 HBM 读写。

### Mamba 2 (mamba2.py)

**主要改进**：
- `d_state=128`（Mamba 1 为 16）
- 引入 `headdim` 和 `nheads` 概念（类似多头注意力）
- `ngroups` 参数支持分组 SSM
- 使用 SSD (Structured State-Space Duality) 的 chunk-based 扫描
- 支持张量并行 (ColumnParallelLinear/RowParallelLinear)
- 使用 RMSNorm（门控版本）
- `chunk_size=256` 的分块扫描

**输入投影布局**：`[z, x, B, C, dt]` = 2*d_inner + 2*ngroups*d_state + nheads

### 关键 CUDA/Triton 算子

- `selective_scan_cuda`：C++/CUDA 实现的选择性扫描，支持前向+反向+重计算
- `causal_conv1d`：融合的因果卷积核
- `mamba_chunk_scan_combined`：Mamba 2 的分块扫描（Triton 实现）
- `mamba_split_conv1d_scan_combined`：Mamba 2 的融合卷积+扫描
- `selective_state_update`：Triton 实现的单步状态更新（推理用）

## 与当前研究的关联

### 对序列建模的启示

1. **选择性 > 复杂度**：Mamba 证明了简单的输入依赖参数化就能带来巨大提升，比复杂的架构改动更有效
2. **压缩视角**：序列建模的核心问题是"如何将上下文压缩到更小的状态"——选择性机制是关键
3. **线性复杂度可行性**：首次证明线性时间模型可以在语言建模上匹配 Transformer

### 对 GRU/RNN 的改进方向

Mamba 的选择性 SSM 可以看作是 RNN 门控机制的理论推广：
- GRU 的门控是启发式的 → Mamba 从连续时间系统离散化推导出门控
- GRU 对所有输入同等对待 → Mamba 的 Δ 可选择性过滤/记忆
- GRU 隐藏状态维度小 → Mamba 的结构化状态 (D×N) 更大且高效

### 局限性

1. **无 KV 缓存**：虽然推理吞吐高，但无法像 Transformer 那样灵活地回溯任意位置
2. **信息压缩瓶颈**：有限状态意味着无法完美记住所有历史信息
3. **硬件依赖**：需要定制 CUDA 核才能高效运行，生态不如 Transformer 成熟
4. **后续发展**：Mamba 2 引入了 SSD 理论框架，建立了与注意力机制的对偶关系
