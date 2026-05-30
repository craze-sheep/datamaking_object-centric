# FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness

## 基本信息

- **作者**: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré
- **机构**: Stanford University, University at Buffalo SUNY
- **年份**: 2022
- **会议**: NeurIPS 2022
- **论文链接**: https://arxiv.org/abs/2205.14135
- **代码仓库**: https://github.com/HazyResearch/flash-attention (FlashAttention-1)
- **最新代码**: FlashAttention-4 (FA4)，使用 CuTeDSL (NVIDIA CUTLASS DSL) 用 Python 编写，支持 Hopper (SM90) 和 Blackwell (SM100/SM110) GPU

## 核心贡献

1. **IO 感知的精确注意力算法**: 首次提出将注意力算法的设计目标从减少 FLOPs 转向减少 HBM（高带宽内存）访问次数，实现精确注意力（非近似）
2. **Tiling + Recomputation 两大技术**: 通过分块计算（tiling）避免在 HBM 中物化 N×N 的注意力矩阵；通过反向传播时重计算（recomputation）避免存储中间结果
3. **显著的加速和内存节省**: 训练速度提升 2-4 倍，注意力计算显存从 O(N²) 降至 O(N)
4. **块稀疏扩展**: 提出 Block-Sparse FlashAttention，IO 复杂度按稀疏比例进一步降低
5. **成为行业标准**: 被 PyTorch、HuggingFace、Megatron-LM、DeepSpeed 等主流框架集成，MLPerf 训练基准中 BERT 训练最快实现

## 模型架构与算法原理

### 问题背景：为什么标准注意力慢

标准注意力计算：
```
S = QK^T ∈ R^{N×N}    （计算并存储到 HBM）
P = softmax(S) ∈ R^{N×N} （计算并存储到 HBM）
O = PV ∈ R^{N×d}
```

- 标准实现需要将 S 和 P 矩阵写入 HBM，消耗 O(N²) 内存
- 在 A100 GPU 上，HBM 带宽 1.5-2.0 TB/s，而片上 SRAM 带宽约 19 TB/s（快 10 倍以上）
- 大多数 Transformer 操作是 **内存受限**（memory-bound）而非计算受限（compute-bound）
- 标准注意力的 HBM 访问量为 Θ(Nd + N²)，对长序列来说 N² 项占主导

### 核心技术 1: Tiling（分块计算）

**关键思想**: 将 Q, K, V 分成小块，每次只加载一小块到 SRAM 中计算，避免物化完整的 N×N 矩阵。

**Online Softmax 技术**: softmax 需要全局归一化常数，但可以通过分块递增计算：

对于向量 x = [x⁽¹⁾, x⁽²⁾]，有：
- `m(x) = max(m(x⁽¹⁾), m(x⁽²⁾))` —— 全局最大值
- `ℓ(x) = e^{m(x⁽¹⁾)-m(x)} ℓ(x⁽¹⁾) + e^{m(x⁽²⁾)-m(x)} ℓ(x⁽²⁾)` —— 全局归一化常数
- `softmax(x) = f(x) / ℓ(x)`

因此只需跟踪额外的统计量 (m, ℓ)，即可分块计算 softmax，无需访问完整输入。

**算法流程** (Algorithm 1):
1. 将 Q 分为 T_r 个块，K/V 分为 T_c 个块
2. 外层循环遍历 K/V 块（加载到 SRAM）
3. 内层循环遍历 Q 块（加载到 SRAM）
4. 在 SRAM 中计算 S_ij = Q_i K_j^T
5. 计算局部 softmax 统计量 m̃, ℓ̃
6. 更新全局统计量 m_new, ℓ_new
7. 增量更新输出 O_i（无需存储完整 P 矩阵）

### 核心技术 2: Recomputation（重计算）

**问题**: 反向传播需要 S 和 P 矩阵来计算梯度，但存储它们需要 O(N²) 内存。

**解决方案**: 只存储输出 O 和 softmax 统计量 (m, ℓ)，反向传播时从 Q, K, V 在 SRAM 中重计算 S 和 P。

- 这是一种 **选择性梯度检查点**（selective gradient checkpointing）
- 虽然增加了 FLOPs（重计算），但由于大幅减少 HBM 访问，实际速度反而更快
- 反向传播的梯度公式经过解析简化，只需要存储 O(N) 额外内存

**反向传播关键推导**:
- `dV = P^T dO`
- `D_i = dO_i^T · O_i`（用 d 维向量点积代替 N 维 softmax 归约）
- `dS_ij = P_ij (dP_ij - D_i)`
- `dQ_i = Σ_j P_ij (dO_i^T v_j - D_i) k_j`
- `dK_j = Σ_i P_ij (dO_i^T v_j - D_i) q_i`

### 核心技术 3: IO-Awareness（IO 感知）

**IO 复杂度分析**:

| 方法 | HBM 访问量 | FLOPs |
|------|-----------|-------|
| 标准注意力 | Θ(Nd + N²) | O(N²d) |
| FlashAttention | Θ(N²d²M⁻¹) | O(N²d) |
| 块稀疏 FlashAttention | Θ(Nd + N²d²M⁻¹·s) | O(N²d·s) |

其中 M 是 SRAM 大小，d 是头维度，s 是非零块比例。

- 对典型值 d=64-128, M≈100KB，FlashAttention 的 HBM 访问量比标准实现少 **最多 9 倍**
- **下界证明**: 不存在对所有 SRAM 大小 M 都能渐进改进 HBM 访问量的精确注意力算法
- HBM 访问量是决定运行时间的主要因素，即使 FLOPs 增加（因重计算），总时间仍更快

### Kernel Fusion（内核融合）

所有注意力操作（矩阵乘法、softmax、mask、dropout、矩阵乘法）融合到 **单个 CUDA kernel** 中执行：
- 避免反复从 HBM 读写输入和输出
- 消除了中间结果的 HBM 存储开销

### 块稀疏扩展 (Block-Sparse FlashAttention)

- 给定块稀疏掩码 M̃，只计算非零块的注意力
- IO 复杂度按稀疏比例 s 缩放
- 使用 butterfly 稀疏模式，可近似任意稀疏结构
- 在 LRA 基准上达到 2.8× 加速，精度与标准注意力相当
- 首次在 Path-X (16K 序列, 61.4%) 和 Path-256 (64K 序列, 63.1%) 上实现超越随机的 Transformer 性能

## 训练细节

- **BERT-large**: 在 Wikipedia 上训练，8×A100 GPU，达到 MLPerf 1.1 目标准确率 72.0%
- **GPT-2**: 在 OpenWebText 数据集上训练，8×A100 GPU
- **长程竞技场 (LRA)**: 序列长度 1024-4096，遵循 Tay et al. 的实验设置
- **长文档分类**: 在 MIMIC-III 和 ECtHR 数据集上微调预训练的 RoBERTa 模型（重复位置编码以支持更长序列）
- **Path-X/Path-256**: 在 Path-64 上预训练，然后通过空间插值位置编码迁移到更长序列

## 实验结果

### 训练加速

| 模型 | 实现 | 训练时间 | 加速比 |
|------|------|---------|--------|
| BERT-large | Nvidia MLPerf 1.1 | 20.0 ± 1.5 min | 1.0× |
| BERT-large | FlashAttention | 17.4 ± 1.4 min | **1.15×** |
| GPT-2 small | HuggingFace | 9.5 days | 1.0× |
| GPT-2 small | Megatron-LM | 4.7 days | 2.0× |
| GPT-2 small | FlashAttention | 2.7 days | **3.5×** |
| GPT-2 medium | HuggingFace | 21.0 days | 1.0× |
| GPT-2 medium | Megatron-LM | 11.5 days | 1.8× |
| GPT-2 medium | FlashAttention | 6.9 days | **3.0×** |

### 模型质量提升

| 任务 | 效果 |
|------|------|
| GPT-2 语言建模 | 上下文从 1K→4K，ppl 改善 0.7，且仍比 Megatron 1K 快 30% |
| MIMIC-III 长文档分类 | 序列长度 512→16K，micro-F1 提升 4.3 点 |
| ECtHR 长文档分类 | 序列长度 512→8K，micro-F1 提升 8.5 点 |
| Path-X (16K) | FlashAttention: 61.4%（首个超越随机的 Transformer） |
| Path-256 (64K) | 块稀疏 FlashAttention: 63.1% |

### 内存与运行时基准

- **注意力计算加速**: 在序列长度 128-2K 范围内，比 PyTorch 标准实现快 **最多 3×**
- **内存节省**: FlashAttention 比精确注意力基线节省 **最多 20×** 显存
- **内存线性增长**: 显存随序列长度线性增长（标准实现为二次增长）
- **GPT-2 注意力计算**: 注意力部分加速 **7.6×**（从 15ms 降至 ~2ms）
- **块稀疏优势**: 块稀疏 FlashAttention 在所有序列长度上都快于所有已知的近似/稀疏注意力方法

### LRA 基准对比

| 方法 | 平均准确率 | 加速比 |
|------|-----------|--------|
| Transformer (标准) | 59.3 | — |
| FlashAttention | 59.8 | 2.4× |
| Block-Sparse FlashAttention | 59.6 | 2.8× |
| Linformer | 54.9 | 2.5× |
| Linear Attention | 59.6 | 2.3× |
| Performer | 58.9 | 1.8× |
| Reformer | 57.6 | 1.3× |

## 代码实现细节

### 仓库结构（FlashAttention-4）

代码库位于 `code/flash_attn/cute/`，使用 CuTeDSL（NVIDIA CUTLASS 的 Python DSL）编写，运行时编译为 PTX/CUBIN。

**核心文件**:

| 文件 | 功能 |
|------|------|
| `interface.py` | 公共 API 入口：`flash_attn_func()` 和 `flash_attn_varlen_func()` |
| `flash_fwd.py` | SM80 (Ampere) 前向 kernel，基线实现 |
| `flash_fwd_sm90.py` | SM90 (Hopper) 前向 kernel |
| `flash_fwd_sm100.py` | SM100 (Blackwell) 前向 kernel，支持 SplitKV、paged KV cache、持久化 kernel、2CTA 指令 |
| `flash_bwd.py` | SM80 反向 kernel |
| `flash_bwd_sm90.py` | SM90 反向 kernel |
| `flash_bwd_sm100.py` | SM100 反向 kernel，支持 2CTA 和块稀疏 |
| `softmax.py` | Online softmax 实现，跟踪 row_max/row_sum，支持 score modifier |
| `mask.py` | 注意力掩码：causal、local/sliding window、块稀疏、自定义 mask_mod |
| `pipeline.py` | 循环缓冲区的流水线状态管理（index/phase） |
| `tile_scheduler.py` | Tile 调度策略（单 tile、varlen 感知、持久化调度） |
| `pack_gqa.py` | GQA 优化：将多个 Q head 打包到同一 KV head |
| `paged_kv.py` | Paged KV cache 管理器（支持 TMA） |
| `block_info.py` | Tile 维度计算，causal/local masking 的块范围计算 |
| `seqlen_info.py` | 变长序列的长度和偏移量跟踪 |

**Tensor 布局**: `(batch, seqlen, num_heads, head_dim)`，最后一维连续，16 字节对齐。

### Online Softmax 实现 (`softmax.py`)

```python
class Softmax:
    scale_log2: Float32      # log2(e) * softmax_scale
    row_max: cute.Tensor     # 每行最大值（寄存器中）
    row_sum: cute.Tensor     # 每行求和（寄存器中）

    def online_softmax(self, acc_S, is_first=False):
        # 1. 计算当前块的 row_max
        row_max_cur = fmax_reduce(acc_S_row, init_val=row_max[r])
        # 2. 更新全局 row_max
        row_max[r] = row_max_cur
        # 3. 计算 exp(S * scale - max)，更新 row_sum
        acc_S_row_exp = exp2(acc_S_row * scale_log2 - row_max_cur_scaled)
        # 4. 返回 row_scale 用于重缩放之前的 O
        row_scale[r] = exp2((row_max_prev - row_max_cur) * scale_log2)
        return row_scale
```

### 前向 Kernel 核心流程 (`flash_fwd.py`)

```
1. 加载 Q tile → 寄存器
2. 循环遍历 K/V 块（流水线加载）:
   a. 加载 K 块到 SRAM
   b. 计算 S = Q @ K^T（Tensor Core GEMM）
   c. 应用 mask（causal/local/custom）
   d. Online softmax: 计算 P̃ = exp(S - max)，更新 row_max, row_sum
   e. 加载 V 块到 SRAM
   f. 计算 O += P̃ @ V（Tensor Core GEMM）
3. 写回 O 和 LSE（log-sum-exp）到 HBM
```

### 编译与缓存

- **JIT 编译**: Kernel 运行时编译，缓存键包括 dtype、head_dim、causal、mask/score_mod 哈希、架构、块大小
- **缓存层级**: 内存 LRU + 可选磁盘缓存
- **测试两阶段流程**:
  1. 编译阶段：`FLASH_ATTENTION_FAKE_TENSOR=1` 并行编译所有 kernel（不需要 GPU 内存）
  2. 执行阶段：使用缓存的编译 kernel 运行测试

### 架构特定优化

- **SM90 (Hopper)**: Warp-group GEMM、TMA (Tensor Memory Accelerator)、named barriers
- **SM100 (Blackwell)**: UMMA-based GEMM、2CTA 指令（两个 CTA 协作）、持久化 kernel
- **Tile 大小选择**: 根据 head_dim 和架构自动选择最优 tile_m/tile_n（如 SM90 上 head_dim≤64 使用 192×128）

### 关键设计模式

- **编译时常量**: 使用 `cutlass.Constexpr[type]` 实现 kernel 特化
- **Score/Mask Modifier**: 用户定义的 `@cute.jit` 可调用对象，在编译时注入 kernel
- **Pipeline 管理**: `PipelineStateSimple` 使用单个 Int32 存储 index 和 phase，通过 divmod 分离
- **共享内存布局**: Q tile 和 K/V tile 分配，精确计算 smem_usage 不超过容量

## 与当前研究的关联

### 后续工作演进

| 版本 | 架构 | 关键特性 |
|------|------|---------|
| FlashAttention-1 (本文) | Ampere (SM80) | Tiling + recomputation, CUDA C++ |
| FlashAttention-2 | SM80/SM90 | 更好的并行化和工作分区 |
| FlashAttention-3 | Hopper (SM90) | 异步执行、warp 专门化、FP8 支持 |
| FlashAttention-4 | SM90/SM100/SM110 | Python CuTeDSL、Blackwell 支持、2CTA、持久化 kernel |

### 广泛的行业影响

- **框架集成**: PyTorch nn.Transformer、HuggingFace Transformers、Megatron-LM、DeepSpeed、PaddlePaddle
- **MLPerf**: 2022 年 MLPerf 训练中云端 BERT 训练最快方案
- **扩散模型**: HuggingFace diffusers、Stable Diffusion 推理加速 3-4×
- **科学计算**: Uni-Fold (蛋白质模型) 比 AlphaFold 快 2.6×、OpenFold 推理快 3×
- **替代实现**: Triton (OpenAI)、xformers (Meta)、Jax、Metal (Apple Silicon)

### 核心思想的普适性

- **IO 感知**: 不仅适用于注意力，可扩展到深度学习中所有内存受限操作
- **Tiling + Recomputation**: 通用的"以计算换内存"策略，适用于任何需要中间结果的计算图
- **Kernel Fusion**: 将多个操作合并为单个 kernel 以减少 HBM 访问，已成为高性能 DL 的标准实践
- **硬件协同设计**: 算法设计需要考虑具体硬件的内存层次（HBM vs SRAM vs 寄存器）

### 当前应用价值

- 任何使用 Transformer 的模型都可以直接使用 FlashAttention 获得 2-4× 训练加速
- 对于长序列任务（文档理解、基因组学、视频处理），FlashAttention 的线性内存增长特性尤为关键
- FlashAttention 使得在相同硬件上训练更大模型或使用更长上下文成为可能
