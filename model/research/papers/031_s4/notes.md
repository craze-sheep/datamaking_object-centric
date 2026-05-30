# S4: Structured State Spaces for Sequence Modeling

## 基本信息
- **论文标题**: Efficiently Modeling Long Sequences with Structured State Spaces (S4)
- **前身论文**: Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers (LSSL)
- **作者**: Albert Gu, Karan Goel, Christopher Ré (Stanford University)
- **年份**: 2022 (ICLR 2022)
- **论文链接**: https://arxiv.org/abs/2110.13985
- **代码仓库**: https://github.com/HazyResearch/state-spaces

---

## 核心贡献

1. **统一三大序列建模范式**: S4 统一了 RNN（循环）、CNN（卷积）和 NDE（神经微分方程）三种序列模型，兼具三者优势：
   - **递推模式**（RNN 优势）：推理时 O(1) 内存/时间每步，支持无限长序列
   - **卷积模式**（CNN 优势）：训练时可完全并行化，通过 FFT 加速
   - **连续时间**（NDE 优势）：处理不规则采样、缺失数据、时间尺度自适应

2. **HiPPO 理论解决长程记忆**: 引入结构化矩阵 A 初始化，使状态空间模型具有理论保证的长程记忆能力

3. **Diagonal Plus Low-Rank (DPLR) 参数化**: 将 A 矩阵分解为对角加低秩形式，使计算复杂度从 O(N³) 降至 O(N log N)

4. **SOTA 结果**: 在 Long Range Arena、语音分类、图像分类等任务上达到最优，尤其在超长序列（16K-38K）上大幅超越 Transformer 和 LSTM

---

## 模型架构

### 1. 状态空间模型 (SSM) 基础

S4 的核心是线性连续时间状态空间模型：

```
连续形式:
  ẋ(t) = A·x(t) + B·u(t)     # 状态方程
  y(t) = C·x(t) + D·u(t)     # 输出方程

离散化后（双线性变换）:
  x_t = Ā·x_{t-1} + B̄·u_t    # 离散递推
  y_t = C·x_t + D·u_t         # 输出
```

- **A ∈ R^(N×N)**: 状态转移矩阵，控制系统的演化动力学（最关键参数）
- **B ∈ R^(N×1)**: 输入投影矩阵
- **C ∈ R^(1×N)**: 输出投影矩阵
- **D ∈ R^(1×1)**: 跳跃连接（skip connection）
- **Δt**: 步长/时间尺度参数，控制模型感知的时间范围
- **N**: 状态维度（d_state），通常取 64

### 2. 三重视角

**卷积视角（训练）**:
```
K_L(A, B, C) = (CB, CAB, CA²B, ..., CA^(L-1)B)  ∈ R^L
y = K_L * u + D·u     # 非循环卷积
```
通过 3 次 FFT 即可高效计算：`y = IFFT(FFT(K) · FFT(u))`

**递推视角（推理）**:
```
x_t = Ā·x_{t-1} + B̄·u_t
y_t = C·x_t + D·u_t
```
逐步递推，常数内存，适合自回归生成

**连续时间视角**:
- 处理不规则采样：只需改变 Δt 离散化
- 时间尺度适应：推理时改变 Δt 可适配不同采样率

### 3. HiPPO 初始化

**核心思想**: HiPPO（High-Order Polynomial Projection Operator）提供了一种理论框架，用于设计能最优记忆历史信息的状态矩阵 A。

**数学表述**: 给定输入函数 u(t) 和概率测度 ω(t)，HiPPO 在每个时刻 t 将 u 的历史投影到正交多项式基上，得到系数向量 x(t) ∈ R^N，表示对历史的最优近似。

**关键 HiPPO 矩阵**:

| 名称 | 测度 ω | 多项式基 | 特点 |
|------|--------|----------|------|
| HiPPO-LegS | 缩放勒让德 (Scaled Legendre) | Legendre | 无限时间窗口，最常用 |
| HiPPO-LegT | 平移勒让德 (Translated Legendre) | Legendre | 滑动窗口，等价于 LMU |
| HiPPO-LagT | 平移拉盖尔 (Translated Laguerre) | Laguerre | 指数衰减测度 |

**HiPPO-LegS 矩阵 A 的形式**:
```
A_{nk} = -(2n+1)^{1/2} (2k+1)^{1/2}  if n > k
A_{nk} = -(n+1)                        if n = k
A_{nk} = 0                             if n < k
```

**为什么 HiPPO 关键**: 随机 A 矩阵的 SSM 在长序列上表现很差（如 pMNIST 仅 62%），而 HiPPO 初始化的 SSM 能有效记忆长程信息。

### 4. Diagonal Plus Low-Rank (DPLR) 参数化

**问题**: 直接使用 HiPPO 矩阵 A 存在两个困难：
1. A 是稠密矩阵，矩阵乘法 O(N²)
2. 学习 A 需反复计算 Krylov 函数 K_L(A,B,C)，朴素方法需 O(LN²)

**解决方案**: 将 A 分解为对角矩阵 + 低秩修正：
```
A = diag(Λ) - P·P^*
```
其中 Λ ∈ C^(N/2) 为对角元素（利用共轭对称性减半），P ∈ C^(r×N/2) 为低秩部分（rank r 通常为 1）。

**初始化策略**（代码 `dplr.py`）:
- `legs`/`hippo`: 从 HiPPO-LegS 矩阵的特征分解得到 A 的虚部
- `diag-inv`: 基于 HiPPO 矩阵渐近行为的逆初始化
- `diag-lin`: 线性间距初始化 `imag_part = π·arange(N/2)`
- `fourier`: 傅里叶基初始化
- 组合初始化 `hippo` = `legs` + `fourier`

**DPLR 的关键性质**:
- 准可分矩阵 (quasiseparable)，具有线性 MVM
- Krylov 函数可在 Õ(N+L) 时间内计算（定理 2）
- 同时保证长程记忆（定理 1）和计算效率（定理 2）

### 5. FFT 卷积实现

**核心流程**（代码 `fftconv.py`）:
```python
# 1. 生成卷积核 K ∈ (C, H, L)
k, k_state = self.kernel(L=l_kernel, rate=rate, state=state)

# 2. FFT 卷积
k_f = torch.fft.rfft(k, n=l_kernel+L)      # (C, H, L+L)
x_f = torch.fft.rfft(x, n=l_kernel+L)      # (B, H, L+L)
y_f = einsum('bhl,chl->bchl', x_f, k_f)    # 频域逐元素乘
y = torch.fft.irfft(y_f, n=l_kernel+L)[..., :L]

# 3. D 项跳跃连接
y = y + einsum('bhl,ch->bchl', x, self.D)
```

**卷积核计算**（SSMKernelDPLR）:
- 利用 DPLR 结构，通过 Cauchy 核和 Vandermonde 矩阵快速计算
- 支持 CUDA 加速扩展（`cauchy_cuda`, `log_vandermonde_cuda`）
- 回退方案：PyKeOps 或纯 PyTorch 朴素实现

### 6. 整体架构（S4Block）

```
输入 x → [可选: bottleneck 降维] → [可选: gate 门控]
       → SSM Layer (FFTConv + SSM Kernel)
       → 激活函数 (GELU)
       → [可选: gate 乘法]
       → Dropout
       → 输出线性层 + GLU
       → 残差连接
       → 输出
```

---

## 训练细节

### 超参数
- **d_state (N)**: 通常 64，对速度影响不大
- **dt_min / dt_max**: 0.001 ~ 0.1，控制时间尺度范围
- **dt_transform**: 默认 `exp`（`Δt = exp(param)`），也可用 `softplus`
- **dt_tie**: 默认 True，所有 N 维共享同一 Δt
- **rank**: 低秩修正的秩，通常为 1；HiPPO-LegT 初始化需要增大
- **n_ssm**: 独立训练的 SSM 副本数，None 表示 H 个完全独立
- **init**: 初始化策略，DPLR 模式推荐 `hippo`，Diag 模式推荐 `diag`

### 优化器设置
- **学习率**: 可使用比 baseline 高得多的学习率
- **参数分组**: A、B、Δt 参数可设置不同学习率（通过 `register` 方法的 `_optim` 属性）
- **权重衰减**: 通常设为 0.0，未使用 gradient clipping、weight norm 等
- **Dropout**: 支持 DropoutNd（沿序列长度共享 mask）

### 双模式切换
- **训练**: 卷积模式，并行计算整个序列
- **推理**: 递推模式，通过 `setup_step()` + `step()` 逐步生成
- 状态管理: `default_state()` → `step()` → `forward_state()`

---

## 实验结果

### 逐像素图像分类

| 模型 | sMNIST | pMNIST | sCIFAR |
|------|--------|--------|--------|
| **S4 (LSSL)** | **99.53** | **98.76** | **84.65** |
| S4-fixed | 99.50 | 98.60 | 81.97 |
| HiPPO-RNN | 98.9 | 98.3 | 61.1 |
| CKConv | 99.32 | 98.54 | 63.74 |
| Transformer | 98.9 | 97.9 | 62.2 |

- sCIFAR 上 S4 超过前 SoTA **10+ 个百分点**
- 参数量仅为前 SoTA 的 1/5

### 医疗时序回归（BDIMC, 长度 4000）

| 模型 | RR | HR | SpO2 |
|------|----|----|------|
| **S4 (LSSL)** | **0.350** | **0.432** | **0.141** |
| S4-fixed | 0.378 | 0.561 | 0.221 |
| UnICORNN | 1.06 | 1.39 | 0.869 |
| CKConv | 1.214 | 2.05 | 1.051 |
| Transformer | 2.61 | 12.2 | 3.02 |

- RMSE 降低超过 **2/3**

### 原始语音分类（Speech Commands, 长度 16000）

| 模型 | 原始信号 | 采样率×1/2 | MFCC |
|------|----------|-----------|------|
| **S4 (LSSL)** | **95.87** | **88.66** | 93.58 |
| S4-fixed | 90.64 | 78.01 | 92.55 |
| CKConv | 71.66 | 65.96 | **95.3** |
| UnICORNN | 11.02 | 11.07 | 90.64 |

- 在原始信号上超越使用 MFCC 特征的 baseline
- 表明 S4 可以**自动学习类似 MFCC 的特征**

### 超长序列 CelebA（长度 38000）

| 属性 | S4-fixed | ResNet-18 (10× 参数) |
|------|----------|---------------------|
| Attractive | 78.89 | 81.35 |
| Mouth Slightly Open | 92.36 | 93.92 |
| Smiling | 90.95 | 92.89 |
| Wearing Lipstick | 90.57 | 93.25 |

- 作为通用序列模型，接近专用视觉架构

### 收敛速度

| 数据集 | 指标 | S4-fixed | CKConv | 加速比 |
|--------|------|----------|--------|--------|
| pMNIST | 98% Acc | 16 ep | 118 ep | **7.4×** |
| BDIMC HR | SoTA | 10 ep | 467 ep | **46.7×** |
| Speech | 65% Acc | 9 ep | 188 ep | **20.9×** |

---

## 代码实现细节

### 文件结构
```
src/models/sequence/
├── kernels/
│   ├── kernel.py          # Kernel 基类，定义卷积核接口
│   ├── ssm.py             # SSM 核心：SSMKernel{Dense,Real,Diag,DPLR}
│   ├── dplr.py            # DPLR 初始化策略
│   └── fftconv.py         # FFTConv：FFT 卷积 + SSM 核
├── modules/
│   ├── s4block.py         # S4Block：完整 S4 层（含门控、残差等）
│   └── ...
├── rnns/cells/
│   └── hippo.py           # HiPPO-RNN 变体实现
└── backbones/
    └── model.py           # 主模型骨架
```

### 关键类层次
```
Kernel (基类)
├── ConvKernel        # 自由卷积核 baseline
├── EMAKernel        # Mega 的 MultiHeadEMA
└── SSMKernel        # SSM 核基类
    ├── SSMKernelDense    # 稠密 A，朴素算法（教学用）
    ├── SSMKernelReal     # 实值 HiPPO 矩阵（测试用）
    ├── SSMKernelDiag     # 对角 A（S4D 模型）
    └── SSMKernelDPLR     # 对角+低秩 A（完整 S4 模型）
```

### SSM 参数化变体
- **SSMKernelDense**: 直接存储稠密 A 矩阵，用朴素 Krylov 计算。慢但直观
- **SSMKernelDiag (S4D)**: A 为对角矩阵，计算最简单高效。通过 Vandermonde 矩阵乘法计算核
- **SSMKernelDPLR (S4)**: 完整的对角+低秩参数化。利用 Cauchy 核和特殊结构加速

### 离散化方法
- **双线性变换 (Bilinear)**: `Ā = (I - Δt/2·A)⁻¹(I + Δt/2·A)`，保持稳定性（主要方法）
- **零阶保持 (ZOH)**: `Ā = exp(Δt·A)`，S4D 默认使用
- **广义双线性变换 (GBT)**: 含参数 α 的一般化形式

### CUDA 加速
- 自定义 CUDA 扩展：Cauchy 矩阵乘法、Vandermonde 矩阵乘法
- 安装：`cd extensions/kernels/ && python setup.py install`
- 回退：PyKeOps 或纯 PyTorch 实现（较慢）

### 参数量分析
- 学习 Δt: 仅增加 O(H) 参数
- 学习 A: 仅增加 O(N) 参数
- 总增加 < 1%（相比 O(HN) 的基础参数）
- 但对性能影响巨大（sCIFAR: 81.97% → 84.65%）

---

## 与当前研究的关联

### 1. SSM 替代 GRU 做时序建模

**当前问题**:
- GRU 的长期依赖会逐步衰减，序列越长性能下降越明显
- 梯度消失问题限制了 RNN 在长序列上的表现

**S4 方案**:
- SSM 的卷积模式可并行训练，递推模式可高效推理
- HiPPO 理论保证了长程记忆能力
- 连续时间建模天然支持不规则采样

### 2. S4 的演化脉络
```
HiPPO (2020) → LMU (2021) → LSSL (2021) → S4 (2022) → S4D (2022) → Mamba (2023)
```
- **HiPPO**: 提出连续时间记忆理论
- **LSSL**: 统一 RNN/CNN/NDE，但计算不可行
- **S4**: DPLR 参数化使 LSSL 实际可用
- **S4D**: 简化为纯对角 A，牺牲少量性能换取大幅简化
- **Mamba**: 引入选择性 SSM（输入依赖的 B、C、Δt）

### 3. 关键启示
- **结构化矩阵**是解决长序列问题的关键：不是任意 A 都有效，需要 HiPPO 等理论指导
- **连续-离散桥接**: SSM 通过离散化连接连续时间和离散序列
- **表达力与效率的平衡**: DPLR 在表达力和计算效率之间找到甜蜜点
- **简单架构**: S4 层非常简单（线性变换 + FFT 卷积 + 激活），不依赖复杂的注意力机制
