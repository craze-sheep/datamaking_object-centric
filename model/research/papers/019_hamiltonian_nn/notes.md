# Hamiltonian Neural Networks (HNN)

## 基本信息
- **作者**: Samuel Greydanus, Misko Dzamba, Jason Yosinski
- **机构**: Google AI Residency
- **年份**: 2019
- **会议**: NeurIPS 2019
- **论文链接**: https://arxiv.org/abs/1906.08387
- **代码**: https://github.com/greydanus/hamiltonian-nn

## 核心贡献
1. **提出 HNN 框架**: 用神经网络学习哈密顿量 H(q,p)，通过哈密顿方程推导运动方程，天然保证能量守恒
2. **无模型约束的学习**: 不需要知道系统解析形式，H 完全由数据驱动学习
3. **统一框架**: 同一架构适用于弹簧振子、摆、N体问题、像素观测等多种系统
4. **Baseline 对比设计**: 同一 MLP 架构，HNN 模式 vs 直接预测 dstate 的 baseline 模式，隔离了"哈密顿归纳偏置"的贡献

## 模型架构

### 1. 哈密顿力学基础
经典哈密顿力学中，系统的状态由广义坐标 q 和广义动量 p 描述，运动由哈密顿量 H(q,p) 决定：

```
dq/dt =  ∂H/∂p    (速度方程)
dp/dt = -∂H/∂q    (力方程)
```

关键性质：
- **能量守恒**: dH/dt = (∂H/∂q)(dq/dt) + (∂H/∂p)(dp/dt) = 0
- **辛结构**: 哈密顿流保持相空间的辛(symplectic)结构
- **可逆性**: 系统在时间反演下对称

### 2. HNN 核心实现 (`hnn.py`)

```python
class HNN(nn.Module):
    def __init__(self, input_dim, differentiable_model, field_type='solenoidal',
                 baseline=False, assume_canonical_coords=True):
        # differentiable_model: 底层 MLP，输出标量 H(q,p) 或 [F1, F2]
        # M: Levi-Civita 置换张量（辛矩阵）
        self.M = self.permutation_tensor(input_dim)
```

**核心前向传播**:
1. MLP 输入 (q, p)，输出两个标量 F1, F2（shape: [batch, 2]）
2. 通过自动微分计算梯度:
   - **保守场 (conservative)**: `dF1/dx @ I` → 对应 ∂H/∂q 方向
   - **螺线场 (solenoidal)**: `dF2/dx @ M^T` → 对应 ∂H/∂p 方向（辛变换）
3. 总向量场 = 保守场 + 螺线场

**Levi-Civita 置换张量** (辛矩阵):
```python
# assume_canonical_coords=True 时:
M = [[0, 1], [-1, 0]]  # 2D 辛矩阵 J
# 推广到 n 维:
M = [I_{n/2}]  →  [[0, I], [-I, 0]]
    [-I_{n/2}]
```

**field_type 参数**:
- `'solenoidal'`: 仅学习螺线场（无耗散的保守系统，如弹簧、摆）
- `'conservative'`: 仅学习保守场（有梯度的场）
- 默认: `'solenoidal'`（大多数物理系统是保守的）

### 3. 时间积分
- **ODE 求解器**: 使用 `scipy.integrate.solve_ivp` 进行长期轨迹积分
- **RK4 (四阶龙格-库塔)**: 代码中实现了 `rk4()` 函数，用于单步推进
  ```python
  def rk4(fun, y0, t, dt):
      k1 = fun(y0, t)
      k2 = fun(y0 + dt/2 * k1, t + dt/2)
      k3 = fun(y0 + dt/2 * k2, t + dt/2)
      k4 = fun(y0 + dt * k3, t + dt)
      return dt/6 * (k1 + 2*k2 + 2*k3 + k4)
  ```
- **训练时**: 直接用 `model.time_derivative(x)` 预测单步 dx/dt
- **推理时**: 用 `solve_ivp` 或 `rk4_time_derivative` 积分多步

### 4. PixelHNN (`hnn.py`)
用于从像素观测中学习哈密顿动力学:
```
像素 x → [Encoder] → 潜在 z=(q,p) → [HNN] → dz/dt → z_next → [Decoder] → 像素 x_next
```

**三重损失**:
```python
loss = ae_loss + cc_loss + 0.1 * hnn_loss
# ae_loss: 自编码器重建损失 ||x - decode(encode(x))||²
# cc_loss: 规范坐标损失 ||dw - (w_next - w)||²，强制 z=(w,dw) 像 (位置,速度)
# hnn_loss: HNN 向量场损失 ||z_next - (z + dz/dt)||²
```

## 训练细节

### 优化器与超参数
| 参数 | 弹簧/摆 | 三体 | 像素 |
|------|---------|------|------|
| optimizer | Adam | Adam | Adam |
| lr | 1e-3 | 1e-3 | 1e-3 |
| weight_decay | 1e-4 | 1e-4 | 1e-5 |
| hidden_dim | 200 | 200 | 200 |
| total_steps | 2000 | 10000 | 10000 |
| batch_size | 全量 | 600 | 200 |
| nonlinearity | tanh | tanh | tanh |
| 权重初始化 | orthogonal | orthogonal | orthogonal |

### 损失函数
- **状态空间实验**: `L2_loss = (dxdt - dxdt_hat)².mean()`
  - 训练目标是匹配 dcoords（状态导数），不是直接匹配下一状态
  - 这是关键设计：让网络学习向量场而非轨迹
- **像素实验**: 见上述三重损失

### 数据生成
- **弹簧振子**: H = p² + q²，50条轨迹，t∈[0,3]，noise_std=0.1
- **摆**: H = 3(1-cos(q)) + p²，50条轨迹，t∈[0,3]，noise_std=0.1
- **三体问题**: 3个粒子，input_dim=12（3×2维位置+动量），使用Lipson数据集
- **像素摆**: 28×28 灰度图像，input_dim=2×28²=1568

## 实验结果

### 定量结果（论文报告）

| 实验 | 指标 | HNN | Baseline MLP | 说明 |
|------|------|-----|-------------|------|
| 弹簧振子 | dstate MSE | **显著更低** | 较高 | HNN 能量守恒，baseline 能量漂移 |
| 摆 | dstate MSE | **显著更低** | 较高 | 非线性系统，HNN 优势更明显 |
| 三体问题 | dstate MSE | **显著更低** | 较高 | 高维混沌系统 |
| 像素摆 | 下一帧 MSE | **更低** | 较高 | 潜在空间 HNN |

### 关键发现
1. **能量守恒**: HNN 学到的总能量在长时间积分中保持恒定，baseline 的能量发散
2. **相空间结构**: HNN 学到的相空间向量场与真实系统高度吻合
3. **长期稳定性**: HNN 积分长轨迹时保持物理合理性，baseline 轨迹发散
4. **泛化性**: 在训练分布外的初始条件上，HNN 仍然表现良好

## 代码实现细节

### 代码结构
```
code/
├── hnn.py              # HNN 和 PixelHNN 核心类
├── nn_models.py        # MLP 和 MLPAutoencoder
├── utils.py            # rk4, L2_loss, integrate_model, 辅助函数
├── experiment-spring/  # 弹簧振子实验
├── experiment-pend/    # 摆实验
├── experiment-2body/   # 二体问题
├── experiment-3body/   # 三体问题
├── experiment-pixels/  # 像素摆实验
├── experiment-real/    # 真实数据(Lipson)
├── analyze-*.ipynb     # 分析notebooks
└── static/             # 图片和GIF
```

### 关键设计决策
1. **orthogonal 初始化**: 所有线性层使用正交初始化，避免梯度消失/爆炸
2. **无 bias 的输出层**: `Linear(hidden, output, bias=None)`，保证输出的零点对应零向量场
3. **tanh 激活函数**: 平滑可微，适合自动微分计算二阶导数
4. **Baseline 模式**: `--baseline` 标志让同一个 MLP 直接输出 (dq/dt, dp/dt)，跳过哈密顿结构，作为消融实验

### HNN vs Baseline 的代码差异
```python
# HNN 模式: MLP 输出标量 H，通过自动微分得到向量场
y = self.differentiable_model(x)  # [batch, 2] → F1, F2
dF = autograd.grad(F.sum(), x)    # 自动微分
field = dF @ M.t()                 # 辛变换

# Baseline 模式: MLP 直接输出向量场
return self.differentiable_model(x)  # 直接输出 [batch, 2]
```

## 与当前研究的关联

### 对物理模拟的启发
1. **能量守恒归纳偏置**: HNN 证明了将物理守恒律编码到网络结构中，比事后加正则化损失更有效
2. **向量场学习**: 学习 dx/dt 而非 x_{t+1}，更符合物理规律，泛化更好
3. **辛积分**: 使用辛积分器（如 leapfrog/symplectic Euler）保持几何结构

### 对 Slot-based 模型的借鉴
1. **每个 slot 可学习独立的哈密顿量**: 不同物体有不同能量函数
2. **交互哈密顿量**: H_total = Σ H_self(i) + Σ H_interact(i,j)，自然建模物体间交互
3. **潜在空间哈密顿动力学**: 类似 PixelHNN，在潜在空间中用 HNN 推进状态
4. **能量守恒作为训练信号**: 可用于检测和纠正物理不合理的预测

### 局限性
1. **仅适用于保守系统**: 耗散系统（摩擦、阻尼）需要扩展到 GENERIC 框架
2. **需要已知坐标**: 代码假设输入已经是 (q, p)，实际应用需要学习坐标变换
3. **低维系统**: 原始实验限于 2-12 维，高维系统（如刚体、流体）需要架构扩展
4. **无外力**: 标准 HNN 不处理时变外力
