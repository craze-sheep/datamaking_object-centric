# Lagrangian Neural Networks (LNN)

## 基本信息
- **作者**：Miles Cranmer (Princeton), Sam Greydanus (Oregon State), Stephan Hoyer (Google Research), Peter Battaglia (DeepMind), David Spergel (Flatiron), Shirley Ho (Flatiron)
- **年份**：2020
- **会议**：ICLR 2020 Workshop
- **arXiv**：https://arxiv.org/abs/2003.04630
- **代码**：https://github.com/MilesCranmer/lagrangian_nns
- **框架**：JAX

## 核心贡献

1. **提出 Lagrangian Neural Networks (LNN)**：用神经网络作为黑盒参数化拉格朗日量 L(q, q̇)，再通过欧拉-拉格朗日方程自动推导运动方程，实现能量守恒的动力学建模。

2. **比 HNN 更通用**：Hamiltonian Neural Networks (HNN) 要求输入坐标为正则坐标 (canonical coordinates)，即广义动量 p = ∂L/∂q̇ 必须已知且满足泊松括号关系。LNN 则**不要求正则坐标**，可直接使用任意广义坐标 (如角度、速度)，因此适用范围更广。

3. **比 DeLaN 更通用**：Deep Lagrangian Networks (DeLaN) 假设动能为 T = q̇ᵀMq̇（速度的二次型），仅适用于刚体动力学。LNN 不限制拉格朗日量的函数形式，可建模任意系统（如带电粒子在磁场中、相对论粒子等）。

4. **Lagrangian Graph Network**：将 LNN 扩展到图结构和连续系统，通过拉格朗日密度的求和建模偏微分方程（如 1D 波动方程）。

## 模型架构

### 理论基础：拉格朗日力学

**拉格朗日量**定义为动能减势能：

$$L \equiv T(q, \dot{q}) - V(q)$$

**作用量**（Action）为拉格朗日量在时间上的积分：

$$S = \int_{t_0}^{t_1} L(q_t, \dot{q}_t) \, dt$$

物理系统的真实路径使作用量取驻值（δS = 0），由此导出**欧拉-拉格朗日方程**：

$$\frac{d}{dt} \frac{\partial L}{\partial \dot{q}_j} = \frac{\partial L}{\partial q_j}$$

### LNN 的前向模型

传统方法将 L 写成解析表达式后展开微分方程。LNN 中 L 是神经网络黑盒，无法解析展开。论文推导了数值求解公式：

1. 将欧拉-拉格朗日方程向量化：$\frac{d}{dt} \nabla_{\dot{q}} L = \nabla_q L$
2. 用链式法则展开时间导数，得到包含 q̈ 的项：
   $$(\nabla_{\dot{q}} \nabla_{\dot{q}}^T L) \ddot{q} + (\nabla_q \nabla_{\dot{q}}^T L) \dot{q} = \nabla_q L$$
3. 求解加速度：
   $$\ddot{q} = (\nabla_{\dot{q}} \nabla_{\dot{q}}^T L)^{-1} [\nabla_q L - (\nabla_q \nabla_{\dot{q}}^T L) \dot{q}]$$

其中：
- $\nabla_{\dot{q}} \nabla_{\dot{q}}^T L$ 是 L 对 q̇ 的 Hessian 矩阵（用 `jax.hessian` 计算）
- $\nabla_q L$ 是 L 对 q 的梯度（用 `jax.grad` 计算）
- $\nabla_q \nabla_{\dot{q}}^T L$ 是混合 Jacobian（用 `jax.jacobian` 嵌套计算）
- 使用伪逆（`pinv`）避免 Hessian 奇异

### 网络结构

- **拉格朗日量网络**：MLP，输入 (q, q̇)，输出标量 L
- 激活函数：**Softplus**（因为需要二阶导数，ReLU 的二阶导为零不可用）
- 架构：4 层全连接，500 隐藏单元（论文实验配置）
- 代码中也支持可配置的隐藏层数和宽度

### Lagrangian Graph Network（连续/图系统扩展）

对于图/网格结构的系统（如 1D 波动方程），将总拉格朗日量分解为拉格朗日密度的求和：

$$L = \sum_i L_i, \quad L_i = L_{\text{density}}(\{\phi_j, \dot{\phi}_j\}_{j \in \mathcal{I}_i})$$

其中 $\mathcal{I}_i$ 是节点 i 的邻域。对 1D 网格，$\mathcal{I}_i = \{i-1, i, i+1\}$。Hessian 矩阵稀疏（仅在"邻居的邻居"位置非零），可线性时间求逆。

## 训练细节

- **损失函数**：状态预测损失 $L = \|(\ddot{q}_t^L - \ddot{q}_t^{\text{true}})\|^2$，即预测加速度与真实加速度的 MSE
- **优化器**：Adam，学习率 10⁻³，分段衰减（1/3 和 2/3 训练进度时各降 10 倍）
- **批大小**：32（论文）/ 100（代码示例）
- **网络**：4 层 MLP，500 隐藏单元
- **初始化**：自定义初始化方案（非 Kaiming/Xavier），通过符号回归拟合最优初始化方差：
  - 第一层：σ = 2.2/√n
  - 隐藏层 i：σ = 0.58i/√n
  - 输出层：σ = n/√n = √n
- **ODE 求解器**：`jax.experimental.ode.odeint`（自适应步长），运行在 CPU 上（因控制流主导）
- **坐标预处理**：角度坐标取模 2π，映射到 [-π, π]

## 实验结果

### 实验 1：双摆（Double Pendulum）

| 指标 | LNN | Baseline MLP |
|------|-----|-------------|
| 最终训练损失 | 7.3×10⁻² | 7.4×10⁻² |
| 能量误差（占最大势能比） | **0.4%** | 8% |

- LNN 和基线在短期轨迹预测上表现相似
- **关键差异**：LNN 几乎精确守恒系统总能量，基线模型能量随时间漂移
- 40 个随机初始条件 × 100 时间步的平均结果

### 实验 2：相对论粒子（Relativistic Particle）

- 拉格朗日量：$L = (1-\dot{q}^2)^{-1/2} - 1 + gq$
- 正则动量为 $\dot{q}(1-\dot{q}^2)^{-3/2}$（非简单的 mass × velocity）
- **HNN 在非正则坐标下失败**（因不满足泊松括号关系）
- HNN 在正则坐标下成功
- **LNN 在非正则坐标下成功**，精度与 HNN 在正则坐标下的表现相当

### 实验 3：1D 波动方程（Lagrangian Graph Network）

- 拉格朗日密度需学习有限差分算子：$L_i = \dot{\phi}_i^2 - \frac{(\phi_{i+1}-\phi_{i-1})^2}{2\Delta x}$
- 100 个网格点，周期性边界条件
- LNN 准确建模波动方程，几乎精确守恒系统总能量

## 代码实现细节

### 核心模块 (`lnn/core.py`)

```python
# 拉格朗日方程的运动方程（EOM）
def lagrangian_eom(lagrangian, state, t=None):
    q, q_t = jnp.split(state, 2)
    q = q % (2*jnp.pi)  # 角度取模
    q_tt = (jnp.linalg.pinv(jax.hessian(lagrangian, 1)(q, q_t))  # Hessian 伪逆
            @ (jax.grad(lagrangian, 0)(q, q_t)                    # ∂L/∂q
               - jax.jacobian(jax.jacobian(lagrangian, 1), 0)(q, q_t) @ q_t))  # 混合 Jacobian
    dt = 1e-1
    return dt * jnp.concatenate([q_t, q_tt])
```

- `jax.hessian(lagrangian, 1)`：对第 1 个参数（q̇）求 Hessian
- `jax.grad(lagrangian, 0)`：对第 0 个参数（q）求梯度
- `jax.jacobian(jax.jacobian(lagrangian, 1), 0)`：先对 q̇ 求 Jacobian，再对 q 求 Jacobian
- 还提供了 `lagrangian_eom_rk4` 使用 RK4 积分器

### 模型定义 (`lnn/models.py`)

```python
def mlp(args=None, input_dim=None, hidden_dim=None, output_dim=None, n_hidden_layers=None):
    layers = []
    for i in range(n_hidden_layers):
        layers.append(stax.Dense(hidden_dim))
        layers.append(stax.Softplus)  # 关键：使用 Softplus 激活
    layers.append(stax.Dense(output_dim))  # 输出标量 L
    return stax.serial(*layers)
```

### 自定义初始化 (`lnn/core.py` - `custom_init`)

```python
# 根据层位置使用不同的初始化标准差
std = 1.0/np.sqrt(n)
std *= 2.2*first + 0.58*mid + n*last  # first/mid/last 层不同系数
# bias 全部初始化为 0
```

### 训练流程 (`examples/double_pendulum/train.py`)

- 数据生成：通过解析力学（`physics.py`中的`analytical_fn`）或拉格朗日量数值求解生成轨迹
- 坐标包装：`wrap_coords` 将角度映射到 [-π, π]
- 损失：预测加速度与真实加速度的 MSE
- 通过 `jax.vmap` 批量计算 EOM

### 物理模块 (`examples/double_pendulum/physics.py`)

- 完整的双摆动能/势能/拉格朗日量/哈密顿量的解析实现
- `analytical_fn`：通过解析力方程计算双摆动力学（用于生成训练数据）

## 与当前研究的关联

### 可借鉴的技术点

1. **物理约束作为归纳偏置**：LNN 通过拉格朗日方程硬编码能量守恒，可作为物理正则化手段引入到状态预测任务中
2. **自动微分求高阶导数**：利用 JAX 的 `hessian`/`jacobian` 自动计算运动方程，无需手动推导
3. **Graph Lagrangian Network**：将拉格朗日密度聚合到图结构的方法，可用于粒子系统或网格物理模拟

### 潜在改进方向

- **物理一致性正则化**：在状态预测损失中添加速度-位置导数一致性约束
- **能量守恒监控**：用 LNN 学到的拉格朗日量作为能量守恒的度量
- **非笛卡尔坐标支持**：LNN 天然支持任意广义坐标，适合处理角度、极坐标等非标准表示

### 局限性

- 需要计算 Hessian 逆矩阵，计算复杂度 O(d³)（d 为坐标维度）
- 仅处理保守系统（无耗散力），需扩展才能处理摩擦等非保守力
- 训练数据需要完整的 (q, q̇) 状态，不直接处理观测数据
