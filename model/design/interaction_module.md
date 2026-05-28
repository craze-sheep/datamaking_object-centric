# T6 交互建模模块设计

> 任务：T6 交互建模模块
> 上游：T5 输入处理模块
> 下游：T7 时序预测模块、T10 `model/models/interaction.py`
> 第一版目标：用显式物体图建模物体间相互作用，输出 interaction-aware object tokens `[B,Th,N,D]`

---

## 1. 设计结论

第一版采用 dense pairwise message passing 的 Interaction Network 风格模块。由于 `N<=7`，不引入 PyG batch graph，直接在 dense `[B,Th,N,N,*]` 张量上计算边特征和消息，简单、可测、显存可控。

关键维度：`De = 16`。这是 T6 明细设计对 T4 初始估算 `De=14` 的更新：新增 `dist_ij` 与 `force_norm_ij` 两个标量 edge feature。

核心公式：

```text
edge_ij = [F_ij, F_ji, rel_pos_ij, rel_vel_ij, static_i, static_j, dist_ij, force_norm_ij]
m_ij = EdgeMLP([z_i, z_j, edge_ij])
a_j = masked_sum_i(m_ij) / degree_j
z'_j = NodeMLP([z_j, a_j]) + residual(z_j)
```

输出：

```python
inter_token: [B,Th,N,D]
interaction_aux: {
    'edge_feat': [B,Th,N,N,De],
    'edge_mask': [B,Th,N,N],
    'messages': [B,Th,N,N,D],
    'agg_message': [B,Th,N,D],
}
```

---

## 2. 输入输出接口

### 2.1 输入

来自 T5：

```python
interaction_module(
    object_token: Tensor,  # [B,Th,N,D]
    state_hist: Tensor,    # [B,Th,N,16], normalized
    force_hist: Tensor,    # [B,Th,N,N,3], normalized
    valid_mask: Tensor,    # [B,N] bool
    static_flag: Tensor,   # [B,N] bool, already valid_mask-filtered
) -> dict
```

### 2.2 输出

```python
{
    'inter_token': Tensor,   # [B,Th,N,D]
    'edge_feat': Tensor,     # [B,Th,N,N,De]
    'edge_mask': Tensor,     # [B,Th,N,N]
    'messages': Tensor,      # [B,Th,N,N,D]，默认最后一层
    'agg_message': Tensor,   # [B,Th,N,D]，默认最后一层
    'valid_mask': Tensor,    # [B,N] bool，透传给T7
    'static_flag': Tensor,   # [B,N] bool，透传给T8/T9需要时使用
}
```

`inter_token` 是传给 T7 的主输出。其余字段用于调试、可视化和单元测试。

---

## 3. 图结构定义

### 3.1 节点

每个有效物体 slot 是一个节点：

```python
z = object_token  # [B,Th,N,D]
```

padding 节点：`valid_mask=False`，不发送也不接收消息，输出 token 保持0。

### 3.2 边

第一版采用 dense directed graph：每个有效 pair `(i,j)` 都建边，但排除自环。

```python
valid_i = valid_mask[:, None, :, None]  # [B,1,N,1]
valid_j = valid_mask[:, None, None, :]  # [B,1,1,N]
not_self = ~torch.eye(N, dtype=torch.bool, device=valid_mask.device)[None, None, :, :]
edge_mask = valid_i & valid_j & not_self  # [B,1,N,N]
edge_mask = edge_mask.expand(B, Th, N, N)
degree = edge_mask.float().sum(dim=2)     # [B,Th,N], sum over sender i for each receiver j
```

方向约定：`i -> j` 表示 i 给 j 发送 message，聚合时对 i 求和更新 j。

### 3.3 为什么 dense 而不是稀疏

- `N<=7`，每帧最多 `N*(N-1)=42` 条有向边。
- `Th=12`，每个样本最多504条边，dense计算很便宜。
- dense张量更容易写单元测试，避免 PyG batch graph 的复杂度。
- 稀疏近邻/非零力图留给后续优化。

---

## 4. 边特征设计

### 4.1 输入 state 切片

T6 使用 T5 透传的 normalized `state_hist`。T10 不应在多个文件散落魔法数字，必须从 `config.py` 或集中常量读取：

```python
STATE_POS_SLICE = slice(0, 3)
STATE_VEL_SLICE = slice(7, 10)
# ATTR_STATIC_INDEX=8 由T5处理；T6只接收 static_flag
pos = state_hist[..., STATE_POS_SLICE]
vel = state_hist[..., STATE_VEL_SLICE]
```

T6 不反归一化。相对状态在 normalized space 中计算，便于尺度稳定。若后续发现固定归一化不准确，由 T3/T10 配置传入统计值统一处理。

### 4.2 force_matrix 方向不确定的处理

数据集 `force_matrix[i][j]` 方向语义尚未完全确认。第一版边特征同时包含 `F_ij` 和 `F_ji`：

```python
F_ij = force_hist[:, :, i, j, :]  # [B,Th,3]
F_ji = force_hist[:, :, j, i, :]  # [B,Th,3]
```

`force_norm_ij` 只是 normalized force 的关系特征，帮助 message passing 表达当前接触强度；碰撞标签阈值不在 T6 中定义，也不能直接用该 normalized 标量，应由 T9 使用原始或反归一化 force 构造 label。

这样即使原始矩阵方向解释反了，EdgeMLP 仍能学习有效关系。

T10 实现前建议抽样检查 2-3 个有接触样本，记录 `force_matrix` 是否近似反对称，但不阻塞第一版。

### 4.3 edge feature 组成

```python
rel_pos_ij = pos_j - pos_i      # [B,Th,N,N,3]
rel_vel_ij = vel_j - vel_i      # [B,Th,N,N,3]
dist_ij = norm(rel_pos_ij)      # [B,Th,N,N,1]
force_norm_ij = norm(F_ij)      # [B,Th,N,N,1]，仅作为edge feature，不用于collision label阈值
static_i = static_flag_i        # [B,Th,N,N,1]
static_j = static_flag_j        # [B,Th,N,N,1]
```

最终：

```python
edge_feat = concat([
    F_ij, F_ji,          # 6
    rel_pos_ij,          # 3
    rel_vel_ij,          # 3
    static_i, static_j,  # 2
    dist_ij,             # 1
    force_norm_ij,       # 1
], dim=-1)
# De = 16
edge_feat = edge_feat * edge_mask[..., None]  # aux输出中padding/self edge显式置零
```

注意：T4 曾初始估算 `De=14`，T6 明细设计决定加入 `dist` 和 `force_norm` 后 `De=16`。已同步要求：T10 `config.py` 必须设置 `EDGE_DIM = 16`，单元测试必须断言 `config.edge_dim == edge_feat.shape[-1] == 16`。

---

## 5. Message Passing 层

### 5.1 单层结构

```python
z_i = z[:, :, :, None, :].expand(B,Th,N,N,D)
z_j = z[:, :, None, :, :].expand(B,Th,N,N,D)
edge_input = concat([z_i, z_j, edge_feat], dim=-1)  # [B,Th,N,N,2D+De]
message = EdgeMLP(edge_input)                       # [B,Th,N,N,D]
message = message * edge_mask[..., None]
agg = message.sum(dim=2) / degree.clamp_min(1)      # aggregate senders i -> receiver j
node_input = concat([z, agg], dim=-1)               # [B,Th,N,2D]
delta = NodeMLP(node_input)                         # [B,Th,N,D]
z_next = LayerNorm(z + dropout(delta))
z_next = z_next * valid_mask[:,None,:,None]
```

聚合维度说明：

- `message[:, :, i, j]` 表示 i -> j。
- 更新接收者 j 时，对 sender i 求和，所以 `sum(dim=2)`。

### 5.2 多层堆叠

默认：

```yaml
num_interaction_layers: 2
edge_hidden_dim: 256
node_hidden_dim: 256
dropout: 0.0 initially
aggregation: mean  # 第一版固定mean；配置项仅记录，不实现sum/max分支
exclude_self_edges: true
```

每层重复构造 message。edge_feat 可每层复用，不必重复计算。

### 5.3 EdgeMLP / NodeMLP

```python
EdgeMLP:
  Linear(2D + De, hidden_dim)
  SiLU
  LayerNorm(hidden_dim)
  Linear(hidden_dim, D)

NodeMLP:
  Linear(2D, hidden_dim)
  SiLU
  LayerNorm(hidden_dim)
  Linear(hidden_dim, D)
```

不使用 BatchNorm，保持小 batch 稳定。

---

## 6. 静态物体策略

静态物体包括 ground/wall 等。第一版策略：

| 行为 | 静态物体 |
|------|----------|
| 发送 message | 是 |
| 接收 message | 是，但其 token 更新只用于视觉/上下文，不用于运动预测主loss |
| state 预测 | T8中 copy last observed state，不由 state head 自由预测 |
| 状态 loss | T9主loss不计算静态物体 |
| collision pair | 保留动态-静态 pair，因为 ground/wall 接触重要 |

T6 不冻结静态 token，也不区分 dynamic/static 来决定是否更新 token：所有 valid object token 都可以在 message passing 中更新。静态物体的运动状态预测和 state loss 在 T8/T9 中通过 copy-last-state 与 `dynamic_mask` 处理。若其他文档出现“只更新动态 token”的旧表述，以本 T6 最终策略为准。

---

## 7. 远距离物体与零力边

第一版保留所有有效非自环边，包括 `force_norm=0` 的边。

理由：

- 零力边也表达“当前无接触/远距离”的信息。
- rel_pos/dist 可帮助模型学习未来可能接触。
- N 小，保留 dense 边没有性能压力。

后续优化：

- 稀疏边：`force_norm > threshold` 或 `dist < radius`。
- attention bias：用 `force_norm` 或 `-dist` 作为 attention/message 权重。
- 但这些都不是第一版必需。

---

## 8. 视觉-物理特征融合

T6 接收的 `object_token` 已由 T5 融合视觉和物理状态。T6 内部不再重新分开 visual/physical，而是在 message passing 中通过 edge feature 注入关系物理量。

避免的问题：

- 不在 T6 直接 concat 原始 RGB/mask，避免模块职责混乱。
- 不在 T6 再编码 attr/state，避免重复参数和接口不一致。
- T6 只负责“节点之间如何交互”。

---

## 9. Mask 约定

### 9.1 edge mask

```python
edge_mask: [B,Th,N,N]
edge_mask[b,t,i,j] = valid_i & valid_j & (i != j)
```

### 9.2 node mask

```python
node_mask = valid_mask[:, None, :, None]  # [B,1,N,1]
inter_token = inter_token * node_mask
```

### 9.3 static / dynamic mask

T6 只使用 `static_flag` 作为 edge feature，不使用 `dynamic_mask` 屏蔽消息。原因：静态物体仍然参与交互。

---

## 10. 输出给 T7

T7 需要：

```python
inter_token: [B,Th,N,D]
valid_mask: [B,N]
```

因此 T6 输出 dict 必须透传 `valid_mask`，避免整体模型集成时漏传。T6 输出不改变物体数量和时间长度。

---

## 11. T10 实现建议

文件：`model/models/interaction.py`

建议类：

```python
@dataclass
class InteractionConfig:
    token_dim: int = 256
    state_dim: int = 16
    force_dim: int = 3
    edge_dim: int = 16
    num_layers: int = 2
    hidden_dim: int = 256
    dropout: float = 0.0
    aggregation: str = 'mean'  # reserved; v1 only implements mean
    exclude_self_edges: bool = True
```

```python
class EdgeFeatureBuilder(nn.Module): ...
class MessagePassingLayer(nn.Module): ...
class InteractionModule(nn.Module): ...
```

组合关系：

```text
InteractionModule
  ├─ edge_builder: EdgeFeatureBuilder
  └─ layers: ModuleList[MessagePassingLayer x num_layers]
```

`EdgeFeatureBuilder.forward(state_hist, force_hist, valid_mask, static_flag)` 返回：

```python
edge_feat: [B,Th,N,N,16]
edge_mask: [B,Th,N,N]
```

`InteractionModule.forward(object_token, state_hist, force_hist, valid_mask, static_flag)` 返回 dict。实现应在入口做轻量断言/规整：

```python
assert object_token.shape[:3] == state_hist.shape[:3]
assert force_hist.shape[:4] == (*object_token.shape[:2], object_token.shape[2], object_token.shape[2])
valid_mask = valid_mask.bool().to(object_token.device)
static_flag = (static_flag.bool().to(object_token.device) & valid_mask)
```

aux 输出 `messages` / `agg_message` 默认表示最后一层；第一版不实现 all-layer debug。

---

## 12. 单元测试要求（T10）

测试文件：`model/tests/test_interaction.py`

必须覆盖：

1. `test_edge_feature_builder_shapes`
   - mock `state_hist [B,Th,N,16]`、`force_hist [B,Th,N,N,3]`
   - 验证 `edge_feat [B,Th,N,N,16]`、`edge_mask [B,Th,N,N]`

2. `test_edge_mask_excludes_padding_and_self_edges`
   - valid_mask 设置部分 False
   - 验证 padding pair 和 self edge 为 False

3. `test_edge_features_include_bidirectional_force`
   - 构造 `force[i,j] != force[j,i]`
   - 验证 edge_feat 中同时包含 F_ij 和 F_ji

4. `test_relative_position_and_velocity_computation`
   - 构造简单 pos/vel
   - 验证 `rel_pos = pos_j - pos_i`，`rel_vel = vel_j - vel_i`

5. `test_message_passing_output_shape_and_padding_zero`
   - 输入 `object_token [B,Th,N,D]`
   - 验证输出 `[B,Th,N,D]`
   - padding object 输出全0

6. `test_static_objects_can_send_messages`
   - static_flag 中某物体为 True
   - 验证其相关边未被 edge_mask 屏蔽（只要 valid 且非自环）

7. `test_gradients_flow_through_interaction_module`
   - `inter_token.sum().backward()`
   - 验证 EdgeMLP/NodeMLP 参数有梯度

8. `test_zero_force_edges_are_kept`
   - force 全0但 valid pair 非自环
   - 验证 edge_mask 仍为 True

9. `test_edge_dim_matches_config`
   - 验证 `config.edge_dim == edge_feat.shape[-1] == 16`

10. `test_mean_aggregation_direction_i_to_j`
   - 构造可控 message 或 mock EdgeMLP
   - 验证 receiver j 聚合 sender i，即 `sum(dim=2) / degree`

11. `test_all_invalid_mask_outputs_zero_without_nan`
   - valid_mask 全 False
   - 验证输出全0且无 NaN

12. `test_valid_mask_is_returned_for_temporal_module`
   - 验证 output dict 包含 `valid_mask [B,N]`

---

## 13. 替代方案与放弃理由

| 方案 | 结论 | 理由 |
|------|------|------|
| PyTorch Geometric | 第一版不用 | N小，dense更简单可控 |
| Graph Attention | 后续可试 | 当前force/rel state已是显式edge feature，MLP message足够；`N<=7` 时 attention softmax 可能稀释显式物理信号，多头参数量对小图不划算 |
| 只保留非零force边 | 放弃 | 远距离/未来接触信息会丢失 |
| 把force concat到node | 放弃 | 丢失pairwise方向和配对关系 |
| 静态物体不参与图 | 放弃 | ground/wall接触对物理预测关键 |
| 自环边 | 默认排除 | node residual 已保留自身信息，自环edge无明确物理意义 |

---

## 14. 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| force_matrix方向语义不明 | 使用 F_ij + F_ji 双向特征；T10前抽样检查 |
| normalized state相对量不是真实距离 | 第一版保持统一normalized空间；评估时反归一化 |
| 静态token更新导致语义漂移 | T8 state对静态物体copy last observed；T9不算静态state主loss |
| dense全边引入噪声 | N小先保留；后续做稀疏边消融 |
| edge_dim与T4估算不同 | T6明细以 edge_dim=16 为准，T10 config固定 |

---

## 15. 迭代记录

### 2026-05-28 OpenCode/子agent第一轮评审

结论：需要修改。

主要问题：
- T6 `edge_dim=16` 与 architecture 初始 `De=14` 不一致。
- 静态 token 是否更新存在跨文档语义冲突。
- state slice/static index 等关键索引需集中常量化。
- edge_mask/device/degree/edge_feat masked 位置等实现细节需补充。
- T6 输出需透传 `valid_mask` 给 T7，避免集成时漏传。

修改结果：
- 同步 `architecture.md` 中 `De=16`，T6 要求 `EDGE_DIM=16` 并测试断言。
- 明确 T6 更新所有 valid object token；静态状态预测/loss 由 T8/T9 处理。
- 增加 `STATE_POS_SLICE`、`STATE_VEL_SLICE` 集中常量要求。
- 补充 `not_self` device/bool 构造、`degree=sum(dim=2)`、`edge_feat *= edge_mask`。
- T6 输出 dict 增加 `valid_mask`、`static_flag` 透传。
- 补充 hidden_dim MLP、all-invalid、mean聚合方向、valid_mask透传等测试。

### 2026-05-28 Claude Code最终评审

结论：PASS。

通过理由：
- T6五个子问题全覆盖：交互方式、力矩阵融入、静态物体、零力/远距离边、视觉-物理融合边界。
- 与 `architecture.md`、`input_module.md`、`dataset.py` 的 shape、接口、静态策略、双向力特征一致。
- OpenCode/子agent指出的 edge_dim、静态 token、config常量、edge_mask、valid_mask透传问题均已修订。

轻微建议（不阻塞）：
- 已补充 aggregation 注释：v1 only implements mean。
- 已补充 degree 聚合方向注释：sum over sender i for each receiver j。

### 2026-05-28 OpenCode补充评审

结论：PASS，无必须修改。

采纳的轻微建议：
- 在设计结论处醒目标注 `De=16`，说明这是 T6 明细设计对 T4 初始 `De=14` 的更新。
- 明确 `force_norm_ij` 只是 normalized edge feature，不用于 collision label 阈值；碰撞标签由 T9 使用原始或反归一化 force。
- 在替代方案表中补充 Graph Attention 放弃理由：小图上 attention softmax 可能稀释显式物理信号，多头参数量不划算。
- 保留并强调 `aggregation: mean` 是 v1 固定实现，配置项仅作标记用途；`InteractionModule -> edge_builder + ModuleList[MessagePassingLayer]` 组合关系已在第11节说明。

---

## 16. 最终状态

当前状态：T6交互建模模块设计已完成，OpenCode/子agent修订后，Claude Code最终评审通过。
