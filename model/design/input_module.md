# T5 输入处理模块设计

> 任务：T5 输入处理模块
> 上游：T3 数据加载器、T4 总体架构
> 下游：T6 交互模块、T7 时序模块、T10 `model/models/encoder.py`
> 第一版目标：把 RGB、GT mask、静态属性、动态状态、力矩阵上下文编码为统一 object token `[B,Th,N,D]`

---

## 1. 设计结论

第一版输入模块采用“轻量 CNN 视觉特征 + GT mask ROI pooling + 属性/状态 MLP + 时间/物体位置编码 + token fusion”的方案。

核心输出：

```python
object_token: [B, Th, N, D]
encoder_aux: {
    'visual_feat': [B, Th, N, Dv],
    'physical_feat': [B, Th, N, Dp],
    'static_flag': [B, N],
    'dynamic_mask': [B, N],
    'valid_mask': [B, N],
}
```

默认配置：

```yaml
history_length: 12
max_objects: 7
image_size: 128
cnn_channels: [32, 64, 128, 256]
feature_stride: 4
feature_map_size: 32
visual_dim: 256
attr_dim: 14
state_dim: 16
attr_embed_dim: 64
state_embed_dim: 128
physical_dim: 128
token_dim: 256
use_dinov2: false
use_scene_embedding: false
use_view_embedding: true  # view_id缺省0
```

---

## 2. 输入与输出接口

### 2.1 输入 batch

来自 `PhysicsVideoDataset.collate_fn`：

| 字段 | Shape | 数值空间 | 用途 |
|------|-------|----------|------|
| `rgb` | `[B,T,3,128,128]` | normalize=True 时约 `[-1,1]` | 历史帧视觉编码 |
| `mask` | `[B,T,N,128,128]` | 0/1 | GT object ROI pooling |
| `obj_attrs` | `[B,N,14]` | 混合数值/one-hot | 静态属性编码 |
| `dyn_state` | `[B,T,N,16]` | 默认 normalized | 动态状态编码 |
| `force_matrix` | `[B,T,N,N,3]` | 默认 normalized | T5不直接编码为node；传给T6 |
| `valid_mask` | `[B,N]` | bool | padding mask |
| `scene_id` | `[B]` | 当前不稳定hash | 第一版默认不用 |
| `sample_id` | `[B]` | int | 不进模型 |
| `view_id` optional | `[B]` | int | 当前缺省0 |
| `depth` future | `[B,T,1,128,128]` | 未定义 | 第一版不用 |

T5 只读取历史帧：

```python
Th = config.history_length
rgb_hist = batch['rgb'][:, :Th]
mask_hist = batch['mask'][:, :Th]
state_hist = batch['dyn_state'][:, :Th]
attrs = batch['obj_attrs']
valid_mask = batch['valid_mask']
view_id = batch.get('view_id', zeros([B]))
```

### 2.2 输出

```python
class InputEncoderOutput(TypedDict):
    object_token: Tensor      # [B,Th,N,D]
    visual_feat: Tensor       # [B,Th,N,Dv]
    physical_feat: Tensor     # [B,Th,N,Dp]
    state_hist: Tensor        # [B,Th,N,16], normalized，供T6计算相对状态
    force_hist: Tensor        # [B,Th,N,N,3], normalized，供T6构造edge feature
    static_flag: Tensor       # [B,N] bool，已与valid_mask相与
    dynamic_mask: Tensor      # [B,N] bool
    valid_mask: Tensor        # [B,N] bool
    resized_mask: Tensor      # [B,Th,N,32,32]
```

`object_token` 是传给 T6 的主输出。`state_hist/force_hist` 是 T6 构造 edge feature 的正式输入透传；`resized_mask` 主要供 T5 调试 ROI pooling 与后续可视化，不作为 T8 mask loss 的监督源（T8/T9 使用原始未来 `mask_target`）。

---

## 3. 子模块数据流

```text
rgb_hist [B,Th,3,128,128]
        │
        ▼
VisualBackbone / LightCNN
feat_map [B,Th,256,32,32]
        │                         mask_hist [B,Th,N,128,128]
        └──────────────┬─────────────────────┘
                       ▼
                MaskROIPool
                       │
visual_feat [B,Th,N,256]
                       │
obj_attrs [B,N,14] ── AttrEncoder ─┐
state_hist [B,Th,N,16] ─ StateEncoder ─┤
time/object/view embeddings ──────────┤
valid_mask/static_flag ────────────────┤
                       ▼
                  TokenFusion
                       │
object_token [B,Th,N,256]
```

---

## 4. RGB 编码器选型

### 4.1 第一版默认：轻量 CNN

虽然 T1 调研推荐 DINOv2 冻结特征，但 T4 已决定 DINOv2 不作为硬依赖。T5 默认使用轻量 CNN，原因：

- 可控、无额外安装风险。
- 128x128 输入下特征图尺寸好控制。
- 便于 T10 TDD 测试 shape 和梯度。
- 12GB GPU 最小配置可跑。

默认 LightCNN：

```python
ConvBlock(3, 32, stride=1)    # [B*Th,32,128,128]
ConvBlock(32, 64, stride=2)   # [B*Th,64,64,64]
ConvBlock(64,128, stride=2)   # [B*Th,128,32,32]
ConvBlock(128,256,stride=1)   # [B*Th,256,32,32]
1x1 Conv -> C=256
```

输出：

```python
feat_map: [B, Th, 256, 32, 32]
```

ConvBlock 建议：`Conv2d + GroupNorm + SiLU`。不用 BatchNorm，避免小 batch 不稳定。

### 4.2 可选：DINOv2 冻结特征

接口保留：

```python
if config.use_dinov2:
    feat_map = DinoV2Backbone(rgb_hist)  # [B,Th,C,h,w]
    feat_map = projection(feat_map)      # -> [B,Th,256,32,32]
```

约束：

- DINOv2 必须冻结参数。
- 默认 `torch.no_grad()` 提取特征。
- 若 DINOv2 原生 patch grid 不是32x32，统一 interpolate 到32x32。
- DINOv2 只作为后续质量配置，不阻塞第一版。

### 4.3 为什么不用 Slot Attention/SAVi

本数据已有 GT mask。T5 的目标不是无监督分割，而是稳定构造物体级 token。Slot/SAVi 留作后续无 mask 消融：只要替换 `MaskROIPool` 为 `SlotExtractor`，保持输出 `[B,Th,N,Dv]` 即可。

---

## 5. 物体级特征提取：GT mask ROI pooling

### 5.1 输入输出

```python
feat_map:  [B,Th,C,32,32]
mask_hist: [B,Th,N,128,128]
output:    [B,Th,N,Dv]
```

### 5.2 算法

1. 将 mask resize 到特征图大小：

```python
mask_32 = interpolate(mask_hist.float(), size=(32,32), mode='nearest')
# [B,Th,N,32,32]
```

2. masked average pooling：

```python
weighted = feat_map[:, :, None] * mask_32[:, :, :, None]
area = mask_32.sum(dim=(-1,-2)).clamp_min(eps)
visual_feat = weighted.sum(dim=(-1,-2)) / area[..., None]
```

3. 对空 mask / padding 物体置零：

```python
empty = area <= eps
visual_feat[empty] = 0
visual_feat *= valid_mask[:, None, :, None]
```

4. projection + normalization：

```python
visual_feat = LayerNorm(Linear(C, Dv)(visual_feat))
```

### 5.3 边界情况

| 情况 | 处理 |
|------|------|
| padding object | `valid_mask=False`，输出0，后续attention mask掉 |
| mask全0 | 输出0，不产生NaN |
| mask面积极小 | `clamp_min(eps)` 防除0；可记录warning计数 |
| mask非二值 | 先 `>0.5` 或直接作为soft mask；第一版用nearest保持0/1 |
| 物体被遮挡导致某帧mask消失 | 输出0，但 state/attr 仍可提供token信息 |

---

## 6. 静态属性编码

### 6.1 attr_dim=14 约定

当前 dataset.py 中 `attr_dim=14`，但实际只填到 index 12，index 13 保留未用。T5 不能假设第13维一定有 color 信息。

索引约定：

| slice/index | 含义 |
|-------------|------|
| `0:3` | size `[x,y,z]` |
| `3` | lateralFriction |
| `4` | rollingFriction |
| `5` | spinningFriction |
| `6` | restitution |
| `7` | mass |
| `8` | static flag |
| `9:13` | object_type one-hot: ground/sphere/box/cylinder |
| `13` | reserved/color slot，当前恒0 |

### 6.2 AttrEncoder

```python
attr_feat = AttrEncoder(obj_attrs)  # [B,N,64]
```

建议结构：

```python
Linear(14, 64) + SiLU + LayerNorm
Linear(64, 64) + SiLU + LayerNorm
```

处理策略：

- 不在 encoder 中硬编码类别逻辑，只用 dataset 编码后的数值向量。
- `static_flag_raw = obj_attrs[..., 8] > 0.5`。
- `static_flag = valid_mask & static_flag_raw`，padding object 不应被单独解释成动态或静态物体。
- `dynamic_mask = valid_mask & (~static_flag_raw)`。
- mass=0 对静态物体是合法值，不额外修正。

---

## 7. 动态状态编码

### 7.1 state_dim=16 约定

| slice | 含义 |
|-------|------|
| `0:3` | position |
| `3:7` | quaternion `[w,x,y,z]` |
| `7:10` | velocity |
| `10:13` | angular velocity |
| `13:16` | resultant force |

输入为 dataset normalized space。T5 不反归一化，保持模型内部统一使用 normalized state。

### 7.2 StateEncoder

```python
state_feat = StateEncoder(state_hist)  # [B,Th,N,128]
```

建议结构：

```python
Linear(16, 128) + SiLU + LayerNorm
Linear(128, 128) + SiLU + LayerNorm
```

### 7.3 quaternion 处理

第一版不做复杂 SO(3) 表示变换，直接使用 dataset 中 quaternion 4维。原因：

- 当前模型是预测视频和状态，不是精确刚体积分器。
- 简单直接，便于 T10 shape 和 loss 测试。

约束：

- 输出预测 quaternion 时由 T8/T9 决定是否 normalize。
- T5 只编码输入 quaternion，不改变其范数。

---

## 8. 力矩阵编码策略

T5 不把 `force_matrix [B,Th,N,N,3]` 融入 node token，避免提前丢失 pairwise 结构。力矩阵原样传给 T6，由 T6 构造 edge feature：

```python
edge_feat = [F_ij, F_ji, rel_pos, rel_vel, static_i, static_j]
```

T5 只做两件事：

1. 保证 `state_hist` 中 position/velocity 可供 T6 计算相对状态。
2. 在 encoder output 中返回 `static_flag`、`valid_mask`、`dynamic_mask`。

T5/T6 正式接口约定：

```python
encoder_out = input_encoder(batch)
interaction_out = interaction_module(
    object_token=encoder_out['object_token'],
    state_hist=encoder_out['state_hist'],       # [B,Th,N,16], normalized
    force_hist=encoder_out['force_hist'],       # [B,Th,N,N,3], normalized
    valid_mask=encoder_out['valid_mask'],
    static_flag=encoder_out['static_flag'],
)
```

T6 使用 normalized `state_hist` 计算 `rel_pos/rel_vel`，这与模型内部 normalized state 空间一致；碰撞标签阈值仍由 T9 使用原始或反归一化 force 构造。

理由：

- 力是边属性，不是节点属性。
- 如果在 T5 直接 pool/concat 到 object token，会丢失方向和配对关系。
- T4 已决定使用双向 `F_ij + F_ji` 特征应对原始矩阵方向不确定。

---

## 9. 变长物体与 padding 策略

- 统一 `N=max_objects=7`。
- 所有 `[B,Th,N,*]` token 对 padding object 必须置零。
- 下游 attention/message passing 必须使用 mask，不能只依赖零向量。

T5 输出 mask：

```python
valid_mask: [B,N]
static_flag_raw: [B,N] = obj_attrs[..., 8] > 0.5
static_flag: [B,N] = valid_mask & static_flag_raw
dynamic_mask: [B,N] = valid_mask & (~static_flag_raw)
object_token = object_token * valid_mask[:, None, :, None]
visual_feat = visual_feat * valid_mask[:, None, :, None]
physical_feat = physical_feat * valid_mask[:, None, :, None]
```

---

## 10. 时序输入组织

T5 不改变时间长度，只处理历史帧：

```python
input:  [B,T, ...]  # T=24 from DataLoader window
use:    [:, :Th]    # Th=12
output: [B,Th,N,D]
```

跳帧/stride：由 T3 dataset 的窗口采样控制，T5 不再二次采样。

时间编码：

```python
time_emb: [Th,D]
object_id_emb: [N,D]
view_emb: [B,D]  # 缺省0
```

注入方式：

```python
object_token += time_emb[None,:,None,:]
object_token += object_id_emb[None,None,:,:]
object_token += view_emb[:,None,None,:]
```

scene embedding：第一版默认关闭，因为当前 `dataset.py` 的 `scene_id = hash(scene) % 8` 跨进程不稳定。若后续修复为 `S1->0...S8->7`，再启用。

---

## 11. Token Fusion

### 11.1 输入

```python
visual_feat:  [B,Th,N,256]
attr_feat:    [B,N,64]
state_feat:   [B,Th,N,128]
physical_feat:[B,Th,N,128]
time/object/view embeddings
```

attr 需要 broadcast 到时间维，并与 state feature 投影成正式 physical feature：

```python
attr_feat_t = attr_feat[:, None].expand(B, Th, N, 64)
physical_feat = PhysicalProjector(concat([attr_feat_t, state_feat], dim=-1))
# concat dim: 64 + 128 = 192 -> physical_dim 128
```

### 11.2 融合结构

```python
fusion_input = concat([visual_feat, physical_feat], dim=-1)
# [B,Th,N,384]
object_token = FusionMLP(fusion_input)
# [B,Th,N,D]
object_token = object_token + position_embeddings
object_token = LayerNorm(object_token)
object_token = object_token * valid_mask[:,None,:,None]
visual_feat = visual_feat * valid_mask[:,None,:,None]
physical_feat = physical_feat * valid_mask[:,None,:,None]
```

建议 FusionMLP：

```python
Linear(384, D) + SiLU + LayerNorm
Linear(D, D) + residual + LayerNorm
```

### 11.3 为什么 concat + MLP 而不是 cross-attention

- N<=7，模态数少，concat足够。
- cross-attention会增加实现复杂度和测试负担。
- 当前设计目标是稳定第一版，复杂融合留给后续消融。

---

## 12. 视角编码

当前 dataset 不输出 view_id。第一版接口：

```python
view_id = batch.get('view_id', torch.zeros(B, dtype=torch.long, device=device))
view_emb = nn.Embedding(num_views, D)(view_id)
```

默认：

```yaml
num_views: 1
use_view_embedding: true
```

虽然只有一个 view，保留 embedding 不影响输出，后续多视角扩展时接口不变。

---

## 13. 深度图扩展接口

第一版不使用 depth。为后续兼容，`InputEncoder.forward` 可接受：

```python
depth = batch.get('depth', None)
```

如果 `depth is None`，走 RGB-only 路径。

后续启用 depth 时：

```python
rgb_feat = RGBBackbone(rgb_hist)
depth_feat = DepthBackbone(depth_hist)
feat_map = Project(concat([rgb_feat, depth_feat], dim=channel))
```

T10 第一版不需要实现 `DepthBackbone`，但接口不能因缺 depth 报错。

---

## 14. T10 实现建议

文件：`model/models/encoder.py`

建议类：

```python
@dataclass
class EncoderConfig:
    image_size: int = 128
    history_length: int = 12
    max_objects: int = 7
    attr_dim: int = 14
    state_dim: int = 16
    visual_dim: int = 256
    attr_embed_dim: int = 64
    state_embed_dim: int = 128
    token_dim: int = 256
    feature_map_size: int = 32
    use_dinov2: bool = False
    use_scene_embedding: bool = False
    use_view_embedding: bool = True
    num_views: int = 1
```

```python
class LightCNNBackbone(nn.Module): ...
class MaskROIPool(nn.Module): ...
class AttrEncoder(nn.Module): ...
class StateEncoder(nn.Module): ...
class PhysicalProjector(nn.Module): ...
class TimeEmbedding(nn.Module): ...
class ObjectIDEmbedding(nn.Module): ...
class ViewEmbedding(nn.Module): ...
class TokenFusion(nn.Module): ...
class InputEncoder(nn.Module): ...
```

`InputEncoder.forward(batch)` 返回 dict，至少包括 `object_token`。

---

## 15. 单元测试要求（T10）

测试文件：`model/tests/test_encoder.py`

必须覆盖：

1. `test_light_cnn_outputs_expected_feature_shape`
   - 输入 `[B*Th,3,128,128]`
   - 输出 `[B*Th,256,32,32]`

2. `test_mask_roi_pool_outputs_object_features`
   - mock mask 覆盖左上角区域
   - 验证输出 shape `[B,Th,N,Dv]`
   - 验证全0 mask 不产生 NaN

3. `test_input_encoder_forward_shapes`
   - mock batch
   - 验证 `object_token [B,Th,N,D]`
   - 验证 `visual_feat/physical_feat/state_hist/force_hist/resized_mask/static_flag/dynamic_mask` shape
   - 验证输入 `T=24` 时 encoder 只输出 `Th=12`

4. `test_padding_objects_are_zeroed`
   - valid_mask 中部分 False
   - 验证对应 `visual_feat`、`physical_feat`、`object_token` 全0

5. `test_static_and_dynamic_masks_from_attrs`
   - attr index 8 设置 static flag
   - 验证 `static_flag = valid_mask & static_flag_raw`
   - 验证 `dynamic_mask = valid_mask & (~static_flag_raw)`

6. `test_missing_view_id_defaults_to_zero`
   - batch 不包含 view_id
   - forward 不报错

7. `test_gradients_flow_through_trainable_encoder`
   - object_token.sum().backward()
   - 验证轻量CNN/MLP参数有梯度

8. `test_encoder_exports_force_and_state_for_interaction`
   - 验证返回的 `state_hist == batch['dyn_state'][:, :Th]`
   - 验证返回的 `force_hist == batch['force_matrix'][:, :Th]`

---

## 16. 替代方案与放弃理由

| 方案 | 结论 | 理由 |
|------|------|------|
| DINOv2 默认启用 | 放弃 | 安装/显存/速度风险，第一版先可跑 |
| Slot Attention/SAVi | 放弃 | 已有GT mask，无需无监督发现 |
| 直接CNN flatten整帧 | 放弃 | 丢失物体级身份，不利于力矩阵和状态融合 |
| force_matrix提前汇聚到node | 放弃 | 力是pairwise edge属性，应留给T6 |
| cross-attention模态融合 | 后续可试 | 第一版concat+MLP足够简单稳定 |
| scene embedding默认启用 | 放弃 | 当前scene_id不稳定，需先修dataset |

---

## 17. 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| LightCNN特征弱于DINOv2 | 先保证闭环；后续启用DINOv2消融 |
| mask ROI pooling 对遮挡物体输出0 | state/attr仍提供token信息；T8 mask预测可补偿 |
| attr_dim第13维未使用 | 明确保留，不在T5硬编码color |
| dataset归一化固定值不精确 | T5不反归一化；T9/T11负责物理量指标反归一化 |
| object id跨帧不一致 | T5假设一致；若数据验证失败必须回到T3做remap |
| view_id缺失 | 默认0，不阻塞单视角训练 |

---

## 18. 迭代记录

### 2026-05-28 OpenCode/子agent第一轮评审

结论：需要修改。

主要问题：
- `physical_feat [B,Th,N,Dp]` 声明了但没有完整定义。
- T5/T6 关于 `force_hist` 和 `state_hist` 的传递边界不清。
- padding/static mask 语义需要更严格。
- T10 测试缺少 `physical_feat`、`resized_mask`、历史切片、force/state透传等关键断言。

修改结果：
- 增加 `state_hist`、`force_hist` 到 `InputEncoderOutput`，明确 T6 调用接口。
- 明确 `physical_feat = PhysicalProjector(concat(attr_feat_t, state_feat))`，并用 `visual_feat + physical_feat` 做 TokenFusion。
- 修正 `static_flag = valid_mask & static_flag_raw` 和 `dynamic_mask = valid_mask & (~static_flag_raw)`。
- 补充 padding 对 `visual_feat/physical_feat/object_token` 全部置零。
- 补充 T10 单元测试要求：历史切片、resized_mask、force/state透传、padding feature置零。
- 明确 `resized_mask` 主要供 T5 ROI pooling 调试和可视化，不作为 T8 mask target。

### 2026-05-28 Claude Code最终评审

结论：PASS。

通过理由：
- T5子问题全部覆盖：RGB编码器、GT mask ROI、属性/状态编码、force_matrix策略、padding、时序组织、view/depth扩展。
- 与 `architecture.md`、`data_design.md`、`dataset.py` 的 shape、索引、归一化空间和接口一致。
- OpenCode/子agent指出的四类问题已修订到位。

轻微建议（不阻塞）：
- T10实现时可省略或推导 `feature_stride`，避免与 `feature_map_size` 重复。
- visual/physical feature 置零可集中在一处实现，避免双重mask维护。
- T5测试可额外验证 `valid_mask` dtype/shape，attention mask细节留给T6/T7测试。

---

## 19. 最终状态

当前状态：T5输入处理模块设计已完成，OpenCode/子agent修订后，Claude Code最终评审通过。
