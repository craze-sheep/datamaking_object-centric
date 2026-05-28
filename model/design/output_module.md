# T8 输出解码模块设计

> 任务：T8 输出解码模块
> 上游：T7 时序预测模块
> 下游：T9 Loss、T10 `model/models/decoder.py`
> 第一版目标：从未来 object latent 解码 RGB 帧、物理状态、碰撞 logits，并保留 mask/depth 扩展接口

---

## 1. 设计结论

第一版采用多头解码器：

1. `StateHead`：预测动态物体未来状态 `[B,Tp,N,16]`，静态物体 copy 最后一帧观测状态。
2. `MaskHead`：预测未来每物体 soft mask `[B,Tp,N,128,128]`，作为空间约束和辅助任务。
3. `AppearanceHead + ObjectRGBDecoder`：基于 future token、pred_state 和 pred_mask 合成 RGB `[B,Tp,3,128,128]`。
4. `CollisionHead`：预测 pairwise collision/contact logits `[B,Tp,N,N]`。
5. `DepthHead`：只预留接口，第一版默认关闭。

核心原则：RGB decoder 不能只把所有 object latent 聚合成一个 frame latent；第一版主方案必须保留 object-conditioned spatial path。弱 frame decoder 只能作为 `rgb_weak_decoder` 临时 baseline，不作为主路径。

---

## 2. 输入输出接口

### 2.1 输入

来自 T7/T5：

```python
decoder(
    future_token: Tensor,   # [B,Tp,N,D]
    valid_mask: Tensor,     # [B,N] bool
    static_flag: Tensor,    # [B,N] bool
    last_state: Tensor,     # [B,N,16], normalized, history last frame
    last_mask: Tensor=None, # [B,N,128,128], optional；来自 batch['mask'][:, Th-1]，仅 static_mask_strategy!='predict' 时使用
) -> dict
```

`last_state = encoder_out['state_hist'][:, -1]`。`last_mask` 若需要 static mask fallback，则由整体模型从原始 batch 取 `batch['mask'][:, Th-1]`，不要从 `resized_mask` 取，因为 decoder 需要128x128原分辨率。静态物体使用 copy-last-state，动态物体使用 StateHead 输出。

### 2.2 输出

```python
{
    'rgb': Tensor,                # [B,Tp,3,128,128]
    'state': Tensor,              # [B,Tp,N,16]
    'collision_logits': Tensor,   # [B,Tp,N,N]
    'pair_mask': Tensor,          # [B,Tp,N,N] bool，T9必须用它mask collision loss
    'mask_logits': Tensor,        # [B,Tp,N,128,128]
    'mask_prob': Tensor,          # [B,Tp,N,128,128]
    'depth': Optional[Tensor],    # [B,Tp,1,128,128] if enabled
}
```

所有 object 维输出对 padding object 必须置零或在 loss 中 mask 掉；`rgb` 是 frame-level，不按 object mask。

---

## 3. StateHead 物理状态预测

### 3.1 输出定义

```python
raw_state_delta = MLP(future_token)  # [B,Tp,N,16]
```

第一版采用 delta prediction：

```python
base = last_state[:, None, :, :]     # [B,1,N,16]
pred_dynamic_state = base + raw_state_delta
```

每个 future step 的 delta 都相对同一个历史最后状态 `last_state`，不是累积 rollout：不做 `state[t] = state[t-1] + delta[t]`。这与 T7 非自回归一次性预测保持一致。

原因：

- 比直接预测绝对状态更稳定。
- 对短期 12 帧预测，delta 学习更容易。
- normalized state space 下 delta 尺度较可控。

说明：architecture.md 中 “MLP `[B,Tp,N,D] -> [B,Tp,N,16]`” 是简化表述；T8 细化为 StateHead 输出 `raw_state_delta`，再与 `last_state` 相加得到最终 state。

### 3.2 静态物体 copy-last-state

```python
static = static_flag[:, None, :, None]
valid = valid_mask[:, None, :, None]
pred_state = where(static, base.expand_as(pred_dynamic_state), pred_dynamic_state)
pred_state = pred_state * valid
```

padding 物体输出0。

### 3.3 quaternion 处理

根据 `dataset.py` 当前归一化策略，quaternion slice `[3:7]` 的 `state_std=1.0`，等价于未缩放，仍处于真实 quaternion 空间。因此第一版允许在 decoder 输出后对 dynamic valid object 做 unit normalize，避免明显非法旋转：

```python
quat = pred_state[..., 3:7]
quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
pred_state[..., 3:7] = quat
```

约束：

- 只对 dynamic valid object normalize；静态 object copy last_state，不改 quaternion。
- 如果未来 T3/T10 改成对 quaternion 使用非1 std或其他 normalization，则 T8 不得在 normalized space 直接 unit normalize，必须改为在反归一化/评估或 T9 angular loss 中处理。
- T9 仍负责定义 quaternion loss 的具体形式。

---

## 4. MaskHead 空间解码

### 4.1 目的

MaskHead 是第一版默认保留的轻量辅助头，用于：

- 为 RGB object compose 提供空间布局。
- 让 object latent 保持 object-centric 语义。
- 为 T9 提供可选 BCE/Dice mask loss。

### 4.2 输入输出

```python
mask_logits = MaskHead(future_token, pred_state)  # [B,Tp,N,128,128]
mask_logits = mask_logits * valid_mask[:,None,:,None,None]
mask_prob = sigmoid(mask_logits) * valid_mask[:,None,:,None,None]
```

padding object 的 `mask_logits` 和 `mask_prob` 都必须置零；T9 若直接对 logits 做 BCE，也必须用 valid object mask。

### 4.3 实现路径

建议用 token-conditioned spatial decoder：

```python
mask_seed = MLP([future_token, pred_state_pos_vel])  # [B,Tp,N,Cm*8*8]
reshape -> [B*Tp*N,Cm,8,8]
ConvTranspose blocks: 8 -> 16 -> 32 -> 64 -> 128
1x1 Conv -> 1 channel logits
reshape -> [B,Tp,N,128,128]
```

默认：

```yaml
mask_base_channels: 64
mask_input_state_slices: position + velocity  # 6 dims
```

静态物体 mask 策略配置：

```python
static_mask_strategy: Literal['predict', 'copy_last', 'blend'] = 'predict'
```

第一版默认 `predict`：静态物体也由 MaskHead 预测；若训练中 static mask 漂移明显，可切换为 `copy_last` 使用 `last_mask`。主 loss 是否监督静态 mask 由 T9 决定。

---

## 5. RGB Decoder

### 5.1 主方案：object-conditioned compose

每个物体生成 appearance feature/color layer，再用 mask_prob 合成 frame：

```python
appearance = AppearanceHead(future_token)  # [B,Tp,N,Ca]
object_rgb = ObjectRGBDecoder(appearance, pred_state)  # [B,Tp,N,3,128,128]
mask_prob = mask_prob * valid_mask[:,None,:,None,None]
obj_alpha, bg_alpha = normalize_masks_with_background(mask_prob, valid_mask)
# obj_alpha: [B,Tp,N,1,128,128]
# bg_alpha:  [B,Tp,1,128,128]
rgb = (obj_alpha * object_rgb).sum(dim=2) + bg_alpha * background[None,None]
```

背景不是无条件相加，必须由 `bg_alpha` 控制可见区域。

### 5.2 mask 归一化

避免多个物体 mask 重叠导致亮度异常，并避免 background 无条件叠加：

```python
mask_prob = mask_prob * valid_mask[:,None,:,None,None]
mask_sum = mask_prob.sum(dim=2, keepdim=True)             # [B,Tp,1,H,W]
obj_alpha_2d = mask_prob / mask_sum.clamp_min(1e-6)       # [B,Tp,N,H,W]
object_coverage = mask_sum.clamp(max=1.0)                 # [B,Tp,1,H,W]
obj_alpha_2d = obj_alpha_2d * object_coverage             # object总alpha <= 1
bg_alpha = 1.0 - object_coverage.squeeze(2)               # [B,Tp,H,W]
obj_alpha = obj_alpha_2d.unsqueeze(3)                     # [B,Tp,N,1,H,W]
bg_alpha = bg_alpha.unsqueeze(2)                          # [B,Tp,1,H,W]
```

如果所有 mask 都为0，则 `object_coverage=0`，RGB 完全来自 background，且不会 NaN。

第一版使用 learnable background：

```python
background = nn.Parameter(torch.zeros(3, 128, 128))
```

后续如背景复杂，可替换为 background decoder from pooled future token。

### 5.3 object_rgb 解码

与 MaskHead 类似：

```python
rgb_seed = MLP([future_token, pred_state_pos_vel])
reshape [B*Tp*N, Cr, 8, 8]
ConvTranspose 8->16->32->64->128
1x1 Conv -> 3 channels
Tanh output -> [-1,1]
```

RGB target 在 dataset normalize=True 时约为 `[-1,1]`，因此 RGB decoder 默认输出 `tanh` 到 `[-1,1]`。若训练使用 normalize=False，config 中关闭 tanh 或转换 target。

### 5.4 弱 baseline

`FrameLatentRGBDecoder`：将 future_token 对 valid objects mean pool 成 `[B,Tp,D]`，上采样到 RGB。仅允许作为 `rgb_weak_decoder` baseline 或 debugging，不作为主架构默认。

---

## 6. CollisionHead

### 6.1 输出定义

```python
collision_logits: [B,Tp,N,N]
```

表示 pairwise contact/collision logit。不是力大小回归，也不是时空定位热图。

### 6.2 pair feature

```python
z_i, z_j = future_token pairwise
state_i, state_j = pred_state pairwise
rel_pos = pos_j - pos_i  # pos slice [0:3]
rel_vel = vel_j - vel_i  # vel slice [7:10]
static_i, static_j
pair_feat = concat([z_i, z_j, rel_pos, rel_vel, static_i, static_j])
# pair_feat_dim = 2*D + 8，D=256时为520
logits_ij = PairMLP(pair_feat)
valid_i = valid_mask[:,None,None,:,None]
valid_j = valid_mask[:,None,None,None,:]
not_self = ~eye(N)
pair_mask = (valid_i & valid_j & not_self).expand(B,Tp,N,N)
```

输出 mask：

```python
collision_logits = collision_logits.masked_fill(~pair_mask, 0.0)
```

T8 必须返回 `pair_mask`。T9 必须用 `pair_mask` mask collision loss；invalid/self pair logits 置0只是安全输出，不代表这些 pair 可以参与 loss。

是否对称：

- 第一版不强制对称，因为 `i->j` 与 `j->i` 特征方向不同。
- T9 计算标签/指标时可以对 pair 取 max/mean 或同时监督双向。

### 6.3 collision label 归属

T8 只输出 logits，不构造 label。label 由 T9 根据未来 force_matrix 的原始/反归一化力构造。

---

## 7. DepthHead 预留接口

第一版默认：

```yaml
use_depth_head: false
```

若启用：

```python
depth = DepthHead(pooled frame feature or composed object depth)  # [B,Tp,1,128,128]
```

未来启用时建议输出 `[0,1]` normalized depth，并由 T9 定义 depth valid mask 和 loss。第一版不训练 depth loss，不要求 dataset 提供 depth target。

---

## 8. 输出数值空间和 mask 约定

| 输出 | 数值空间 | mask策略 |
|------|----------|----------|
| `rgb` | 默认 `[-1,1]` | frame-level，不按object mask |
| `state` | normalized state space | padding=0，静态copy last |
| `collision_logits` | raw logits | invalid/self pair=0，T9必须用返回的 `pair_mask` |
| `mask_logits` | raw logits | padding object logits必须置0，loss仍用valid mask |
| `mask_prob` | `[0,1]` | padding object=0 |
| `depth` | 未定义 | disabled |

T9 负责具体 loss mask。T8 输出 dict 必须返回 `pair_mask`；`dynamic_mask` 可由 T9 根据 `valid_mask/static_flag` 重建。

---

## 9. T10 实现建议

文件：`model/models/decoder.py`

建议类：

```python
@dataclass
class DecoderConfig:
    token_dim: int = 256
    state_dim: int = 16
    image_size: int = 128
    max_objects: int = 7
    predict_length: int = 12
    mask_base_channels: int = 64
    rgb_base_channels: int = 64
    use_mask_head: bool = True
    use_depth_head: bool = False
    rgb_output_activation: str = 'tanh'
```

```python
class StateHead(nn.Module): ...
class SpatialTokenDecoder(nn.Module): ...  # shared pattern for mask/rgb
class MaskHead(nn.Module): ...
class ObjectRGBDecoder(nn.Module): ...
class CollisionHead(nn.Module): ...
class OutputDecoder(nn.Module): ...
```

组合关系：

```text
OutputDecoder
  ├─ state_head
  ├─ mask_head
  ├─ object_rgb_decoder
  ├─ collision_head
  └─ optional depth_head
```

---

## 10. 单元测试要求（T10）

测试文件：`model/tests/test_decoder.py`

必须覆盖：

1. `test_state_head_output_shape_and_static_copy`
   - dynamic object 使用预测，static object 等于 last_state
   - padding object 输出0

2. `test_quaternion_normalization_for_dynamic_objects`
   - dynamic quaternion norm 接近1
   - static quaternion 保持 last_state

3. `test_mask_head_output_shape_and_padding_zero`
   - `mask_logits/mask_prob [B,Tp,N,128,128]`
   - padding object mask_prob 全0

4. `test_rgb_decoder_output_shape_and_range`
   - `rgb [B,Tp,3,128,128]`
   - tanh 时范围在 `[-1,1]`

5. `test_collision_head_shape_and_pair_mask`
   - logits `[B,Tp,N,N]`
   - self/padding pair logits 为0或 pair_mask False

6. `test_decoder_forward_full_output_keys`
   - 输出包含 `rgb/state/collision_logits/pair_mask/mask_logits/mask_prob`

7. `test_gradients_flow_through_decoder`
   - loss = rgb.sum() + state.sum() + collision_logits.sum()
   - 验证 decoder 参数有梯度

8. `test_depth_head_disabled_by_default`
   - `use_depth_head=False` 时 depth 为 None 或不在输出中

9. `test_weak_rgb_decoder_not_default`
   - 默认 config 不使用 `rgb_weak_decoder`

10. `test_rgb_compose_alpha_shape`
   - 验证 `obj_alpha [B,Tp,N,1,H,W]` 可与 `object_rgb [B,Tp,N,3,H,W]` 广播

11. `test_rgb_background_not_added_unconditionally`
   - mask 全1时背景权重接近0；mask 全0时输出来自背景

12. `test_padding_masks_do_not_contribute_to_rgb_alpha`
   - padding object 的 mask 不参与 `mask_sum` 和 RGB compose

13. `test_all_zero_mask_uses_background_without_nan`
   - 所有 mask 为0时 RGB 无 NaN，输出等于/接近 background

---

## 11. 替代方案与放弃理由

| 方案 | 结论 | 理由 |
|------|------|------|
| frame latent RGB decoder | baseline only | 空间信息不足，容易模糊 |
| object compose RGB decoder | 第一版主方案 | 保持object-centric，利用mask空间约束 |
| 直接预测RGB不预测mask | 放弃 | 缺少空间监督，object token易退化 |
| collision力大小回归 | 放弃 | T8目标是碰撞检测logits，力大小监督留给state/force loss扩展 |
| 强制collision logits对称 | 第一版不做 | direction feature不同，T9可统一双向标签 |
| depth head默认开启 | 放弃 | 用户明确第一版暂不用深度图 |
| GAN/diffusion decoder | 放弃 | 训练成本和复杂度过高，第一版先监督闭环 |

---

## 12. 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| mask质量差影响RGB | mask loss低权重辅助；可fallback weak decoder调试 |
| object_rgb重叠/遮挡处理粗糙 | alpha normalize；后续可加depth ordering |
| learnable background过于简单 | 后续改为背景decoder或使用静态物体层 |
| quaternion normalize影响loss | T9定义 quaternion loss；先保证合法旋转 |
| collision类别不平衡 | T9 focal loss处理 |
| static mask预测漂移 | 可用 last_mask fallback 或 T9静态mask监督 |

---

## 13. 迭代记录

### 2026-05-28 OpenCode/子agent第一轮评审

结论：需要修改。

主要问题：
- RGB alpha shape 缺 channel 维，background 被无条件相加。
- mask normalize 需要先应用 valid_mask。
- quaternion normalize 与 normalized state space 语义需澄清。
- CollisionHead 必须返回 `pair_mask`，避免 T9 误把 invalid/self pair 当负样本。
- `last_mask` 来源不明确，类名存在 `ObjectComposeRGBDecoder/ObjectRGBDecoder` 混用。

修改结果：
- 修正 RGB compose：`obj_alpha [B,Tp,N,1,H,W]`，`bg_alpha` 控制 background 可见区域。
- 明确 `mask_prob` 先 valid-mask 再 normalize，padding mask logits/prob 均置零。
- 澄清 quaternion 当前因 `state_std[3:7]=1` 可 normalize；若未来改归一化策略则不得直接 normalize。
- 输出 dict 增加 `pair_mask`，并要求 T9 必须使用。
- 明确 `last_mask` 来自原始 `batch['mask'][:, Th-1]`，统一类名为 `ObjectRGBDecoder`。
- 补充 RGB compose 单元测试要求。

### 2026-05-28 Claude Code最终评审

结论：PASS。

通过理由：
- T8五个子问题全覆盖：RGB、state、collision、mask、depth预留。
- 与 `architecture.md`、`temporal_module.md`、`input_module.md` 的 shape、数值空间、静态策略、非自回归接口一致。
- OpenCode/子agent指出的 RGB alpha/background、valid mask、quaternion、pair_mask、last_mask、类名问题均已修订。

轻微建议（不阻塞）：
- 已补充 collision pair feature 中 pos/vel slice 注释。
- T10实现时统一 `normalize_masks_with_background` 函数名即可。

---

## 14. 最终状态

当前状态：T8输出解码模块设计已完成，OpenCode/子agent修订后，Claude Code最终评审通过。
