# T9 Loss设计

> 任务：T9 多任务损失函数设计
> 上游：T8 输出解码模块
> 下游：T10 `model/models/loss.py`、训练循环、T11评估
> 第一版目标：稳定训练 RGB、物理状态、碰撞检测、mask辅助任务；默认关闭LPIPS/GAN

---

## 1. 设计结论

第一版使用可解释、稳定的监督多任务 loss：

```text
L_total = λ_rgb * L_rgb
        + λ_state * L_state
        + λ_collision * L_collision
        + λ_mask * L_mask
        + λ_lpips * L_lpips
```

默认权重：

```python
lambda_rgb = 1.0
lambda_state = 0.1
lambda_collision = 0.5
lambda_mask = 0.1
lambda_lpips = 0.0  # 第一版关闭
```

默认不使用 GAN/adversarial loss。原因：训练复杂度高、容易不稳定，且第一版重点是多模态监督闭环。

---

## 2. 输入输出接口

### 2.1 输入

```python
loss_fn(
    pred: dict,       # decoder output
    batch: dict,      # dataloader batch
    valid_mask: Tensor=None,   # [B,N]；若None，从batch/obj_attrs推导
    static_flag: Tensor=None,  # [B,N]；若None，从 batch['obj_attrs'][..., ATTR_STATIC_INDEX] 推导
) -> dict
```

需要的 `pred` keys：

```python
pred['rgb']                # [B,Tp,3,128,128]
pred['state']              # [B,Tp,N,16]
pred['collision_logits']   # [B,Tp,N,N]
pred['pair_mask']          # [B,Tp,N,N] bool
pred['mask_logits']        # [B,Tp,N,128,128]
pred['mask_prob']          # [B,Tp,N,128,128]
```

从 batch 切 targets：

```python
Th = config.history_length
Tp = config.predict_length
rgb_tgt = batch['rgb'][:, Th:Th+Tp]
state_tgt = batch['dyn_state'][:, Th:Th+Tp]
force_tgt = batch['force_matrix'][:, Th:Th+Tp]
mask_tgt = batch['mask'][:, Th:Th+Tp]
```

注意：`batch` 中 `state_tgt/force_tgt` 默认是 normalized space。碰撞标签阈值需要原始/反归一化 force。

### 2.2 输出

```python
{
    'loss': total_loss,
    'loss_rgb': L_rgb,
    'loss_state': L_state,
    'loss_collision': L_collision,
    'loss_mask': L_mask,
    'loss_lpips': L_lpips,
    'collision_pos_rate': scalar,
}
```

---

## 3. Mask 约定

```python
ATTR_STATIC_INDEX = 8
if valid_mask is None:
    valid_mask = batch['valid_mask']
if static_flag is None:
    static_flag = batch['obj_attrs'][..., ATTR_STATIC_INDEX] > 0.5
static_flag = static_flag & valid_mask

valid_obj = valid_mask                    # [B,N]
static_obj = static_flag & valid_mask      # [B,N]
dynamic_obj = valid_mask & (~static_flag)  # [B,N]
```

时间扩展：

```python
obj_mask_t = valid_obj[:,None,:,None]        # [B,1,N,1]
dyn_mask_t = dynamic_obj[:,None,:,None]      # [B,1,N,1]
mask2d = valid_obj[:,None,:,None,None]       # [B,1,N,1,1]
pair_mask = pred['pair_mask']                # [B,Tp,N,N]
```

这里假设 object slot 在历史帧和未来帧中顺序恒定，与 `architecture.md` 的 object id 跨帧一致约定、`dataset.py` 的固定 slot/padding 组织一致。

所有除 RGB 外的 object/pair loss 都必须用 mask 归一化，不能让 padding 参与。

### 3.1 masked weighted mean 约定

所有带 mask/time weight 的 loss 使用统一归一化：

```python
# future_step_weights 采用 mean=1 约定，保持loss尺度稳定
if future_step_weights is None:
    future_step_weights = torch.ones(Tp, device=device)
else:
    future_step_weights = torch.tensor(future_step_weights, device=device)
    future_step_weights = future_step_weights / future_step_weights.mean().clamp_min(1e-6)
```

广播规则：

```python
w_rgb = future_step_weights[None,:,None,None,None]      # [1,Tp,1,1,1]
w_state = future_step_weights[None,:,None,None]         # [1,Tp,1,1]
w_collision = future_step_weights[None,:,None,None]     # [1,Tp,1,1]
w_mask = future_step_weights[None,:,None,None,None]     # [1,Tp,1,1,1]
w_mask_dice = future_step_weights[None,:,None]          # [1,Tp,1]
```

实现模式：先 reduce 掉非时间维，保留 `[B,Tp]` / `[B,Tp,N]` / `[B,Tp,N,N]` 的 per-step loss 或 mask，再乘对应 `w_*`，最后 `weighted_sum / valid_weight_sum.clamp_min(1e-6)`。对于 RGB 没有 object mask，valid_weight_sum 是 `B * sum(time_weights)`。

---

## 4. RGB 重建 loss

### 4.1 默认 L1 + MSE

```python
err_l1 = abs(pred_rgb - rgb_tgt).mean(dim=(2,3,4))     # [B,Tp]
err_mse = square(pred_rgb - rgb_tgt).mean(dim=(2,3,4)) # [B,Tp]
err_rgb = 0.8 * err_l1 + 0.2 * err_mse
L_rgb = (err_rgb * future_step_weights[None, :]).sum() / (B * future_step_weights.sum()).clamp_min(1e-6)
```

原因：

- L1 对模糊更鲁棒。
- MSE 保留 PSNR 对齐。
- RGB 是 frame-level，不使用 object mask。

### 4.2 数值空间

`dataset.py normalize=True` 时 RGB target 约为 `[-1,1]`。T8 默认 `tanh` 输出 `[-1,1]`。若训练使用 normalize=False，需要在 config 中统一：

```python
rgb_value_range = 'minus_one_to_one' or 'zero_to_one'
```

### 4.3 Perceptual / LPIPS

默认关闭：

```python
lambda_lpips = 0.0
```

启用条件：

- RGB/MSE/state/collision 已稳定下降。
- GPU显存允许。
- 需要更好视觉质量。

LPIPS 输入通常需要 `[0,1]` 或 `[-1,1]`，T10 实现时必须按库要求转换。LPIPS 不作为第一版通过条件。

实现约束：

- `lpips_weight == 0` 时，不 import、不初始化 LPIPS module，`L_lpips = pred_rgb.new_tensor(0.0)`。
- 启用时，将 `[B,Tp,3,H,W]` reshape 为 `[B*Tp,3,H,W]`。
- 按 LPIPS 库要求转换数值范围，常见实现要求 `[-1,1]`。
- LPIPS 输出 reshape 回 `[B,Tp]` 后应用 `future_step_weights`。

---

## 5. 物理状态 loss

### 5.1 基本 loss

使用 SmoothL1：

```python
state_error = smooth_l1(pred_state, state_tgt, reduction='none')  # [B,Tp,N,16]
state_weight_vec = build_state_weight_vec(device)                 # [16]
state_error = state_error * state_weight_vec[None,None,None,:]
state_mask = dyn_mask_t.float() * w_state                         # [B,Tp,N,1]
L_state = (state_error * state_mask).sum() / (state_mask.sum() * state_weight_vec.sum()).clamp_min(1e-6)
```

只对 dynamic valid object 计算主 loss。静态物体 copy last_state，不参与主 state loss。

### 5.2 分量权重

默认在 normalized space 计算 state loss，但不同分量仍可加权：

```python
state_weight_vec = torch.ones(16, device=device)
state_weight_vec[0:3] = 1.0       # position
state_weight_vec[3:7] = 0.5       # quaternion
state_weight_vec[7:10] = 0.5      # velocity
state_weight_vec[10:13] = 0.25    # angular_velocity
state_weight_vec[13:16] = 0.25    # resultant_force
```

切片：

```python
pos: 0:3
quat: 3:7
vel: 7:10
angvel: 10:13
force: 13:16
```

### 5.3 quaternion loss

第一版可使用 SmoothL1 on quaternion slice。T8 已对 dynamic quaternion unit normalize（当前 dataset quaternion std=1），因此 loss稳定。

后续可替换为 angular loss：

```python
L_quat = 1 - abs(dot(q_pred, q_tgt))
```

但不是第一版必需。

### 5.4 能量/动量守恒约束

第一版不加硬物理守恒 loss。原因：

- 数据场景包含碰撞、摩擦、外力、静态约束，简单能量守恒可能不成立。
- 当前目标先保证监督状态预测。

后续可在特定场景加 soft constraint。

---

## 6. 碰撞检测 loss

### 6.1 标签构造

碰撞标签来自未来 force_matrix。必须使用原始力或反归一化力，不能直接用 normalized force 默认阈值。

当前 `dataset.py normalize=True` 时固定 `force_std=[50,50,50]`，`force_mean=0`。LossConfig 的 `force_mean/std` 必须与 dataset/config 保持一致：若 `dataset.normalize=False`，则 `force_std=(1,1,1)`、`force_mean=(0,0,0)`，不能在 loss.py 中无条件硬编码50。T10 可先反归一化：

```python
force_raw = force_tgt * force_std + force_mean
force_norm = norm(force_raw, dim=-1)  # [B,Tp,N,N]
collision_label = (force_norm > collision_threshold).float().to(logits.device)
```

默认阈值：

```python
collision_threshold = 1e-6
```

若发现噪声导致正样本过多，再按训练集非零力分布调整到 P5/P10。

### 6.2 Focal BCE

碰撞极稀疏，使用 Focal BCE：

```python
bce = binary_cross_entropy_with_logits(logits, labels, reduction='none')
p = sigmoid(logits)
pt = where(labels==1, p, 1-p)
alpha_t = labels * alpha + (1 - labels) * (1 - alpha)
focal = alpha_t * (1-pt)**gamma * bce
collision_mask = pair_mask.float() * w_collision
L_collision = (focal * collision_mask).sum() / collision_mask.sum().clamp_min(1e-6)
```

默认：

```python
alpha = 0.25
gamma = 2.0
```

### 6.3 pair mask

必须使用 T8 返回的 `pair_mask [B,Tp,N,N]`，并默认进一步过滤 static-static pair：

```python
pair_mask = pred['pair_mask']
dynamic_pair = dynamic_obj[:,None,:,None] | dynamic_obj[:,None,None,:]
pair_mask = pair_mask & dynamic_pair
```

对角线 self pair、padding pair、static-static pair 不参与 loss。动态-静态 pair 保留，因为 ground/wall 接触重要。若后续确认 static-static 接触有意义，可配置 `include_static_static_pairs=True`。

---

## 7. Mask 辅助 loss

### 7.1 BCE + Dice

```python
mask_bce = BCEWithLogitsLoss(reduction='none')(mask_logits, mask_tgt)
mask_weight = mask2d.float() * w_mask
mask_bce = (mask_bce * mask_weight).sum() / (mask_weight.sum() * H * W).clamp_min(1e-6)
```

Dice：

```python
pred_prob = sigmoid(mask_logits) * mask2d  # mask2d [B,1,N,1,1] broadcasts over Tp,H,W
tgt = mask_tgt * mask2d
intersection = (pred_prob * tgt).sum(dim=(-1,-2))
union = pred_prob.sum(dim=(-1,-2)) + tgt.sum(dim=(-1,-2))
dice = 1 - (2*intersection + eps) / (union + eps)  # [B,Tp,N]
dice_mask = valid_obj[:,None,:].float() * w_mask_dice
L_dice = (dice * dice_mask).sum() / dice_mask.sum().clamp_min(1e-6)
```

组合：

```python
L_mask = 0.5 * mask_bce + 0.5 * L_dice
```

### 7.2 静态物体 mask

第一版 mask loss 对所有 valid object 计算，包括静态物体。理由：ground/wall/静态物体的空间布局有助于 RGB compose。

如果静态 mask target 不稳定，可在配置中改为只监督 dynamic object。

---

## 8. 时序权重

第一版默认等权，并采用 mean=1 约定：

```python
future_step_weights = torch.ones(Tp)  # mean=1, sum=Tp
```

可选衰减：

```python
future_step_weights = exp(-k * arange(Tp))
future_step_weights /= future_step_weights.mean()
```

T9 在每个 loss 的 `[Tp]` 维度上应用。具体乘法位置遵循 §3.1：先得到 per-step loss，再乘 `future_step_weights` 后归一化。第一版先等权，T11按 t=0..11 分帧评估远帧退化后再调整。

---

## 9. 多任务权重策略

第一版固定权重，简单可控：

```python
loss = (
    1.0 * L_rgb +
    0.1 * L_state +
    0.5 * L_collision +
    0.1 * L_mask +
    0.0 * L_lpips
)
```

后续可选 uncertainty weighting：

```python
L = sum(exp(-s_i) * L_i + s_i)
```

第一版不默认启用，避免多引入可训练标量导致调试困难。

---

## 10. GAN / adversarial loss

第一版不使用。原因：

- 小模型/小分辨率先保证监督闭环。
- GAN 会引入判别器、训练不稳定、额外调参。
- 物理状态和碰撞检测比视觉锐度更核心。

后续若 RGB 过于模糊，优先尝试 LPIPS、mask/object compose 改进，再考虑 GAN。

---

## 11. T10 实现建议

文件：`model/models/loss.py`

建议类：

```python
@dataclass
class LossConfig:
    history_length: int = 12
    predict_length: int = 12
    state_dim: int = 16
    rgb_weight: float = 1.0
    state_weight: float = 0.1
    collision_weight: float = 0.5
    mask_weight: float = 0.1
    lpips_weight: float = 0.0
    collision_threshold: float = 1e-6
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    force_mean: tuple = (0.0, 0.0, 0.0)  # T10内部转为tensor(device/dtype)
    force_std: tuple = (50.0, 50.0, 50.0) # T10内部转为tensor(device/dtype)
    state_component_weights: tuple = (1.0,1.0,1.0, 0.5,0.5,0.5,0.5, 0.5,0.5,0.5, 0.25,0.25,0.25, 0.25,0.25,0.25)
    future_step_weights: Optional[list[float]] = None
```

```python
class PhysicsPredictionLoss(nn.Module):
    def forward(self, pred, batch, valid_mask, static_flag): ...
```

辅助函数：

```python
build_collision_labels(force_tgt, force_mean, force_std, threshold)
masked_mean(values, mask, eps=1e-6)
focal_bce_with_logits(logits, labels, mask, alpha, gamma)
dice_loss(mask_prob, mask_tgt, valid_mask)
```

---

## 12. 单元测试要求（T10）

测试文件：`model/tests/test_loss.py`

必须覆盖：

1. `test_loss_returns_all_components`
   - 输出包含 total 和各 component。

2. `test_rgb_loss_finite_and_positive`
   - mock pred/target，RGB loss 有限。

3. `test_state_loss_masks_static_and_padding_objects`
   - static/padding object 上巨大误差不影响 L_state。

4. `test_collision_labels_use_denormalized_force`
   - normalized force 乘 std 后超过阈值才为正。

5. `test_collision_loss_uses_pair_mask`
   - invalid/self pair 上巨大logit不影响 loss。

6. `test_mask_loss_uses_valid_object_mask`
   - padding object mask误差不影响 loss。

7. `test_future_step_weights_apply_time_dimension`
   - 修改某个 future step 权重会改变 loss。

8. `test_zero_valid_dynamic_objects_no_nan`
   - dynamic_mask 全 False 时 state loss 为0且 total 无 NaN。

9. `test_zero_pair_mask_no_nan`
   - pair_mask 全 False 时 collision loss 为0且无 NaN。

10. `test_lpips_disabled_by_default`
   - lpips_weight=0 时不需要导入/运行 LPIPS。

11. `test_total_loss_weighted_sum`
   - total 等于各 component * weight 之和。

---

## 13. 替代方案与放弃理由

| 方案 | 结论 | 理由 |
|------|------|------|
| RGB L1+MSE | 第一版使用 | 稳定、简单、与PSNR相关 |
| LPIPS | 默认关闭 | 显存和依赖额外成本，后续可开 |
| SSIM loss | 后续可试 | 实现略复杂，第一版先不用 |
| GAN loss | 放弃 | 训练不稳定，非第一版重点 |
| 能量/动量守恒 | 后续场景化 | 摩擦/碰撞/外力下简单守恒不可靠 |
| uncertainty weighting | 后续可试 | 第一版固定权重更易调试 |
| collision重采样 | 后续可试 | Focal loss先处理类别不平衡 |

---

## 14. 已知风险与缓解

| 风险 | 缓解 |
|------|------|
| loss尺度不平衡 | 固定初始权重，记录各component，必要时调权重 |
| collision正样本极少 | Focal BCE，记录 pos_rate |
| force阈值不合适 | 先1e-6，后按训练集非零力分布调P5/P10 |
| quaternion loss不准确 | 第一版SmoothL1，后续 angular loss |
| mask loss压过RGB | mask_weight=0.1低权重 |
| LPIPS依赖/显存问题 | 默认关闭 |
| 无有效dynamic/pair导致NaN | 所有 masked mean clamp，空mask返回0 |

---

## 15. 迭代记录

### 2026-05-28 OpenCode/子agent第一轮评审

结论：需要修改。

主要问题：
- state component weights 定义了但未进入公式。
- future_step_weights 概念存在但未精确定义到各 loss 的 broadcast/归一化。
- Focal BCE 的 alpha 写法没有区分正负样本。
- collision pair 监督范围未明确 static-static 是否参与。
- force_std 默认值需与 dataset.normalize 保持一致。
- static_flag 来源未写清。

修改结果：
- State loss 加入 16维 `state_weight_vec` 和加权分母。
- 统一 future_step_weights 为 mean=1，并定义 RGB/state/collision/mask broadcast 规则。
- Focal BCE 改为标准 `alpha_t = y*alpha + (1-y)*(1-alpha)`。
- 默认过滤 static-static pair，保留 dynamic-static pair。
- 明确 force_mean/std 必须来自 dataset/config；normalize=False 时 std=(1,1,1)。
- 明确 static_flag 从 `batch['obj_attrs'][...,8]` 推导，LossConfig 补充 state_component_weights。

### 2026-05-28 Claude Code最终评审

结论：PASS。

通过理由：
- T9九个子问题全覆盖：RGB、LPIPS、状态、碰撞、mask、多任务权重、时序权重、padding、GAN取舍。
- 与 `architecture.md`、`output_module.md`、`dataset.py` 的默认权重、shape、归一化、static/collision约定一致。
- OpenCode/子agent指出的 state weights、time weights、focal alpha、static-static pair、force_std、static_flag来源问题均已修订。

轻微建议（不阻塞）：
- 已补充 LossConfig `state_component_weights`。
- T10实现中可简化 `static_obj=static_flag`，因为 static_flag 已 valid-gated。

### 2026-05-28 OpenCode补充评审

结论：PASS，反馈均为轻微建议或已修订项，无必须修改。

采纳的轻微建议：
- 在 mask 时间扩展处补充 object slot 跨帧顺序恒定假设。
- 明确 time weight 的实现模式：先保留 per-step loss，再乘 `future_step_weights` 并归一化。
- 在 Dice loss 中标注 `mask2d [B,1,N,1,1]` 会广播到 `Tp,H,W`。
- 在 `LossConfig.force_mean/std` 注释中明确 T10 内部转 tensor(device/dtype)。

已确认无需再改：
- `static_flag` 来源已在 §2.1/§3 明确：默认从 `batch['obj_attrs'][..., ATTR_STATIC_INDEX] > 0.5` 推导。
- RGB loss 已用 `mean(dim=(2,3,4))` 明确 per-frame per-pixel/channel reduction。
- Focal BCE 已采用经典 `alpha_t = y*alpha + (1-y)*(1-alpha)`，不是统一乘 alpha。

---

## 16. 最终状态

当前状态：T9 Loss设计已完成，OpenCode/子agent修订后，Claude Code最终评审通过。
