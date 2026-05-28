# T11 评估指标定义

## 目标

评估模块覆盖三类指标：

1. 视频质量：MSE、PSNR、SSIM
2. 物理一致性：动态有效物体的 state / position / velocity MSE
3. 碰撞检测：由未来 force_matrix 反归一化构造标签，计算 precision / recall / F1

评估结果同时支持按 `scene_id` 分组汇总。

---

## 视频质量

输入：

```python
pred_rgb:   [B,Tp,3,H,W]
target_rgb: [B,Tp,3,H,W]
```

指标：

```python
mse = mean((pred_rgb - target_rgb)^2)
psnr = 20 * log10(value_range) - 10 * log10(mse)
```

默认 `value_range=2.0`，对应 RGB normalized 到 `[-1,1]`。
当 `mse=0` 时，`psnr=inf`。

SSIM 第一版使用无额外依赖的 global SSIM，用于训练趋势和模型对比。后续若需要论文级报告，可替换为 windowed SSIM，保持输出 key 不变。

---

## 物理状态指标

输入：

```python
pred_state:   [B,Tp,N,16]
target_state: [B,Tp,N,16]
valid_mask:   [B,N]
static_flag:  [B,N]
```

只评估 dynamic valid object：

```python
dynamic_mask = valid_mask & (~static_flag)
```

输出：

- `state_mse`: 16维状态整体 MSE
- `position_mse`: state `[0:3]` 的 MSE
- `velocity_mse`: state `[7:10]` 的 MSE

若没有 dynamic valid object，返回 0，不返回 NaN。

---

## 碰撞检测指标

碰撞标签来自未来 `force_matrix`，必须使用原始力或反归一化力：

```python
force_raw = force_tgt * force_std + force_mean
label = norm(force_raw, dim=-1) > collision_threshold
```

默认：

```python
force_mean = (0,0,0)
force_std = (50,50,50)
collision_threshold = 1e-6
```

预测：

```python
pred_collision = collision_logits > 0
```

只在 `pair_mask=True` 的 pair 上统计：

```python
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
```

若 `pair_mask` 为空，precision/recall/f1/TP/FP/FN 均返回 0。

---

## 场景分组

`evaluate_model` 会读取 batch 中的 `scene_id`，输出类似：

```text
scene_0/mse
scene_0/collision_f1
scene_1/mse
...
```

当前实现按 per-sample video/state 指标聚合，因此混合 scene 的 batch 不会把 batch 均值复制给每个 scene。
碰撞指标按 scene 累加 TP/FP/FN 后再计算 precision/recall/F1，避免直接平均 F1。

---

## 设备与归一化

`evaluate_model` 默认使用模型参数所在 device；如果显式传入 `device`，会执行 `model.to(device)` 并把 batch tensor 移到同一 device。

`force_mean/std` 优先从 `dataloader.dataset.force_mean/std` 读取；没有 dataset 属性时使用传入参数或默认 `(0,0,0)/(50,50,50)`。

---

## 验证

单元测试：

```bash
conda run -n model pytest model/tests/test_eval.py -q
```

覆盖：

- 完全相同 RGB 的 MSE=0、PSNR=inf、SSIM=1
- 已知 MSE/PSNR 数值
- state metric 正确忽略 static/padding object
- 无 dynamic object 不产生 NaN
- collision label 使用反归一化 force
- empty pair_mask 不产生 NaN
- scene 分组 key 正确
- mock dataloader + mock model 的 evaluate_model 端到端运行
