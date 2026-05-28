# 物理视频预测模型 README

这个目录实现的是一个“物体中心 + 显式物理交互 + 时序预测 + 多头解码”的物理视频预测 baseline。目标是：给定历史 12 帧 RGB、物体 mask、静态属性、动态状态和力矩阵，预测未来 12 帧 RGB、未来物理状态和碰撞事件。

当前 T5-T12 已完成，核心测试在 `model` conda 环境下通过：

```bash
conda run -n model pytest \
  model/tests/test_config.py \
  model/tests/test_encoder.py \
  model/tests/test_interaction.py \
  model/tests/test_temporal.py \
  model/tests/test_decoder.py \
  model/tests/test_loss.py \
  model/tests/test_physics_pred.py \
  model/tests/test_eval.py \
  model/tests/test_inference.py \
  -q
```

最近验证结果：`63 passed`。

---

## 1. 你应该从哪里开始看

建议按这个顺序读：

1. `分工.md`
   - 看整个任务如何拆成 T1-T12。
   - T5-T12 的完成状态、产出文件、测试命令都记录在里面。

2. `architecture.md`
   - 看总体架构为什么这样设计。
   - 重点看输入输出 shape、模块边界、静态物体、padding mask、loss 设计。

3. `design/*.md`
   - 每个模块的详细设计文档：
     - `design/input_module.md`：T5 输入编码
     - `design/interaction_module.md`：T6 交互建模
     - `design/temporal_module.md`：T7 时序预测
     - `design/output_module.md`：T8 输出解码
     - `design/loss_design.md`：T9 Loss

4. 代码实现：
   - `config.py`
   - `dataset.py`
   - `models/*.py`
   - `eval.py`
   - `inference.py`
   - `visualization.py`

5. 测试：
   - `tests/test_*.py`
   - 测试是理解接口和边界条件最快的入口。

---

## 2. 整体数据流

模型输入来自 `dataset.py`，一个 batch 主要包含：

```text
rgb          [B,T,3,128,128]
mask         [B,T,N,128,128]
obj_attrs    [B,N,14]
dyn_state    [B,T,N,16]
force_matrix [B,T,N,N,3]
valid_mask   [B,N]
scene_id     [B]
sample_id    [B]
```

默认：

```text
T = history_length + predict_length = 12 + 12 = 24
N <= 7
state_dim = 16
attr_dim = 14
```

模型输出：

```text
rgb              [B,Tp,3,128,128]
state            [B,Tp,N,16]
collision_logits [B,Tp,N,N]
pair_mask        [B,Tp,N,N]
mask_logits      [B,Tp,N,128,128]
mask_prob        [B,Tp,N,128,128]
valid_mask       [B,N]
static_flag      [B,N]
dynamic_mask     [B,N]
```

`Tp = predict_length`，默认 12。

---

## 3. 模块分解

### T5 Input Encoder

文件：

```text
models/encoder.py
tests/test_encoder.py
```

职责：

- 输入 RGB、GT mask、物体属性、动态状态、力矩阵。
- 用 mask 做物体 ROI pooling。
- 编码视觉特征、属性特征、状态特征。
- 融合为 object token：

```text
object_token [B,Th,N,D]
```

同时输出：

```text
state_hist   [B,Th,N,16]
force_hist   [B,Th,N,N,3]
valid_mask   [B,N]
static_flag  [B,N]
dynamic_mask [B,N]
```

注意：

- padding object 的 token 会被 mask 掉。
- static flag 来自 `obj_attrs[..., 8]`。
- 当前第一版默认不用 DINOv2，使用轻量 CNN 特征，便于先跑通 baseline。

---

### T6 Interaction Module

文件：

```text
models/interaction.py
tests/test_interaction.py
```

职责：

- 在每一帧内对物体做 dense pairwise message passing。
- 边特征来自：
  - `force_matrix[t,i,j]`
  - `force_matrix[t,j,i]`
  - 相对位置/速度
  - 距离
  - 力大小
  - static flag
- 输出交互后的 token：

```text
inter_token [B,Th,N,D]
```

关键设计：

- `N <= 7`，所以不用 PyG 稀疏图，直接 dense `[B,Th,N,N,*]` 更简单可靠。
- 保留零力边，因为“无接触”本身也是有意义的信息。
- static object 参与发送和接收 message，但后续 state loss 不要求它运动。
- static-static pair 在 collision loss / eval 中会被过滤。

---

### T7 Temporal Predictor

文件：

```text
models/temporal.py
tests/test_temporal.py
```

职责：

- 用历史 object-time token 预测未来 object latent。
- 当前第一版使用 Transformer encoder/decoder。
- 输入：

```text
inter_token [B,Th,N,D]
valid_mask  [B,N]
```

输出：

```text
future_token [B,Tp,N,D]
valid_mask   [B,N]
```

关键修复：

- Transformer 在 all-invalid key padding mask 下可能产生 NaN。
- 当前实现会在内部临时 unmask 一个 dummy token，避免 attention 全 mask；最终输出仍按原始 `valid_mask` 置零。
- 有回归测试覆盖这个边界条件。

---

### T8 Output Decoder

文件：

```text
models/decoder.py
tests/test_decoder.py
```

职责：

从 `future_token` 解码出：

```text
rgb              [B,Tp,3,H,W]
state            [B,Tp,N,16]
collision_logits [B,Tp,N,N]
pair_mask        [B,Tp,N,N]
mask_logits      [B,Tp,N,H,W]
mask_prob        [B,Tp,N,H,W]
```

关键设计：

- RGB decoder 是轻量空间 decoder。
- state head 预测每个物体未来状态。
- collision head 输出 pairwise logits。
- pair_mask 排除 padding 和 self-edge。
- static object 的 state 由 decoder copy last observed state，避免静态物体被预测成乱动。

---

### T9 Loss

文件：

```text
models/loss.py
tests/test_loss.py
```

包含：

- RGB loss：L1 + MSE
- state loss：Smooth L1，只统计 dynamic valid object
- collision loss：Focal BCE
- mask loss：BCE + Dice
- LPIPS：接口保留，默认权重为 0

重要 mask 规则：

- padding object 不计入 state/mask/collision。
- static object 不计入 state loss。
- static-static pair 不计入 collision loss。
- 空 mask / 无 dynamic object 时返回 0，避免 NaN。

碰撞标签：

```python
force_raw = force_tgt * force_std + force_mean
collision_label = norm(force_raw) > threshold
```

也就是说：碰撞阈值基于原始力或反归一化力，不直接用 normalized force。

---

### T10 End-to-End Model

文件：

```text
models/physics_pred.py
tests/test_physics_pred.py
```

`PhysicsVideoPredictor` 串起：

```text
InputEncoder
  -> InteractionModule
  -> TemporalPredictor
  -> OutputDecoder
  -> PhysicsPredictionLoss
```

典型用法：

```python
from config import ModelConfig
from models.physics_pred import PhysicsVideoPredictor

cfg = ModelConfig.minimal_12gb()
model = PhysicsVideoPredictor(cfg).cuda()

pred = model(batch)
loss_dict = model.compute_loss(batch)
```

配置说明：

- `ModelConfig.minimal_12gb()`：面向 12GB GPU 的最小真实配置。
- `ModelConfig.tiny_test()`：单元测试用 tiny 配置，不代表真实训练配置。

---

### T11 Evaluation

文件：

```text
eval.py
evaluation_metrics.md
tests/test_eval.py
```

实现指标：

视频质量：

```text
mse
psnr
global ssim
```

物理状态：

```text
state_mse
position_mse
velocity_mse
```

碰撞检测：

```text
collision_precision
collision_recall
collision_f1
collision_tp / fp / fn
collision_pos_rate
```

特点：

- video/state 指标按 per-sample 聚合。
- scene 分组不会把 batch 均值复制给每个 scene。
- collision 指标按 TP/FP/FN 累加后再算 precision/recall/F1。
- evaluation 的 collision mask 与 loss 一致，会过滤 static-static pair。
- force_mean/std 优先从 dataloader.dataset 读取。

---

### T12 Inference and Visualization

文件：

```text
inference.py
visualization.py
tests/test_inference.py
```

`inference.py` 提供：

```python
load_model_from_checkpoint(checkpoint_path, device=None, config=None)
run_inference(model, batch, history_length=None, predict_length=None, device=None)
save_prediction_outputs(pred, metrics, output_path)
```

checkpoint 支持：

- `{'model_state_dict': ..., 'config': ...}`
- `{'state_dict': ..., 'config': ...}`
- raw tensor state_dict
- 自动 strip `module.` 前缀

注意：`torch.load(weights_only=False)` 只能加载可信 checkpoint。

`visualization.py` 提供：

```python
save_prediction_comparison(pred_rgb, target_rgb, output_path)
save_state_curves(pred_state, target_state, valid_mask, output_path)
save_collision_confusion(collision_logits, force_tgt, pair_mask, output_path)
```

可视化内容：

- 预测帧 vs GT 帧对比图。
- 物理状态曲线：position、velocity、force。
- 碰撞检测混淆矩阵。

---

## 4. 环境

请使用 `model` conda 环境。

当前已验证：

```text
torch 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

检查命令：

```bash
conda run -n model python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

不要再用旧的 `slotformer` 环境跑这个模型测试。

---

## 5. 常用命令

### 跑 T10 模型相关测试

```bash
conda run -n model pytest \
  model/tests/test_config.py \
  model/tests/test_encoder.py \
  model/tests/test_interaction.py \
  model/tests/test_temporal.py \
  model/tests/test_decoder.py \
  model/tests/test_loss.py \
  model/tests/test_physics_pred.py \
  -q
```

### 跑评估测试

```bash
conda run -n model pytest model/tests/test_eval.py -q
```

### 跑推理和可视化测试

```bash
conda run -n model pytest model/tests/test_inference.py -q
```

### 跑 T10-T12 全量测试

```bash
conda run -n model pytest \
  model/tests/test_config.py \
  model/tests/test_encoder.py \
  model/tests/test_interaction.py \
  model/tests/test_temporal.py \
  model/tests/test_decoder.py \
  model/tests/test_loss.py \
  model/tests/test_physics_pred.py \
  model/tests/test_eval.py \
  model/tests/test_inference.py \
  -q
```

---

## 6. 最小推理示例

```python
import torch
from inference import load_model_from_checkpoint, run_inference, save_prediction_outputs
from visualization import save_prediction_comparison, save_state_curves, save_collision_confusion

model = load_model_from_checkpoint("checkpoint.pt", device="cuda")

# batch 来自 PhysicsVideoDataset / DataLoader
result = run_inference(model, batch, device="cuda")

pred = result["pred"]
metrics = result["metrics"]

save_prediction_outputs(pred, metrics, "outputs/pred.pt")

# 如果 batch 有未来 GT，可做对比图
history_length = model.config.history_length
tp = model.config.predict_length
target_rgb = batch["rgb"][:, history_length:history_length + tp]
target_state = batch["dyn_state"][:, history_length:history_length + tp]
force_tgt = batch["force_matrix"][:, history_length:history_length + tp]

save_prediction_comparison(pred["rgb"].cpu(), target_rgb.cpu(), "outputs/rgb_compare.png")
save_state_curves(pred["state"].cpu(), target_state.cpu(), pred["valid_mask"].cpu(), "outputs/state_curves.png")
save_collision_confusion(pred["collision_logits"].cpu(), force_tgt.cpu(), pred["pair_mask"].cpu(), "outputs/collision_confusion.png")
```

---

## 7. 关键设计注意事项

### 7.1 static object 怎么处理

- static object 参与 interaction message passing。
- static object 可作为环境影响 dynamic object。
- static object 不计入 state loss。
- static-static pair 不计入 collision loss / eval。
- decoder 中 static object 的 state 复制最后一帧观测。

### 7.2 padding object 怎么处理

- `valid_mask [B,N]` 表示哪些 slot 是真实物体。
- padding object 的 token、输出、loss、eval 都要 mask 掉。
- pairwise 模块中 self-edge 和 padding pair 都被排除。

### 7.3 force_matrix 怎么处理

- 模型输入中 normalized force 可以作为 edge feature。
- 碰撞标签必须用原始 force 或反归一化 force：

```python
force_raw = force_tgt * force_std + force_mean
```

### 7.4 为什么不是纯 RGB 视频预测

因为这个数据集有强监督的物体状态、mask、force matrix。第一版目标是建立结构化物理预测 baseline，而不是只追求 RGB 画质。

### 7.5 为什么不用 GAN / Diffusion

第一版先保证：

- shape 稳定
- loss 可训练
- 物体身份明确
- state / collision 任务闭环
- 12GB GPU 可跑

GAN/Diffusion 后续可以作为 RGB 质量增强，不放在第一版 baseline 里。

---

## 8. 当前完成状态

已完成：

```text
T5 输入处理模块设计与实现
T6 交互建模模块设计与实现
T7 时序预测模块设计与实现
T8 输出解码模块设计与实现
T9 Loss 设计与实现
T10 模型端到端实现与测试
T11 评估模块实现与测试
T12 推理和可视化模块实现与测试
```

当前还没有完整训练脚本和真实 checkpoint 训练结果。也就是说，现在完成的是：

- 数据接口
- 模型结构
- loss
- eval
- inference
- visualization
- 单元测试闭环

下一步如果继续推进，建议优先做：

1. `train.py`：训练循环、checkpoint、日志。
2. 小数据 overfit：确认 loss 能下降。
3. val/test 评估脚本：调用 `evaluate_model`。
4. baseline 对比：RGB-only / 无 force / 无 interaction ablation。
5. 真实样本可视化报告。

---

## 9. 文件索引

```text
model/
  README.md                  # 本文件
  分工.md                    # 任务拆解与完成记录
  architecture.md            # 总体架构设计
  dataset.py                 # 数据集和 collate_fn
  config.py                  # 所有配置 dataclass 和常量
  eval.py                    # T11 评估指标与 evaluate_model
  evaluation_metrics.md      # T11 指标说明
  inference.py               # T12 推理/checkpoint/保存输出
  visualization.py           # T12 可视化

  design/
    input_module.md
    interaction_module.md
    temporal_module.md
    output_module.md
    loss_design.md

  models/
    encoder.py
    interaction.py
    temporal.py
    decoder.py
    loss.py
    physics_pred.py

  tests/
    test_dataset.py
    test_config.py
    test_encoder.py
    test_interaction.py
    test_temporal.py
    test_decoder.py
    test_loss.py
    test_physics_pred.py
    test_eval.py
    test_inference.py
```

---

## 10. 快速判断代码是否还健康

只要改了 T5-T12 相关代码，至少跑：

```bash
conda run -n model pytest model/tests/test_inference.py model/tests/test_eval.py -q
```

如果改了模型结构，跑全量：

```bash
conda run -n model pytest \
  model/tests/test_config.py \
  model/tests/test_encoder.py \
  model/tests/test_interaction.py \
  model/tests/test_temporal.py \
  model/tests/test_decoder.py \
  model/tests/test_loss.py \
  model/tests/test_physics_pred.py \
  model/tests/test_eval.py \
  model/tests/test_inference.py \
  -q
```

通过后再继续下一步。