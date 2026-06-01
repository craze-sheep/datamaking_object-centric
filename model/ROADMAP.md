# Model Optimization ROADMAP

## 当前主线状态

`ai_model/` 现在作为优化版主线维护，不再是只读 baseline。后续大改仍放到 `experiments/` 独立验证，验证通过后再考虑合入主线。

### 已合入 ai_model/

- [x] GAT-style 图注意力：GNN 聚合从均值聚合升级为 masked attention。
- [x] GNN 残差 + LayerNorm：稳定多层消息传递。
- [x] GNN 默认 3 层：增强多跳/链式物体交互建模。
- [x] FP16-safe attention mask：attention masked fill 从 `-1e9` 改为 `-1e4`，降低 AMP 溢出风险。
- [x] 时序 GRU 后接 history self-attention：保留 GRU 的轻量性，同时增强历史帧访问。
- [x] SSIM RGB loss：补充像素 L1/MSE 的结构约束。
- [x] Focal collision BCE + 有效 pair `pos_weight`：缓解碰撞类别不平衡。
- [x] Kendall 多任务不确定性加权：自动平衡 RGB/state/collision/mask 主损失。
- [x] 派生物理特征：PhysicsEncoder 拼接动能、势能、动量大小、角动量 proxy。
- [x] 能量守恒辅助 loss：惩罚预测未来帧的动能突变，已加入 `total_loss`。
- [x] Collision effect 辅助头/loss：碰撞头额外预测 pairwise force/effect 向量，已加入 `total_loss`。
- [x] 可选 LPIPS perceptual loss：默认 `lpips_weight=0.0`，安装 `lpips` 后可打开。
- [x] AMP、梯度累积、checkpoint resume、checkpoint config 保存。
- [x] TensorBoard 标量可视化：只保留训练/验证 loss，中文标签，`01 总损失` 排在最前。

## 当前真实训练

本次训练已经启动：

- 启动时间：`2026-05-31 23:19:54`
- 进程 PID：`37420`
- 训练数据：`/home/lzy/project/slot-datamaking/train`
- 验证/测试数据：`/home/lzy/project/slot-datamaking/test`
- TensorBoard 日志：`/home/lzy/project/slot-datamaking/model/runs/main_train_test_20260531_231954`
- 文本日志：`/home/lzy/project/slot-datamaking/model/runs/main_train_test_20260531_231954.log`
- Checkpoint：`/home/lzy/project/slot-datamaking/model/ai_model/checkpoints/main_train_test_20260531_231954`
- 当前 batch 策略：RTX 4060 Laptop 8GB 上实测 `batch_size=16`，AMP 开启，启动探测峰值显存约 `38.7%`；`batch_size=32` 会 OOM，因此正式训练固定上限为 `16`。
- TensorBoard 写入策略：训练第 1 step 立即写入所有 loss，之后每 `20` step 写一次；验证 loss 在每个 epoch 验证结束后写入。

查看训练是否还在运行：

```bash
ps -p 37420 -o pid,etime,pcpu,pmem,rss,cmd
```

查看训练日志：

```bash
tail -f /home/lzy/project/slot-datamaking/model/runs/main_train_test_20260531_231954.log
```

本次启动命令：

```bash
RUN_NAME=main_train_test_20260531_231954
LOG_DIR="/home/lzy/project/slot-datamaking/model/runs/$RUN_NAME"
CKPT_DIR="/home/lzy/project/slot-datamaking/model/ai_model/checkpoints/$RUN_NAME"
LOG_FILE="/home/lzy/project/slot-datamaking/model/runs/$RUN_NAME.log"

setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/lzy/miniconda3/envs/model/bin/python -u /home/lzy/project/slot-datamaking/model/ai_model/train.py \
  --mode train --epochs 10 \
  --train-root /home/lzy/project/slot-datamaking/train \
  --val-root /home/lzy/project/slot-datamaking/test \
  --batch-size 16 \
  --target-vram-fraction 0.82 \
  --max-batch-size 16 \
  --tb-log-every 20 \
  --log-dir "$LOG_DIR" \
  --checkpoint-dir "$CKPT_DIR" \
  > "$LOG_FILE" 2>&1 < /dev/null &
```

以后重新开一次正式训练，可以只改 `RUN_NAME`：

```bash
RUN_NAME=main_train_test_$(date +%Y%m%d_%H%M%S)
LOG_DIR="/home/lzy/project/slot-datamaking/model/runs/$RUN_NAME"
CKPT_DIR="/home/lzy/project/slot-datamaking/model/ai_model/checkpoints/$RUN_NAME"
LOG_FILE="/home/lzy/project/slot-datamaking/model/runs/$RUN_NAME.log"
mkdir -p /home/lzy/project/slot-datamaking/model/runs "$CKPT_DIR"

setsid env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /home/lzy/miniconda3/envs/model/bin/python -u /home/lzy/project/slot-datamaking/model/ai_model/train.py \
  --mode train --epochs 10 \
  --train-root /home/lzy/project/slot-datamaking/train \
  --val-root /home/lzy/project/slot-datamaking/test \
  --batch-size 16 \
  --target-vram-fraction 0.82 \
  --max-batch-size 16 \
  --tb-log-every 20 \
  --log-dir "$LOG_DIR" \
  --checkpoint-dir "$CKPT_DIR" \
  > "$LOG_FILE" 2>&1 < /dev/null &
```

从本次 checkpoint 继续训练：

```bash
/home/lzy/miniconda3/envs/model/bin/python -u /home/lzy/project/slot-datamaking/model/ai_model/train.py \
  --mode train --epochs 20 \
  --train-root /home/lzy/project/slot-datamaking/train \
  --val-root /home/lzy/project/slot-datamaking/test \
  --resume /home/lzy/project/slot-datamaking/model/ai_model/checkpoints/main_train_test_20260531_231954/best.pt \
  --log-dir /home/lzy/project/slot-datamaking/model/runs/main_train_test_resume \
  --checkpoint-dir /home/lzy/project/slot-datamaking/model/ai_model/checkpoints/main_train_test_resume
```

## TensorBoard 可视化

训练脚本默认写入 TensorBoard 日志：

```bash
/home/lzy/miniconda3/envs/model/bin/tensorboard \
  --logdir /home/lzy/project/slot-datamaking/model/runs \
  --port 6006 \
  --host localhost
```

浏览器打开：

```text
http://localhost:6006
```

新训练日志会使用中文标签。旧日志如果已经生成过，仍会保留英文标签；重新训练一次就会出现中文曲线。

可查看内容只保留 loss：

- `训练/*`：每个 epoch 的训练集平均 loss。
- `验证/*`：每个 epoch 的验证集平均 loss。

不会写入 `学习率`、`教师强制比例`、`模型参数量`、`每步 loss` 这些杂项，避免 TensorBoard 页面太乱。

### TensorBoard 标签含义

核心先看这几条：

- `01 总损失`：最重要的总指标，越低通常越好；如果训练降、验证升，可能过拟合。数字前缀用于让它在 TensorBoard 里排第一。

图像相关：

- `02 图像重建损失`：预测 RGB 画面和真实画面的差距，越低画面越接近。
- `03 结构相似度损失`：SSIM 对应的结构损失，越低说明画面轮廓/结构更像。
- `04 感知相似度损失`：LPIPS 对应的感知损失；默认 `lpips_weight=0.0` 时通常为 0，打开 LPIPS 后才有意义。
- `08 分割掩码损失`：预测物体 mask 和真实 mask 的差距，越低说明物体轮廓/位置更准。

物理相关：

- `05 物理状态损失`：位置、速度、旋转、力等低维状态预测误差，越低说明物体运动状态更准。
- `06 碰撞分类损失`：判断物体对是否发生碰撞的误差，越低越好；但碰撞很稀疏时不能只看这个，后续还需要 F1/Recall。
- `07 碰撞效果损失`：碰撞发生时预测力/效果方向的误差，越低说明模型更理解“碰了以后怎么动”。
- `09 能量守恒损失`：预测未来运动的动能变化惩罚，越低说明预测更平滑、更不容易乱加速。

建议查看顺序：

1. 先看 `验证/01 总损失`，判断整体有没有变好。
2. 再看 `验证/02 图像重建损失`、`验证/05 物理状态损失`、`验证/08 分割掩码损失`，判断是哪一部分在拖后腿。
3. 如果总损失突然变大，看 `训练/01 总损失` 和 `训练/09 能量守恒损失`，排查是否训练发散。
4. 如果打开 LPIPS，再看 `感知相似度损失`；没打开时不用管它。

常用参数：

```bash
# 指定日志和 checkpoint 目录
/home/lzy/miniconda3/envs/model/bin/python -u /home/lzy/project/slot-datamaking/model/ai_model/train.py \
  --mode train \
  --train-root /home/lzy/project/slot-datamaking/train \
  --val-root /home/lzy/project/slot-datamaking/test \
  --batch-size 16 \
  --max-batch-size 16 \
  --log-dir /home/lzy/project/slot-datamaking/model/runs/main_train \
  --checkpoint-dir /home/lzy/project/slot-datamaking/model/ai_model/checkpoints/main_train

# 临时关闭 TensorBoard
/home/lzy/miniconda3/envs/model/bin/python -u /home/lzy/project/slot-datamaking/model/ai_model/train.py \
  --mode smoke \
  --train-root /home/lzy/project/slot-datamaking/train \
  --val-root /home/lzy/project/slot-datamaking/test \
  --no-tensorboard
```

## 近期验证队列

这些是已经进入主线或低风险的思路，下一步重点做消融和权重调参。

### P0: 主线消融

- [ ] ablation_derived_physics — 对比开启/关闭派生物理特征。
- [ ] ablation_gnn_layers — 对比 2 层 vs 3 层 GNN，重点看链式碰撞场景。
- [ ] ablation_energy_weight — 扫描 `energy_weight`：`0, 0.001, 0.01, 0.05`。
- [ ] ablation_collision_effect — 扫描 `collision_effect_weight`：`0, 0.01, 0.05, 0.1`。
- [ ] ablation_lpips — 安装 `lpips` 后扫描 `lpips_weight`：`0, 0.01, 0.05`，重点看 RGB 感知质量。

### P1: 评估指标补齐

- [ ] metrics_psnr_ssim — 增加 RGB PSNR/SSIM 评估脚本。
- [ ] metrics_state_mse — 分位置、速度、角速度统计 state MSE。
- [ ] metrics_collision_f1 — 增加 collision precision/recall/F1。
- [ ] metrics_mask_iou — 增加 mask IoU/Dice 评估。
- [ ] metrics_fvd_optional — FVD 作为可选视频质量指标，先不作为训练依赖。

### P2: 训练策略

- [ ] curriculum_learning — 课程学习保留为独立实验；需要真正接入 sampler/epoch schedule，而不是只计算 complexity。
- [ ] gradient_checkpointing — 模型或 batch 放大后再启用。
- [ ] longer_horizon_eval — 增加 Tp=24 或滚动预测评估，观察误差累积。

## 暂缓的大分支

这些方向潜力大，但都需要接口重构和严格验证，暂不进入主线。

### Scheduled Sampling

- 当前状态：接口保留，默认禁用。
- 暂缓原因：不能直接喂未来 encoder/interacted tokens，否则会泄漏未来 RGB/mask/force/state。
- 后续实现要求：
  - 设计 decoder-feedback teacher path，只允许使用模型上一帧预测得到的 state/token。
  - 明确 teacher token 的来源和维度，不读取未来视觉/force 目标。
  - 增加泄漏检查：训练 forward 中不得访问 `batch[:, history_length:]` 的视觉/force 编码作为 teacher。

### DINOv2 Encoder

- 当前状态：作为 `experiments/exp008_dinov2_encoder` 原型保留。
- 暂缓原因：DINOv2 patch token 与当前 mask ROI pooling 接口不完全匹配，引入外部权重和显存开销。
- 后续实现要求：
  - 明确 `get_intermediate_layers` 输出是否包含 CLS token。
  - 将 patch token 恢复到稳定的 feature grid，再做 GT mask pooling。
  - 增加 fallback/cache，避免每次训练都在线下载权重。
  - 先冻结 DINOv2，只训练投影和后续模块。

### Mamba Temporal

- 当前状态：作为 `experiments/exp009_mamba_gru` 原型保留。
- 暂缓原因：`mamba_ssm` 依赖 CUDA 编译；当前原型对 history state 的条件解码不够扎实。
- 后续实现要求：
  - 明确没有 `mamba_ssm` 时是报错还是显式 fallback，不能静默变成 GRU。
  - 让 history encoding 真正条件化 future decoding。
  - 在 Tp=12 和更长 horizon 上分别比较，短序列未必有优势。

### Slot Attention / SAVi

- 当前状态：作为 `experiments/exp010_slot_attention` 原型保留。
- 暂缓原因：当前主线依赖 GT mask 对齐 object index；Slot 输出无序，直接拼物理状态会错配。
- 后续实现要求：
  - 增加 slot-to-object matching 或完全改为 slot-centric 数据流。
  - 加 mask reconstruction / diversity loss 防止 slot collapse。
  - 明确训练时是否仍使用 GT mask 监督，以及推理时如何摆脱 GT mask。

## 实验目录约定

```text
experiments/expNNN_name/
  model/              # 从当前 ai_model 复制，保持独立
  config_override.py  # 本实验参数
  run.sh              # 一键训练/评估
  README.md           # 设计说明、改动文件、风险
  result.md           # 指标和结论
```

要求：

- 每个实验必须能说明相对当前 `ai_model/` 改了什么。
- 新 loss 必须进入 `total_loss`，否则只能算诊断指标，不能称为训练优化。
- 新模块必须有 smoke forward/backward 验证。
- 大分支先在 `experiments/` 中跑通，再决定是否合入 `ai_model/`。
