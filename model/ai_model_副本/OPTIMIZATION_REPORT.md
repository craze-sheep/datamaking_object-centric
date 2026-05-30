# ai_model_副本优化落地报告

## 核实后的优化方向

### P0：已直接落地
- GAT 风格注意力聚合：替代 GNN mean 聚合，让接触/近邻关系有更高权重。
- GNN 残差 + LayerNorm：来自 Graph Networks / GAT 代码中的残差实践，提升多层消息传递稳定性。
- GRU 历史 self-attention：保留轻量 GRU，同时让最终隐藏态能访问全部历史帧。
- Scheduled Sampling：保留接口但默认禁用；当前不能喂未来 encoder token，否则会泄漏未来 RGB/mask/force。
- SSIM 辅助 RGB loss：补充 L1/MSE 对结构相似性的约束。
- Kendall 多任务不确定性权重：用可学习 log variance 自动平衡 RGB/state/collision/mask loss。
- Loss 尺度修正：RGB target 反归一化到 [0,1]；state/mask/collision 分母按真实元素数归一。
- Checkpoint 可移植性：config 改为纯 dict，并在 best checkpoint 中保存 optimizer state。

## 二审修复记录

- 修复 resume Blocking：训练脚本现在先读取 checkpoint config，再构建模型；resume 时跳过 auto-tune，避免模型尺寸被当前 mode 或 OOM shrink 改坏。
- 修复 checkpoint 完整性：保存/恢复 optimizer、scheduler、AMP scaler；epoch checkpoint 的 `best_val_loss` 在更新后写入。
- 修复 scheduled sampling 泄漏：`model.py` 不再把未来 `interacted` tokens 作为 teacher tokens 传入 temporal。
- 修复单物体/无邻居 GNN 更新：没有有效邻居的节点保持原 token，不再被 `GRUCell(0, old)` 和 residual norm 改写。
- 修复 RGB target 反归一化：固定执行 `rgb * 0.5 + 0.5`，不再用 `min() < 0` 猜测。
- 修复 collision class imbalance：`pos_weight` 现在基于有效 pair 计算并传入 BCE。
- 修复 RGB compositing padding leak：背景 `total_mask` 使用 `mask_prob_valid`。
- 修复梯度累积尾批：按 `min(max_steps, len(train_loader))` flush 最后不足一组的梯度。

### P1：建议作为下一轮实验
- Inception 多尺度 VisualEncoder：低风险，但会改变参数量和视觉特征分布，需要对比实验。
- 能量/速度平滑正则：可提升物理一致性，但要先确认数据中静态/碰撞阶段分布。
- CollisionHead 加相对状态/力辅助输入：当前 token 已含交互信息，增益需消融验证。
- Gradient checkpointing：当前 8GB smoke 可跑，只有放大模型或 batch 时再启用更合适。

### P2：暂不直接写入主路径
- DINOv2 冻结视觉特征：收益潜力大，但引入外部权重/网络依赖，且 ROI 特征接口要重做。
- Slot Attention / SAVi：会改变“使用 GT mask 作为监督和 ROI”的核心假设，应单独建实验分支。
- SlotFormer / 非自回归 temporal decoder / Mamba：属于时序主干替换，不适合和本轮稳定性修复混在一起。
- LPIPS / FVD / CLIP loss：依赖额外模型或评估协议，本轮不增加训练依赖。

## 已验证

- `python -m py_compile model/ai_model_副本/*.py`
- 合成 batch 前向 + backward 通过。
- 真实 database smoke train 通过：
  - `conda run -n model python model/ai_model_副本/train.py --mode smoke`
  - Val total loss: `0.6735`
  - Checkpoint: `model/ai_model_副本/checkpoints/best.pt`
- Resume 通过：
  - `conda run -n model python model/ai_model_副本/train.py --mode smoke --resume model/ai_model_副本/checkpoints/best.pt`

## 主要代码入口

- `config.py`：新增优化开关与 loss/training 配置。
- `interaction.py`：GNN attention aggregation 与 residual norm。
- `temporal.py`：GRU input projection 修正、history attention、scheduled sampling 输入。
- `loss.py`：SSIM、uncertainty weighting、RGB/分母尺度修正。
- `decoder.py`：collision pair mask 显式扩展到未来时间维。
- `train.py`：teacher ratio schedule、gradient accumulation、checkpoint 改进。
