# 实验报告：PhysicsObjectGraphPredictor

## 1. 模型设计概述

### 设计理念

不同于现有 baseline `PhysicsVideoPredictor`（Transformer encoder-decoder + dense pairwise MLP），
本模型采用 **物理感知的物体中心图神经网络** 架构，核心差异：

| 组件 | Baseline | 本模型 |
|------|----------|--------|
| 视觉编码 | CNN + ROI pooling（单一编码流） | CNN + ROI pooling + 独立物理编码流（双流融合） |
| 交互建模 | Dense pairwise MLP | Force-aware GNN（显式力矩阵作为图边特征） |
| 时序预测 | Transformer encoder-decoder | GRU 自回归解码（更轻量、天然适合序列物理） |
| 物理先验 | 隐式学习 | 显式编码质量/摩擦/几何属性 |

### 架构流程

```
输入 batch
  ↓
[PhysicsObjectEncoder] 双流编码
  ├── VisualEncoder: RGB → CNN → ROI pooling → 每物体视觉特征
  └── PhysicsEncoder: 静态属性(质量/摩擦/几何) + 动态状态(位置/速度) → 物理特征
  → 融合: concat + project → per-object tokens [B,T,N,D]
  ↓
[ForceAwareGNN] 力感知图网络（2层）
  ├── EdgeNetwork: 从 force_matrix + 相对位置/速度构建边特征（14维→64维）
  └── GNNSingleLayer: message(node_i, node_j, edge_ij) → GRU更新
  → 交互后 tokens [B,T,N,D]
  ↓
[TemporalGRU] 时序预测
  ├── Encode: GRU编码历史帧 → 最终隐状态
  └── Decode: 自回归生成未来帧 tokens [B,Tp,N,D]
  ↓
[MultiHeadDecoder] 多头解码
  ├── StateHead: MLP → 16维物理状态
  ├── CollisionHead: pairwise MLP → 碰撞logits
  ├── MaskHead: token → ConvTranspose → 128×128 mask
  └── RGBDecoder: 外观+mask合成 + 背景生成 → RGB图像
```

### 参数量

| 配置 | 参数量 | 用途 |
|------|--------|------|
| tiny (fused=32) | 132,032 | Smoke test |
| medium (fused=64) | 363,904 | 小规模训练 |
| small_8gb (fused=96) | 847,796 | 8GB GPU 满量训练（未测试完） |

## 2. 使用的数据字段

| 字段 | 来源 | 用途 |
|------|------|------|
| RGB 帧 | `dynamic/{frame}/{frame}.png` | 视觉编码器输入，RGB loss 监督目标 |
| 分割 mask | `object_segment/{id}.npz` | ROI pooling 的空间权重，mask loss 监督目标 |
| 静态属性 | `object_static.json` | 物理编码器输入（质量/摩擦/几何/类型/静态标记） |
| 动态状态 | `object_dynamicjson/{id}.json` | 物理编码器输入 + GNN边特征（位置/速度提取） |
| 力矩阵 | `force_matrix.json` | GNN边特征（直接作为力信号），碰撞标签生成 |
| 视角元数据 | `video.json` | 数据加载使用（未直接输入模型） |

### 属性编码维度 (attr_dim=14)

```
[0:3]   size (3D尺寸)
[3:6]   lateralFriction, rollingFriction, spinningFriction
[6]     restitution (恢复系数)
[7]     mass (质量)
[8]     static flag (是否静态)
[9:13]  object_type one-hot (ground/sphere/box/cylinder)
```

### 状态编码维度 (state_dim=16)

```
[0:3]   position (世界坐标)
[3:7]   quaternion (姿态四元数)
[7:10]  velocity (线速度)
[10:13] angular_velocity (角速度)
[13:16] resultant force (合力)
```

## 3. 暂时未使用的字段

| 字段 | 原因 | 后续计划 |
|------|------|----------|
| 深度图 `{id}.npz` | 题目要求暂不考虑 | 可作为额外监督信号或 3D 特征输入 |
| `.mp4` 视频文件 | 已使用单帧 PNG，mp4 冗余 | 视频编码器（如 SlowFast）可直接用 mp4 |
| `scene_id` / `sample_id` | 当前模型无场景特化模块 | 可用于场景级条件化或 curriculum learning |
| `subtask` / `main_variable` | 字符串字段，未编码 | 可用 text embedding 注入任务先验 |
| 相机参数 (`position`, `focal_length`) | 单视角固定，暂无多视角模块 | 多视角融合时可编码视角信息 |
| 角速度 (`angular_velocity`) | 已编码在状态中但未单独建模 | 旋转预测需要更精细的刚体动力学模块 |

## 4. 训练结果

### 环境

- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB VRAM)
- CUDA: 12.1
- PyTorch: 2.5.1+cu121
- Conda env: `model`

### Smoke Test（tiny config，真实数据）

```
Config: fused_dim=32, gru=32, gnn=32, history=4, predict=4
Batch size: 2, Parameters: 132,032
Train samples: 22,315 | Val samples: 4,783
Steps: 3 | Loss: 13809 → 13642 (下降)
Checkpoint: model/ai_model/checkpoints/smoke_test.pt
```

### 小规模训练（medium config，50 steps）

```
Config: fused_dim=64, gru=64, gnn=64, history=6, predict=6
Batch size: 4, Parameters: 363,904
Loss weights: rgb=1.0, state=1.0, coll=1.0, mask=0.001

Step  1: total=71.46  rgb=1.232  state=6.549  coll=0.897
Step 50: total=60.66  rgb=0.446  state=0.121  coll=0.032

收敛:
  - state loss:   6.55 → 0.12 (↓ 98%)
  - collision loss: 0.90 → 0.03 (↓ 97%)
  - rgb loss:     1.23 → 0.45 (↓ 64%)

Peak GPU: 205 MB
Checkpoint: model/ai_model/checkpoints/balanced_train.pt
```

### 关键验证

1. ✅ 使用真实 database/ 数据（22,315 train samples），非 mock
2. ✅ GPU 训练（RTX 4060 Laptop, CUDA 12.1）
3. ✅ Forward + Loss + Backward + Optimizer step 完整
4. ✅ Loss 收敛（state ↓98%, collision ↓97%, rgb ↓64%）
5. ✅ Checkpoint 保存（含 model_state_dict + config）
6. ✅ OOM 自动降 batch size 机制已实现

## 5. 已知问题

1. **Mask loss 量级问题**：原始 mask loss（BCE + Dice）约 40K-60K，远大于其他 loss。
   即使 weight=0.001 仍贡献 ~60。原因：128×128 mask 的像素级 BCE 天然量级大。
   **修复方向**：使用 focal loss 替代 BCE，或降采样 mask 到 32×32 计算 loss。

2. **模型容量**：当前 363K 参数的小模型不足以充分利用 32K 样本。
   GPU 只用了 205MB / 8GB，大幅未饱和。
   **修复方向**：增大 fused_dim 到 128-256，加入预训练视觉 backbone。

3. **RGB 解码质量**：当前 RGB decoder 采用"外观+mask 合成"方式，表达能力有限。
   **修复方向**：用更强的 decoder（如 U-Net）或 diffusion-based 生成。

4. **自回归误差累积**：GRU 自回归解码时，预测误差会累积到后续帧。
   **修复方向**：加入 teacher forcing + scheduled sampling。

## 6. 后续迭代计划

### 短期（1-2 周）

1. 修复 mask loss 量级问题，使用降采样或 focal loss
2. 增大模型到 ~1M 参数，测试 batch_size=2 在 8GB 上的极限
3. 跑完整 1-2 epoch 训练，记录 train/val loss 曲线
4. 加入评估指标（PSNR, SSIM, state MSE, collision F1）

### 中期（2-4 周）

5. 引入 DINOv2 预训练视觉特征（baseline 已有接口）
6. 加入深度图作为额外监督信号
7. 多视角训练（利用 5 视角数据）
8. Curriculum learning：先简单场景 S1-S4，再复杂 S5-S8

### 长期

9. 扩散模型用于视频预测
10. 物理约束损失（能量守恒、动量守恒）
11. Few-shot 泛化到新物体类型
