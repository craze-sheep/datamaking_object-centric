# 物理视频预测模型总体架构设计

> 任务：T4 架构总体设计
> 数据集：31,880条样本，S1-S8八类物理场景，128x128，36帧
> 输入模态：RGB、GT分割mask、物体静态属性、动态状态、力矩阵
> 第一版目标：预测未来RGB帧、物理状态、碰撞事件
> 第一版暂不使用：深度图、多视角融合、扩散/GAN生成器、端到端无监督物体发现

---

## 1. 设计结论

第一版采用“物体中心 + 显式物理交互 + 时序预测 + 多头解码”的混合架构：

```text
DataLoader batch
  ├─ rgb:          [B,T,3,128,128]
  ├─ mask:         [B,T,N,128,128]
  ├─ obj_attrs:    [B,N,14]
  ├─ dyn_state:    [B,T,N,16]
  ├─ force_matrix: [B,T,N,N,3]
  └─ valid_mask:   [B,N]
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ T5 Input Encoder                                                 │
│  RGB → frozen/light CNN features → mask ROI pooling              │
│  attrs/state → MLP projection                                    │
│  object token z[t,n] = fuse(visual, attr, state, scene/time/view) │
└─────────────────────────────────────────────────────────────────┘
        │ object tokens [B,T,N,D]
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ T6 Interaction Module                                            │
│  per-frame object graph                                          │
│  node: object token + static flag + valid mask                   │
│  edge: force_matrix[t,i,j] + relative state                      │
│  message passing updates all valid object tokens; static state loss is masked │
└─────────────────────────────────────────────────────────────────┘
        │ interaction-aware tokens [B,T,N,D]
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ T7 Temporal Predictor                                            │
│  history 12 frames → predict 12 future latent object states       │
│  default: object-time Transformer encoder/decoder                │
│  optional later: Mamba/SSM替换时间模块                            │
└─────────────────────────────────────────────────────────────────┘
        │ future object latents [B,Tp,N,D]
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ T8 Output Heads                                                  │
│  RGB decoder: object-conditioned spatial decoder                 │
│  physics head: per-object future state [B,Tp,N,16]               │
│  collision head: pairwise/event logits [B,Tp,N,N]                │
│  optional auxiliary mask head [B,Tp,N,128,128]                   │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
T9 Loss: RGB重建 + 物理状态 + 碰撞Focal + 可选mask/感知loss
```

核心取舍：第一版不追求最复杂的视频生成质量，而优先保证物体身份、力矩阵、物理状态和碰撞检测能被结构化使用。这样可以直接利用当前数据集最强的监督信号，避免把任务退化成纯RGB视频预测。

---

## 2. 设计目标与成功标准

### 2.1 模型要解决什么

给定历史12帧：

```python
rgb_hist          [B, 12, 3, 128, 128]
mask_hist         [B, 12, N, 128, 128]
obj_attrs         [B, N, 14]
dyn_state_hist    [B, 12, N, 16]
force_hist        [B, 12, N, N, 3]
valid_mask        [B, N]
```

预测未来12帧：

```python
pred_rgb          [B, 12, 3, 128, 128]
pred_state        [B, 12, N, 16]
pred_collision    [B, 12, N, N]       # pairwise collision/contact probability
pred_mask optional[B, 12, N, 128, 128]
```

### 2.2 第一版成功标准

1. 能在单张12GB GPU上训练最小配置，batch size下限为2；24GB GPU上目标batch size为4-8。
2. forward pass shape稳定，支持N<=7的padding物体。
3. RGB、状态、碰撞三个任务都能输出并计算loss。
4. 静态物体不被当成需要预测运动的动态物体，但仍可作为施力/碰撞环境参与交互。
5. 架构模块边界清晰，T5-T9可分别实现和测试。
6. 可跑baseline对比：SimVP/PredRNN作为RGB-only baseline，PhyDNet作为需适配数据格式的physics-aware RGB baseline。

### 2.3 第一版明确做/不做

第一版做：

- 单视角、单窗口预测：历史12帧预测未来12帧。
- 使用GT mask做物体ROI pooling。
- 使用静态属性、动态状态、力矩阵。
- 使用dense pairwise message passing建模物体交互。
- 使用非自回归Transformer一次性预测未来12帧latent。
- 默认启用state head和collision head。
- 默认启用轻量mask head作为RGB解码辅助，但mask loss权重可设为0或很小。

第一版不做：

- 不使用深度图作为训练输入，但预留depth输入/输出接口。
- 不做多视角融合；若数据加载器没有`view_id`，模型默认所有样本`view_id=0`。
- 不做Slot Attention/SAVi无监督物体发现。
- 不做Diffusion/GAN。
- 不做24帧或36帧长期rollout；长期预测留给T7后续扩展。

---

## 3. 为什么选这个架构

### 3.1 优先使用GT mask而不是Slot Attention/SAVi做物体发现

数据集已经有每帧每物体分割mask，且目标是物理视频预测，不是无监督分割。第一版直接用GT mask做ROI pooling：

优点：
- 物体身份和空间区域直接可用，降低训练难度。
- 避免Slot Attention在36帧、N<=7、多场景上的不稳定收敛。
- 后续如果要研究无GT mask版本，可以把GT mask ROI替换成Slot/SAVi模块。

保留接口：`ObjectFeatureExtractor`只要求输出`[B,T,N,Dv]`，内部可以从GT mask ROI切换到Slot Attention。

### 3.2 使用显式GNN而不是把力矩阵直接concat进Transformer

力矩阵天然是物体对物体的关系：`[B,T,N,N,3]`。如果直接flatten concat到时序模型，会丢失“谁对谁施力”的结构。GNN/message passing更贴合Interaction Networks和Graph Network Simulator的思想。

优点：
- edge feature就是力向量和相对状态，语义清晰。
- 可通过`valid_mask`过滤padding物体。
- 可区分静态物体对动态物体施加影响，但静态物体自身不预测运动。

### 3.3 时序模块首选Transformer，Mamba作为后续替换选项

当前历史12帧、预测12帧，总序列不长：`T*N <= 24*7 = 168`个object-time token。Transformer的计算量可控，且实现和调试简单。

选择Transformer的理由：
- 易于处理object-time token的全局依赖。
- 能同时建模同一物体跨时间、同一时间跨物体的关系。
- 测试和shape验证简单。

暂不首选Mamba的理由：
- 当前序列长度不长，Mamba的线性长序列优势不明显。
- mamba-ssm安装和CUDA兼容可能增加实现风险。
- 可在T7中保留`TemporalPredictor`接口，后续替换。

### 3.4 第一版不用Diffusion/GAN

本任务不仅要视频质量，还要物理状态和碰撞检测。Diffusion/GAN会显著增加训练成本和调试复杂度，不利于先建立可验证baseline。第一版用确定性解码器和监督loss，先保证多模态任务闭环。

---

## 4. 总体模块划分

### 4.1 文件划分

```text
model/models/
  encoder.py       # T5: RGB/mask ROI、属性、动态状态编码与融合
  interaction.py   # T6: 图交互/message passing
  temporal.py      # T7: object-time Transformer预测未来latent
  decoder.py       # T8: RGB/state/collision/mask输出头
  loss.py          # T9: 多任务loss
  physics_pred.py  # 整体模型包装 PhysicsVideoPredictor
```

### 4.2 整体forward接口

```python
class PhysicsVideoPredictor(nn.Module):
    def forward(self, batch):
        # batch from PhysicsVideoDataset.collate_fn
        # use first history_length frames as context
        return {
            'rgb': pred_rgb,                  # [B,Tp,3,128,128]
            'state': pred_state,              # [B,Tp,N,16]
            'collision_logits': pred_coll,    # [B,Tp,N,N]
            'mask': pred_mask,                # optional [B,Tp,N,128,128]
        }
```

输入中的未来GT只用于loss，不进入模型预测路径。训练时由loss函数从`batch`中切分target：

```python
rgb_target       = batch['rgb'][:, Th:Th+Tp]
state_target     = batch['dyn_state'][:, Th:Th+Tp]
force_target     = batch['force_matrix'][:, Th:Th+Tp]
mask_target      = batch['mask'][:, Th:Th+Tp]
```

---

## 5. 输入输出shape表

符号：

| 符号 | 含义 | 第一版默认 |
|------|------|------------|
| B | batch size | 12GB最小2，24GB目标4-8 |
| T | 总窗口长度 | 24 |
| Th | 历史帧数 | 12 |
| Tp | 预测帧数 | 12 |
| N | padding后最大物体数 | 7 |
| H,W | 图像大小 | 128,128 |
| C,h,w | CNN feature map | 256,32,32（默认轻量CNN） |
| Dv | 视觉特征维度 | 256 |
| Dp | 物理特征维度 | 128 |
| De | 边特征维度 | 16 = Fij3 + Fji3 + rel_pos3 + rel_vel3 + static2 + dist1 + force_norm1 |
| D | 融合token维度 | 默认256；minimal_12gb配置可降到128 |

| 阶段 | 张量 | Shape | 说明 |
|------|------|-------|------|
| DataLoader | rgb | [B,T,3,128,128] | normalize=True时约[-1,1]；normalize=False时[0,1] |
| DataLoader | mask | [B,T,N,128,128] | 0/1 mask |
| DataLoader | obj_attrs | [B,N,14] | 静态属性；当前第13维保留/未使用 |
| DataLoader | dyn_state | [B,T,N,16] | 默认normalized state |
| DataLoader | force_matrix | [B,T,N,N,3] | 默认normalized force；碰撞标签需用原始或反归一化力 |
| DataLoader | valid_mask | [B,N] | True为真实物体 |
| DataLoader optional | view_id | [B] | 当前dataset无该字段，缺省全0 |
| DataLoader future | depth | [B,T,1,128,128] | 第一版不用，未来扩展 |
| Encoder | feat_map | [B,Th,C,32,32] | 默认轻量CNN输出 |
| Encoder | resized_mask | [B,Th,N,32,32] | ROI pooling用 |
| Encoder | visual_feat | [B,Th,N,Dv] | mask ROI后物体视觉特征 |
| Encoder | phys_feat | [B,Th,N,Dp] | attr/state编码 |
| Encoder | object_token | [B,Th,N,D] | 融合后的物体token |
| Interaction | edge_feat | [B,Th,N,N,De] | Fij/Fji/相对状态/static标记 |
| Interaction | message | [B,Th,N,N,D] | pairwise message；默认排除自环聚合 |
| Interaction | inter_token | [B,Th,N,D] | 交互更新后的token |
| Temporal | flat_hist | [B,Th*N,D] | Transformer memory，padding token mask掉 |
| Temporal | future_query | [B,Tp*N,D] | learned [Tp,N,D] query broadcast到B |
| Temporal | future_token | [B,Tp,N,D] | 未来物体latent |
| Decoder | pred_rgb | [B,Tp,3,128,128] | RGB预测，数值空间需匹配target |
| Decoder | pred_state | [B,Tp,N,16] | normalized物理状态预测；静态物体默认copy last observed state |
| Decoder | pred_collision | [B,Tp,N,N] | pairwise碰撞logits |
| Decoder default | pred_mask | [B,Tp,N,128,128] | 轻量辅助mask预测/空间解码辅助 |
| Decoder future optional | pred_depth | [B,Tp,1,128,128] | 深度预测预留接口 |
| Loss mask | dynamic_mask | [B,N] | valid且非静态物体 |
| Loss mask | collision_pair_mask | [B,Tp,N,N] | valid_i & valid_j & i!=j |

---

## 6. 模块职责

### 6.1 T5 输入处理模块

职责：把原始多模态输入变成统一object token。

子组件：

1. `VisualEncoder`
   - 输入：`rgb_hist [B,Th,3,128,128]`
   - 输出：`feat_map [B,Th,C,h,w]`
   - 第一版建议：轻量CNN默认实现；DINOv2作为可选冻结特征抽取器。
   - 理由：DINOv2特征强，但128x128输入、安装/显存和batch速度需要实际验证。为了T10能稳定落地，先提供轻量CNN路径，保留DINOv2接口。

2. `MaskROIPool`
   - 输入：`feat_map`和`mask_hist [B,Th,N,128,128]`
   - 输出：`visual_feat [B,Th,N,Dv]`
   - 方法：mask resize到feature map大小，对每个物体区域做masked average pooling。
   - 边界：空mask或padding物体输出0向量。

3. `PhysicalEncoder`
   - 输入：`obj_attrs [B,N,14]`、`dyn_state_hist [B,Th,N,16]`
   - 输出：`phys_feat [B,Th,N,Dp]`
   - 方法：属性MLP、状态MLP分别编码后concat/project。

4. `TokenFusion`
   - 输入：visual、physical、time embedding、scene embedding可选
   - 输出：`object_token [B,Th,N,D]`
   - 需加入`valid_mask`，padding物体token置0或后续attention mask掉。

### 6.2 T6 交互建模模块

职责：用物体图建模每帧内物体间交互。

图定义：
- 节点：每个真实物体`n`的object token。
- 边：`i -> j`，第一版不假设原始`force_matrix[i,j]`方向语义一定可靠，边特征同时使用`F_ij`和`F_ji`：
  - `force_matrix[t,i,j] [3]`
  - `force_matrix[t,j,i] [3]`
  - 相对位置：`pos_j - pos_i [3]`
  - 相对速度：`vel_j - vel_i [3]`
  - 静态/动态标记：`static_i, static_j`

T6实现前需要用少量样本核对`force_matrix`方向：确认JSON中`matrix[i][j]`表示“i对j的力”还是“j对i的力”。若仍不确定，保留双向力特征是默认安全方案。

message passing：

```text
m_ij = EdgeMLP([z_i, z_j, edge_ij])
a_j  = aggregate_i(m_ij, mask=valid_i & valid_j)
z'_j = NodeMLP([z_j, a_j])
```

静态物体策略：
- 静态物体参与发送message，例如ground/wall影响球/箱子。
- 静态物体的`pred_state`默认直接copy最后一帧观测状态，不由state head自由预测；动态物体才由state head预测。
- 状态主loss只计算`dynamic_mask = valid_mask & (~static_flag)`；静态状态可作为监控项低权重记录。
- 静态物体token可以被更新用于视觉解码，但RGB decoder使用的静态空间信息应来自最后观测mask/state，避免ground/wall漂移。

力为0的边：
- 第一版保留所有有效物体pair的边，用edge MLP学习零力/无接触关系。
- message passing默认排除自环`i==j`，自身信息由NodeMLP residual保留。
- 后续可做稀疏图优化：只保留非零力或近邻边。

### 6.3 T7 时序预测模块

职责：从历史object token预测未来object token。

第一版方案：object-time Transformer。

输入组织：

```python
x = inter_token.reshape(B, Th*N, D)
```

位置编码：
- 第一版采用标准encoder-decoder Transformer结构：history tokens作为encoder memory，learned future queries作为decoder query并cross-attend历史memory。
- history token加入time embedding和object id embedding。
- future query加入future time embedding和object id embedding。
- scene embedding默认关闭；若启用必须先修复稳定scene_id映射。

预测方式：
- 第一版采用非自回归一次性预测全部未来12帧latent。
- 用learned future query `[Tp,N,D]`作为decoder query，broadcast成`[B,Tp*N,D]`后cross-attend历史tokens。
- future query必须包含未来time embedding和object id embedding。

理由：
- 训练稳定，不需要teacher forcing。
- 避免自回归误差累积先影响第一版闭环。
- 推理速度快。

局限：
- 非自回归对24/36帧长期一致性可能弱。
- 若后续预测长度扩展到24帧以上，再加入自回归rollout、scheduled sampling或Mamba时序模块。

### 6.4 T8 输出解码模块

职责：把未来object latent解码为RGB、物理状态、碰撞事件。

1. `StateHead`
   - MLP：`[B,Tp,N,D] -> [B,Tp,N,16]`
   - padding物体和静态物体loss用mask处理。

2. `CollisionHead`
   - pairwise MLP或bilinear：`z_i,z_j,relative_state -> logit`
   - 输出：`[B,Tp,N,N]`
   - label来源：future force_matrix的范数是否超过阈值。
   - 对角线`i==j`和padding pair不计loss。

3. `RGBDecoder`
   - 第一版采用object-conditioned spatial decoder，而不是只把所有object latent聚合成一个frame latent。
   - 最小可实现路径：
     1. `StateHead`先输出未来动态物体state，静态物体state copy最后观测值。
     2. 用`future_token + pred_state(position/velocity) + object_id`生成每物体低分辨率spatial feature或heatmap `[B,Tp,N,Cs,32,32]`。
     3. `MaskHead`输出每物体mask logits `[B,Tp,N,128,128]`，作为空间约束和辅助监督。
     4. appearance head输出每物体appearance feature，经alpha/object compose聚合到frame feature，再上采样为RGB。
   - 若T8实现阶段为了快速跑通需要降级，可临时使用frame-latent decoder作为弱baseline，但必须在文档和实验名中标记为`rgb_weak_decoder`，不能作为最终主方案。
   - RGB target的数值范围必须匹配DataLoader归一化：normalize=True时预测[-1,1]空间或在loss前统一变换。

4. `MaskHead`（第一版默认保留，可低权重）
   - 输出未来mask `[B,Tp,N,128,128]`。
   - 作用：给RGB decoder提供空间布局约束，让object latent保持空间语义，改善RGB解码和碰撞定位。

### 6.5 T9 Loss设计接口

多任务loss：

```text
L = λ_rgb * L_rgb
  + λ_state * L_state
  + λ_collision * L_collision
  + λ_mask * L_mask(default low weight)
  + λ_perceptual * L_lpips(default 0)
```

初始权重建议：

```python
lambda_rgb = 1.0
lambda_state = 0.1
lambda_collision = 0.5
lambda_mask = 0.1
lambda_lpips = 0.0
```

这些只是T9/T10起点，最终需要按各loss实际量级调参。

默认建议：
- `L_rgb`: L1或MSE，第一版先L1/MSE；计算空间必须与`pred_rgb`和target一致。
- `L_state`: SmoothL1，只对valid且动态物体计入主loss；模型输出默认为normalized state，若需要物理量指标则在评估时反归一化。
- `L_collision`: Focal BCE，处理碰撞稀疏和类别不平衡。
  - label由未来`force_matrix`构造，但必须使用原始力或反归一化力，不直接用normalized force默认阈值。
  - 初始阈值建议：`||force|| > 1e-6`视为接触/碰撞；若噪声导致假阳性，再按训练集非零力分布选P5/P10阈值。
  - pair mask：`valid_i & valid_j & i != j`；是否统计静态pair由T9详细设计决定，但默认保留动态-静态pair，因为ground/wall接触重要。
- `L_mask`: BCE/Dice，默认作为低权重辅助。
- `L_lpips`: 训练稳定后再加；第一版默认关闭。

---

## 7. 与已有方案/代码库的关系

| 来源 | 如何使用 | 第一版决策 | 为什么不直接照搬 |
|------|----------|------------|----------------|
| DINOv2 | 冻结视觉特征抽取 | 保留接口，可选；默认轻量CNN保证可跑 | 128x128、小batch和安装/显存风险需先验证 |
| Slot Attention/SAVi | 无监督物体分解 | 不作为第一版主路径，因为已有GT mask | 无监督发现会增加训练不稳定性，且当前任务重点不是分割 |
| SlotFormer | object-centric时序预测思想 | 借鉴object token + Transformer，不照搬无监督部分 | 原方法只用视觉slot，不利用状态/力矩阵 |
| Interaction Networks/GNS | 物体交互message passing | 作为T6核心思想 | 粒子/图模拟器不直接输出RGB视频，需要结合视觉decoder |
| PyTorch Geometric | 图网络实现库 | 可用，但N<=7时手写dense message passing更简单 | PyG batch图构造会增加复杂度，dense N*N足够便宜 |
| SimVP/PredRNN | RGB-only baseline | 用于RGB预测对比，不作为主架构 | 不使用物体属性、状态、力矩阵和碰撞监督 |
| PhyDNet | 物理感知视频预测baseline | 参考loss/物理解耦思想，需适配数据 | 原接口不是object-centric，也不直接支持force matrix |
| Mamba | 高效时序模块 | 后续替换T7，不作为第一版必需依赖 | 当前序列短，Transformer更易实现和调试 |
| LPIPS | 感知loss | 后续可加，第一版可关闭 | 增加显存和训练不稳定风险，先跑通监督闭环 |

关键原则：优先复用思想和成熟组件，但不硬套外部repo的数据结构。T10实现应保持本项目接口干净。

---

## 8. 计算资源预算

### 8.1 显存控制策略

第一版默认配置：

```yaml
image_size: 128
history_length: 12
predict_length: 12
max_objects: 7
hidden_dim: 256
visual_dim: 256
num_interaction_layers: 2
num_temporal_layers: 4
num_attention_heads: 4
rgb_decoder_base_channels: 128
use_dinov2: false
use_lpips: false initially
```

预算考虑：
- 目标硬件分两档：12GB GPU必须能跑最小配置；24GB GPU跑默认配置。
- 粗略参数量目标：默认配置控制在15M-35M trainable parameters；启用DINOv2时DINOv2冻结，不计入可训练参数。
- object-time token数量小，Transformer不是主要瓶颈：`Th*N=84`历史token，`Tp*N=84`未来query。
- 主要显存来自RGB/mask decoder activation和LPIPS/DINOv2特征图。
- 如果启用DINOv2，应冻结参数并在`torch.no_grad()`下提取特征，必要时缓存视觉特征。

粗略显存预算（需要T10用真实profile校准）：

| 配置 | hidden_dim | layers | mask head | DINOv2 | LPIPS | 目标显存 | batch |
|------|------------|--------|-----------|--------|-------|----------|-------|
| minimal_12gb | 128 | GNN1 + Transformer2 | on low-weight | off | off | <=12GB | 2 |
| default_24gb | 256 | GNN2 + Transformer4 | on | off | off | <=24GB | 4-8 |
| quality_24gb+ | 256/384 | GNN2-3 + Transformer4-6 | on | optional frozen | optional | >24GB可能需要accumulation | 2-4 |

降级顺序：
1. 关闭LPIPS/DINOv2。
2. batch size降到2。
3. `hidden_dim`从256降到128，temporal layers从4降到2。
4. mask head只保留低分辨率辅助或暂时只算state/collision。
5. 若仍OOM，先训练state+collision，RGB decoder单独调试。

### 8.2 训练建议

初始训练配置：
- batch size：从4开始，OOM则降到2。
- AMP混合精度：开启。
- gradient accumulation：需要大等效batch时开启。
- 先训练短预测/低loss组合：只开RGB+state，稳定后加入collision/mask/LPIPS。

---

## 9. 多视角和深度图扩展策略

### 9.1 多视角

当前T3未把多视角作为主输入展开。第一版按单视角样本训练，但接口上保留`view_id`：

```python
view_id = batch.get('view_id', torch.zeros(B, dtype=torch.long, device=device))
view_embedding: [num_views, D]
```

第一版约定：
- 当前`dataset.py`没有`view_id/camera_id`字段，因此模型必须把缺省view视为0。
- 单视角输入shape保持`[B,T,3,H,W]`，不额外增加V维。
- 不做跨视角一致性loss。

后续多视角扩展shape约定：

```python
rgb:   [B,V,T,3,H,W]
mask:  [B,V,T,N,H,W]
depth: [B,V,T,1,H,W] optional
view_id: [B,V]
```

多视角融合策略：
- early fusion：每个视角独立ROI pooling得到`[B,V,T,N,Dv]`，加入view embedding后对同一物体做attention/mean聚合成`[B,T,N,Dv]`。
- late fusion：每视角独立预测，再用一致性loss约束state/collision。
- 前置假设：同一样本内物体id/order跨视角一致；若不一致，DataLoader必须提供object remapping。

第一版不做多视角融合，避免数据加载器和模型复杂度同时扩大。

### 9.2 深度图

深度图第一版暂不用，但接口上预留输入和输出：

```python
depth_hist = batch.get('depth', None)  # future extension: [B,T,1,128,128]
pred_depth optional: [B,Tp,1,128,128]
```

扩展方式：
- `DepthEncoder`与RGB encoder并行，输出`depth_feat_map [B,Th,Cd,h,w]`。
- 与RGB feature map concat后project，再进入同一个MaskROIPool。
- 深度输入归一化策略必须在DataLoader中定义，建议按有效深度范围缩放到[0,1]或标准化。
- 若未来需要预测depth，T8增加`DepthHead`，loss使用L1/SmoothL1并mask无效深度。

第一版不读取depth，`depth_hist=None`时所有模块必须正常工作。

---

## 10. 接口约定

### 10.1 batch切分

所有模块只接收历史帧作为模型输入：

```python
Th = config.history_length
Tp = config.predict_length
rgb_hist   = batch['rgb'][:, :Th]
mask_hist  = batch['mask'][:, :Th]
state_hist = batch['dyn_state'][:, :Th]
force_hist = batch['force_matrix'][:, :Th]
```

目标由loss函数内部切出：

```python
rgb_tgt   = batch['rgb'][:, Th:Th+Tp]
state_tgt = batch['dyn_state'][:, Th:Th+Tp]
force_tgt = batch['force_matrix'][:, Th:Th+Tp]
mask_tgt  = batch['mask'][:, Th:Th+Tp]
```

### 10.2 mask和身份约定

- `valid_mask [B,N]`用于所有object维度计算。
- `dynamic_mask = valid_mask & (~static_flag)`用于状态预测主loss。
- collision pair mask：`valid_i & valid_j & (i != j)`。
- padding物体的输出可以存在，但loss必须mask掉。
- 关键假设：同一样本内object slot/object id跨时间一致。T7的object id embedding和temporal reshape都依赖该假设。
- 若后续发现mask/object_dynamicjson编号跨帧不一致，必须先在DataLoader中做object remapping，不能在模型里隐式修正。

### 10.3 静态属性中的static flag

当前`obj_attrs[:, :, 8]`是静态标记，T10中不要硬编码太多位置，建议在`config.py`定义：

```python
ATTR_STATIC_INDEX = 8
STATE_POS_SLICE = slice(0, 3)
STATE_QUAT_SLICE = slice(3, 7)
STATE_VEL_SLICE = slice(7, 10)
STATE_ANGVEL_SLICE = slice(10, 13)
STATE_FORCE_SLICE = slice(13, 16)
```

### 10.4 scene_id约定

当前`dataset.py`使用`hash(scene) % 8`生成`scene_id`，Python hash跨进程可能不稳定。架构层面约定：

- 第一版默认不依赖scene embedding，避免不稳定scene_id影响训练复现。
- 如果T10要启用scene embedding，必须先把dataset改为稳定映射：`S1->0, S2->1, ..., S8->7`。
- 配置中保留`use_scene_embedding: false`作为默认值。

---

## 11. 风险与缓解

| 风险 | 影响 | 缓解 |
|------|------|------|
| RGB decoder输出模糊 | 视频质量差 | 第一版采用object-conditioned spatial decoder + mask辅助；弱frame decoder只能作临时baseline |
| force_matrix稀疏导致collision类别不平衡 | 碰撞检测召回低 | 使用Focal Loss、正负样本mask、按场景统计阈值 |
| DINOv2显存/安装成本高 | T10阻塞 | 默认轻量CNN；DINOv2作为可选配置 |
| mask索引跨帧不一致 | 物体token时序错乱 | 架构假设object id跨帧一致；若不一致先在DataLoader remap |
| 静态物体处理错误 | ground/wall被预测乱动 | 静态state copy最后观测值；dynamic_mask控制状态loss；静态物体仍参与message passing |
| 多任务loss互相干扰 | 训练不稳定 | 分阶段训练，先RGB+state，再加collision/mask/LPIPS |
| 归一化固定值不准确 | 状态/力loss尺度不稳定 | T10配置中允许外部传入统计值；collision标签使用原始/反归一化力 |
| scene_id不稳定 | 复现实验困难 | 第一版默认关闭scene embedding；启用前改为S1-S8稳定映射 |

---

## 12. T10实现顺序建议

为减少返工，T10建议按以下顺序实现和测试：

1. `config.py`：统一shape、维度、loss权重、索引常量。
2. `encoder.py`：写test验证`[B,Th,N,D]`输出、padding mask生效。
3. `interaction.py`：写mock force/state测试message passing shape和mask。
4. `temporal.py`：写Transformer输入输出shape测试。
5. `decoder.py`：分别测试RGB/state/collision输出shape。
6. `physics_pred.py`：整合forward pass，用mock batch端到端跑通。
7. `loss.py`：构造mock target验证loss有限、mask不计padding。
8. `train.py`：最小训练循环，跑1-2个batch验证无NaN。

每个模块都必须先写单元测试，再实现代码，再由OpenCode/子agent和Claude Code审查。

---

## 13. 设计决策记录

### 决策1：第一版是object-centric supervised，不是unsupervised slot discovery

原因：数据已有GT mask，优先解决预测任务本身。无监督slot发现可作为后续消融实验。

### 决策2：T6使用dense pairwise message passing

原因：N<=7，dense `N*N`计算便宜，避免PyG batch图构造复杂度。后续需要扩展N时再换稀疏图/PyG。

### 决策3：T7首版非自回归预测全部未来帧

原因：训练稳定、接口简单、容易测试。自回归和scheduled sampling留到性能优化阶段。

### 决策4：DINOv2不作为硬依赖

原因：调研建议DINOv2冻结层，但T10需要先保证代码可运行。默认轻量CNN + 可选DINOv2接口更稳。

### 决策5：碰撞检测从force_matrix构造标签

原因：数据没有单独碰撞标签，但力矩阵是最直接的接触/相互作用监督信号。阈值策略在T8/T9中进一步定义。

---

## 14. OpenCode / Claude Code评审记录

### 2026-05-28 OpenCode/子agent第一轮评审

结论：未通过，需要修改。

主要问题：
- 计算资源预算不够具体，缺少显存目标、参数量和降级方案。
- RGB decoder空间路径不明确，不能只聚合为frame latent。
- 多视角和深度图扩展接口过于笼统，与当前dataset字段不对齐。
- shape表缺少中间张量、edge feature、mask等关键shape。
- force_matrix方向语义未确认，collision label阈值和归一化空间未定义。
- scene_id使用Python hash存在不稳定风险。
- 静态物体状态预测策略不够严格。

修改结果：
- 增加第一版做/不做清单。
- 增加12GB/24GB两档资源预算、参数量目标和OOM降级顺序。
- shape表补充feat_map、edge_feat、message、future_query、dynamic_mask、collision_pair_mask、depth/view预留shape。
- RGB decoder改为第一版object-conditioned spatial decoder + mask辅助，弱frame decoder仅作为临时baseline。
- force edge采用`F_ij + F_ji`双向特征，并要求T6前核对方向。
- collision label规定使用原始/反归一化力，给出初始阈值策略。
- 多视角补充`view_id`缺省0和未来`[B,V,T,...]`shape约定。
- 深度图补充`depth_hist`和`pred_depth`接口。
- 静态物体state默认copy最后观测值，状态主loss只算dynamic_mask。
- scene embedding默认关闭，启用前必须修复S1-S8稳定映射。

### 2026-05-28 Claude Code最终评审

结论：通过 PASS。

通过理由：
- T4要求逐项满足：第一版范围、资源预算、深度/多视角扩展、架构图、模块职责、数据流、shape表、已有方案对比均已覆盖。
- 与`papers.md`调研结论一致：GT mask ROI、GNN交互、Transformer首版、DINOv2/Mamba/LPIPS可选。
- 与`data_design.md`和`dataset.py`对齐：输入shape、static flag索引、state切片、归一化空间和scene_id风险均已处理。
- OpenCode/子agent第一轮指出的8类问题均已修复。

轻微建议（不阻塞）：
- T8详细设计时补充RGB decoder各子模块shape和通道数。
- T9实现时可显式区分dynamic-static collision pair。
- T5设计时如使用attr_dim第13维，需要明确color_name索引映射。
- 已按建议补充：D默认256，minimal_12gb配置可降到128。

---

## 15. 最终状态

当前状态：T4架构总体设计已完成，OpenCode/子agent第一轮修订后，Claude Code最终评审通过。
