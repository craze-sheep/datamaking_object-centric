# MoCo: Momentum Contrast for Unsupervised Visual Representation Learning

## 基本信息
- **作者**：Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, Ross Girshick
- **机构**：Facebook AI Research (FAIR)
- **年份**：2019（arXiv），2020（CVPR 正式发表）
- **会议**：CVPR 2020
- **论文链接**：https://arxiv.org/abs/1911.05722
- **代码**：https://github.com/facebookresearch/moco
- **引用量**：自监督学习领域最具影响力的工作之一

## 核心贡献

1. **将对比学习重新解读为字典查找问题**：将对比学习中的"正负样本匹配"统一为在一个动态字典中查找匹配 key 的 query
2. **提出动量对比（Momentum Contrast）机制**：通过动量更新的编码器 + 队列，构建一个**大且一致**的动态字典
3. **在多个下游任务上超越监督预训练**：在 PASCAL VOC、COCO 等 7 个检测/分割任务上，无监督预训练首次超越 ImageNet 监督预训练
4. **大规模数据验证**：在 Instagram-1B（~10亿张图片）上验证了方法的有效性，展示了向真实世界场景扩展的能力
5. **成为自监督学习的里程碑**：MoCo 及其后续版本（MoCo v2, v3）成为视觉自监督学习的基础框架

## 模型架构

### 整体思路：对比学习 = 字典查找

将对比学习类比为字典查找任务：
- **Query（查询）**：由 query encoder $f_q$ 编码的图像表示
- **Key（键）**：由 key encoder $f_k$ 编码的图像表示
- **正样本对**：同一张图片的不同增强视图（query 与 key 匹配）
- **负样本对**：字典中其他所有 key（query 与之不匹配）

### 三种对比学习机制的对比

论文系统比较了三种维护字典的机制：

| 机制 | 字典大小 | 一致性 | 问题 |
|------|---------|--------|------|
| **(a) End-to-End** | 受限于 batch size | 高（同一 encoder） | 字典小，大 batch 训练困难 |
| **(b) Memory Bank** | 大（全数据集） | 低（不同时间步的 encoder） | 表示更新不同步 |
| **(c) MoCo** | 大（队列） | 高（动量 encoder） | ✅ 两全其美 |

### 核心组件

#### 1. 动量编码器（Momentum Encoder）

**目的**：保持 key encoder 与 query encoder 的一致性，同时允许缓慢演化。

**更新公式**：
$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

其中：
- $\theta_q$：query encoder 参数（通过反向传播正常更新）
- $\theta_k$：key encoder 参数（通过动量更新，不参与梯度计算）
- $m$：动量系数（默认 0.999，论文实验表明 $m \in [0.99, 0.9999]$ 效果较好）

**直觉**：$m$ 越大，$\theta_k$ 变化越缓慢，队列中不同 mini-batch 编码的 key 之间的差异越小，字典一致性越好。当 $m=0$ 时（直接复制），训练发散；$m=0.9$ 时精度明显下降。

#### 2. 队列（Queue）

**目的**：解耦字典大小和 mini-batch size，支持维护大量负样本。

**机制**：
- 当前 mini-batch 编码的 key 入队
- 最早的 mini-batch 的 key 出队
- 队列大小 K 远大于 mini-batch size（默认 K=65536）

**优势**：
- 字典大小可以灵活设置，不受 GPU 内存限制
- 移除最旧的 mini-batch 是有益的（其编码由最旧的 encoder 生成，一致性最差）

#### 3. InfoNCE 对比损失

$$\mathcal{L}_q = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)}$$

其中：
- $q$：query 表示
- $k_+$：正样本 key
- $k_i$：字典中所有 key（1 个正样本 + K 个负样本）
- $\tau$：温度超参数（默认 0.07）

**本质**：(K+1)-分类的交叉熵损失，试图将 query 分类为其匹配的正样本 key。

### 数据增强策略（Pretext Task）

采用**实例判别**任务：同一张图片的两个随机增强视图构成正样本对。

增强操作（借鉴 BYOL 的增强方案）：
- **视图 1**：RandomResizedCrop(224) → ColorJitter(0.4,0.4,0.2,0.1, p=0.8) → RandomGrayscale(p=0.2) → GaussianBlur(p=1.0) → RandomHorizontalFlip → Normalize
- **视图 2**：RandomResizedCrop(224) → ColorJitter(0.4,0.4,0.2,0.1, p=0.8) → RandomGrayscale(p=0.2) → GaussianBlur(p=0.1) → Solarize(p=0.2) → RandomHorizontalFlip → Normalize

两个视图的增强策略不同（特别是 GaussianBlur 和 Solarize 的概率差异），增加了多样性。

### Shuffling Batch Normalization

**问题**：标准 BN 会在同一 mini-batch 内的样本间泄漏信息（通过 batch 统计量），导致模型"作弊"——pretext task 训练精度迅速达到 99.9%，但验证精度下降。

**解决方案**：Shuffling BN
- 在 key encoder 编码前，打乱 mini-batch 内样本顺序
- 每个 GPU 独立计算 BN 统计量
- 编码后再打乱回来
- 确保 query 和其正样本 key 使用不同的 batch 统计量

## 训练细节

### ImageNet-1M（IN-1M）训练配置
| 参数 | 值 |
|------|-----|
| 数据集 | ImageNet 训练集（~128万张，1000类） |
| 骨干网络 | ResNet-50（默认）/ R50w2× / R50w4× |
| 优化器 | SGD（momentum=0.9, weight decay=0.0001） |
| Mini-batch size | 256（8 GPU） |
| 初始学习率 | 0.03 |
| 训练轮数 | 200 epochs |
| 学习率调度 | 在 120 和 160 epoch 乘以 0.1 |
| 字典大小 K | 65536 |
| 动量系数 m | 0.999 |
| 温度 τ | 0.07 |
| 特征维度 | 128-D（L2 归一化） |
| 训练时长 | ~53 小时（ResNet-50, 8 GPU） |

### Instagram-1B（IG-1B）训练配置
| 参数 | 值 |
|------|-----|
| 数据集 | Instagram ~10亿张图片（940M），~1500 个 hashtag |
| Mini-batch size | 1024（64 GPU） |
| 初始学习率 | 0.12 |
| 学习率调度 | 每 62.5k 次迭代指数衰减 0.9× |
| 训练迭代数 | 1.25M 次（~1.4 epochs） |
| 训练时长 | ~6 天（ResNet-50, 64 GPU） |

### 线性评估协议
- 冻结预训练特征
- 训练一个线性分类器（FC + softmax）
- 100 epochs，初始学习率 30，weight decay 0
- 报告 ImageNet 验证集 1-crop Top-1 精度

## 实验结果

### 1. 线性分类协议（ImageNet, 表1）

| 方法 | 架构 | 参数量(M) | 精度(%) |
|------|------|-----------|---------|
| InstDisc [61] | R50 | 24 | 54.0 |
| LocalAgg [66] | R50 | 24 | 58.8 |
| BigBiGAN [16] | R50 | 24 | 56.6 |
| CPC v2 [35] | R170wider | 303 | 65.9 |
| CMC [56] | R50w2×L+ab | 188 | 68.4† |
| AMDIM [2] | AMDIMlarge | 626 | 68.1† |
| **MoCo** | **R50** | **24** | **60.6** |
| **MoCo** | **R50w4×** | **375** | **68.6** |

† 标记的方法使用了 FastAutoAugment（ImageNet 监督标签）

**关键发现**：MoCo 在标准 ResNet-50 上达到 60.6%，不需要特殊架构设计（如 patchified inputs、定制感受野等），且随着模型增大效果持续提升。

### 2. PASCAL VOC 目标检测（表2, 表4）

使用 Faster R-CNN，在 trainval07+12 上微调，test2007 上评估：

**R50-C4 骨干（trainval07+12 微调）**：
| 预训练 | AP₅₀ | AP | AP₇₅ |
|--------|-------|-----|------|
| 监督 IN-1M | 81.3 | 53.5 | 58.8 |
| MoCo IN-1M | 81.5 (+0.2) | 55.9 (+2.4) | 62.6 (+3.8) |
| MoCo IG-1B | 82.2 (+0.9) | 57.2 (+3.7) | 63.7 (+4.9) |

**R50-C4 骨干（trainval2007 微调，~5k 图片）**：
| 预训练 | AP₅₀ | AP | AP₇₅ |
|--------|-------|-----|------|
| 监督 IN-1M | 74.4 | 42.4 | 42.7 |
| MoCo IN-1M | 74.9 (+0.5) | 46.6 (+4.2) | 50.1 (+7.4) |
| MoCo IG-1B | 75.6 (+1.2) | 47.6 (+5.2) | 51.7 (+9.0) |

### 3. COCO 目标检测与实例分割（表5）

使用 Mask R-CNN，R50-FPN/C4 骨干：

**R50-FPN, 2× schedule**：
| 预训练 | APᵇᵇ | APᵇᵇ₅₀ | APᵇᵇ₇₅ | APᵐᵏ | APᵐᵏ₅₀ | APᵐᵏ₇₅ |
|--------|-------|---------|---------|-------|---------|---------|
| 监督 IN-1M | 40.6 | 61.3 | 44.4 | 36.8 | 58.1 | 39.5 |
| MoCo IN-1M | 40.8 | 61.6 | 44.7 | 36.9 | 58.4 | 39.7 |
| MoCo IG-1B | 41.1 (+0.5) | 61.8 (+0.5) | 45.1 (+0.7) | 37.4 (+0.6) | 59.1 (+1.0) | 40.2 (+0.7) |

**R50-C4, 2× schedule**：
| 预训练 | APᵇᵇ | APᵇᵇ₅₀ | APᵇᵇ₇₅ | APᵐᵏ | APᵐᵏ₅₀ | APᵐᵏ₇₅ |
|--------|-------|---------|---------|-------|---------|---------|
| 监督 IN-1M | 40.0 | 59.9 | 43.1 | 34.7 | 56.5 | 36.9 |
| MoCo IN-1M | 40.7 (+0.7) | 60.5 (+0.6) | 44.1 (+1.0) | 35.4 (+0.7) | 57.3 (+0.8) | 37.6 (+0.7) |
| MoCo IG-1B | 41.1 (+1.1) | 60.7 (+0.8) | 44.8 (+1.7) | 35.6 (+0.9) | 57.4 (+0.9) | 38.1 (+1.2) |

### 4. 更多下游任务（表6）

| 任务 | 指标 | 监督 IN-1M | MoCo IN-1M | MoCo IG-1B |
|------|------|-----------|-----------|-----------|
| COCO 关键点检测 | APᵏᵖ | 65.8 | 66.8 (+1.0) | 66.9 (+1.1) |
| COCO DensePose | APᵈᵖ₇₅ | 50.6 | 53.9 (+3.3) | 54.3 (+3.7) |
| LVIS 实例分割 | APᵐᵏ | 24.4† | 24.1 (-0.3) | 24.9 (+0.5) |
| Cityscapes 实例分割 | APᵐᵏ₅₀ | 59.6 | 59.3 (-0.3) | 60.3 (+0.7) |
| Cityscapes 语义分割 | mIoU | 74.6 | 75.3 (+0.7) | 75.5 (+0.9) |
| VOC 语义分割 | mIoU | 74.4 | 72.5 (-1.9) | 73.6 (-0.8) |
| iNaturalist 分类 | Acc | 66.1 | 65.6 | 65.8 |

**总结**：MoCo 在 **7 个检测/分割任务上超越监督预训练**，仅在 VOC 语义分割上表现稍差。IG-1B 预训练始终优于 IN-1M。

### 5. 动量消融实验

| 动量 m | 0 | 0.9 | 0.99 | 0.999 | 0.9999 |
|--------|---|-----|------|-------|--------|
| 精度(%) | 发散 | 55.2 | 57.8 | **59.0** | 58.9 |

**结论**：缓慢演化的 key encoder（大动量）是核心，$m=0.999$ 为最优默认值。

### 6. 字典大小 K 的影响（图3）

所有三种机制（end-to-end, memory bank, MoCo）都受益于更大的 K：
- K=65536 时 MoCo 达到最佳
- MoCo 在所有 K 值上都优于 memory bank（~2.6%差距）
- End-to-end 在小 K 时与 MoCo 相当，但受限于 batch size

## 代码实现细节

### 仓库结构
```
code/
├── main_moco.py          # 主训练脚本
├── main_lincls.py        # 线性分类评估脚本
├── moco/
│   ├── builder.py        # MoCo 模型核心实现
│   ├── loader.py         # 数据增强（TwoCropsTransform, GaussianBlur, Solarize）
│   └── optimizer.py      # LARS 优化器实现
├── vits.py               # ViT 骨干网络
├── convert_to_deit.py    # 权重转换工具
└── transfer/             # 下游任务迁移代码
```

### 核心实现：`moco/builder.py`

**MoCo 基类**：
- `__init__`：初始化 base_encoder（query encoder）和 momentum_encoder（key encoder），复制参数，冻结动量 encoder 的梯度
- `_update_momentum_encoder(m)`：动量更新 $\theta_k = m \cdot \theta_k + (1-m) \cdot \theta_q$，用 `@torch.no_grad()` 装饰器
- `contrastive_loss(q, k)`：
  1. L2 归一化 q 和 k
  2. `concat_all_gather(k)`：跨 GPU 收集所有 key（分布式训练）
  3. 计算 logits = q·kᵀ / T
  4. 生成标签（每个 query 的正样本在其对应 GPU 的对应位置）
  5. 交叉熵损失 × 2T
- `forward(x1, x2, m)`：
  1. 计算 q1 = predictor(base_encoder(x1)), q2 = predictor(base_encoder(x2))
  2. `@torch.no_grad()` 更新动量 encoder
  3. 计算 k1 = momentum_encoder(x1), k2 = momentum_encoder(x2)
  4. 返回 contrastive_loss(q1, k2) + contrastive_loss(q2, k1)（交叉视图匹配）

**MoCo_ResNet 子类**：
- 移除原始 FC 层
- 添加 2 层 MLP projector（hidden_dim → 4096 → 256）
- 添加 2 层 MLP predictor（256 → 4096 → 256）
- Projector 最后一层使用无参数的 BN（仿 SimCLR 设计）

**MoCo_ViT 子类**：
- 移除原始 head
- 3 层 MLP projector（隐藏维度 → 4096 → 256）
- 2 层 MLP predictor（256 → 4096 → 256）

**关键代码**：
```python
# 动量更新（@torch.no_grad() 装饰器确保不计算梯度）
def _update_momentum_encoder(self, m):
    for param_b, param_m in zip(self.base_encoder.parameters(), 
                                 self.momentum_encoder.parameters()):
        param_m.data = param_m.data * m + param_b.data * (1. - m)

# 对比损失
def contrastive_loss(self, q, k):
    q = nn.functional.normalize(q, dim=1)  # L2 归一化
    k = nn.functional.normalize(k, dim=1)
    k = concat_all_gather(k)  # 跨 GPU 收集 key
    logits = torch.einsum('nc,mc->nm', [q, k]) / self.T  # 相似度矩阵
    N = logits.shape[0]
    labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
    return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
```

### 训练脚本：`main_moco.py`

**关键配置**：
- 优化器：LARS（默认）或 AdamW
- 学习率调度：warmup（10 epochs）+ 余弦退火
- 动量系数：支持余弦调度逐渐增大到 1（`--moco-m-cos`）
- 混合精度训练：`torch.cuda.amp`
- 分布式训练：PyTorch DDP + SyncBN

**数据增强**：
- 采用 TwoCropsTransform 生成两个不同增强的视图
- 视图 1：强 GaussianBlur（p=1.0），无 Solarize
- 视图 2：弱 GaussianBlur（p=0.1），有 Solarize（p=0.2）

### LARS 优化器：`moco/optimizer.py`

- Layer-wise Adaptive Rate Scaling
- 对参数 >1D 的（如卷积权重）应用 trust coefficient 缩放
- 对参数 ≤1D 的（如 BN 的 gamma/beta、bias）不做 weight decay 和 rate scaling
- Trust coefficient 默认 0.001

### 数据加载器：`moco/loader.py`

- `TwoCropsTransform`：对同一图片应用两种不同增强
- `GaussianBlur`：高斯模糊增强（σ ∈ [0.1, 2.0]）
- `Solarize`：曝光反转增强（来自 BYOL）

## 与当前研究的关联

### 对自监督学习发展的影响
1. **MoCo v2** [Chen et al., 2020]：在 MoCo 基础上加入 SimCLR 的 MLP projection head 和更强的数据增强，R50 精度从 60.6% 提升到 71.1%
2. **MoCo v3** [Chen et al., 2021]：将 MoCo 框架扩展到 ViT，训练更稳定（移除 BN，使用 kNN monitor）
3. **SimCLR** [Chen et al., 2020]：同期工作，使用 end-to-end 机制 + 大 batch，MoCo 在相同计算预算下通常更高效
4. **BYOL** [Grill et al., 2020]：受 MoCo 动量更新启发，但去掉了负样本

### 核心思想的传承
- **动量更新**已成为自监督学习的标准组件（BYOL、DINO、MAE 等都采用）
- **队列机制**启发了后续的大字典对比学习方法
- **字典一致性**的概念深刻影响了自监督学习的理论理解

### 与视觉 Slot Learning 的潜在关联
- MoCo 的动量更新机制可借鉴用于 slot 表示的稳定学习
- 对比学习思想可用于区分不同的 slot（正负样本对应）
- 队列机制可用于维护历史 slot 表示，增强时序一致性
