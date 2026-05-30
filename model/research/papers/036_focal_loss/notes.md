# Focal Loss for Dense Object Detection

## 基本信息
- **作者**：Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár
- **机构**：Facebook AI Research (FAIR)
- **年份**：2017
- **会议**：ICCV 2017 (Best Student Paper)
- **论文链接**：https://arxiv.org/abs/1708.02002
- **代码**：https://github.com/facebookresearch/Detectron

## 核心贡献

1. **发现关键问题**：单阶段检测器精度落后于两阶段检测器的根本原因是**训练过程中极端的前景-背景类别不平衡**（约1:1000）
2. **提出 Focal Loss**：通过动态缩放交叉熵损失，降低易分类样本的权重，聚焦于难分类样本，是硬样本挖掘（hard example mining）的更优替代方案
3. **设计 RetinaNet**：简洁的单阶段检测器，首次在精度上匹配甚至超越当时最好的两阶段检测器
4. **成为标准**：Focal Loss 后续被广泛应用于不平衡分类、目标检测、语义分割等多个领域

## 模型架构

### 1. Focal Loss 公式

**标准交叉熵（CE）**：

```
CE(p, y) = -log(p_t)，其中 p_t = p (if y=1) 或 1-p (if y=-1)
```

**α-balanced 交叉熵**（基线）：

```
CE(p_t) = -α_t · log(p_t)
```

- α ∈ [0,1] 为正样本权重，1-α 为负样本权重
- 仅平衡正负样本，不区分难易样本

**Focal Loss**（核心创新）：

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

- **调制因子 (1-p_t)^γ**：γ ≥ 0 为聚焦参数
- 当样本被**误分类**时（p_t 小），调制因子 ≈ 1，损失不受影响
- 当样本被**正确分类**时（p_t → 1），调制因子 → 0，损失被大幅降低
- **γ=0**：等价于标准 CE
- **γ=2**（默认最优值）：p_t=0.9 时损失降低100倍，p_t≈0.968 时降低1000倍
- 误分类样本（p_t ≤ 0.5）的损失最多降低4倍（γ=2 时）

### 2. 类别不平衡与模型初始化

- 默认初始化下，模型输出正负类概率相等，在严重不平衡时会导致早期训练不稳定
- **先验概率 π**：将分类子网络最后一层 bias 初始化为 `b = -log((1-π)/π)`，设 π=0.01
- 确保训练初期模型对前景的预测置信度较低，避免大量背景 anchor 产生不稳定的损失
- 该初始化对 CE 和 Focal Loss 都有帮助

### 3. 与两阶段检测器的对比

两阶段检测器通过两种机制解决类别不平衡：
1. **级联筛选**（RPN/Selective Search）：将候选区域从无穷缩减到1-2千，过滤掉大量简单负样本
2. **偏置采样**：如正负样本1:3比例，类似隐式的α平衡

Focal Loss 将这些机制统一通过**损失函数本身**实现，无需采样策略。

## RetinaNet 架构

RetinaNet = **骨干网络（ResNet-FPN）** + **分类子网络** + **回归子网络**

### 骨干网络：Feature Pyramid Network (FPN)

- 基于 ResNet（50/101），构建特征金字塔 P3-P7
- P3-P5：从 ResNet 对应残差阶段 C3-C5 通过自顶向下路径和横向连接获得
- P6：对 C5 施加 stride-2 的 3×3 卷积
- P7：对 P6 先 ReLU 再 stride-2 的 3×3 卷积
- 所有金字塔层级通道数 C=256

### Anchor 设计

- 每个空间位置 A=9 个 anchor：3种宽高比 {1:2, 1:1, 2:1} × 3种尺度 {2^0, 2^(1/3), 2^(2/3)}
- 金字塔各层级 anchor 面积：P3=32², P4=64², P5=128², P6=256², P7=512²
- 覆盖尺度范围 32-813 像素
- IoU 匹配规则：与 GT IoU ≥ 0.5 为正样本，IoU ∈ [0, 0.4) 为负样本，中间忽略

### 分类子网络（Classification Subnet）

- 4层 3×3 卷积（256通道）+ ReLU → 1层 3×3 卷积（K×A 通道）→ Sigmoid
- 参数在所有金字塔层级间**共享**
- K=类别数，A=anchor 数

### 回归子网络（Box Regression Subnet）

- 结构与分类子网络相同，输出改为 4×A 个线性输出
- **类别无关**（class-agnostic），参数更少且效果等同
- 两个子网络结构相同但参数独立

## 训练细节

### 优化设置
- **优化器**：SGD，momentum=0.9，weight decay=0.0001
- **学习率**：初始 0.01，在 60k 和 80k 迭代时各除以10
- **总迭代**：90k（消融实验），1.5× 更长用于最佳结果
- **批量大小**：8 GPU，每 GPU 2张图，总计16张/批
- **数据增强**：仅水平翻转
- **图像尺度**：默认 600 像素（短边）

### 损失计算
- **分类损失**：Focal Loss 应用于所有约 100k 个 anchor
- **回归损失**：Smooth L1 Loss，仅对正样本计算
- **总损失** = (loc_loss + cls_loss) / num_pos（用正 anchor 数归一化）
- 推断时：置信度阈值 0.05 → 每层最多取 top 1k → 跨层合并 → NMS (阈值 0.5)

### 关键超参数
- **γ = 2**（最优），对 γ ∈ [0.5, 5] 都较鲁棒
- **α = 0.25**（γ=2 时最优），γ 增大时 α 应略减小
- **π = 0.01**（初始化先验）

## 实验结果

### 消融实验（COCO minival，ResNet-50-FPN，600px）

| 方法 | AP | AP50 | AP75 |
|------|-----|------|------|
| CE (γ=0) | 31.1 | 49.4 | 33.0 |
| α-balanced CE (α=0.75) | 31.1 | 49.4 | 33.0 |
| Focal Loss (γ=2, α=0.25) | **34.0** | 52.5 | 36.5 |
| OHEM 最优 | 32.8 | 50.3 | 35.1 |

- Focal Loss 比最优 OHEM 高 **3.2 AP**
- Focal Loss 比 α-balanced CE 高 **2.9 AP**

### 主要结果（COCO test-dev）

| 方法 | Backbone | AP | AP50 | AP75 | APS | APM | APL |
|------|----------|-----|------|------|------|------|------|
| Faster R-CNN w FPN | ResNet-101-FPN | 36.2 | 59.1 | 39.0 | 18.2 | 39.0 | 48.2 |
| SSD513 | ResNet-101 | 31.2 | 50.4 | 33.3 | 10.2 | 34.5 | 49.8 |
| DSSD513 | ResNet-101 | 33.2 | 53.3 | 35.2 | 13.0 | 35.4 | 51.1 |
| **RetinaNet-101-800** | ResNet-101-FPN | **39.1** | 59.1 | 42.3 | 21.8 | 42.7 | 50.2 |
| **RetinaNet (最终)** | ResNeXt-101-FPN | **40.8** | 61.1 | 44.1 | 24.1 | 44.2 | 51.2 |

- 单阶段检测器**首次**全面超越两阶段检测器
- 速度/精度权衡：ResNet-101 600px = 122ms/张（与 Faster R-CNN 相当精度但更快）

### 损失分布分析
- **正样本**：γ 增大对正样本损失分布影响不大（约20%最难正样本贡献约一半损失）
- **负样本**：γ=2 时，绝大多数损失集中在极少数难负样本上，有效抑制了简单负样本

## 代码实现细节

代码位于 `code/` 目录，是论文的 PyTorch 简化实现。

### 核心文件

| 文件 | 功能 |
|------|------|
| `loss.py` | Focal Loss 实现 |
| `retinanet.py` | RetinaNet 网络结构 |
| `fpn.py` | FPN 骨干网络（ResNet-50/101 + 特征金字塔） |
| `train.py` | 训练脚本 |
| `encoder.py` | Anchor 编码/解码 |
| `datagen.py` | 数据加载 |
| `transform.py` | 数据增强 |

### Focal Loss 实现（loss.py）

```python
class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        alpha = 0.25
        gamma = 2
        # one-hot 编码目标
        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # 去掉背景类 → [N,20]
        t = Variable(t).cuda()
        # 计算 p_t
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t>0 else 1-p
        # 计算权重 w = α_t * (1-p_t)^γ
        w = alpha*t + (1-alpha)*(1-t)  # α for positive, 1-α for negative
        w = w * (1-pt).pow(gamma)      # 乘以调制因子
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)
```

**关键实现要点**：
1. **数值稳定性**：将 sigmoid 和损失计算合并（`binary_cross_entropy_with_logits`）
2. **α-balanced**：正样本用 α=0.25，负样本用 1-α=0.75
3. **focal 调制**：`w = w * (1-pt).pow(gamma)` 对已正确分类样本降权
4. 提供了 `focal_loss_alt` 替代实现（使用 x_t 变换）
5. 前向传播中：`loss = (SmoothL1Loss + FocalLoss) / num_pos`，用正 anchor 数归一化

### RetinaNet 实现（retinanet.py）

```python
class RetinaNet(nn.Module):
    num_anchors = 9
    def __init__(self, num_classes=20):
        self.fpn = FPN50()                    # 骨干网络
        self.loc_head = self._make_head(9*4)  # 回归头：9 anchors × 4 坐标
        self.cls_head = self._make_head(9*num_classes)  # 分类头：9 anchors × K 类

    def _make_head(self, out_planes):
        # 4层 Conv(256→256, 3×3) + ReLU + 最后一层 Conv(256→out)
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, 3, 1, 1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, 3, 1, 1))
        return nn.Sequential(*layers)
```

**关键实现要点**：
1. **FPN 输出 5 层特征**：P3-P7，每层独立过分类/回归头
2. **Anchor 数 = 9**：3 宽高比 × 3 尺度
3. 分类和回归头**参数独立**但**结构相同**（4层共享）
4. 训练时 `freeze_bn()`：冻结 BatchNorm

### 训练流程（train.py）

- 数据集：VOC 2012，输入尺寸 600px
- 优化器：SGD (lr=1e-3, momentum=0.9, weight_decay=1e-4)
- 加载预训练 backbone 权重，DataParallel 多卡训练
- 200 个 epoch，保存最优 loss 的 checkpoint

## 与当前研究的关联

### 对现有项目的价值
- 当前项目使用 **Focal Loss 做碰撞预测**（二分类不平衡场景），这正是本文推荐的场景
- Focal Loss 已成为处理类别不平衡分类问题的**标准损失函数**

### 可借鉴的改进方向

1. **γ 参数调优**：γ=2 是通用最优，但特定场景可能需要调整
   - 不平衡程度越大 → γ 可适当增大
   - 可尝试自适应 γ（作为可学习参数）

2. **α 参数优化**：α=0.25 配合 γ=2 最优，但与 γ 存在交互
   - γ 增大时 α 应略减小（因简单负样本已自动降权，正样本无需过多强调）

3. **初始化策略**：先验概率 π=0.01 的 bias 初始化，对任何不平衡分类任务都有参考价值

4. **替代损失形式 FL***：`p_t* = σ(γ·x_t + β)`，效果与标准 FL 相当，提供了更灵活的调节空间

5. **损失归一化**：用正样本数而非总样本数归一化，避免大量简单负样本稀释梯度信号
