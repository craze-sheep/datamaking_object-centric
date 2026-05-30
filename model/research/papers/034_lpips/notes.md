# LPIPS: The Unreasonable Effectiveness of Deep Features as a Perceptual Metric

## 基本信息

- **作者**: Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
- **年份**: 2018
- **会议**: CVPR 2018
- **论文链接**: https://arxiv.org/abs/1801.03924
- **代码仓库**: https://github.com/richzhang/PerceptualSimilarity
- **安装**: `pip install lpips`

---

## 核心贡献

1. **提出 LPIPS 指标**: 通过深度网络内部特征计算感知相似度，大幅超越传统 PSNR/SSIM
2. **构建 BAPPS 数据集**: Berkeley-Adobe Perceptual Patch Similarity 数据集，包含两种感知判断任务:
   - **2AFC (Two Alternative Forced Choice)**: 给定1个参考图+2个失真图，人类选择哪个更接近参考
   - **JND (Just Noticeable Differences)**: 人类判断两张图是否相同
3. **关键发现 - 感知性是深度表征的涌现属性**: 不仅限于 ImageNet 训练的 VGG，不同架构（SqueezeNet/AlexNet/VGG）和不同监督信号（有监督/自监督/无监督）的深度特征都能很好地衡量感知相似度
4. **可微分，可作为损失函数**: 可直接用于训练中的反向传播

---

## 模型架构

### 整体流程

```
输入图像对 (x, y) ∈ [-1, 1]
        │
   ScalingLayer (归一化到 ImageNet 统计量)
        │
  ┌─────┴─────┐
  │           │
  ▼           ▼
Trunk Net   Trunk Net  (共享权重的预训练骨干网络)
  │           │
  ▼           ▼
多层特征 {F_l(x)}  {F_l(y)}
  │           │
  └─────┬─────┘
        │
  逐层计算: diff_l = ||normalize(F_l(x)) - normalize(F_l(y))||²
        │
  线性校准层 w_l (1×1 Conv): res_l = w_l · diff_l
        │
  空间平均 → 求和: d(x,y) = Σ_l spatial_avg(res_l)
```

### 1. 骨干网络 (Trunk Network)

将预训练网络按层切成若干 slice，提取多层中间特征:

| 网络 | 参数量 | 层数 | 各层通道数 |
|------|--------|------|-----------|
| AlexNet | 9.1 MB | 5 层 (relu1-relu5) | [64, 192, 384, 256, 256] |
| VGG16 | 58.9 MB | 5 层 (relu1_2-relu5_3) | [64, 128, 256, 512, 512] |
| SqueezeNet | 2.8 MB | 7 层 (relu1-relu7) | [64, 128, 256, 384, 384, 512, 512] |

代码实现（`pretrained_networks.py`）: 将 torchvision 预训练模型的 features 切分为多个 `nn.Sequential` slice，每个 slice 的输出即为该层特征。

### 2. 输入归一化 (ScalingLayer)

```python
class ScalingLayer(nn.Module):
    def __init__(self):
        self.shift = [-.030, -.088, -.188]  # ImageNet 均值 (归一化到[-1,1]后)
        self.scale = [.458, .448, .450]     # ImageNet 标准差

    def forward(self, inp):
        return (inp - self.shift) / self.scale
```

**作用**: 将输入从 [-1, 1] 归一化到 ImageNet 的统计分布，使预训练特征提取更准确。v0.0 版本缺少此步骤（bug），v0.1 修复。

### 3. 特征归一化 (normalize_tensor)

```python
def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)
```

**作用**: 在通道维度上做 L2 归一化，使不同层的特征处于同一尺度。

### 4. 线性校准层 (Linear Calibration, NetLinLayer)

```python
class NetLinLayer(nn.Module):
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)
```

**核心设计**:
- 每层一个 1×1 卷积（等价于逐通道线性加权），将通道维度压缩到 1
- 可选 Dropout（训练时使用）防止过拟合
- **权重非负约束**: 训练时通过 `clamp_weights()` 将 1×1 卷积的权重 clamp 到 ≥ 0，保证距离非负

### 5. 完整前向传播

```python
def forward(self, in0, in1, retPerLayer=False, normalize=False):
    if normalize:  # 输入 [0,1] → [-1,1]
        in0 = 2 * in0 - 1
        in1 = 2 * in1 - 1

    # ScalingLayer 归一化
    in0_input = self.scaling_layer(in0)
    in1_input = self.scaling_layer(in1)

    # 提取多层特征
    outs0 = self.net.forward(in0_input)  # tuple of features
    outs1 = self.net.forward(in1_input)

    # 逐层: 归一化 → 计算差异
    for kk in range(self.L):
        feats0[kk] = normalize_tensor(outs0[kk])
        feats1[kk] = normalize_tensor(outs1[kk])
        diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

    # 线性校准 + 空间平均
    res = [spatial_average(self.lins[kk](diffs[kk])) for kk in range(self.L)]

    # 求和
    val = sum(res)
    return val
```

### 6. 空间模式 (Spatial Mode)

当 `spatial=True` 时，不进行空间平均，而是将各层差异上采样回原始分辨率，输出空间距离图（可用于定位差异区域）。

---

## 训练细节

### 数据集: BAPPS

- **2AFC 训练集**:
  - `train/traditional`: 56.6k 三元组（传统图像处理失真）
  - `train/cnn`: 38.1k 三元组（CNN 生成的失真）
  - `train/mix`: 56.6k 三元组（混合失真）
  - 每个三元组收集 2 个人类判断
- **2AFC 验证集**: 各 4.7k-10.9k 三元组，每个收集 5 个人类判断
  - 子集: traditional, cnn, superres, deblur, color, frameinterp
- **JND 验证集**: traditional, cnn 各若干

### 训练策略

- **任务**: 2AFC 排序学习（不是回归）
- **损失函数**: BCE Ranking Loss
  - 给定参考图 ref, 两张失真图 p0, p1, 人类判断 judge ∈ [0,1]
  - 计算 d0 = LPIPS(ref, p0), d1 = LPIPS(ref, p1)
  - 通过 Dist2LogitLayer 将 (d0, d1) 映射为预测概率
  - Dist2LogitLayer 输入: concat([d0, d1, d0-d1, d0/d1, d1/d0])，经过 3 层 1×1 Conv
  - 用 BCE Loss 与人类判断对齐
- **优化器**: Adam, lr=0.0001, beta1=0.5
- **训练轮次**: 5 epochs base lr + 5 epochs 线性衰减
- **Batch size**: 50
- **图像尺寸**: 64×64 patches
- **图像预处理**: Resize → ToTensor → Normalize(0.5, 0.5)
- **权重约束**: 每步优化后 clamp 1×1 卷积权重 ≥ 0

### 评估指标

- **2AFC 准确率**: 度量函数是否与人类选择一致（距离小的更接近参考图）
- **JND mAP**: 用距离预测"相同/不同"，计算 precision-recall 曲线下面积

---

## 实验结果

### 2AFC 验证集准确率（与人类偏好一致的比例）

| 方法 | 传统失真 | CNN失真 | 超分辨率 | 去模糊 | 上色 | 帧插值 | **平均** |
|------|---------|---------|---------|--------|------|--------|---------|
| PSNR | 57.3 | 54.0 | 55.1 | 55.6 | 53.2 | 54.6 | 55.0 |
| SSIM | 63.2 | 56.7 | 57.9 | 59.0 | 54.2 | 56.8 | 58.0 |
| LPIPS-Alex (ours) | **72.5** | **66.9** | **70.6** | **69.8** | **65.4** | **68.9** | **69.0** |
| LPIPS-VGG (ours) | 72.0 | 66.7 | 70.1 | 69.4 | 65.2 | 68.5 | 68.7 |

### 关键发现

1. **深度特征大幅超越传统指标**: LPIPS 比 SSIM 高约 11 个百分点
2. **跨架构一致性好**: AlexNet/VGG/SqueezeNet 表现相近
3. **跨监督信号一致**: 有监督、自监督（BiGAN, DeepCluster, RotNet）、甚至无监督（随机初始化后用 k-means 聚类）的特征都显著优于传统指标
4. **线性校准有帮助**: 加上学习的线性层（lpips=True）比直接平均各层（baseline）提升 1-3 个百分点
5. **AlexNet 作为前向度量最佳**: VGG 更适合用作优化损失（更平滑）

---

## 代码实现细节

### 文件结构

```
code/
├── lpips/
│   ├── __init__.py          # normalize_tensor, l2, psnr, dssim, voc_ap 等工具函数
│   ├── lpips.py             # 核心 LPIPS 模型: ScalingLayer, NetLinLayer, LPIPS, BCERankingLoss
│   ├── pretrained_networks.py  # 骨干网络封装: alexnet, vgg16, squeezenet, resnet
│   └── trainer.py           # Trainer 类: 训练/测试逻辑, 评估函数 score_2afc_dataset, score_jnd_dataset
├── train.py                 # 训练入口
├── test_dataset_model.py    # 在数据集上评估
├── test_network.py          # 快速测试两张图的 LPIPS 距离
├── lpips_2imgs.py           # 命令行: 两张图比较
├── lpips_2dirs.py           # 命令行: 两个目录的对应图比较
├── lpips_1dir_allpairs.py   # 命令行: 一个目录内所有图对比较
├── lpips_loss.py            # demo: 用 LPIPS 作为损失函数优化图像
├── data/
│   └── dataset/
│       ├── twoafc_dataset.py  # 2AFC 数据加载: ref, p0, p1, judge
│       └── jnd_dataset.py     # JND 数据加载: p0, p1, same
└── weights/
    ├── v0.1/                # 最新校准权重 (alex.pth, vgg.pth, squeeze.pth)
    └── v0.0/                # 原始权重 (有归一化 bug)
```

### 核心类设计

| 类 | 功能 |
|---|------|
| `LPIPS(nn.Module)` | 主模块。可配置: net/版本/是否用线性层/是否随机初始化/是否微调trunk/空间模式 |
| `ScalingLayer(nn.Module)` | ImageNet 统计量归一化 (v0.1 特有) |
| `NetLinLayer(nn.Module)` | 1×1 卷积 + 可选 Dropout，每层的线性校准 |
| `Dist2LogitLayer(nn.Module)` | 训练用: 将两个距离映射为 [0,1] 概率 (3层1×1Conv + LeakyReLU + Sigmoid) |
| `BCERankingLoss(nn.Module)` | 训练用: BCE 排序损失 |
| `Trainer` | 训练管理器: 优化循环、学习率衰减、模型保存 |

### 使用方式

```python
import lpips

# 前向计算距离
loss_fn = lpips.LPIPS(net='alex')  # 或 'vgg', 'squeeze'
d = loss_fn(img0, img1)           # img ∈ [-1, 1], shape=[B,3,H,W]

# 如果输入是 [0,1]:
d = loss_fn(img0, img1, normalize=True)

# 作为训练损失
loss_fn = lpips.LPIPS(net='vgg', eval_mode=False)  # 保持训练模式
loss = loss_fn(pred, target).mean()
loss.backward()  # 可微分，支持反向传播

# 获取逐层距离
val, per_layer = loss_fn(img0, img1, retPerLayer=True)

# 空间距离图
loss_fn = lpips.LPIPS(net='alex', spatial=True)  # 输出 [B,1,H,W] 距离图
```

### 重要注意事项

- 输入必须归一化到 **[-1, 1]**（或设置 `normalize=True` 让模型自动从 [0,1] 转换）
- 默认 `eval_mode=True`，用于训练损失时建议设为 `eval_mode=False`
- 权重文件位于 `lpips/weights/v0.1/` 目录下
- v0.0 → v0.1 的变化: 增加了 ScalingLayer（ImageNet 归一化修复）

---

## 与当前研究的关联

### 直接应用: 作为感知损失提升图像重建质量

**当前问题**: 模型使用 L1 + MSE 做 RGB 损失，容易导致模糊，不符合人眼感知。

**改进方案**: 添加 LPIPS 损失项：

```python
class LPIPSLoss(nn.Module):
    def __init__(self, net='alex'):
        super().__init__()
        self.lpips = lpips.LPIPS(net=net)
        self.lpips.eval()
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        # pred, target: [B, 3, H, W], 范围 [0, 1]
        return self.lpips(pred, target, normalize=True).mean()
```

**推荐组合**: `Loss = λ₁·L1 + λ₂·LPIPS`，其中 λ₂ 通常设为 0.01-0.1 量级。

### 更广泛的启发

1. **深度特征 = 通用感知表征**: 即使不用 LPIPS 损失，也可以用其他预训练网络的中间特征作为感知监督信号
2. **线性校准思路**: 在固定骨干网络上加轻量线性层进行校准，是一种高效的迁移学习方式
3. **2AFC 训练范式**: 通过人类偏好排序（而非绝对评分）来学习度量，数据效率更高
4. **非负权重约束**: 保证距离度量的非负性，是设计可学习距离函数的重要技巧
