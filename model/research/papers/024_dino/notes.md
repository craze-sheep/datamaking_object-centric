# DINO: Emerging Properties in Self-Supervised Vision Transformers

## 基本信息

- **标题**: Emerging Properties in Self-Supervised Vision Transformers
- **作者**: Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin
- **机构**: Facebook AI Research, Inria, Sorbonne University
- **年份**: 2021
- **会议**: ICCV 2021
- **论文链接**: https://arxiv.org/abs/2104.14294
- **代码**: https://github.com/facebookresearch/dino

---

## 核心贡献

1. **自监督ViT涌现语义分割能力**：自监督训练的ViT的[CLS] token自注意力图自动学习到物体边界和场景布局，这一能力在监督训练的ViT和CNN中都不明显
2. **自监督ViT特征具有极强的k-NN能力**：仅用k-NN分类器（无需微调、线性层或数据增强），ViT-S/8在ImageNet上达到78.3% top-1
3. **提出DINO框架**：一种自蒸馏（self-distillation）无标签学习方法，将知识蒸馏直接作为自监督目标
4. **发现momentum encoder、multi-crop和小patch size的关键作用**
5. **在ImageNet线性评估上达到80.1%**（ViT-B/8），超越之前的自监督SOTA

---

## 模型架构

### 整体框架：自蒸馏（Self-Distillation with No Labels）

DINO的核心思想是将知识蒸馏应用于自监督学习：student网络学习匹配teacher网络的输出，两者都没有标签。

```
输入图像 x
    ├── 随机变换 → x1 (全局裁剪1)
    ├── 随机变换 → x2 (全局裁剪2)
    └── 随机变换 → x3...xn (局部裁剪)

Student网络 gθs: 处理所有裁剪 (x1, x2, x3...xn)
Teacher网络 gθt: 仅处理全局裁剪 (x1, x2)

损失: 交叉熵 H(Pt(x), Ps(x'))
- Pt: teacher输出的概率分布（经过centering和sharpening）
- Ps: student输出的概率分布
```

**关键设计**：
- Student和Teacher共享相同架构，但参数不同
- **不需要预测器（predictor）**，与BYOL不同
- **不需要对比损失**，不需要负样本
- **不需要Batch Normalization**，整个系统BN-free

### 网络架构组成

```
g = h ∘ f

f: backbone (ViT或ResNet)
h: projection head (DINOHead)
```

**Projection Head (DINOHead)**：
- 3层MLP，隐藏维度2048，GELU激活
- L2归一化瓶颈层（bottleneck_dim=256）
- 权重归一化的全连接层，输出维度K=65536
- 设计灵感来自SwAV的prototype layer

```python
# vision_transformer.py 中的DINOHead
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, 
                 nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        # MLP: in_dim → 2048 → 2048 → 256
        # L2 normalization
        # Weight-normalized linear: 256 → out_dim(65536)
```

### Momentum Encoder（动量编码器）

Teacher网络不是预先给定的，而是通过student网络的EMA（指数移动平均）构建：

```
θt ← λθt + (1 − λ)θs

λ: 从0.996按余弦调度递增到1
```

**核心发现**：
- Momentum teacher在整个训练过程中**持续优于student**
- 这解释为Polyak-Ruppert平均（模型集成）的一种形式
- Teacher不断为student提供更高质量的目标特征，形成良性循环
- 不使用momentum时，框架会崩溃（collapse）

**与其他方法的对比**：
- BYOL/MoCo中momentum encoder的作用是替代队列
- DINO中momentum encoder类似Mean Teacher，用于持续构建更好的模型集成

### Multi-Crop（多裁剪策略）

```
全局裁剪（Global crops）：2个，分辨率224×224，覆盖>50%原图
局部裁剪（Local crops）：默认8个，分辨率96×96，覆盖<50%原图

Student处理所有裁剪 → Teacher仅处理全局裁剪
→ 鼓励"局部到全局"的对应关系
```

**关键作用**：
- Multi-crop是DINO的核心组件，不是简单的"附加项"
- DINO从multi-crop中受益最大（+3.4% linear eval）
- 不同框架对multi-crop的响应不同：BYOL直接加multi-crop反而会降低性能
- 全局裁剪scale: (0.4, 1.0)，局部裁剪scale: (0.05, 0.4)

### Centering 和 Sharpening（防止崩溃的机制）

**两种崩溃形式**：
1. **维度主导**：一个维度的输出占主导地位
2. **均匀分布**：所有维度的输出趋于相同

**Centering**（中心化）：
```
gt(x) ← gt(x) - c

c ← m*c + (1-m) * mean(gθt(xi))   # EMA更新center
```
- 防止维度主导崩溃
- 但鼓励输出趋向均匀分布
- 仅依赖一阶batch统计量，对batch size不敏感

**Sharpening**（锐化）：
```
Pt(x) = softmax((gt(x) - c) / τt)
```
- 使用低温度τt使teacher输出更锐利
- 与centering的效果互补
- τt从0.04线性warmup到0.07（前30个epoch）

**互补性分析**（公式5）：
```
H(Pt, Ps) = h(Pt) + DKL(Pt || Ps)
```
- KL散度=0表示常数输出（崩溃）
- 没有centering → KL→0，熵→0（维度主导崩溃）
- 没有sharpening → KL→0，熵→-log(1/K)（均匀分布崩溃）
- 两者结合 → 平衡效果，避免崩溃

---

## 训练细节

### 优化配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW |
| Batch size | 1024（分布在16 GPU上） |
| 学习率 | 0.0005 × batchsize/256（线性缩放） |
| 学习率调度 | 10 epoch warmup + 余弦衰减 |
| 权重衰减 | 余弦调度从0.04到0.4 |
| Student温度 τs | 0.1 |
| Teacher温度 τt | 0.04→0.07（前30 epoch warmup） |
| Momentum λ | 0.996→1（余弦调度） |
| Center momentum m | 0.9 |
| 训练轮数 | 默认100/300/800 epoch |
| 梯度裁剪 | 3.0 |
| 混合精度 | 默认开启（FP16） |
| Stochastic depth | 0.1 |

### 数据增强

**全局裁剪增强流水线**：
1. RandomResizedCrop(224, scale=(0.4, 1.0), bicubic)
2. RandomHorizontalFlip(p=0.5)
3. ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8)
4. RandomGrayscale(p=0.2)
5. GaussianBlur(p=1.0 for crop1, p=0.1 for crop2)
6. Solarization(p=0.2, 仅crop2)
7. Normalize(ImageNet均值/标准差)

**局部裁剪增强**：
1. RandomResizedCrop(96, scale=(0.05, 0.4), bicubic)
2. 同上颜色增强
3. GaussianBlur(p=0.5)
4. Normalize

### 训练资源

| 配置 | 训练时长 | 精度 |
|------|---------|------|
| ViT-S/16, 100ep, 2×224² | 15.3h | 67.8% |
| ViT-S/16, 300ep, 2×224² | 45.9h, 9.3G | 72.5% |
| ViT-S/16, 300ep, 2×224²+10×96² | 72.6h, 15.4G | 76.1% |
| 2×8-GPU服务器，3天 | - | 76.1% |

---

## 实验结果

### ImageNet线性评估与k-NN评估

| 方法 | 架构 | 参数量 | Linear | k-NN |
|------|------|--------|--------|------|
| 监督学习 | RN50 | 23M | 79.3 | 79.3 |
| SimCLR | RN50 | 23M | 69.1 | 60.7 |
| BYOL | RN50 | 23M | 74.4 | 64.8 |
| SwAV | RN50 | 23M | 75.3 | 65.7 |
| **DINO** | **RN50** | **23M** | **75.3** | **67.5** |
| 监督学习 | ViT-S | 21M | 79.8 | 79.8 |
| BYOL | ViT-S | 21M | 71.4 | 66.6 |
| SwAV | ViT-S | 21M | 73.5 | 66.3 |
| **DINO** | **ViT-S** | **21M** | **77.0** | **74.5** |
| **DINO** | **ViT-B/16** | **85M** | **78.2** | **76.1** |
| **DINO** | **ViT-S/8** | **21M** | **79.7** | **78.3** |
| **DINO** | **ViT-B/8** | **85M** | **80.1** | **77.4** |

**关键发现**：
- 在ResNet上DINO与SOTA持平
- 在ViT上DINO大幅超越其他方法（k-NN +7.9%）
- k-NN性能几乎接近线性评估（74.5% vs 77.0%），这种性质仅在DINO+ViT组合中出现
- 减小patch size（/8 vs /16）性能提升显著，无需增加参数

### 迁移学习

| 数据集 | 监督ViT-S | DINO ViT-S | 监督ViT-B | DINO ViT-B |
|--------|-----------|------------|-----------|------------|
| CIFAR-10 | 99.0 | 99.0 | 99.0 | 99.1 |
| CIFAR-100 | 89.5 | 90.5 | 90.8 | 91.7 |
| iNaturalist18 | 70.7 | 72.0 | 73.2 | 72.6 |
| iNaturalist19 | 76.6 | 78.2 | 77.7 | 78.6 |
| Flowers | 98.2 | 98.5 | 98.4 | 98.8 |
| Cars | 92.1 | 93.0 | 92.1 | 93.0 |
| ImageNet微调 | 79.9 | 81.5 | 81.8 | 82.8 |

- 自监督预训练的DINO在大多数下游任务上优于监督预训练

### 图像检索

- DINO ViT特征在Oxford和Paris数据集上超越监督特征
- 在Google Landmarks v2上训练的DINO超越之前所有off-the-shelf方法
- 拷贝检测任务上DINO ViT-B/16达到81.7% mAP（监督为76.4%）

### 视频物体分割（DAVIS 2017）

- DINO ViT-B/8: (J&F)m = 71.4，无需任何微调
- 小patch（/8）比大patch（/16）提升巨大（+9.1%）
- 超越监督ViT-S/8（66.0 vs 71.4）

### 消融实验核心发现

| 变体 | k-NN | Linear | 说明 |
|------|------|--------|------|
| DINO完整版 | 72.8 | 76.1 | baseline |
| 无Momentum | 0.1 | 0.1 | **完全崩溃** |
| 无Multi-crop | 67.9 | 72.5 | **下降3.6%** |
| 无Centering+Sharpening | - | - | **崩溃** |
| 用MSE替代CE | 52.6 | 62.4 | 大幅下降 |
| 加Predictor | 71.8 | 75.6 | 几乎无影响 |
| BYOL | 66.6 | 71.4 | - |
| MoCo-v2 | 62.0 | 71.6 | - |
| SwAV | 64.7 | 71.8 | - |

---

## 代码实现细节

### 核心文件结构

```
code/
├── main_dino.py          # 主训练脚本
├── vision_transformer.py  # ViT和DINOHead实现
├── utils.py              # 工具函数（数据增强、调度器等）
├── eval_knn.py           # k-NN评估
├── eval_linear.py        # 线性评估
├── visualize_attention.py # 注意力可视化
└── hubconf.py            # PyTorch Hub配置
```

### DINO Loss实现（main_dino.py）

```python
class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        self.student_temp = student_temp          # 0.1
        self.center_momentum = center_momentum    # 0.9
        self.ncrops = ncrops                      # 2 + local_crops_number
        self.register_buffer("center", torch.zeros(1, out_dim))  # 65536维
        
        # Teacher温度调度：warmup + 常数
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        # Student: 除以温度后分chunk（每个crop一个chunk）
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        
        # Teacher: centering + sharpening + softmax
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)  # 2个全局裁剪
        
        # 计算交叉熵：每个teacher view vs 每个student view（跳过相同view）
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:  # 跳过student和teacher处理同一view的情况
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        
        # 更新center
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        # 跨GPU同步求均值
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        # EMA更新
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
```

### 训练循环核心（main_dino.py: train_one_epoch）

```python
for it, (images, _) in enumerate(data_loader):
    images = [im.cuda(non_blocking=True) for im in images]
    
    with torch.cuda.amp.autocast(fp16_scaler is not None):
        # Teacher只看前2个（全局裁剪）
        teacher_output = teacher(images[:2])
        # Student看所有裁剪
        student_output = student(images)
        loss = dino_loss(student_output, teacher_output, epoch)
    
    # 反向传播只更新student
    loss.backward()
    clip_gradients(student, args.clip_grad)
    cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
    optimizer.step()
    
    # EMA更新teacher
    with torch.no_grad():
        m = momentum_schedule[it]
        for param_q, param_k in zip(student.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
```

### ViT架构（vision_transformer.py）

```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=[224], patch_size=16, embed_dim=768, 
                 depth=12, num_heads=12, ...):
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans=3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([Block(...) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.prepare_tokens(x)  # patch embed + cls token + pos embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]  # 返回[CLS] token
    
    def get_last_selfattention(self, x):
        # 用于可视化：返回最后一个block的注意力图
        ...
    
    def get_intermediate_layers(self, x, n=1):
        # 返回最后n个block的输出
        ...
```

**预定义ViT配置**：
| 模型 | embed_dim | depth | heads | 参数量 |
|------|-----------|-------|-------|--------|
| vit_tiny | 192 | 12 | 3 | ~5M |
| vit_small | 384 | 12 | 6 | 21M |
| vit_base | 768 | 12 | 12 | 85M |

### Multi-Crop数据增强实现

```python
class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # 两个全局裁剪（增强策略略有不同）
        self.global_transfo1 = Compose([
            RandomResizedCrop(224, scale=global_crops_scale, interpolation=BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),  # p=1.0 必定模糊
            normalize,
        ])
        self.global_transfo2 = Compose([
            RandomResizedCrop(224, scale=global_crops_scale, interpolation=BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),      # p=0.1 低概率模糊
            Solarization(0.2),      # 额外的solarization
            normalize,
        ])
        # 多个局部裁剪
        self.local_transfo = Compose([
            RandomResizedCrop(96, scale=local_crops_scale, interpolation=BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])
```

---

## 与当前研究的关联

### 对Slot/物体发现研究的价值

1. **涌现的语义分割能力**：DINO ViT的自注意力图自动对应物体边界（PASCAL VOC上Jaccard相似度达44.7-45.9），这意味着DINO特征天然适合物体发现任务

2. **无需监督的物体级特征**：
   - [CLS] token聚合全局语义信息
   - Patch tokens保留空间信息
   - 不同attention head关注不同语义区域/物体

3. **k-NN友好的特征**：DINO+ViT的特征空间具有极强的最近邻特性，这对基于聚类的物体发现方法非常有利

### 对视觉Transformer研究的价值

1. **自监督是释放ViT潜力的关键**：监督训练的ViT没有展现出这些涌现性质，说明自监督预训练对ViT特别重要

2. **小patch size的权衡**：
   - /8比/16性能提升显著（+2-3%线性精度，+3-7% k-NN）
   - 代价是吞吐量大幅下降（180 vs 1007 im/s for ViT-S）
   - 不增加参数但增加计算量

3. **BN-free系统的优势**：DINO+ViT完全不需要Batch Normalization，避免了分布式训练中BN同步的开销

### 对自监督学习方法论的贡献

1. **Centering + Sharpening替代对比损失**：简单的一阶统计量就足以防止崩溃，无需复杂的对比学习机制

2. **Momentum encoder的新理解**：不仅是MoCo中的队列替代品，更是一种持续的模型集成（Polyak-Ruppert平均）

3. **Multi-crop不是通用的**：不同框架对multi-crop的响应不同，DINO受益最大

### 潜在的改进方向

1. **DINO特征作为预训练backbone**：用DINO预训练的ViT替代随机初始化或监督预训练，可获得+1% ImageNet精度
2. **弱监督分割**：自注意力图中的物体边界信息可用于弱监督语义分割
3. **大规模无标注数据预训练**：论文提到未来计划在随机未筛选的图像上预训练大型ViT
