# CLIP: Contrastive Language-Image Pre-training

## 基本信息
- **标题**: Learning Transferable Visual Models From Natural Language Supervision
- **作者**: Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh 等 (OpenAI)
- **年份**: 2021
- **会议**: ICML 2021
- **论文链接**: https://arxiv.org/abs/2103.00020
- **代码**: https://github.com/OpenAI/CLIP

## 核心贡献
1. **对比语言-图像预训练**: 在 4 亿 (image, text) 对（WIT 数据集）上训练，学习通用视觉-语言表征
2. **高效对比目标函数**: 用对比学习替代预测式目标，效率提升 4x（比 bag-of-words 预测）和 12x（比语言模型预测）
3. **零样本迁移能力**: 零样本 ImageNet 达 76.2%，匹配原始 ResNet-50 监督性能
4. **自然语言监督的可行性**: 首次大规模验证自然语言监督可学习 SOTA 图像表征
5. **鲁棒性优势**: 零样本 CLIP 对分布偏移的鲁棒性远超 ImageNet 训练的模型（有效鲁棒性提升高达 75%）

## 模型架构

### 整体结构
CLIP 由两个编码器组成，通过对比学习对齐到共享多模态嵌入空间：

```
图像 → Image Encoder → 线性投影 → 图像嵌入 (L2归一化)
                                        ↓ 余弦相似度
文本 → Text Encoder  → 线性投影 → 文本嵌入 (L2归一化)
```

### 图像编码器（两种选择）
- **Modified ResNet**: 基于 ResNet-D + antialiased blur pooling，最后用注意力池化替代全局平均池化
  - RN50, RN101, RN50x4, RN101x16, RN50x64
- **Vision Transformer (ViT)**: 在 patch+position embedding 前添加 LayerNorm
  - ViT-B/32, ViT-B/16, ViT-L/14, ViT-L/14@336px

### 文本编码器
- 12 层 Transformer，512 宽，8 注意力头（63M 参数）
- 输入：小写 BPE 编码，词表 49,152，最大序列长度 76
- 输出：[EOS] token 的最高层激活 → LayerNorm → 线性投影到多模态空间
- 使用因果（masked）自注意力

### 对比损失（InfoNCE / 对称交叉熵）
```
给定 batch 中 N 个 (image, text) 对：
1. 提取特征: I_f = image_encoder(I), T_f = text_encoder(T)
2. 线性投影 + L2 归一化: I_e = L2(I_f @ W_i), T_e = L2(T_f @ W_t)
3. 计算相似度矩阵: logits = (I_e @ T_e.T) * exp(τ)
4. 对称损失:
   loss_i = cross_entropy(logits, labels, axis=0)  # 图像→文本
   loss_t = cross_entropy(logits, labels, axis=1)  # 文本→图像
   loss = (loss_i + loss_t) / 2
```
- τ 是可学习的温度参数，初始化为 log(1/0.07)，clip 到最大 logit scale 100
- 等价于 multi-class N-pair loss (Sohn 2016) / InfoNCE loss (Oord 2018)

### 零样本推理
- 将类别名填入 prompt 模板（如 "A photo of a {label}."）
- 用文本编码器生成分类器权重
- 本质是：L2 归一化的多类逻辑回归，无偏置，带温度缩放
- 文本编码器充当 hypernetwork，根据自然语言描述生成线性分类器

### Prompt Engineering & Ensembling
- **Prompt 模板**: "A photo of a {label}." 比直接用类名提升 1.3%（ImageNet）
- **任务特定模板**: 细粒度分类加类别描述（如 "a type of pet"），OCR 加引号等
- **Ensemble**: 80 个不同 prompt 的嵌入平均，额外提升 3.5%（ImageNet）
- prompt 工程 + ensemble 合计提升约 5%

## 训练细节
- **数据集**: WIT (WebImageText)，4 亿 (image, text) 对，从互联网收集
- **优化器**: Adam + 解耦权重衰减
- **学习率**: cosine schedule
- **Batch size**: 32,768（超大 batch）
- **Epochs**: 32
- **混合精度训练**: fp16，半精度 Adam 统计，梯度检查点
- **数据增强**: 仅随机裁剪（resize 后）
- **最大模型**: RN50x64 训练 18 天（592 V100），ViT-L/14 训练 12 天（256 V100）
- **ViT-L/14@336px**: 额外 1 epoch 高分辨率微调

## 实验结果

### 零样本迁移（vs Visual N-Grams）
| 数据集 | Visual N-Grams | CLIP |
|--------|---------------|------|
| aYahoo | 72.4 | 98.4 |
| ImageNet | 11.5 | **76.2** |
| SUN | 23.0 | 58.5 |

### 零样本 vs 全监督线性探针（27 数据集）
- 零样本 CLIP 在 16/27 数据集上超过 ResNet-50 全监督线性分类器
- 优势最大：Stanford Cars (+28.9), Country211 (+23.2), Food101 (+22.5), Kinetics700 (+14.5)
- 劣势：Flowers102 (-12.5), FGVCAircraft (-11.3), MNIST (-10.0)

### 表征学习（线性探针）
- 最佳模型 ViT-L/14@336px 在 27 数据集上超越所有现有模型平均 5%
- CLIP ViT 比 CLIP ResNet 计算效率高约 3x

### 鲁棒性（分布偏移）
- 零样本 CLIP 在 7 个自然分布偏移数据集上大幅超越 ImageNet 模型
- 有效鲁棒性提升高达 75%
- ImageNet 适应后（线性分类器）：ImageNet 准确率 +9.2%，但分布偏移性能下降
- 说明过拟合 ImageNet 分布会损害泛化

### 数据效率
- 零样本 CLIP ≈ 4-shot 线性分类器（同特征空间）
- 中位数：5.4 标注样本/类别即可匹配零样本性能
- ImageNet 上等效于 16-shot

## 代码实现细节

### 模型变体（`clip/model.py`）
- `CLIP` 主类：包含 `visual`（图像编码器）和 `transformer`（文本编码器）
- `ModifiedResNet`: 3 层 stem → 4 层 residual blocks → `AttentionPool2d`
- `VisionTransformer`: Conv patch embedding → [CLS] + positional embedding → Transformer → LN → 投影
- `AttentionPool2d`: 单层多头 QKV 注意力，query 基于全局平均池化
- `QuickGELU`: `x * sigmoid(1.702 * x)`，比标准 GELU 更快
- `LayerNorm`: fp16 兼容版本（先转 fp32 再转回）
- `build_model()`: 从 state_dict 自动推断架构参数

### 加载与推理（`clip/clip.py`）
- 支持 9 个预训练模型（5 ResNet + 4 ViT）
- `tokenize()`: BPE 编码，context_length=77，SOS/EOS token 包围
- 图像预处理：Resize → CenterCrop → RGB → Normalize（均值和标准差固定）
- 支持 JIT 和非 JIT 加载模式

### 关键设计选择
- 不使用非线性投影头（仅线性投影）——与 SimCLR/BYOL 不同
- 不使用图像编码器 ImageNet 预训练初始化
- 不使用文本变换函数（如随机采样句子）
- 温度参数可学习，避免超参搜索

## 与当前研究的关联

### 对比学习在视觉-语言中的奠基作用
- CLIP loss（对称 InfoNCE）成为后续多模态模型的标准损失：
  - CLIP4Clip（视频-文本检索）
  - BLIP/BLIP-2（图像-文本生成）
  - CLIPSeg（图像分割）
  - StyleCLIP（图像编辑）

### 在 slot-based 模型中的应用潜力
1. **特征对比约束**: 当前 slot 模型缺乏对比约束，不同物体的 slot 特征可能不够区分
2. **跨模态对齐**: CLIP 的对齐思想可用于将 slot 特征与文本/语义对齐
3. **零样本评估**: 利用 CLIP 嵌入空间评估 slot 质量
4. **温度参数设计**: CLIP 的可学习温度参数可借鉴用于 slot 间的相似度计算

### 局限性
- 不擅长细粒度空间推理、计数等复杂任务
- 仍会利用虚假相关性（尽管比监督模型好很多）
- 4 亿数据集的需求限制了可复现性
