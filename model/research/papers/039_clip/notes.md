# CLIP: Learning Transferable Visual Models From Natural Language Supervision

## 基本信息
- 作者：Alec Radford, Jong Wook Kim, Chris Hallacy, et al.
- 年份：2021
- 会议/期刊：ICML 2021
- 论文链接：https://arxiv.org/abs/2103.00020
- PDF：需下载

## 核心贡献
- 4 亿图文对对比学习，视觉-语言共享嵌入空间
- 零样本迁移能力强，无需微调即可分类
- 在多种下游任务上达到 SOTA
- 开源模型和代码，可直接使用

## 模型架构
- **双编码器**：
  - 图像编码器：ViT 或 ResNet
  - 文本编码器：Transformer
- **对比学习**：
  - 正样本：匹配的图文对
  - 负样本：不匹配的图文对
  - 损失：InfoNCE

## 与当前模型的对比
- **当前模型用 4 层 CNN，无预训练**
- **CLIP 提供语义对齐的视觉特征**
- **改进**：用 CLIP 替代 CNN

## 可借鉴的点

### 1. CLIP 特征 → 改进物体语义理解
**映射位置**：`model/ai_model/encoder.py`

**当前问题**：
- CNN 特征缺乏语义信息
- 无法理解物体的语义类别

**具体改进**：
```python
# CLIP 特征编码器
class CLIPVisualEncoder(nn.Module):
    def __init__(self):
        self.clip, _ = clip.load("ViT-B/16")
        self.clip.eval()
        
        for param in self.clip.parameters():
            param.requires_grad = False
        
        self.resize = nn.AdaptiveAvgPool2d((224, 224))
    
    def forward(self, rgb):
        B, T, C, H, W = rgb.shape
        x = rgb.reshape(B*T, C, H, W)
        x = self.resize(x)
        
        # 提取特征
        feat = self.clip.encode_image(x)  # [B*T, 512]
        
        return feat.reshape(B, T, 512)
```

**预期收益**：
- 语义对齐的视觉特征
- 零样本分类能力
- 预计物体分类准确率提升 10-15%

**实现难度**：中（需要安装 CLIP）

## 实验结果
| 数据集 | 零样本 Acc |
|--------|-----------|
| ImageNet | **76.2%** |
| CIFAR-10 | 88.9% |
| Food-101 | 85.3% |

- CLIP 零样本分类性能优秀
