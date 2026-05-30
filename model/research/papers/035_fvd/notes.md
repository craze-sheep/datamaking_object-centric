# FVD: Fréchet Video Distance

## 基本信息
- **论文标题**: Towards Accurate Generative Models of Video: A New Metric & Challenges
- **作者**: Thomas Unterthiner*, Sjoerd van Steenkiste*, Karol Kurach, Raphaël Marinier, Marcin Michalski, Sylvain Gelly (*共同一作)
- **机构**: Johannes Kepler University, IDSIA/SUPSI/USI, Google Brain
- **年份**: 2018 (arXiv: 1812.01717)
- **代码**: https://git.io/fpuEH
- **关键词**: 视频生成评估、Fréchet距离、I3D网络、StarCraft 2基准

## 核心贡献

1. **提出FVD指标**: 将FID (Fréchet Inception Distance) 从图像扩展到视频领域，利用I3D网络提取时空特征，计算真实视频与生成视频分布之间的Fréchet距离
2. **大规模人类评估验证**: 通过大规模人类评估实验，证实FVD与人类对生成视频质量的主观判断高度一致，显著优于PSNR、SSIM等传统指标
3. **引入SCV基准数据集**: 提出StarCraft 2 Videos (SCV) 基准套件，包含4个不同复杂度的游戏场景，专门测试长时记忆和关系推理能力
4. **综合基准测试**: 在BAIR、KTH和SCV数据集上评估了3000+个模型（超过100 GPU年的计算量），提供了全面的模型对比

## 模型架构

### 1. 整体思路

FVD的核心思想是：一个好的视频生成模型应该生成与真实视频分布相近的视频。通过比较两个分布的距离来评估生成质量。

### 2. Fréchet距离计算

基于Wasserstein-2距离（也称Fréchet距离），当分布假设为多元高斯分布时有闭式解：

```
d(P_R, P_G) = |μ_R - μ_G|² + Tr(Σ_R + Σ_G - 2(Σ_R · Σ_G)^{1/2})
```

其中：
- `μ_R, μ_G`: 真实和生成视频特征的均值向量
- `Σ_R, Σ_G`: 真实和生成视频特征的协方差矩阵
- `Tr`: 矩阵的迹

### 3. 特征提取：I3D网络

- **选择I3D (Inflated 3D Convnet)** 而非图像分类网络(如Inception)
- I3D将Inception架构扩展到时序数据，在Kinetics数据集上训练动作识别
- 动作识别需要同时考虑视觉上下文和时间演化，天然适合捕捉视频的时序一致性
- I3D在UCF101和HMDB51上取得SOTA结果

**特征提取流程**：
1. 使用在Kinetics-400上预训练的I3D网络
2. 将视频序列输入I3D，提取最后一层pooling层或logits层的激活值作为特征
3. 对真实视频和生成视频分别估计均值μ和协方差Σ
4. 代入Fréchet距离公式计算FVD

**关键发现**: logits层的I3D (Kinetics-400预训练) 在噪声检测实验中表现最佳，后续实验均采用此配置。

### 4. 替代方案：KVD (Kernel Video Distance)

- 使用MMD (Maximum Mean Discrepancy) 替代高斯假设
- 采用多项式核函数 `k(a,b) := (a^T b + 1)^3`
- 不假设分布的具体形式，直接在I3D特征上计算核距离
- 实验中KVD与FVD表现相近，但FVD在大多数场景下略优

### 5. 与FID的区别

| 方面 | FID (图像) | FVD (视频) |
|------|-----------|-----------|
| 特征网络 | Inception (ImageNet) | I3D (Kinetics) |
| 输入 | 单帧图像 | 视频序列 |
| 时序建模 | 无 | 有 (3D卷积) |
| 评估粒度 | 帧级 | 视频级 |

## 训练细节

### 评估的模型
- **CDNA**: Convolutional Dynamic Neural Advection
- **SV2P**: Stochastic Variational Video Prediction
- **SVP-FP**: SVP with Fixed Prior
- **SAVP**: Stochastic Adversarial Video Prediction

### 训练设置
- 使用Tensor2Tensor实现
- 学习率网格搜索: 10⁻³, 10⁻⁴, 10⁻⁵
- VAE模型的β (重建损失与KL散度权衡): 10⁻⁶, 10⁻⁵, 10⁻⁴, 10⁻³
- SV2P使用β退火策略
- SAVP额外调优GAN损失 (10⁻⁶ ~ 10⁻³)
- 所有模型训练300,000步

### FVD计算设置
- BAIR: 256个验证样本，2帧上下文 + 14帧输出
- KTH: 1024个样本，10帧上下文 + 10帧输出
- SCV: 1024个样本，2帧上下文 + 14/32帧输出
- **重要**: 比较不同模型的FVD时必须使用相同的样本数量

## 实验结果

### 1. 噪声敏感性实验

对真实视频添加8种噪声类型，测试FVD检测能力：

**静态噪声** (逐帧):
- 黑色矩形、高斯模糊、高斯噪声、椒盐噪声

**时序噪声** (整序列):
- 局部交换 (相邻帧互换)
- 全局交换 (随机帧互换)
- 交错 (多视频帧混合)
- 切换 (中途切换到另一视频)

**结果**: FVD对所有噪声类型都有良好检测能力，尤其在时序噪声上远超基于图像的FID。

### 2. 样本量影响

- 样本量越大，FVD估计越准确
- 固定样本量时标准误差很小，结果可复现
- 当底层分布相同时，FVD > 0（因参数估计噪声）
- **关键**: 不同模型比较时必须使用相同样本量

### 3. 人类评估 (核心实验)

**实验设计**:
- "One Metric Equal": 选择单一指标上值相近的10个模型，测试其他指标能否区分
- "One Metric Spread": 选择单一指标上值分散的10个模型，测试排名是否与人类一致

**主要结果** (与人类判断的一致率):

| 指标 | eq. FVD | eq. SSIM | eq. PSNR | spr. FVD | spr. SSIM | spr. PSNR |
|------|---------|----------|----------|----------|-----------|-----------|
| FVD | N/A | 74.9% | 81.0% | 71.9% | 58.4% | 63.5% |
| SSIM | 51.5% | N/A | 44.6% | 61.8% | 51.2% | 45.9% |
| PSNR | 56.3% | 21.4% | N/A | 54.1% | 37.0% | 44.8% |
| KVD | 40.6% | 70.4% | 73.8% | 69.4% | 56.8% | 63.8% |
| Avg. FID | 35.5% | 71.2% | 52.0% | 62.4% | 62.7% | 57.6% |
| 人类间一致率 | 79.3% | 77.8% | 84.4% | 83.3% | 69.9% | 72.5% |

**关键发现**:
- FVD在几乎所有场景下都优于其他指标
- FVD与PSNR的关联较弱(r=-0.278)，说明FVD捕捉了PSNR遗漏的信息
- SSIM与PSNR高度相关(r=0.730)，但两者与FVD相关性均较弱
- FVD差异≥50时，人类通常能感知到质量差异

### 4. SCV基准结果

| 模型 | BAIR | KTH | SCV-MUtB(64/128) | SCV-CMS | SCV-Brawl | SCV-RTwM |
|------|------|-----|-------------------|---------|-----------|----------|
| CDNA | 296.5 | 150.8 | 486.1/51.4 | 440.8/515.3 | 877.1/1016.6 | 1089.3/1295.4 |
| SV2P | 262.5 | 136.8 | 423.9/710.5 | 430.4/316.0 | 859.9/995.9 | 1068.7/1026.1 |
| SVP-FP | 315.5 | 208.4 | 276.7/121.3 | 379.8/442.1 | 714.5/1240.7 | 1022.9/2031.4 |
| SAVP | **116.4** | **78.0** | 479.7/204.4 | **188.8/192.5** | **192.9/150.3** | **698.6/1055.4** |

- SAVP在大多数场景下表现最好
- 所有模型在RTwM（需要长时记忆）上表现最差
- CMS场景中模型难以准确建模矿晶消失的时序关系
- Brawl场景中模型生成模糊的大色块，无法处理多实体交互

## 代码实现细节

代码地址: https://git.io/fpuEH (Google提供的官方实现)

核心计算流程：
```python
import tensorflow as tf
import tensorflow_hub as hub
from scipy.linalg import sqrtm
import numpy as np

def compute_fvd(real_videos, generated_videos):
    """
    计算 Fréchet Video Distance
    
    Args:
        real_videos: 真实视频 [N, T, H, W, 3], 像素值[0, 1]
        generated_videos: 生成视频 [N, T, H, W, 3], 像素值[0, 1]
    
    Returns:
        fvd: Fréchet Video Distance
    """
    # 1. 加载预训练I3D模型 (Kinetics-400, logits层)
    i3d = hub.Module('https://tfhub.dev/deepmind/i3d-kinetics-400/1')
    
    # 2. 预处理: resize到224x224, 像素值缩放到[-1, 1]
    # I3D输入: [batch, num_frames, 224, 224, 3]
    
    # 3. 提取特征
    real_features = i3d(real_videos)  # [N, 400] logits
    gen_features = i3d(generated_videos)
    
    # 4. 估计均值和协方差
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # 5. 计算Fréchet距离
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    
    # 处理数值误差(取实部)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fvd = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fvd
```

**注意事项**:
- 视频需resize到224×224
- I3D要求输入至少有一定数量的帧（通常16帧以上）
- 协方差矩阵计算需要足够的样本数，否则会不稳定
- `sqrtm`可能产生复数结果，需取实部

## 与当前研究的关联

### 对视频生成评估的启示
1. **FVD已成为视频生成领域的标准评估指标**，被广泛用于视频预测、视频生成、视频插帧等任务
2. **优于帧级指标**: PSNR/SSIM逐帧计算，无法评估时序一致性和整体分布质量
3. **无参考评估**: FVD不需要ground-truth配对，可用于无条件生成评估

### 对Slot-based视频理解的潜在价值
1. **评估Slot模型的时序一致性**: Slot-based模型需要保持物体slot在时间上的对应关系，FVD可评估这种一致性
2. **评估生成质量**: 如果Slot模型用于视频生成/预测，FVD是更合适的评估指标
3. **物体级评估**: Slot模型的物体分离特性可能使FVD在物体级别的特征空间中更有效

### 局限性
1. FVD对样本量敏感，不同样本量的结果不可直接比较
2. 高斯假设可能不完全成立，尤其在高维特征空间中
3. I3D在Kinetics上训练，可能不完全适合所有视频域（如游戏、医学图像等）
4. 计算FVD需要大量生成样本，计算成本较高
5. FVD差异<50时，人类几乎无法区分质量差异
