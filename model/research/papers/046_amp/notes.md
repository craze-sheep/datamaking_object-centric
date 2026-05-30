# AMP: Mixed Precision Training

## 基本信息
- **标题**: Mixed Precision Training
- **作者**: Paulius Micikevicius\*, Sharan Narang\*, Jonah Alben, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, Hao Wu (NVIDIA); Gregory Diamos, Erich Elsen (Baidu Research)
- **年份**: 2018
- **会议**: ICLR 2018
- **论文链接**: https://arxiv.org/abs/1710.03740
- **代码仓库**: NVIDIA Apex (`/code/` 目录为 apex 库)

## 核心贡献

1. **系统性混合精度训练方法**: 提出了一套完整的 FP16 训练方法论，不损失模型精度，无需调整超参数
2. **三大关键技术**:
   - FP32 主权重副本 (Master Weights)
   - Loss Scaling（损失缩放）
   - FP16 算术 + FP32 累加
3. **广泛验证**: 在分类、检测、语音识别、机器翻译、语言建模、GAN 生成等任务上验证有效性
4. **成为行业标准**: AMP 混合精度训练已成为大模型训练的标配技术

## 模型架构

### FP16/BF16 精度选择

- **IEEE FP16 (半精度)**: 1位符号 + 5位指数 + 10位尾数，动态范围较窄（最小正规化数 ≈ 2⁻¹⁴，最大值 ≈ 65,504）
- **BF16**: 1位符号 + 8位指数 + 7位尾数，动态范围与 FP32 相同但精度更低
- **本文使用 FP16**，权重、激活值、梯度均以 FP16 存储和计算

### 三大核心技术

#### 1. FP32 主权重副本
- **问题**: FP16 的权重更新值可能过小（绝对值 < 2⁻²⁴ 的值在 FP16 中变为零），或权重值与更新值比值过大（> 2048）导致更新被舍入为零
- **方案**: 维护一份 FP32 主权重副本，每次优化器步进时在 FP32 精度下累积梯度更新；前向/反向传播使用 FP16 副本
- **内存影响**: 权重内存增加 50%，但训练内存主要由激活值占据，激活值以 FP16 存储，总体内存仍减半

#### 2. Loss Scaling
- **问题**: 激活梯度值大量集中在小幅度区间，FP16 表示范围的低端未被利用，大量梯度值下溢为零（如 Multibox SSD 中 67% 的梯度为零）
- **方案**: 前向传播后将 loss 乘以缩放因子 S，通过链式法则使所有梯度值放大 S 倍，保留原本下溢的小梯度；反向传播后将梯度除以 S 恢复原始量级
- **缩放因子选择**:
  - 常数缩放: 经验选择 8~32K，确保最大梯度值 × S < 65,504
  - **动态缩放**: 监测梯度溢出，溢出时跳过更新并减小 S，无溢出时逐步增大 S（论文提到作为未来工作）
- **实验验证**: Multibox SSD 无 loss scaling 时训练发散，缩放因子 8 即可匹配 FP32 精度

#### 3. 算术精度
- **向量点积** (卷积/全连接): FP16 乘法 + FP32 累加，再转回 FP16 存储（Volta Tensor Core 原生支持）
- **大范围归约** (BatchNorm 统计、Softmax): 在 FP32 中执行，读写仍为 FP16
- **逐点操作** (激活函数、逐元素乘): 内存带宽受限，FP16 或 FP32 均可

### 训练流程图解
```
FP32 主权重 ──(转FP16)──> FP16 权重 ──> 前向传播(FP16) ──> FP16 激活值
                                                              │
                                                              ▼
                                                        计算 loss (FP16)
                                                              │
                                                              ▼
                                                     loss × scale_factor
                                                              │
                                                              ▼
                                               反向传播(FP16, FP32累加)
                                                              │
                                                              ▼
                                              FP16 梯度 ÷ scale_factor
                                                              │
                                                              ▼
                                            更新 FP32 主权重(梯度以FP32累积)
```

## 训练细节

- **框架**: Caffe (分类/检测)、PyTorch (ResNet50)、TensorFlow (翻译)
- **硬件**: NVIDIA Volta V100 (Tensor Core)、Maxwell GPU (伪 FP16 模式)
- **超参数**: 与 FP32 基线完全一致，无需调整学习率、动量等
- **优化器**: SGD (分类/语音)、Adam (GAN)、Adagrad (语言模型)
- **BatchNorm**: 建议保持 FP32（keep_batchnorm_fp32）以维持数值稳定性
- **数据格式**: 支持 channels_last 内存格式以进一步加速

## 实验结果

### ILSVRC 图像分类
| 模型 | FP32 基线 | 混合精度 | 参考 |
|------|-----------|----------|------|
| AlexNet | 56.77% | **56.93%** | Krizhevsky 2012 |
| VGG-D | 65.40% | **65.43%** | Simonyan 2014 |
| GoogLeNet | 68.33% | **68.43%** | Szegedy 2015 |
| Inception v2 | 70.03% | 70.02% | Ioffe 2015 |
| Inception v3 | 73.85% | **74.13%** | Szegedy 2016 |
| ResNet50 | 75.92% | **76.04%** | He 2016 |

- 不需要 Loss Scaling，均使用 FP32 主权重

### 目标检测
| 模型 | FP32 基线 | MP (无 loss-scale) | MP (有 loss-scale) |
|------|-----------|--------------------|--------------------|
| Faster R-CNN | 69.1% | 68.6% | **69.7%** |
| Multibox SSD | 76.9% | **发散** | **77.1%** |

- SSD 必须使用 loss scaling（缩放因子 8），否则梯度下溢导致发散

### 语音识别 (DeepSpeech 2)
| 数据集 | FP32 CER | 混合精度 CER |
|--------|----------|-------------|
| 英语 | 2.20 | **1.99** |
| 普通话 | 15.82 | **15.01** |

- 模型规模: 英语 115M 参数，普通话 215M 参数（本文最大模型）
- 混合精度结果略优于 FP32，可能是因为 FP16 存储起到了正则化作用

### 机器翻译 (英法)
- 3 层 / 5 层 LSTM + Attention，使用 WMT15 数据集
- 带 loss scaling 的混合精度匹配 FP32 结果
- 不带 loss scaling 有轻微退化

### 语言建模 (bigLSTM)
- 两层 8192 LSTM + 1024 维投影，793K 词表，1B Word 数据集
- **必须使用 loss scaling**（缩放因子 128），否则 300K 迭代后发散

### DCGAN 人脸生成
- 定性评估: FP32 与混合精度输出质量相当
- 不需要 loss scaling

## 代码实现细节

### 代码仓库: NVIDIA Apex
- `/code/` 目录为 **NVIDIA Apex** 库，是早期实现 AMP 的主要工具

### 核心组件

#### 1. `apex.amp` (自动混合精度)
- 旧版 API: `model, optimizer = amp.initialize(model, optimizer, opt_level="O1")`
- `opt_level` 选项:
  - **O0**: 纯 FP32（基线）
  - **O1**: 推荐，白名单操作用 FP16，其余 FP32
  - **O2**: 尽可能用 FP16，保留 FP32 BatchNorm
  - **O3**: 纯 FP16（调试用）

#### 2. `torch.amp` (PyTorch 原生，当前推荐)
- **GradScaler**: 自动 loss scaling
  ```python
  scaler = torch.amp.GradScaler("cuda")
  ```
- **autocast**: 自动选择精度上下文
  ```python
  with torch.autocast(device_type="cuda"):
      output = model(input)
      loss = criterion(output, target)
  ```
- 完整训练循环:
  ```python
  scaler = torch.amp.GradScaler("cuda")
  for input, target in loader:
      optimizer.zero_grad()
      with torch.autocast(device_type="cuda"):
          output = model(input)
          loss = criterion(output, target)
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
  ```

#### 3. `apex.optimizers.FusedAdam`
- 融合 Adam 的逐元素操作 + multi-tensor apply，减少 kernel 启动开销
- 支持 `master_weights=True` 维护 FP32 主权重
- 支持 `capturable=True` 用于 CUDA Graphs
- 分别处理 FP16、BF16、FP32 参数组

#### 4. `apex.parallel.DDP` & `SyncBatchNorm`
- 分布式训练支持，SyncBN 在多 GPU 间同步统计量

### 关键实现模式
```python
# 旧版 Apex 方式
from apex import amp
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
with amp.scale_loss(loss, optimizer) as scaled_loss:
    scaled_loss.backward()

# 现代 PyTorch 原生方式 (推荐)
scaler = torch.amp.GradScaler("cuda")
with torch.autocast(device_type="cuda"):
    loss = model(input).sum()
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # 可选，用于梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

## 与当前研究的关联

1. **大模型训练基础设施**: AMP 是所有现代大模型 (GPT, LLaMA, etc.) 训练的标准组件，节省 50% 显存和通信带宽
2. **BF16 的崛起**: 论文使用 FP16 + loss scaling，但随着 A100/H100 GPU 普及，BF16 因无需 loss scaling（动态范围与 FP32 相同）而逐渐成为主流
3. **多精度训练范式**: FP8 训练 (Transformer Engine)、INT8 推理等技术均建立在混合精度训练的基础上
4. **分布式训练优化**: 梯度通信使用 FP16/BF16 可将 AllReduce 通信量减半
5. **内存-速度权衡**: 激活值用低精度存储 → 可用更大 batch size → 提高吞吐量，同时 gradient checkpointing 等技术可进一步配合使用
