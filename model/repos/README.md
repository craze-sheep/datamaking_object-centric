# 代码收集记录

> 收集时间：2026-05-28
> 用途：物理视频预测模型开发参考

---

## 已收集代码库

### 视觉编码器

| 代码库 | 来源 | 用途 | 可直接用 |
|--------|------|------|----------|
| `dinov2` | facebookresearch | DINOv2视觉编码器，冻结层提取特征 | ✅ 直接import |
| `slot-attention-pytorch` | evelinehong | Slot Attention PyTorch实现 | ⚠️ 参考实现 |

### 视频预测Baseline

| 代码库 | 来源 | 用途 | 可直接用 |
|--------|------|------|----------|
| `SimVPv2` | chengtan9907 | SimVP视频预测模型 | ✅ 可跑baseline |
| `predrnn-pytorch` | thuml | PredRNN视频预测 | ✅ 可跑baseline |
| `PhyDNet` | vincent-leguen | 物理感知视频预测 | ⚠️ 需适配数据格式 |
| `earth-forecasting-transformer` | amazon-science | Earthformer时空预测 | ⚠️ 参考架构 |

### 物体中心学习

| 代码库 | 来源 | 用途 | 可直接用 |
|--------|------|------|----------|
| `SAVi-pytorch` | junkeun-yi | SAVi PyTorch实现 | ⚠️ 参考实现 |

### GNN/图网络

| 代码库 | 来源 | 用途 | 可直接用 |
|--------|------|------|----------|
| `pytorch_geometric` | pyg-team | PyTorch Geometric图神经网络 | ✅ 直接import |
| `graph_nets` | deepmind | DeepMind GNS物理模拟参考 | ⚠️ 参考实现 |
| `mamba` | state-spaces | Mamba SSM时序模型 | ⚠️ 参考架构 |

### Loss函数

| 代码库 | 来源 | 用途 | 可直接用 |
|--------|------|------|----------|
| `PerceptualSimilarity` | richzhang | LPIPS感知损失 | ✅ 直接import |

---

## 依赖安装

```bash
# PyTorch Geometric
pip install torch_geometric

# DINOv2
pip install dinov2

# LPIPS
pip install lpips

# Mamba
pip install mamba-ssm
```

## 使用说明

- ✅ 标记的库可直接用于模型开发
- ⚠️ 标记的库需要参考其实现，适配我们的数据格式
- Baseline代码可用于对比实验
