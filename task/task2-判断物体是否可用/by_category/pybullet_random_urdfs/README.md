# pybullet_random_urdfs

## 中文介绍

**PyBullet 随机 URDF 物体库**

这一类代表 pybullet_data 中的 1000 个随机几何组合物体。它们适合做杂物、干扰物和 OOD 测试，但单个物体的形状、尺度和稳定性差异较大。

**使用建议**：后期使用。纳入数据集前应先做自动筛查，剔除尺寸异常、碰撞不稳定或视觉效果差的样本。

## 检查摘要

- 数量：1
- 状态统计：OK=1
- 缩略图目录：`thumbnails/`
- 总览图：`contact_sheet.png`

## 对象明细

| object_id | 中文描述 | scenes | status | source | 结论 |
|---|---|---|---|---|---|
| `random_urdfs/000-999` | 1000 个随机 URDF 物体集合，适合作为杂物或 OOD 测试库。 | S16 | OK | pybullet_data/random_urdfs | 当前环境可用；本机检测到 1000 个随机 URDF。 |
