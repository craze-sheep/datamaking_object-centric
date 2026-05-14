# pybullet_urdf

## 中文介绍

**PyBullet 常用 URDF 物体**

这一类是 pybullet_data 中已经实测可加载的日常物体和结构物，包括足球、乐高、杯子、托盘、多米诺和 Jenga。它们适合把数据集从抽象几何体扩展到更真实的刚体外观和复杂碰撞。

**使用建议**：第二批使用。建议先抽小样本检查尺寸、贴地高度、质量和碰撞稳定性。

## 检查摘要

- 数量：10
- 状态统计：OK=10
- 缩略图目录：`thumbnails/`
- 总览图：`contact_sheet.png`

## 对象明细

| object_id | 中文描述 | scenes | status | source | 结论 |
|---|---|---|---|---|---|
| `soccerball.urdf` | 足球外观球体，适合替代抽象球做真实外观碰撞。 | S10 | OK | pybullet_data | 当前环境可直接加载。 |
| `cube.urdf` | 标准方块 URDF，适合和原生方块做一致性对照。 | S10 | OK | pybullet_data | 当前环境可直接加载。 |
| `block.urdf` | 细长块状物，适合测试小尺寸目标被撞后的位移和旋转。 | S10 | OK | pybullet_data | 当前环境可直接加载。 |
| `lego/lego.urdf` | 乐高积木，适合引入轻量真实物体外观和凸起结构。 | S10 | OK | pybullet_data | 当前环境可直接加载。 |
| `duck_vhacd.urdf` | 小黄鸭，带 VHACD 碰撞近似，适合日常物体碰撞扩展。 | S10 | OK | pybullet_data | 当前环境可直接加载。 |
| `teddy_vhacd.urdf` | 泰迪熊，带 VHACD 碰撞近似，适合非规则外形测试。 | S10 | OK | pybullet_data | 当前环境可直接加载。 |
| `objects/mug.urdf` | 杯子，典型日常容器物体，适合真实外观和非规则接触测试。 | S10 | OK | pybullet_data | 当前环境可直接加载。 |
| `tray/tray.urdf` | 托盘，尺寸较大且有边缘结构，适合作为容器、障碍或被撞目标。 | S10 | OK | pybullet_data | 当前环境可直接加载。 |
| `domino/domino.urdf` | 多米诺骨牌，适合连锁倒塌和接触传播场景。 | S11 | OK | pybullet_data | 当前环境可直接加载。 |
| `jenga/jenga.urdf` | Jenga 木块，适合堆叠稳定性和坍塌事件场景。 | S12 | OK | pybullet_data | 当前环境可直接加载。 |
