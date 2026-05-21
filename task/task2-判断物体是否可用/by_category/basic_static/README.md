# basic_static

## 中文介绍

**基础静态物体**

这一类主要作为环境、约束和遮挡使用，包括墙、斜面、遮挡板和柱体。它们通常不主动运动，用来制造反弹、滑动、遮挡、障碍物绕行等物理情境。

**使用建议**：优先使用。适合和基础动态物体组合，构成第一版可控场景。

## 检查摘要

- 数量：5
- 状态统计：OK*=5
- 缩略图目录：`thumbnails/`
- 总览图：`contact_sheet.png`

## 对象明细

| object_id | 中文描述 | scenes | status | source | 结论 |
|---|---|---|---|---|---|
| `wall_x` | 沿 y 方向延展的竖直墙，用于墙面反弹和边界碰撞。 | S4/S13 | OK* | kb.Cube / PyBullet primitive | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 |
| `wall_y` | 沿 x 方向延展的竖直墙，可作为另一个方向的边界或反弹面。 | optional | OK* | kb.Cube / PyBullet primitive | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 |
| `ramp` | 斜面，用于生成重力驱动的滑动、滚动和底部停止事件。 | S3 | OK* | kb.Cube / PyBullet primitive | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 |
| `occluder` | 遮挡板，用于让物体短暂消失，测试身份保持和遮挡后推理。 | S13 | OK* | kb.Cube / PyBullet primitive | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 |
| `pillar` | 圆柱障碍物，用于制造绕行、碰撞、遮挡和复杂接触。 | S13 | OK* | kb.Cylinder / PyBullet primitive | 可用；属于基础静态碰撞几何，需在 Kubric 环境补齐依赖后批量渲染。 |
