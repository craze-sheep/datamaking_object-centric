# basic_dynamic

## 中文介绍

**基础动态物体**

这一类是数据集第一版最核心的可动物体，包括球、方块和圆柱。它们形状简单、物理稳定、属性容易控制，适合覆盖质量、摩擦、恢复系数、初始速度和碰撞结果等基础因果变量。

**使用建议**：优先使用。建议先用这些物体完成 S1-S8 的基础可控物理场景。

## 检查摘要

- 数量：9
- 状态统计：OK*=9
- 缩略图目录：`thumbnails/`
- 总览图：`contact_sheet.png`

## 对象明细

| object_id | 中文描述 | scenes | status | source | 结论 |
|---|---|---|---|---|---|
| `sphere_s` | 小球，适合自由落体、反弹、墙面碰撞和小尺度遮挡测试。 | S1/S4 | OK* | kb.Sphere / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
| `sphere_m` | 中球，基础场景主力物体，适合球撞球、球撞方块、负样本和遮挡场景。 | S1/S3/S5-S8/S13/S14 | OK* | kb.Sphere / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
| `sphere_l` | 大球，适合测试尺度变化对落体、反弹和墙面碰撞的影响。 | S1/S4 | OK* | kb.Sphere / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
| `cube_s` | 小方块，适合滑动停止、落体翻滚和多物体混合碰撞。 | S1/S2/S14 | OK* | kb.Cube / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
| `cube_m` | 中方块，球撞方块和遮挡场景中的主要目标物体。 | S1-S3/S6/S8/S13/S14 | OK* | kb.Cube / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
| `cube_l` | 大方块，适合测试尺度、质量和摩擦变化下的滑动距离。 | S1/S2/S14 | OK* | kb.Cube / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
| `cylinder_s` | 小圆柱，适合测试滚动、侧翻和接触面变化。 | S1/S2 | OK* | kb.Cylinder / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
| `cylinder_m` | 中圆柱，适合斜面滑动和复杂几何碰撞的基础对照。 | S1-S3/S9 | OK* | kb.Cylinder / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
| `cylinder_l` | 大圆柱，适合测试较大尺寸柱体的滚动和停止行为。 | S1/S2 | OK* | kb.Cylinder / PyBullet primitive | 物体定义可用；当前 Kubric Python 环境缺 pyquaternion，需补依赖后用 Kubric 渲染。 |
