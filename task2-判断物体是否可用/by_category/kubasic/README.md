# kubasic

## 中文介绍

**KuBasic 复杂几何体**

这一类来自 Kubric 官方 KuBasic 资产，包含圆锥、圆环、齿轮、茶壶、猴头等复杂形状。它们比基础几何体更能测试模型对形状差异、旋转和复杂接触面的泛化能力。

**使用建议**：第二批使用。当前已确认远端 asset id 存在，但本地还需要补齐 Kubric 环境并缓存资产后再批量生成。

## 检查摘要

- 数量：8
- 状态统计：OK*=8
- 缩略图目录：`thumbnails/`
- 总览图：`contact_sheet.png`

## 对象明细

| object_id | 中文描述 | scenes | status | source | 结论 |
|---|---|---|---|---|---|
| `cone` | 圆锥体，接触面不对称，适合测试复杂形状碰撞和旋转。 | S9 | OK* | gs://kubric-public/assets/KuBasic/KuBasic.json | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 |
| `torus` | 圆环体，适合测试非凸外观、滚动和孔洞形状带来的视觉泛化。 | S9 | OK* | gs://kubric-public/assets/KuBasic/KuBasic.json | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 |
| `gear` | 齿轮形物体，边缘复杂，适合测试复杂接触和旋转。 | S9 | OK* | gs://kubric-public/assets/KuBasic/KuBasic.json | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 |
| `torus_knot` | 扭结圆环，外形复杂，适合作为形状泛化和视觉分割难例。 | S9 | OK* | gs://kubric-public/assets/KuBasic/KuBasic.json | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 |
| `sponge` | 多孔复杂几何体，适合作为复杂形状碰撞扩展。 | S9 | OK* | gs://kubric-public/assets/KuBasic/KuBasic.json | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 |
| `spot` | KuBasic 中的 Spot 模型，适合测试非基础几何体的外观泛化。 | S9 | OK* | gs://kubric-public/assets/KuBasic/KuBasic.json | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 |
| `teapot` | 茶壶模型，典型复杂网格物体，适合真实物体前的过渡测试。 | S9 | OK* | gs://kubric-public/assets/KuBasic/KuBasic.json | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 |
| `suzanne` | Blender 猴头模型，常用复杂几何基准，适合测试视角和形状泛化。 | S9 | OK* | gs://kubric-public/assets/KuBasic/KuBasic.json | 远端 manifest 中存在该 asset id；本地未下载，需用 Kubric AssetSource 下载/缓存。 |
