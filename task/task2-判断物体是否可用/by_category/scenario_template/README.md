# scenario_template

## 中文介绍

**场景模板**

这一类不是单个物体资产，而是可复用的生成脚本或场景配置。比如 bouncing_balls 可以快速生成多物体弹跳和碰撞视频，用来验证数据流程、轨迹标注和模型输入格式。

**使用建议**：可作为调试模板使用。正式数据集仍建议把物体、属性和事件标注显式写入 manifest。

## 检查摘要

- 数量：1
- 状态统计：OK*=1
- 缩略图目录：`thumbnails/`
- 总览图：`contact_sheet.png`

## 对象明细

| object_id | 中文描述 | scenes | status | source | 结论 |
|---|---|---|---|---|---|
| `bouncing_balls: sphere/cube/mixed` | 多物体弹球场景模板，用于快速验证多物体碰撞和轨迹预测流程。 | S14 | OK* | kubric examples/bouncing_balls.py | 场景模板和物体类型设计可用；仍受当前 Kubric 依赖缺失影响。 |
