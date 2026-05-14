# gso

## 中文介绍

**Google Scanned Objects**

这一类是真实扫描物体资产，适合构造更接近真实世界的碰撞和遮挡场景。它们的外观和几何复杂度高，但下载、缓存、尺寸归一化和碰撞稳定性都需要额外验证。

**使用建议**：暂缓使用。本次当前工作区没有本地 GSO manifest，远端请求也失败，所以暂时标为 CHECK。

## 检查摘要

- 数量：1
- 状态统计：CHECK=1
- 缩略图目录：`thumbnails/`
- 总览图：`contact_sheet.png`

## 对象明细

| object_id | 中文描述 | scenes | status | source | 结论 |
|---|---|---|---|---|---|
| `Google Scanned Objects` | 真实扫描物体集合，适合高真实感场景，但当前尚未完成本地验证。 | S15/S16 | CHECK | gs://kubric-public/assets/GSO/GSO.json | 当前工作区没有本地 GSO manifest；远端 manifest 本次请求失败，暂不算已验证可用。 |
