# task1_副本

这个副本把任务拆成三个阶段：

1. `output/video/generate_video.py`
   - 生成 `frame/`
   - 合成 `rgb.mp4`
   - 生成 `object_segment/`，包含所有物体的 segment
   - 生成 `label/labels.json`，其中 `object_labels` 包含所有物体

2. `output/slot/extract_slots.py`
   - 读取 `video/label/labels.json` 的 `object_labels`
   - object 数记为 `n`
   - slot 数固定为 `6`
   - 使用 `SlotFormer-master/pretrained/steve_physion_params/model_10.pth`
   - 默认提取 6 帧
   - 为每个采样帧生成一个 `frame_XXXX/` slot 向量文件夹
   - 在 `segment/` 下保存每帧 slot segment

3. `output/alignment/align_objects_slots.py`
   - 读取 `video/object_segment/`
   - 读取 `slot/segment/`
   - 用像素重叠得分做 `n:6` 匈牙利匹配，从 6 个 slot 中选 n 个最佳匹配
   - 生成 `alignment_preview.png`、`alignment_result.json` 和 `alignment_result.md`

推荐运行顺序：

```bash
/root/project/kubric-main/run_with_blender.sh \
  /root/project/task1_副本/output/video/generate_video.py -- \
  --output_dir /root/project/task1_副本/output/video

python /root/project/task1_副本/output/slot/extract_slots.py

python /root/project/task1_副本/output/alignment/align_objects_slots.py
```
