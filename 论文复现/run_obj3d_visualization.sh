#!/usr/bin/env bash
set -euo pipefail

cd /home/lzy/project/slot-datamaking/论文复现

conda run -n slotformer python test_vp_repro.py \
  --params /home/lzy/project/slot-datamaking/SlotFormer-master/slotformer/video_prediction/configs/slotformer_obj3d_params.py \
  --weight pretrained/slotformer_obj3d_params/model_200.pth \
  --batch_size 1 \
  --save_num 2

src_video="/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params.mp4"
ffmpeg -y -loglevel error -i "$src_video" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params_h264.mp4"
ffmpeg -y -loglevel error -i "$src_video" \
  -vf "fps=4,scale=272:-1:flags=lanczos" \
  "/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params.gif"

echo "Saved visualization to:"
echo "/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params.mp4"
echo "/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params_h264.mp4"
echo "/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params.gif"
