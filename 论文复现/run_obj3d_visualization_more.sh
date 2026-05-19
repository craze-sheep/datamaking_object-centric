#!/usr/bin/env bash
set -euo pipefail

cd /home/lzy/project/slot-datamaking/论文复现

SAVE_NUM="${SAVE_NUM:-8}"
PARAMS="/home/lzy/project/slot-datamaking/SlotFormer-master/slotformer/video_prediction/configs/slotformer_obj3d_params.py"
WEIGHT="/home/lzy/project/slot-datamaking/SlotFormer-master/pretrained/slotformer_obj3d_params/model_200.pth"
OUT_DIR="/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params"

/home/lzy/miniconda3/envs/slotformer/bin/python test_vp_repro.py \
  --params "$PARAMS" \
  --weight "$WEIGHT" \
  --batch_size 1 \
  --save_num "$SAVE_NUM"

src_video="$OUT_DIR/slotformer_obj3d_params.mp4"
ffmpeg -y -loglevel error -i "$src_video" \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  "$OUT_DIR/slotformer_obj3d_params_${SAVE_NUM}videos_h264.mp4"
ffmpeg -y -loglevel error -i "$src_video" \
  -vf "fps=4,scale=544:-1:flags=lanczos" \
  "$OUT_DIR/slotformer_obj3d_params_${SAVE_NUM}videos.gif"

echo "Saved ${SAVE_NUM} OBJ3D SlotFormer visualizations to:"
echo "$src_video"
echo "$OUT_DIR/slotformer_obj3d_params_${SAVE_NUM}videos_h264.mp4"
echo "$OUT_DIR/slotformer_obj3d_params_${SAVE_NUM}videos.gif"
