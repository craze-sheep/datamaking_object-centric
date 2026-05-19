#!/usr/bin/env bash
set -euo pipefail

cd /home/lzy/project/slot-datamaking/论文复现

conda run -n slotformer python test_vp_repro.py \
  --params /home/lzy/project/slot-datamaking/SlotFormer-master/slotformer/video_prediction/configs/slotformer_obj3d_params.py \
  --weight pretrained/slotformer_obj3d_params/model_200.pth \
  --batch_size 1
