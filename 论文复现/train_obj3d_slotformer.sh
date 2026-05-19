#!/usr/bin/env bash
set -euo pipefail

cd /home/lzy/project/slot-datamaking/论文复现

export WANDB_MODE=offline
export PYTHONPATH=/home/lzy/project/slot-datamaking/SlotFormer-master:${PYTHONPATH:-}

conda run -n slotformer python /home/lzy/project/slot-datamaking/SlotFormer-master/scripts/train.py \
  --task video_prediction \
  --params /home/lzy/project/slot-datamaking/SlotFormer-master/slotformer/video_prediction/configs/slotformer_obj3d_params.py \
  --fp16 \
  --cudnn
