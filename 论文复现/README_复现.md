# SlotFormer 论文复现工作区

本目录用于复现 `SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models`。

## 目录结构

- `data -> ../SlotFormer-master/data`: 复用已有数据。
- `pretrained -> ../SlotFormer-master/pretrained`: 复用已有预训练权重。
- `test_vp_repro.py`: 本地复现 runner，调用原始 `SlotFormer-master` 代码，但把输出写到本目录。
- `run_obj3d_visualization.sh`: 用预训练权重生成 OBJ3D 预测可视化。
- `run_obj3d_visualization_more.sh`: 生成更多 OBJ3D 预测可视化，默认 8 个样本。
- `run_obj3d_full_eval.sh`: 运行 OBJ3D 完整指标评测。
- `train_obj3d_slotformer.sh`: 从已有 OBJ3D slots 训练 SlotFormer 动力学模型。
- `vis/obj3d/slotformer_obj3d_params/`: 当前保留的复现结果。

## 当前已验证

已在 `slotformer` conda 环境中验证：

```bash
conda run -n slotformer python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

环境为 PyTorch 1.10.1，CUDA 可用，检测到 1 张 NVIDIA GeForce RTX 4060 Laptop GPU。

已跑通 OBJ3D 预训练模型的最小可视化：

```bash
bash /home/lzy/project/slot-datamaking/论文复现/run_obj3d_visualization.sh
```

输出：

```text
/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params.mp4
```

已额外跑通 8 个 OBJ3D 样本的可视化：

```text
/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params_8videos_h264.mp4
/home/lzy/project/slot-datamaking/论文复现/vis/obj3d/slotformer_obj3d_params/slotformer_obj3d_params_8videos.gif
```

## 复现命令

生成 OBJ3D 预测视频：

```bash
bash /home/lzy/project/slot-datamaking/论文复现/run_obj3d_visualization.sh
```

生成更多 OBJ3D 预测视频：

```bash
SAVE_NUM=8 bash /home/lzy/project/slot-datamaking/论文复现/run_obj3d_visualization_more.sh
```

运行 OBJ3D 完整评测：

```bash
bash /home/lzy/project/slot-datamaking/论文复现/run_obj3d_full_eval.sh
```

注意：完整评测会初始化 LPIPS，需要本机已有或能下载 VGG16 权重：

```text
~/.cache/torch/hub/checkpoints/vgg16-397923af.pth
```

从已有 slots 训练 OBJ3D SlotFormer：

```bash
bash /home/lzy/project/slot-datamaking/论文复现/train_obj3d_slotformer.sh
```

## 数据状态

当前可直接复现的是 OBJ3D 视频预测链路：

- `data/OBJ3D/obj3d_slots.pkl` 已存在。
- `data/OBJ3D/OBJ3D.zip` 已解压为 `train/val/test` 帧目录。
- `pretrained/savi_obj3d_params/model_40.pth` 和 `pretrained/slotformer_obj3d_params/model_200.pth` 已存在。

CLEVRER 目前只有 `data/CLEVRER/clevrer_slots.pkl`。若要复现 CLEVRER 视频预测完整指标，还需要 CLEVRER 原始视频帧、mask/attribute 标注；若要复现 VQA，还需要 questions/answers 和 rollout slots。
