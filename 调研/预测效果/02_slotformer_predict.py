#!/usr/bin/env python3
"""Extract STEVE slots and make a SlotFormer prediction comparison video."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


PROJECT_ROOT = Path("/home/lzy/project/slot-datamaking")
REPO_ROOT = PROJECT_ROOT / "SlotFormer-master"
NERV_ROOT = PROJECT_ROOT / "nerv-0.1.0"
for path in (str(REPO_ROOT), str(NERV_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from slotformer.base_slots.configs.steve_physion_params import SlotFormerParams as SteveParams  # noqa: E402
from slotformer.base_slots.models import build_model as build_steve_model  # noqa: E402
from slotformer.video_prediction.configs.slotformer_physion_params import SlotFormerParams as VPParams  # noqa: E402
from slotformer.video_prediction.models import build_model as build_vp_model  # noqa: E402


def read_video_rgb(path: Path, resolution: tuple[int, int], max_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, resolution[::-1], interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    if len(frames) < max_frames:
        raise RuntimeError(f"{path} only has {len(frames)} frames; need {max_frames}")
    return np.stack(frames, axis=0)


def frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    tensor = tensor * 2.0 - 1.0
    return tensor.unsqueeze(0).to(device)


def tensor_to_uint8(video: torch.Tensor) -> np.ndarray:
    video = video.detach().cpu().float()
    video = (video * 0.5 + 0.5).clamp(0, 1)
    video = video.permute(0, 2, 3, 1).numpy()
    return (video * 255.0).round().astype(np.uint8)


def add_border(frame: np.ndarray, color: tuple[int, int, int], width: int = 3) -> np.ndarray:
    out = frame.copy()
    out[:width, :, :] = color
    out[-width:, :, :] = color
    out[:, :width, :] = color
    out[:, -width:, :] = color
    return out


def label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (5, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def make_comparison(frames: np.ndarray, pred_future: np.ndarray, history_len: int) -> np.ndarray:
    green = (40, 190, 95)
    red = (220, 65, 55)
    bottom = []
    for idx in range(len(frames)):
        if idx < history_len:
            panel = label_frame(frames[idx], "input history")
            panel = add_border(panel, green)
        else:
            panel = label_frame(pred_future[idx - history_len], "SlotFormer prediction")
            panel = add_border(panel, red)
        bottom.append(panel)
    out_frames = []
    for idx, (gt, pred) in enumerate(zip(frames, bottom)):
        gt_panel = add_border(label_frame(gt, "ground truth"), green if idx < history_len else red)
        out_frames.append(np.concatenate([gt_panel, pred], axis=0))
    return np.stack(out_frames, axis=0)


def make_contact_sheet(frames: np.ndarray, pred_future: np.ndarray, history_len: int, output_path: Path) -> None:
    indices = [0, history_len - 1, history_len, len(frames) - 1]
    tiles = []
    for idx in indices:
        top = label_frame(frames[idx], f"GT frame {idx}")
        if idx < history_len:
            bottom = label_frame(frames[idx], f"input frame {idx}")
        else:
            bottom = label_frame(pred_future[idx - history_len], f"pred frame {idx}")
        tiles.append(np.concatenate([top, bottom], axis=0))
    sheet = np.concatenate(tiles, axis=1)
    imageio.imwrite(output_path, sheet)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", default="/home/lzy/project/slot-datamaking/调研/预测效果/kubric_input.mp4")
    parser.add_argument("--output_dir", default="/home/lzy/project/slot-datamaking/调研/预测效果")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This Physion STEVESlotFormer decoder expects CUDA in the upstream code.")

    history_len = 15
    rollout_len = 10
    total_frames = history_len + rollout_len
    resolution = (128, 128)

    frames = read_video_rgb(Path(args.input_video), resolution, total_frames)
    video_tensor = frames_to_tensor(frames, device)

    steve_params = SteveParams()
    steve_params.dvae_dict["dvae_ckp_path"] = str(REPO_ROOT / "pretrained/dvae_physion_params/model_20.pth")
    steve = build_steve_model(steve_params).to(device).eval()
    steve.load_state_dict(torch.load(REPO_ROOT / "pretrained/steve_physion_params/model_10.pth", map_location="cpu")["state_dict"])
    steve.testing = True
    steve_out = steve({"img": video_tensor})
    slots = steve_out["slots"].detach()
    masks = steve_out["masks"].detach().cpu().numpy()

    vp_params = VPParams()
    vp_params.dvae_dict["dvae_ckp_path"] = str(REPO_ROOT / "pretrained/dvae_physion_params/model_20.pth")
    vp_params.dec_dict["dec_ckp_path"] = str(REPO_ROOT / "pretrained/steve_physion_params/model_10.pth")
    vp_params.loss_dict["use_img_recon_loss"] = False
    vp = build_vp_model(vp_params).to(device).eval()
    vp.load_state_dict(torch.load(REPO_ROOT / "pretrained/slotformer_physion_params/model_25.pth", map_location="cpu")["state_dict"])

    out = vp({"slots": slots})
    pred_slots = out["pred_slots"]
    soft_recon, hard_recon = vp.decode(pred_slots.flatten(0, 1))
    pred_future = tensor_to_uint8(hard_recon)

    np.save(output_dir / "steve_slots.npy", slots.detach().cpu().numpy())
    np.savez_compressed(output_dir / "steve_masks.npz", masks=masks)
    np.save(output_dir / "slotformer_pred_slots.npy", pred_slots.detach().cpu().numpy())

    comparison = make_comparison(frames, pred_future, history_len)
    imageio.mimsave(output_dir / "comparison.mp4", list(comparison), fps=8, quality=8, macro_block_size=1)
    imageio.mimsave(output_dir / "prediction_future_only.mp4", list(pred_future), fps=8, quality=8, macro_block_size=1)
    make_contact_sheet(frames, pred_future, history_len, output_dir / "comparison_keyframes.png")

    summary = {
        "input_video": str(args.input_video),
        "steve_weight": str(REPO_ROOT / "pretrained/steve_physion_params/model_10.pth"),
        "slotformer_weight": str(REPO_ROOT / "pretrained/slotformer_physion_params/model_25.pth"),
        "dvae_weight": str(REPO_ROOT / "pretrained/dvae_physion_params/model_20.pth"),
        "history_len": history_len,
        "rollout_len": rollout_len,
        "video_tensor_shape": list(video_tensor.shape),
        "slots_shape": list(slots.shape),
        "masks_shape": list(masks.shape),
        "pred_slots_shape": list(pred_slots.shape),
        "outputs": {
            "comparison_video": str(output_dir / "comparison.mp4"),
            "prediction_future_only": str(output_dir / "prediction_future_only.mp4"),
            "keyframes": str(output_dir / "comparison_keyframes.png"),
            "slots": str(output_dir / "steve_slots.npy"),
            "masks": str(output_dir / "steve_masks.npz"),
        },
    }
    with open(output_dir / "slotformer_prediction_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
