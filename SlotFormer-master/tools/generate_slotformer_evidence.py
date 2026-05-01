#!/usr/bin/env python3
"""Generate a small, presentation-ready SlotFormer evidence bundle.

This script uses the pretrained STEVE Physion checkpoint to process one local
video, then saves:
  - a slot mask visualization PNG
  - a text log with loss / tensor shapes
  - a compact output directory that can be shown in slides
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SLOTS_DIR = REPO_ROOT / "slotformer" / "base_slots"
if str(BASE_SLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_SLOTS_DIR))

from configs.steve_physion_params import SlotFormerParams  # noqa: E402
from models import build_model  # noqa: E402


def load_video_rgb(video_path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def sample_frame_indices(num_frames: int, count: int) -> np.ndarray:
    return np.linspace(0, num_frames - 1, count, dtype=int)


def preprocess_frames(frames: np.ndarray, resolution: tuple[int, int]) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resolution),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    tensors = [transform(frame.astype(np.uint8)) for frame in frames]
    return torch.stack(tensors, dim=0)


def unnormalize_frame(frame_tensor: torch.Tensor) -> np.ndarray:
    frame = frame_tensor.detach().cpu()
    frame = frame * 0.5 + 0.5
    frame = frame.clamp(0, 1).permute(1, 2, 0).numpy()
    return (frame * 255).astype(np.uint8)


def colorize_mask(mask: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    mask = np.clip(mask, 0.0, 1.0)[..., None]
    color = np.array(rgb, dtype=np.float32).reshape(1, 1, 3)
    canvas = np.ones((*mask.shape[:2], 3), dtype=np.float32) * 255.0
    canvas = canvas * (1.0 - mask) + color * mask
    return canvas.astype(np.uint8)


def build_contact_sheet(
    original_frames: list[np.ndarray],
    slot_masks: np.ndarray,
    output_path: Path,
) -> None:
    colors = [
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
    ]
    tiles: list[Image.Image] = []

    for idx, frame in enumerate(original_frames):
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
        draw.text((6, 5), f"frame_{idx}", fill=(255, 255, 255))
        tiles.append(image)

    for slot_idx in range(slot_masks.shape[0]):
        image = Image.fromarray(colorize_mask(slot_masks[slot_idx], colors[slot_idx % len(colors)]))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, image.width, 22), fill=(0, 0, 0))
        draw.text((6, 5), f"slot_{slot_idx}", fill=(255, 255, 255))
        tiles.append(image)

    tile_w, tile_h = tiles[0].size
    columns = 3
    rows = int(np.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), color=(245, 245, 245))
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_w
        y = (idx // columns) * tile_h
        sheet.paste(tile, (x, y))
    sheet.save(output_path)


def main() -> None:
    input_video = Path(
        os.environ.get(
            "SLOTFORMER_EVIDENCE_VIDEO",
            "/root/project/kubric-main/output/mini_physics_clean/collision/rgb.mp4",
        )
    )
    output_dir = Path(
        os.environ.get(
            "SLOTFORMER_EVIDENCE_DIR",
            str(REPO_ROOT / "artifacts" / "slotformer_evidence"),
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    params = SlotFormerParams()
    weight_path = REPO_ROOT / "pretrained" / "steve_physion_params" / "model_10.pth"
    model = build_model(params)
    checkpoint = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.testing = False

    frames = load_video_rgb(input_video)
    frame_indices = sample_frame_indices(len(frames), params.n_sample_frames)
    sampled_frames = frames[frame_indices]
    video_tensor = preprocess_frames(sampled_frames, params.resolution).unsqueeze(0)

    with torch.no_grad():
        out_dict = model({"img": video_tensor})
        loss_dict = model.calc_train_loss({"img": video_tensor}, out_dict)

    slots = out_dict["slots"].cpu()
    masks = out_dict["masks"].cpu()
    token_loss = float(loss_dict["token_recon_loss"].item())

    middle_idx = sampled_frames.shape[0] // 2
    original_panels = [
        sampled_frames[0],
        sampled_frames[middle_idx],
        sampled_frames[-1],
    ]
    slot_masks = masks[0, middle_idx].numpy()
    vis_path = output_dir / "slots_visualization.png"
    build_contact_sheet(original_panels, slot_masks, vis_path)

    np.save(output_dir / "slots.npy", slots.numpy())

    summary = {
        "input_video": str(input_video),
        "sampled_frame_indices": frame_indices.tolist(),
        "video_tensor_shape": list(video_tensor.shape),
        "slots_shape": list(slots.shape),
        "masks_shape": list(masks.shape),
        "token_recon_loss": token_loss,
        "weight_path": str(weight_path),
        "visualization": str(vis_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log_path = output_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("SlotFormer STEVE evidence run\n")
        f.write(f"input_video: {input_video}\n")
        f.write(f"sampled_frame_indices: {frame_indices.tolist()}\n")
        f.write(f"video_tensor_shape: {tuple(video_tensor.shape)}\n")
        f.write(f"slots_shape: {tuple(slots.shape)}\n")
        f.write(f"masks_shape: {tuple(masks.shape)}\n")
        f.write(f"token_recon_loss: {token_loss:.6f}\n")
        f.write(f"weight_path: {weight_path}\n")
        f.write(f"saved_visualization: {vis_path}\n")

    print(f"Saved SlotFormer evidence to {output_dir}")
    print(f"token_recon_loss={token_loss:.6f}")
    print(f"slots_shape={tuple(slots.shape)}")
    print(f"masks_shape={tuple(masks.shape)}")


if __name__ == "__main__":
    main()
