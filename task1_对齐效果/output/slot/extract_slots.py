#!/usr/bin/env python3
"""Extract STEVE slots for task1_副本.

This script reads video/label/labels.json, sets num_slots to the number of
object_labels, runs STEVE with model_10.pth, and saves per-frame slots plus
slot segments.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms


REPO_ROOT = Path("/root/project")
DEFAULT_SLOTFORMER_ROOT = REPO_ROOT / "SlotFormer-master"
DEFAULT_WEIGHT_PATH = DEFAULT_SLOTFORMER_ROOT / "pretrained" / "steve_physion_params" / "model_10.pth"
FIXED_NUM_SLOTS = 6


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
    if count >= num_frames:
        return np.arange(num_frames, dtype=int)
    return np.linspace(0, num_frames - 1, count, dtype=int)


def preprocess_frames(frames: np.ndarray, resolution: tuple[int, int]) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tensors = [transform(frame.astype(np.uint8)) for frame in frames]
    return torch.stack(tensors, dim=0)


def load_state_dict_compatible(model: torch.nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_sd and model_sd[key].shape == value.shape:
            filtered[key] = value
        else:
            skipped.append(key)
    missing = [key for key in model_sd if key not in filtered]
    model.load_state_dict(filtered, strict=False)
    return skipped, missing


def palette_color(idx: int) -> tuple[int, int, int]:
    palette = [
        (231, 76, 60),
        (46, 204, 113),
        (52, 152, 219),
        (241, 196, 15),
        (155, 89, 182),
        (230, 126, 34),
        (26, 188, 156),
        (241, 90, 34),
    ]
    return palette[idx % len(palette)]


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for slot_idx in np.unique(mask):
        out[mask == slot_idx] = palette_color(int(slot_idx))
    return out


def overlay(rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(rgb * (1.0 - alpha) + mask_rgb * alpha, 0, 255).astype(np.uint8)


def add_title(image: Image.Image, title: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 22), fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))
    return out


def save_overview(sampled_frames: np.ndarray, hard_masks: np.ndarray, output_path: Path) -> None:
    tiles = []
    h, w = hard_masks.shape[-2:]
    for idx, frame in enumerate(sampled_frames):
        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        hard_rgb = mask_to_rgb(hard_masks[idx])
        tiles.extend(
            [
                add_title(Image.fromarray(resized), f"frame_{idx:04d}"),
                add_title(Image.fromarray(hard_rgb), "slot_segment"),
                add_title(Image.fromarray(overlay(resized, hard_rgb)), "overlay"),
            ]
        )

    columns = 3
    tile_w, tile_h = tiles[0].size
    rows = int(np.ceil(len(tiles) / columns))
    sheet = Image.new("RGB", (columns * tile_w, rows * tile_h), (245, 245, 245))
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % columns) * tile_w, (idx // columns) * tile_h))
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    video_dir = script_dir.parent / "video"
    parser = argparse.ArgumentParser()
    parser.add_argument("--slotformer_root", default=str(DEFAULT_SLOTFORMER_ROOT))
    parser.add_argument("--video_dir", default=str(video_dir))
    parser.add_argument("--label_json", default="")
    parser.add_argument("--video_path", default="")
    parser.add_argument("--output_dir", default=str(script_dir))
    parser.add_argument("--params_module", default="configs.steve_physion_params")
    parser.add_argument("--weight_path", default=str(DEFAULT_WEIGHT_PATH))
    parser.add_argument("--sample_count", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir).resolve()
    label_json = Path(args.label_json).resolve() if args.label_json else video_dir / "label" / "labels.json"
    video_path = Path(args.video_path).resolve() if args.video_path else video_dir / "rgb.mp4"
    output_dir = Path(args.output_dir).resolve()
    segment_dir = output_dir / "segment"
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.name == Path(__file__).name:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    segment_dir.mkdir(parents=True, exist_ok=True)

    with open(label_json, "r", encoding="utf-8") as f:
        labels = json.load(f)
    object_labels = labels.get("object_labels", [])
    object_count = len(object_labels)
    if object_count <= 0:
        raise ValueError(f"No object_labels found in {label_json}")
    num_slots = FIXED_NUM_SLOTS

    slotformer_root = Path(args.slotformer_root).resolve()
    os.chdir(slotformer_root)
    base_slots_dir = slotformer_root / "slotformer" / "base_slots"
    if str(base_slots_dir) not in sys.path:
        sys.path.insert(0, str(base_slots_dir))

    from models import build_model  # pylint: disable=import-outside-toplevel

    params_mod = importlib.import_module(args.params_module)
    params = params_mod.SlotFormerParams()
    params.slot_dict["num_slots"] = int(num_slots)
    params.input_frames = int(params.n_sample_frames)

    model = build_model(params)
    checkpoint = torch.load(args.weight_path, map_location="cpu")
    skipped_keys, missing_keys = load_state_dict_compatible(model, checkpoint["state_dict"])
    model.eval()
    model.testing = False

    frames = load_video_rgb(video_path)
    sample_count = int(args.sample_count) if int(args.sample_count) > 0 else 6
    frame_indices = sample_frame_indices(len(frames), sample_count)
    sampled_frames = frames[frame_indices]
    video_tensor = preprocess_frames(sampled_frames, params.resolution).unsqueeze(0)

    with torch.no_grad():
        out_dict = model({"img": video_tensor})

    slot_key = "post_slots" if "post_slots" in out_dict else "slots"
    mask_key = "post_masks" if "post_masks" in out_dict else "masks"
    slots = out_dict[slot_key].detach().cpu().numpy()
    masks = out_dict[mask_key].detach().cpu().numpy()
    if masks.ndim == 6:
        masks = masks[:, :, :, 0]
    slots = slots[0]
    masks = np.clip(masks[0], 0.0, 1.0)
    hard_masks = masks.argmax(axis=1).astype(np.int32)

    np.save(output_dir / "slots.npy", slots)
    np.save(output_dir / "slot_masks.npy", masks)
    np.save(segment_dir / "slot_hard_mask.npy", hard_masks)

    for local_idx, source_frame_idx in enumerate(frame_indices):
        frame_dir = output_dir / f"frame_{int(source_frame_idx):04d}"
        frame_segment_dir = segment_dir / f"frame_{int(source_frame_idx):04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_segment_dir.mkdir(parents=True, exist_ok=True)
        np.save(frame_dir / "all_slots.npy", slots[local_idx])
        for slot_idx in range(num_slots):
            np.save(frame_dir / f"slot_{slot_idx:02d}.npy", slots[local_idx, slot_idx])
            prob = np.clip(masks[local_idx, slot_idx] * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(prob).save(frame_segment_dir / f"slot_{slot_idx:02d}.png")
            np.save(frame_segment_dir / f"slot_{slot_idx:02d}.npy", masks[local_idx, slot_idx])
        Image.fromarray(mask_to_rgb(hard_masks[local_idx])).save(frame_segment_dir / "slot_hard_segment.png")

    save_overview(sampled_frames, hard_masks, output_dir / "slot_segment_overview.png")

    metadata = {
        "video_path": str(video_path),
        "label_json": str(label_json),
        "frame_indices": frame_indices.tolist(),
        "num_slots": int(num_slots),
        "object_count_from_labels": int(object_count),
        "matching_mode": f"{object_count}:{num_slots}",
        "object_labels": object_labels,
        "params_module": args.params_module,
        "weight_path": str(Path(args.weight_path).resolve()),
        "slots_shape": list(slots.shape),
        "slot_masks_shape": list(masks.shape),
        "hard_mask_shape": list(hard_masks.shape),
        "num_skipped_checkpoint_keys": len(skipped_keys),
        "num_missing_model_keys": len(missing_keys),
        "artifacts": {
            "slots_npy": str(output_dir / "slots.npy"),
            "slot_masks_npy": str(output_dir / "slot_masks.npy"),
            "segment_dir": str(segment_dir),
            "overview": str(output_dir / "slot_segment_overview.png"),
        },
    }
    with open(output_dir / "slot_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Wrote slot bundle to: {output_dir}")
    print(f"object_count={object_count}")
    print(f"num_slots={num_slots}")
    print(f"frame_indices={frame_indices.tolist()}")
    print(f"slots_shape={slots.shape}")
    print(f"slot_masks_shape={masks.shape}")


if __name__ == "__main__":
    main()
