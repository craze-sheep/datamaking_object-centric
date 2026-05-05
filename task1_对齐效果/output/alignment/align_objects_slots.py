#!/usr/bin/env python3
"""Align Kubric object segments to STEVE slot segments with n:n matching."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment


def load_labels(label_json: Path) -> dict:
    with open(label_json, "r", encoding="utf-8") as f:
        return json.load(f)


def load_slot_metadata(metadata_json: Path) -> dict:
    with open(metadata_json, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rgb_frame(video_dir: Path, frame_idx: int, width: int, height: int) -> np.ndarray:
    frame_path = video_dir / "frame" / f"frame_{frame_idx:04d}.png"
    if frame_path.exists():
        frame = np.array(Image.open(frame_path).convert("RGB"))
    else:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def read_object_masks(video_dir: Path, object_labels: list[dict], frame_indices: list[int], width: int, height: int) -> np.ndarray:
    all_masks = []
    for frame_idx in frame_indices:
        frame_masks = []
        for obj in object_labels:
            name = obj["name"]
            png_path = video_dir / "object_segment" / f"frame_{frame_idx:04d}" / f"{name}.png"
            npy_path = video_dir / "object_segment" / f"frame_{frame_idx:04d}" / f"{name}.npy"
            if npy_path.exists():
                mask = np.load(npy_path).astype(bool)
            elif png_path.exists():
                mask = np.array(Image.open(png_path).convert("L")) > 127
            else:
                raise FileNotFoundError(f"Missing object segment for frame {frame_idx}: {name}")
            mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
            frame_masks.append(mask)
        all_masks.append(np.stack(frame_masks, axis=0))
    return np.stack(all_masks, axis=0)


def read_slot_masks(slot_segment_dir: Path, num_slots: int, frame_indices: list[int], width: int, height: int) -> np.ndarray:
    all_masks = []
    for frame_idx in frame_indices:
        frame_masks = []
        for slot_idx in range(num_slots):
            npy_path = slot_segment_dir / f"frame_{frame_idx:04d}" / f"slot_{slot_idx:02d}.npy"
            png_path = slot_segment_dir / f"frame_{frame_idx:04d}" / f"slot_{slot_idx:02d}.png"
            if npy_path.exists():
                mask = np.load(npy_path).astype(np.float32)
            elif png_path.exists():
                mask = np.array(Image.open(png_path).convert("L")).astype(np.float32) / 255.0
            else:
                raise FileNotFoundError(f"Missing slot segment for frame {frame_idx}: slot_{slot_idx:02d}")
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
            frame_masks.append(np.clip(mask, 0.0, 1.0))
        all_masks.append(np.stack(frame_masks, axis=0))
    return np.stack(all_masks, axis=0)


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


def masks_to_rgb(binary_or_hard: np.ndarray) -> np.ndarray:
    if binary_or_hard.ndim == 3:
        hard = binary_or_hard.argmax(axis=0)
    else:
        hard = binary_or_hard
    out = np.zeros((*hard.shape, 3), dtype=np.uint8)
    for idx in np.unique(hard):
        out[hard == idx] = palette_color(int(idx))
    return out


def aligned_to_rgb(aligned_object_idx: np.ndarray, num_objects: int) -> np.ndarray:
    out = np.zeros((*aligned_object_idx.shape, 3), dtype=np.uint8)
    for idx in range(num_objects):
        out[aligned_object_idx == idx] = palette_color(idx)
    return out


def overlay(rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(rgb * (1.0 - alpha) + mask_rgb * alpha, 0, 255).astype(np.uint8)


def add_title(image: Image.Image, title: str) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 22), fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))
    return out


def save_preview(video_dir: Path, frame_indices: list[int], object_masks: np.ndarray, slot_masks: np.ndarray, aligned: np.ndarray, output_path: Path) -> None:
    rows = []
    height, width = object_masks.shape[-2:]
    for local_idx, frame_idx in enumerate(frame_indices):
        rgb = load_rgb_frame(video_dir, frame_idx, width, height)
        object_rgb = masks_to_rgb(object_masks[local_idx])
        slot_hard_rgb = masks_to_rgb(slot_masks[local_idx])
        aligned_rgb = aligned_to_rgb(aligned[local_idx], object_masks.shape[1])
        rows.extend(
            [
                add_title(Image.fromarray(rgb), f"frame_{frame_idx:04d}"),
                add_title(Image.fromarray(object_rgb), "object_segment"),
                add_title(Image.fromarray(slot_hard_rgb), "slot_segment"),
                add_title(Image.fromarray(aligned_rgb), "aligned_slot"),
                add_title(Image.fromarray(overlay(rgb, aligned_rgb)), "overlay"),
            ]
        )

    columns = 5
    tile_w, tile_h = rows[0].size
    sheet = Image.new("RGB", (columns * tile_w, len(frame_indices) * tile_h), (245, 245, 245))
    for idx, tile in enumerate(rows):
        sheet.paste(tile, ((idx % columns) * tile_w, (idx // columns) * tile_h))
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    output_root = script_dir.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default=str(output_root / "video"))
    parser.add_argument("--slot_dir", default=str(output_root / "slot"))
    parser.add_argument("--label_json", default="")
    parser.add_argument("--slot_metadata_json", default="")
    parser.add_argument("--output_dir", default=str(script_dir))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir).resolve()
    slot_dir = Path(args.slot_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.name == Path(__file__).name:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    label_json = Path(args.label_json).resolve() if args.label_json else video_dir / "label" / "labels.json"
    slot_metadata_json = Path(args.slot_metadata_json).resolve() if args.slot_metadata_json else slot_dir / "slot_metadata.json"
    labels = load_labels(label_json)
    slot_metadata = load_slot_metadata(slot_metadata_json)

    object_labels = labels.get("object_labels", [])
    num_objects = len(object_labels)
    num_slots = int(slot_metadata.get("num_slots", 0))
    if num_objects <= 0:
        raise ValueError(f"No object_labels found in {label_json}")
    if num_slots < num_objects:
        raise ValueError(f"Need n:6 matching, but object count={num_objects}, slot count={num_slots}")

    frame_indices = [int(x) for x in slot_metadata["frame_indices"]]
    object_masks = read_object_masks(video_dir, object_labels, frame_indices, args.width, args.height)
    slot_masks = read_slot_masks(slot_dir / "segment", num_slots, frame_indices, args.width, args.height)

    score_matrix = np.zeros((num_objects, num_slots), dtype=np.float64)
    eps = 1e-8
    for obj_idx in range(num_objects):
        obj_all = object_masks[:, obj_idx].astype(np.float32)
        obj_area = float(obj_all.sum())
        for slot_idx in range(num_slots):
            slot_all = slot_masks[:, slot_idx].astype(np.float32)
            inter = float((obj_all * slot_all).sum())
            union = float(obj_all.sum() + slot_all.sum() - inter) + eps
            iou = inter / union
            pixel_recall = inter / (obj_area + eps)
            score_matrix[obj_idx, slot_idx] = 0.7 * iou + 0.3 * pixel_recall

    object_rows, slot_cols = linear_sum_assignment(-score_matrix)
    object_to_slot = {int(obj_idx): int(slot_idx) for obj_idx, slot_idx in zip(object_rows, slot_cols)}
    slot_to_object = {int(slot_idx): int(obj_idx) for obj_idx, slot_idx in object_to_slot.items()}

    hard_slots = slot_masks.argmax(axis=1)
    aligned_object_idx = np.full_like(hard_slots, -1, dtype=np.int32)
    for slot_idx, obj_idx in slot_to_object.items():
        aligned_object_idx[hard_slots == slot_idx] = obj_idx

    per_object_iou = {}
    for obj_idx, obj in enumerate(object_labels):
        pred = aligned_object_idx == obj_idx
        gt = object_masks[:, obj_idx]
        inter = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        per_object_iou[obj["name"]] = float(inter / union) if union > 0 else 0.0
    mean_iou = float(np.mean(list(per_object_iou.values()))) if per_object_iou else 0.0

    np.save(output_dir / "score_matrix.npy", score_matrix)
    np.save(output_dir / "aligned_object_index.npy", aligned_object_idx)
    save_preview(video_dir, frame_indices, object_masks, slot_masks, aligned_object_idx, output_dir / "alignment_preview.png")

    result = {
        "label_json": str(label_json),
        "slot_metadata_json": str(slot_metadata_json),
        "frame_indices": frame_indices,
        "num_objects": int(num_objects),
        "num_slots": int(num_slots),
        "matching_mode": f"{num_objects}:{num_slots}",
        "object_names": [obj["name"] for obj in object_labels],
        "score_matrix": score_matrix.tolist(),
        "object_to_slot": {
            object_labels[obj_idx]["name"]: int(slot_idx)
            for obj_idx, slot_idx in object_to_slot.items()
        },
        "slot_to_object": {
            f"slot_{slot_idx:02d}": object_labels[obj_idx]["name"]
            for slot_idx, obj_idx in slot_to_object.items()
        },
        "per_object_iou": per_object_iou,
        "mean_iou": mean_iou,
        "artifacts": {
            "alignment_preview": str(output_dir / "alignment_preview.png"),
            "score_matrix_npy": str(output_dir / "score_matrix.npy"),
            "aligned_object_index_npy": str(output_dir / "aligned_object_index.npy"),
        },
    }
    with open(output_dir / "alignment_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with open(output_dir / "alignment_result.md", "w", encoding="utf-8") as f:
        f.write("# Object 与 Slot 对齐结果\n\n")
        f.write(f"- 帧: {frame_indices}\n")
        f.write(f"- object 数: {num_objects}\n")
        f.write(f"- slot 数: {num_slots}\n")
        f.write(f"- 匹配方式: `{num_objects}:{num_slots}`，从 {num_slots} 个 slot 中为每个 object 选一个最佳 slot\n")
        f.write(f"- 平均 IoU: `{mean_iou:.4f}`\n\n")
        f.write("## 映射\n")
        for obj_idx in sorted(object_to_slot):
            obj_name = object_labels[obj_idx]["name"]
            f.write(f"- {obj_name} -> slot_{object_to_slot[obj_idx]:02d}\n")
        f.write("\n## 每个物体 IoU\n")
        for name, iou in per_object_iou.items():
            f.write(f"- {name}: {iou:.4f}\n")

    print(f"Wrote alignment bundle to: {output_dir}")
    print(f"mean_iou={mean_iou:.4f}")
    print(f"object_to_slot={result['object_to_slot']}")


if __name__ == "__main__":
    main()
