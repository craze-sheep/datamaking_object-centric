#!/usr/bin/env python3
"""Extract SlotFormer slots/slot_masks and align slots to Kubric object IDs."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from torchvision import transforms


def load_video_rgb(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def sample_frame_indices(num_frames: int, count: int) -> np.ndarray:
    return np.linspace(0, num_frames - 1, count, dtype=int)


def preprocess_frames(frames: np.ndarray, resolution: tuple[int, int]) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    tensors = [tfm(frame.astype(np.uint8)) for frame in frames]
    return torch.stack(tensors, dim=0)


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


def hardmask_to_rgb(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in np.unique(mask):
        out[mask == idx] = palette_color(int(idx))
    return out


def ids_to_rgb(mask: np.ndarray, id_color_map: dict[int, tuple[int, int, int]]) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in np.unique(mask):
        out[mask == idx] = id_color_map.get(int(idx), (180, 80, 220))
    return out


def overlay(rgb: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip(rgb * (1.0 - alpha) + mask_rgb * alpha, 0, 255).astype(np.uint8)


def add_title(im: Image.Image, text: str) -> Image.Image:
    out = im.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle((0, 0, out.width, 20), fill=(0, 0, 0))
    draw.text((5, 4), text, fill=(255, 255, 255))
    return out


def save_row_grid(rows: list[list[np.ndarray]], titles: list[str], save_path: Path) -> None:
    tiles = []
    for row in rows:
        row_tiles = []
        for col_idx, img in enumerate(row):
            pil = Image.fromarray(img)
            row_tiles.append(add_title(pil, titles[col_idx]))
        tiles.append(row_tiles)

    tile_w, tile_h = tiles[0][0].size
    grid = Image.new("RGB", (tile_w * len(titles), tile_h * len(rows)), (245, 245, 245))
    for r, row in enumerate(tiles):
        for c, tile in enumerate(row):
            grid.paste(tile, (c * tile_w, r * tile_h))
    grid.save(save_path)


def save_slot_channels(slot_prob: np.ndarray, save_path: Path) -> None:
    # slot_prob: [N, H, W], in [0, 1]
    n_slots = slot_prob.shape[0]
    cols = 3
    rows = int(np.ceil(n_slots / cols))
    h, w = slot_prob.shape[1:]
    canvas = Image.new("RGB", (cols * w, rows * h), (245, 245, 245))
    for s in range(n_slots):
        arr = np.clip(slot_prob[s] * 255.0, 0, 255).astype(np.uint8)
        rgb = np.stack([arr, arr, arr], axis=-1)
        tile = add_title(Image.fromarray(rgb), f"slot_{s}")
        x = (s % cols) * w
        y = (s // cols) * h
        canvas.paste(tile, (x, y))
    canvas.save(save_path)


def resize_ids(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slotformer_root", required=True)
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--kubric_mask_npy", required=True)
    parser.add_argument("--kubric_labels_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--params_module", default="configs.savi_obj3d_params")
    parser.add_argument("--params_file", default="")
    parser.add_argument("--sample_count_override", type=int, default=0)
    parser.add_argument("--drop_first_n_frames", type=int, default=0)
    parser.add_argument(
        "--weight_path",
        default="/root/project/SlotFormer-master/pretrained/savi_obj3d_params/model_40.pth",
    )
    parser.add_argument("--num_slots_override", type=int, default=0)
    return parser.parse_args()


def load_state_dict_compatible(model: torch.nn.Module, state_dict: dict) -> tuple[list[str], list[str]]:
    """Load only shape-compatible keys from a checkpoint state_dict."""
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            skipped.append(k)
    missing = [k for k in model_sd.keys() if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    return skipped, missing


def main():
    args = parse_args()
    slotformer_root = Path(args.slotformer_root).resolve()
    os.chdir(slotformer_root)
    base_slots_dir = slotformer_root / "slotformer" / "base_slots"
    if str(base_slots_dir) not in sys.path:
        sys.path.insert(0, str(base_slots_dir))

    from models import build_model  # pylint: disable=import-outside-toplevel

    if args.params_file:
        params_file = Path(args.params_file).resolve()
        spec = importlib.util.spec_from_file_location("slotformer_params_module", str(params_file))
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load params file: {params_file}")
        params_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params_mod)
    else:
        params_mod = importlib.import_module(args.params_module)
    params = params_mod.SlotFormerParams()
    if args.num_slots_override and args.num_slots_override > 0:
        if hasattr(params, "slot_dict") and isinstance(params.slot_dict, dict):
            params.slot_dict["num_slots"] = int(args.num_slots_override)

    model = build_model(params)
    ckp = torch.load(args.weight_path, map_location="cpu")
    skipped_keys, missing_keys = load_state_dict_compatible(model, ckp["state_dict"])
    model.eval()
    model.testing = False  # we need slot masks

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_video_rgb(Path(args.video_path))
    sample_count = int(args.sample_count_override) if int(args.sample_count_override) > 0 else int(params.n_sample_frames)
    frame_indices = sample_frame_indices(len(frames), sample_count)
    if args.drop_first_n_frames > 0:
        if args.drop_first_n_frames >= len(frame_indices):
            raise ValueError("drop_first_n_frames is too large for sampled frame count.")
        frame_indices = frame_indices[int(args.drop_first_n_frames):]
    sampled_frames = frames[frame_indices]

    video_tensor = preprocess_frames(sampled_frames, params.resolution).unsqueeze(0)

    with torch.no_grad():
        out_dict = model({"img": video_tensor})

    slot_key = "post_slots" if "post_slots" in out_dict else "slots"
    mask_key = "post_masks" if "post_masks" in out_dict else "masks"
    slots = out_dict[slot_key].cpu().numpy()  # [1, T, N, C]
    masks = out_dict[mask_key].cpu().numpy()  # [1, T, N, 1, H, W] or [1, T, N, H, W]

    if masks.ndim == 6:
        masks = masks[:, :, :, 0]  # [1, T, N, H, W]
    masks = np.clip(masks, 0.0, 1.0)

    slots_np = slots[0]
    slot_masks_np = masks[0]  # [T, N, H, W]
    hard_slot = slot_masks_np.argmax(axis=1).astype(np.int32)  # [T, H, W]

    np.save(out_dir / "slots.npy", slots_np)
    np.save(out_dir / "slot_masks.npy", slot_masks_np)
    np.save(out_dir / "slot_hard_mask.npy", hard_slot)

    # Build slot-mask visualizations.
    h, w = slot_masks_np.shape[-2:]
    sampled_resized = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in sampled_frames], axis=0)
    rows = []
    for t in range(slot_masks_np.shape[0]):
        slot_rgb = hardmask_to_rgb(hard_slot[t])
        rows.append(
            [
                sampled_resized[t],
                slot_rgb,
                overlay(sampled_resized[t], slot_rgb),
            ]
        )
        save_slot_channels(slot_masks_np[t], out_dir / f"slot_masks_frame_{t:02d}.png")
    save_row_grid(rows, ["rgb", "slot_hard_mask", "overlay"], out_dir / "slot_mask_overview.png")

    # Load Kubric masks and labels for alignment.
    kubric_masks = np.load(args.kubric_mask_npy)  # [F, Hk, Wk]
    with open(args.kubric_labels_json, "r", encoding="utf-8") as f:
        kubric_labels = json.load(f)

    id_to_name = {}
    object_ids = []
    for obj in kubric_labels.get("object_labels", []):
        obj_id = int(obj["segmentation_id"])
        id_to_name[obj_id] = obj["name"]
        if not bool(obj.get("static", False)):
            object_ids.append(obj_id)
    object_ids = sorted(set(object_ids))
    if not object_ids:
        # fallback: all non-background ids
        present = sorted(int(x) for x in np.unique(kubric_masks) if int(x) != 0)
        object_ids = present

    sampled_kubric = kubric_masks[frame_indices]
    sampled_kubric = np.stack([resize_ids(m, w, h) for m in sampled_kubric], axis=0)

    # Weighted Hungarian matching across sampled frames.
    t_len, num_slots = slot_masks_np.shape[:2]
    num_obj = len(object_ids)
    score_mat = np.zeros((num_slots, num_obj), dtype=np.float64)
    eps = 1e-8

    for t in range(t_len):
        k_mask = sampled_kubric[t]
        total_obj_px = float(sum((k_mask == oid).sum() for oid in object_ids)) + eps
        for s in range(num_slots):
            s_prob = slot_masks_np[t, s]
            s_sum = float(s_prob.sum())
            for o_idx, oid in enumerate(object_ids):
                obj_bin = (k_mask == oid).astype(np.float32)
                obj_sum = float(obj_bin.sum())
                if obj_sum < 1:
                    continue
                inter = float((s_prob * obj_bin).sum())
                union = s_sum + obj_sum - inter + eps
                iou = inter / union
                weight = obj_sum / total_obj_px
                score_mat[s, o_idx] += weight * iou
    score_mat /= max(t_len, 1)

    row_ind, col_ind = linear_sum_assignment(-score_mat)
    slot_to_object = {}
    object_to_slot = {}
    for r, c in zip(row_ind, col_ind):
        slot_to_object[int(r)] = int(object_ids[c])
        object_to_slot[int(object_ids[c])] = int(r)

    # Create aligned mask by remapping hard-slot IDs -> Kubric object IDs.
    aligned = np.zeros_like(hard_slot, dtype=np.int32)
    for s, oid in slot_to_object.items():
        aligned[hard_slot == s] = oid

    # IoU evaluation on sampled frames.
    per_object_iou = {}
    for oid in object_ids:
        inter = np.logical_and(aligned == oid, sampled_kubric == oid).sum()
        union = np.logical_or(aligned == oid, sampled_kubric == oid).sum()
        per_object_iou[oid] = float(inter / union) if union > 0 else 0.0
    mean_iou = float(np.mean(list(per_object_iou.values()))) if per_object_iou else 0.0

    np.save(out_dir / "aligned_slot_mask.npy", aligned)
    np.save(out_dir / "sampled_kubric_mask.npy", sampled_kubric)

    kubric_color = {
        0: (18, 20, 24),
        1: (230, 55, 45),
        2: (245, 205, 55),
        10: (140, 145, 155),
    }
    rows = []
    for t in range(t_len):
        rows.append(
            [
                sampled_resized[t],
                ids_to_rgb(sampled_kubric[t], kubric_color),
                hardmask_to_rgb(hard_slot[t]),
                ids_to_rgb(aligned[t], kubric_color),
            ]
        )
    save_row_grid(
        rows,
        ["rgb", "kubric_mask", "slot_hard_mask", "aligned_slot_mask"],
        out_dir / "alignment_preview.png",
    )

    report = {
        "video_path": str(Path(args.video_path).resolve()),
        "frame_indices": frame_indices.tolist(),
        "params_module": args.params_module,
        "params_file": str(Path(args.params_file).resolve()) if args.params_file else "",
        "num_slots_override": int(args.num_slots_override),
        "sample_count_used": int(sample_count),
        "drop_first_n_frames": int(args.drop_first_n_frames),
        "weight_path": str(Path(args.weight_path).resolve()),
        "num_skipped_ckp_keys": len(skipped_keys),
        "num_missing_model_keys": len(missing_keys),
        "slots_shape": list(slots_np.shape),
        "slot_masks_shape": list(slot_masks_np.shape),
        "object_ids": object_ids,
        "object_names": {str(k): id_to_name.get(k, f"id_{k}") for k in object_ids},
        "score_matrix": score_mat.tolist(),
        "slot_to_object": {str(k): int(v) for k, v in slot_to_object.items()},
        "object_to_slot": {str(k): int(v) for k, v in object_to_slot.items()},
        "per_object_iou": {str(k): float(v) for k, v in per_object_iou.items()},
        "mean_iou": mean_iou,
        "artifacts": {
            "slot_mask_overview": str(out_dir / "slot_mask_overview.png"),
            "alignment_preview": str(out_dir / "alignment_preview.png"),
            "slot_masks_npy": str(out_dir / "slot_masks.npy"),
            "aligned_slot_mask_npy": str(out_dir / "aligned_slot_mask.npy"),
        },
    }
    with open(out_dir / "alignment_result.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(out_dir / "alignment_result.md", "w", encoding="utf-8") as f:
        f.write("# Slot 与物体对齐结果\n\n")
        f.write(f"- 采样帧: {frame_indices.tolist()}\n")
        f.write(f"- 模型: `{args.params_module}`\n")
        f.write(f"- 权重: `{args.weight_path}`\n")
        f.write(f"- 平均 IoU: `{mean_iou:.4f}`\n\n")
        f.write("## 映射\n")
        for s in sorted(slot_to_object):
            oid = slot_to_object[s]
            name = id_to_name.get(oid, f"id_{oid}")
            f.write(f"- slot_{s} -> object_id={oid} ({name})\n")
        f.write("\n## 每个物体 IoU\n")
        for oid in object_ids:
            name = id_to_name.get(oid, f"id_{oid}")
            f.write(f"- object_id={oid} ({name}): {per_object_iou.get(oid, 0.0):.4f}\n")

    print(f"Wrote slot extraction and alignment results to: {out_dir}")
    print(f"mean_iou={mean_iou:.4f}")
    print(f"slot_to_object={slot_to_object}")


if __name__ == "__main__":
    main()
