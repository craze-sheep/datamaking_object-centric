"""Visualization utilities for inference outputs."""
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image, ImageDraw

try:
    from eval import build_collision_labels
except ImportError:  # pragma: no cover
    from .eval import build_collision_labels  # type: ignore


def _ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _rgb_to_uint8(frame: torch.Tensor) -> torch.Tensor:
    frame = frame.detach().cpu().float()
    if frame.ndim != 3:
        raise ValueError(f"frame must be [3,H,W], got {frame.shape}")
    # Support either [-1,1] normalized or [0,1] tensors.
    if frame.min() < 0:
        frame = (frame + 1.0) / 2.0
    frame = frame.clamp(0.0, 1.0)
    return (frame.permute(1, 2, 0) * 255).byte()


def _frame_image(frame: torch.Tensor) -> Image.Image:
    return Image.fromarray(_rgb_to_uint8(frame).numpy(), mode="RGB")


def save_prediction_comparison(pred_rgb: torch.Tensor, target_rgb: torch.Tensor, output_path, sample_idx: int = 0, max_steps: int = 6):
    """Save a grid with GT row and prediction row."""
    output_path = _ensure_parent(output_path)
    pred = pred_rgb[sample_idx].detach().cpu()
    target = target_rgb[sample_idx].detach().cpu()
    steps = min(max_steps, pred.shape[0], target.shape[0])
    frames = []
    for row in (target, pred):
        frames.append([_frame_image(row[t]) for t in range(steps)])
    w, h = frames[0][0].size
    label_w = 48
    canvas = Image.new("RGB", (label_w + steps * w, 2 * h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, h // 2 - 6), "GT", fill=(0, 0, 0))
    draw.text((4, h + h // 2 - 6), "Pred", fill=(0, 0, 0))
    for r in range(2):
        for t in range(steps):
            canvas.paste(frames[r][t], (label_w + t * w, r * h))
    canvas.save(output_path)
    return str(output_path)


def _normalize_series(values):
    vals = [float(v) for v in values]
    lo = min(vals) if vals else 0.0
    hi = max(vals) if vals else 1.0
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    return [(v - lo) / (hi - lo) for v in vals]


def _draw_polyline(draw, values, box, color):
    x0, y0, x1, y1 = box
    if len(values) == 1:
        draw.ellipse((x0, y0, x0 + 3, y0 + 3), fill=color)
        return
    norm = _normalize_series(values)
    pts = []
    for i, v in enumerate(norm):
        x = x0 + i * (x1 - x0) / max(1, len(norm) - 1)
        y = y1 - v * (y1 - y0)
        pts.append((x, y))
    draw.line(pts, fill=color, width=2)


def save_state_curves(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    valid_mask: torch.Tensor,
    output_path,
    sample_idx: int = 0,
    object_idx: int = 0,
    dim_indices=(0, 1, 2, 7, 8, 9, 13, 14, 15),
    dim_labels=("x", "y", "z", "vx", "vy", "vz", "fx", "fy", "fz"),
):
    """Save position, velocity, and force curves for one valid object."""
    output_path = _ensure_parent(output_path)
    valid = valid_mask[sample_idx].detach().cpu().bool()
    if not valid[object_idx]:
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        if len(valid_indices) == 0:
            raise ValueError("No valid object available for state curve visualization")
        object_idx = int(valid_indices[0].item())
    pred = pred_state[sample_idx, :, object_idx].detach().cpu()
    tgt = target_state[sample_idx, :, object_idx].detach().cpu()
    rows = len(dim_indices)
    row_h = 80
    canvas = Image.new("RGB", (720, 36 + rows * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"Object {object_idx} state curves: pred red, GT blue", fill=(0, 0, 0))
    for row, dim in enumerate(dim_indices):
        y0 = 32 + row * row_h
        box = (70, y0 + 8, 700, y0 + row_h - 8)
        draw.rectangle(box, outline=(180, 180, 180))
        label = dim_labels[row] if row < len(dim_labels) else f"d{dim}"
        draw.text((10, y0 + 30), label, fill=(0, 0, 0))
        _draw_polyline(draw, pred[:, dim].tolist(), box, (220, 0, 0))
        _draw_polyline(draw, tgt[:, dim].tolist(), box, (0, 0, 220))
    canvas.save(output_path)
    return str(output_path)


def save_collision_confusion(
    collision_logits: torch.Tensor,
    force_tgt: torch.Tensor,
    pair_mask: torch.Tensor,
    output_path,
    force_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    force_std: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    threshold: float = 1e-6,
    logit_threshold: float = 0.0,
):
    """Save a 2x2 confusion matrix image for collision predictions."""
    output_path = _ensure_parent(output_path)
    logits = collision_logits.detach().cpu()
    labels = build_collision_labels(force_tgt.detach().cpu().to(logits.dtype), force_mean, force_std, threshold)
    mask = pair_mask.detach().cpu().bool()
    pred = logits > logit_threshold
    tp = int((pred & labels & mask).sum().item())
    fp = int((pred & (~labels) & mask).sum().item())
    fn = int(((~pred) & labels & mask).sum().item())
    tn = int(((~pred) & (~labels) & mask).sum().item())
    canvas = Image.new("RGB", (360, 280), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 15), "Collision confusion", fill=(0, 0, 0))
    cells = [((80, 60, 180, 140), "TP", tp, (180, 240, 180)), ((190, 60, 290, 140), "FP", fp, (250, 210, 180)), ((80, 150, 180, 230), "FN", fn, (250, 180, 180)), ((190, 150, 290, 230), "TN", tn, (220, 220, 220))]
    for box, name, value, color in cells:
        draw.rectangle(box, fill=color, outline=(0, 0, 0))
        draw.text((box[0] + 30, box[1] + 20), name, fill=(0, 0, 0))
        draw.text((box[0] + 35, box[1] + 45), str(value), fill=(0, 0, 0))
    canvas.save(output_path)
    return str(output_path)
