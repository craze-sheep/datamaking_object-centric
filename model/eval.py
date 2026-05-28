"""Evaluation metrics for physics video prediction."""
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import torch
from torch.nn import functional as F

try:
    from config import ATTR_STATIC_INDEX, STATE_POS_SLICE, STATE_VEL_SLICE
except ImportError:  # pragma: no cover
    from .config import ATTR_STATIC_INDEX, STATE_POS_SLICE, STATE_VEL_SLICE  # type: ignore


def _to_float(value) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _safe_mean(values: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return values.mean() if values.numel() else values.new_tensor(0.0)
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum()
    if denom <= 0:
        return values.new_tensor(0.0)
    return (values * mask).sum() / denom.clamp_min(1e-6)


def _global_ssim_per_sample(pred: torch.Tensor, target: torch.Tensor, value_range: float = 2.0) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2
    dims = tuple(range(1, pred.ndim))
    mu_x = pred.mean(dim=dims)
    mu_y = target.mean(dim=dims)
    view_shape = (-1,) + (1,) * (pred.ndim - 1)
    sigma_x = ((pred - mu_x.view(view_shape)) ** 2).mean(dim=dims)
    sigma_y = ((target - mu_y.view(view_shape)) ** 2).mean(dim=dims)
    cov = ((pred - mu_x.view(view_shape)) * (target - mu_y.view(view_shape))).mean(dim=dims)
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim.clamp(0.0, 1.0)


def compute_video_metrics_per_sample(pred_rgb: torch.Tensor, target_rgb: torch.Tensor, value_range: float = 2.0) -> Dict[str, torch.Tensor]:
    pred_rgb = pred_rgb.detach().float()
    target_rgb = target_rgb.detach().float().to(pred_rgb.device)
    if pred_rgb.shape != target_rgb.shape:
        raise ValueError(f"pred_rgb and target_rgb shape mismatch: {pred_rgb.shape} vs {target_rgb.shape}")
    mse = (pred_rgb - target_rgb).pow(2).flatten(1).mean(dim=1)
    psnr = torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        20.0 * math.log10(value_range) - 10.0 * torch.log10(mse),
    )
    return {"mse": mse, "psnr": psnr, "ssim": _global_ssim_per_sample(pred_rgb, target_rgb, value_range)}


def compute_video_metrics(pred_rgb: torch.Tensor, target_rgb: torch.Tensor, value_range: float = 2.0) -> Dict[str, float]:
    per = compute_video_metrics_per_sample(pred_rgb, target_rgb, value_range)
    mse = _to_float(per["mse"].mean())
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(value_range) - 10.0 * math.log10(mse)
    return {"mse": mse, "psnr": psnr, "ssim": _to_float(per["ssim"].mean())}


def compute_state_metrics_per_sample(
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    valid_mask: torch.Tensor,
    static_flag: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    pred_state = pred_state.detach().float()
    target_state = target_state.detach().float().to(pred_state.device)
    if pred_state.shape != target_state.shape:
        raise ValueError(f"pred_state and target_state shape mismatch: {pred_state.shape} vs {target_state.shape}")
    valid = valid_mask.to(device=pred_state.device, dtype=torch.bool)
    static = torch.zeros_like(valid) if static_flag is None else (static_flag.to(device=pred_state.device, dtype=torch.bool) & valid)
    dynamic = valid & (~static)
    state_sq = (pred_state - target_state).pow(2)
    mask = dynamic[:, None, :, None]

    def per_sample_mean(values: torch.Tensor, expanded_mask: torch.Tensor) -> torch.Tensor:
        b = values.shape[0]
        vals = values.reshape(b, -1)
        m = expanded_mask.expand_as(values).reshape(b, -1).to(values.dtype)
        denom = m.sum(dim=1)
        return torch.where(denom > 0, (vals * m).sum(dim=1) / denom.clamp_min(1e-6), torch.zeros_like(denom))

    return {
        "state_mse": per_sample_mean(state_sq, mask),
        "position_mse": per_sample_mean(state_sq[..., STATE_POS_SLICE], mask.expand(*state_sq.shape[:-1], 3)),
        "velocity_mse": per_sample_mean(state_sq[..., STATE_VEL_SLICE], mask.expand(*state_sq.shape[:-1], 3)),
    }


def compute_state_metrics(pred_state: torch.Tensor, target_state: torch.Tensor, valid_mask: torch.Tensor, static_flag: Optional[torch.Tensor] = None) -> Dict[str, float]:
    per = compute_state_metrics_per_sample(pred_state, target_state, valid_mask, static_flag)
    return {key: _to_float(value.mean()) for key, value in per.items()}


def build_collision_labels(
    force_tgt: torch.Tensor,
    force_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    force_std: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    threshold: float = 1e-6,
) -> torch.Tensor:
    mean = torch.tensor(force_mean, dtype=force_tgt.dtype, device=force_tgt.device)
    std = torch.tensor(force_std, dtype=force_tgt.dtype, device=force_tgt.device)
    force_raw = force_tgt * std + mean
    return (torch.linalg.norm(force_raw, dim=-1) > threshold).bool()


def collision_counts(
    collision_logits: torch.Tensor,
    force_tgt: torch.Tensor,
    pair_mask: torch.Tensor,
    force_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    force_std: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    threshold: float = 1e-6,
    logit_threshold: float = 0.0,
) -> Dict[str, int]:
    logits = collision_logits.detach().float()
    labels = build_collision_labels(force_tgt.detach().to(device=logits.device, dtype=logits.dtype), force_mean, force_std, threshold)
    mask = pair_mask.to(device=logits.device, dtype=torch.bool)
    pred_pos = logits > logit_threshold
    return {
        "tp": int((pred_pos & labels & mask).sum().item()),
        "fp": int((pred_pos & (~labels) & mask).sum().item()),
        "fn": int(((~pred_pos) & labels & mask).sum().item()),
        "pos": int((labels & mask).sum().item()),
        "count": int(mask.sum().item()),
    }


def _collision_rates(tp: int, fp: int, fn: int, pos: int, count: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "collision_precision": precision,
        "collision_recall": recall,
        "collision_f1": f1,
        "collision_tp": tp,
        "collision_fp": fp,
        "collision_fn": fn,
        "collision_pos_rate": (pos / count if count else 0.0),
    }


def compute_collision_metrics(
    collision_logits: torch.Tensor,
    force_tgt: torch.Tensor,
    pair_mask: torch.Tensor,
    force_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    force_std: Tuple[float, float, float] = (50.0, 50.0, 50.0),
    threshold: float = 1e-6,
    logit_threshold: float = 0.0,
) -> Dict[str, float]:
    c = collision_counts(collision_logits, force_tgt, pair_mask, force_mean, force_std, threshold, logit_threshold)
    return _collision_rates(c["tp"], c["fp"], c["fn"], c["pos"], c["count"])


def group_metrics_by_scene(per_sample_metrics: List[Mapping[str, float]]) -> Dict[str, float]:
    grouped = defaultdict(list)
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "pos": 0, "count": 0})
    for sample in per_sample_metrics:
        scene = int(sample.get("scene_id", -1))
        for key, value in sample.items():
            if key == "scene_id":
                continue
            if key in {"collision_tp", "collision_fp", "collision_fn", "collision_pos", "collision_count"}:
                short = key.replace("collision_", "")
                counts[scene][short] += int(value)
            elif isinstance(value, (int, float)) and not key.startswith("collision_"):
                grouped[f"scene_{scene}/{key}"].append(float(value))
            elif key in {"collision_precision", "collision_recall", "collision_f1", "collision_pos_rate"} and isinstance(value, (int, float)):
                # Backward-compatible path for caller-provided per-sample rates.
                grouped[f"scene_{scene}/{key}"].append(float(value))
    out = {key: sum(values) / len(values) for key, values in grouped.items() if values}
    for scene, c in counts.items():
        rates = _collision_rates(c["tp"], c["fp"], c["fn"], c["pos"], c["count"])
        out.update({f"scene_{scene}/{k}": v for k, v in rates.items()})
    return out


def _dataset_force_stats(dataloader, default_mean, default_std):
    ds = getattr(dataloader, "dataset", None)
    mean = getattr(ds, "force_mean", default_mean)
    std = getattr(ds, "force_std", default_std)
    if isinstance(mean, torch.Tensor):
        mean = tuple(float(x) for x in mean.flatten().tolist())
    if isinstance(std, torch.Tensor):
        std = tuple(float(x) for x in std.flatten().tolist())
    return tuple(mean), tuple(std)


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: Iterable[Mapping[str, torch.Tensor]],
    history_length: int = 12,
    predict_length: int = 12,
    device: Optional[torch.device] = None,
    value_range: float = 2.0,
    force_mean: Optional[Tuple[float, float, float]] = None,
    force_std: Optional[Tuple[float, float, float]] = None,
    collision_threshold: float = 1e-6,
) -> Dict[str, float]:
    if device is None:
        device = _get_model_device(model)
    else:
        device = torch.device(device)
        model.to(device)
    force_mean, force_std = _dataset_force_stats(dataloader, force_mean or (0.0, 0.0, 0.0), force_std or (50.0, 50.0, 50.0))

    was_training = model.training
    model.eval()
    sums = defaultdict(float)
    n_samples = 0
    collision_total = {"tp": 0, "fp": 0, "fn": 0, "pos": 0, "count": 0}
    per_sample = []

    for batch in dataloader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        pred = model(batch)
        rgb_tgt = batch["rgb"][:, history_length : history_length + predict_length]
        state_tgt = batch["dyn_state"][:, history_length : history_length + predict_length]
        force_tgt = batch["force_matrix"][:, history_length : history_length + predict_length]
        valid = pred.get("valid_mask", batch["valid_mask"]).to(device=device, dtype=torch.bool)
        static = pred.get("static_flag", batch["obj_attrs"][..., ATTR_STATIC_INDEX] > 0.5).to(device=device, dtype=torch.bool) & valid
        dynamic = valid & (~static)
        pair_mask = pred["pair_mask"].to(device=device, dtype=torch.bool)
        dynamic_pair = dynamic[:, None, :, None] | dynamic[:, None, None, :]
        pair_mask = pair_mask & dynamic_pair

        video = compute_video_metrics_per_sample(pred["rgb"], rgb_tgt, value_range=value_range)
        state = compute_state_metrics_per_sample(pred["state"], state_tgt, valid, static)
        b = rgb_tgt.shape[0]
        n_samples += b
        for key, value in {**video, **state}.items():
            finite = torch.where(torch.isinf(value), torch.zeros_like(value), value)
            sums[key] += float(finite.sum().item())

        c_all = collision_counts(pred["collision_logits"], force_tgt, pair_mask, force_mean, force_std, collision_threshold)
        for key in collision_total:
            collision_total[key] += c_all[key]

        scene_ids = batch.get("scene_id")
        if scene_ids is not None:
            for idx, scene_id in enumerate(scene_ids.detach().cpu().tolist()):
                sample_pair = pair_mask[idx : idx + 1]
                c = collision_counts(
                    pred["collision_logits"][idx : idx + 1],
                    force_tgt[idx : idx + 1],
                    sample_pair,
                    force_mean,
                    force_std,
                    collision_threshold,
                )
                row = {"scene_id": int(scene_id)}
                for key, value in {**video, **state}.items():
                    row[key] = _to_float(value[idx]) if not torch.isinf(value[idx]) else float("inf")
                row.update({
                    "collision_tp": c["tp"],
                    "collision_fp": c["fp"],
                    "collision_fn": c["fn"],
                    "collision_pos": c["pos"],
                    "collision_count": c["count"],
                })
                per_sample.append(row)

    if was_training:
        model.train()

    metrics = {key: (value / n_samples if n_samples else 0.0) for key, value in sums.items()}
    if "mse" in metrics:
        metrics["psnr"] = float("inf") if metrics["mse"] == 0.0 else 20.0 * math.log10(value_range) - 10.0 * math.log10(metrics["mse"])
    metrics.update(_collision_rates(**collision_total))
    metrics.update(group_metrics_by_scene(per_sample))
    return metrics
