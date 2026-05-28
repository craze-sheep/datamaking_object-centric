"""Inference helpers for physics video prediction."""
from pathlib import Path
from typing import Mapping, Optional

import torch

try:
    from config import ModelConfig, ATTR_STATIC_INDEX
    from eval import compute_collision_metrics, compute_state_metrics, compute_video_metrics
    from models.physics_pred import PhysicsVideoPredictor
except ImportError:  # pragma: no cover
    from .config import ModelConfig, ATTR_STATIC_INDEX  # type: ignore
    from .eval import compute_collision_metrics, compute_state_metrics, compute_video_metrics  # type: ignore
    from .models.physics_pred import PhysicsVideoPredictor  # type: ignore


def _move_batch(batch: Mapping, device: torch.device):
    return {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in batch.items()}


def load_model_from_checkpoint(checkpoint_path, device: Optional[str] = None, config: Optional[ModelConfig] = None):
    """Load PhysicsVideoPredictor from a trusted checkpoint path.

    This uses ``torch.load(weights_only=False)`` so checkpoint files must be trusted.
    Supported formats:
    - {'model_state_dict': state_dict, 'config': ModelConfig}
    - {'state_dict': state_dict, 'config': ModelConfig}
    - raw state_dict, with config passed separately or default ModelConfig.
    DataParallel/DDP ``module.`` prefixes are stripped automatically.
    """
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(checkpoint_path, map_location=device_obj, weights_only=False)
    if isinstance(ckpt, dict) and "config" in ckpt:
        config = ckpt["config"]
    config = config or ModelConfig()

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        state = ckpt
    else:
        raise ValueError("Checkpoint must contain 'model_state_dict' or 'state_dict', or be a raw tensor state_dict")
    state = {key.removeprefix("module."): value for key, value in state.items()}

    model = PhysicsVideoPredictor(config)
    model.load_state_dict(state)
    model.to(device_obj)
    model.eval()
    return model


def _resolve_lengths(model, history_length, predict_length):
    cfg = getattr(model, "config", None)
    cfg_h = getattr(cfg, "history_length", None)
    cfg_p = getattr(cfg, "predict_length", None)
    if history_length is None:
        history_length = cfg_h if cfg_h is not None else 12
    if predict_length is None:
        predict_length = cfg_p if cfg_p is not None else 12
    if cfg_h is not None and history_length != cfg_h:
        raise ValueError(f"history_length={history_length} does not match model.config.history_length={cfg_h}")
    if cfg_p is not None and predict_length != cfg_p:
        raise ValueError(f"predict_length={predict_length} does not match model.config.predict_length={cfg_p}")
    return int(history_length), int(predict_length)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    batch: Mapping,
    history_length: Optional[int] = None,
    predict_length: Optional[int] = None,
    device: Optional[str] = None,
    compute_metrics: bool = True,
    value_range: float = 2.0,
    force_mean=(0.0, 0.0, 0.0),
    force_std=(50.0, 50.0, 50.0),
    collision_threshold: float = 1e-6,
):
    """Run a single inference batch and optionally compute metrics when GT exists."""
    if device is None:
        try:
            device_obj = next(model.parameters()).device
        except StopIteration:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)
        model.to(device_obj)
    history_length, predict_length = _resolve_lengths(model, history_length, predict_length)
    was_training = model.training
    model.eval()
    batch_dev = _move_batch(batch, device_obj)
    pred = model(batch_dev)
    metrics = {}
    if compute_metrics and all(key in batch_dev for key in ("rgb", "dyn_state", "force_matrix", "valid_mask", "obj_attrs")):
        rgb_tgt = batch_dev["rgb"][:, history_length : history_length + predict_length]
        state_tgt = batch_dev["dyn_state"][:, history_length : history_length + predict_length]
        force_tgt = batch_dev["force_matrix"][:, history_length : history_length + predict_length]
        valid = pred.get("valid_mask", batch_dev["valid_mask"])
        static = pred.get("static_flag", batch_dev["obj_attrs"][..., ATTR_STATIC_INDEX] > 0.5)
        metrics.update(compute_video_metrics(pred["rgb"], rgb_tgt, value_range=value_range))
        metrics.update(compute_state_metrics(pred["state"], state_tgt, valid, static))
        if "collision_logits" in pred and "pair_mask" in pred:
            dynamic = valid.to(torch.bool) & (~static.to(torch.bool))
            pair_mask = pred["pair_mask"].to(torch.bool) & (dynamic[:, None, :, None] | dynamic[:, None, None, :])
            metrics.update(compute_collision_metrics(pred["collision_logits"], force_tgt, pair_mask, force_mean=force_mean, force_std=force_std, threshold=collision_threshold))
    if was_training:
        model.train()
    return {"pred": pred, "metrics": metrics, "batch": batch_dev}


def save_prediction_outputs(pred: Mapping, metrics: Mapping, output_path):
    """Save predictions and metrics to a torch .pt file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_cpu = {key: (value.detach().cpu() if torch.is_tensor(value) else value) for key, value in pred.items()}
    torch.save({"pred": pred_cpu, "metrics": dict(metrics)}, output_path)
    return str(output_path)
