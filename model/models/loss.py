"""Multi-task losses for physics video prediction."""
try:
    from config import LossConfig, ATTR_STATIC_INDEX
except ImportError:  # pragma: no cover
    from ..config import LossConfig, ATTR_STATIC_INDEX  # type: ignore

import torch
from torch import nn
from torch.nn import functional as F


def normalize_future_step_weights(weights, tp, device):
    if weights is None:
        w = torch.ones(tp, device=device)
    else:
        w = torch.tensor(weights, dtype=torch.float32, device=device)
    return w / w.mean().clamp_min(1e-6)


def build_collision_labels(force_tgt, force_mean=(0.0, 0.0, 0.0), force_std=(50.0, 50.0, 50.0), threshold=1e-6):
    mean = torch.tensor(force_mean, dtype=force_tgt.dtype, device=force_tgt.device)
    std = torch.tensor(force_std, dtype=force_tgt.dtype, device=force_tgt.device)
    force_raw = force_tgt * std + mean
    force_norm = torch.linalg.norm(force_raw, dim=-1)
    return (force_norm > threshold).float()


def focal_bce_with_logits(logits, labels, mask, alpha=0.25, gamma=2.0):
    if mask.sum() == 0:
        return logits.new_tensor(0.0)
    labels = labels.to(dtype=logits.dtype, device=logits.device)
    mask = mask.to(dtype=logits.dtype, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    pt = torch.where(labels > 0.5, p, 1 - p)
    alpha_t = labels * alpha + (1 - labels) * (1 - alpha)
    focal = alpha_t * (1 - pt).pow(gamma) * bce
    return (focal * mask).sum() / mask.sum().clamp_min(1e-6)


class PhysicsPredictionLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or LossConfig()
        self.lpips = None

    def _masks(self, batch, valid_mask, static_flag, device):
        if valid_mask is None:
            valid_mask = batch["valid_mask"]
        if static_flag is None:
            static_flag = batch["obj_attrs"][..., ATTR_STATIC_INDEX] > 0.5
        valid = valid_mask.to(device=device, dtype=torch.bool)
        static = static_flag.to(device=device, dtype=torch.bool) & valid
        dynamic = valid & (~static)
        return valid, static, dynamic

    def _state_weight_vec(self, device, dtype):
        return torch.tensor(self.config.state_component_weights, device=device, dtype=dtype)

    def forward(self, pred, batch, valid_mask=None, static_flag=None):
        cfg = self.config
        pred_rgb = pred["rgb"]
        device = pred_rgb.device
        dtype = pred_rgb.dtype
        b = pred_rgb.shape[0]
        tp = cfg.predict_length
        th = cfg.history_length
        valid, static, dynamic = self._masks(batch, valid_mask, static_flag, device)
        time_w = normalize_future_step_weights(cfg.future_step_weights, tp, device).to(dtype)

        rgb_tgt = batch["rgb"][:, th : th + tp].to(device=device, dtype=dtype)
        err_l1 = (pred_rgb - rgb_tgt).abs().mean(dim=(2, 3, 4))
        err_mse = (pred_rgb - rgb_tgt).pow(2).mean(dim=(2, 3, 4))
        err_rgb = 0.8 * err_l1 + 0.2 * err_mse
        loss_rgb = (err_rgb * time_w[None, :]).sum() / (b * time_w.sum()).clamp_min(1e-6)

        pred_state = pred["state"]
        state_tgt = batch["dyn_state"][:, th : th + tp].to(device=device, dtype=pred_state.dtype)
        state_error = F.smooth_l1_loss(pred_state, state_tgt, reduction="none")
        state_weight = self._state_weight_vec(device, pred_state.dtype)
        state_mask = dynamic[:, None, :, None].to(pred_state.dtype) * time_w[None, :, None, None].to(pred_state.dtype)
        denom_state = (state_mask.sum() * state_weight.sum()).clamp_min(1e-6)
        if state_mask.sum() == 0:
            loss_state = pred_state.new_tensor(0.0)
        else:
            loss_state = (state_error * state_weight[None, None, None, :] * state_mask).sum() / denom_state

        force_tgt = batch["force_matrix"][:, th : th + tp].to(device=device, dtype=pred["collision_logits"].dtype)
        labels = build_collision_labels(force_tgt, cfg.force_mean, cfg.force_std, cfg.collision_threshold)
        pair_mask = pred["pair_mask"].to(device=device, dtype=torch.bool)
        dynamic_pair = dynamic[:, None, :, None] | dynamic[:, None, None, :]
        pair_mask = pair_mask & dynamic_pair
        collision_mask = pair_mask.to(pred["collision_logits"].dtype) * time_w[None, :, None, None].to(pred["collision_logits"].dtype)
        loss_collision = focal_bce_with_logits(
            pred["collision_logits"],
            labels,
            collision_mask,
            cfg.focal_alpha,
            cfg.focal_gamma,
        )
        if pair_mask.sum() == 0:
            collision_pos_rate = pred_rgb.new_tensor(0.0)
        else:
            collision_pos_rate = labels[pair_mask].mean().to(dtype)

        mask_logits = pred["mask_logits"]
        mask_tgt = batch["mask"][:, th : th + tp].to(device=device, dtype=mask_logits.dtype)
        mask2d = valid[:, None, :, None, None].to(mask_logits.dtype)
        mask_weight = mask2d * time_w[None, :, None, None, None].to(mask_logits.dtype)
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, mask_tgt, reduction="none")
        loss_mask_bce = (mask_bce * mask_weight).sum() / (mask_weight.sum() * mask_logits.shape[-1] * mask_logits.shape[-2]).clamp_min(1e-6)
        mask_prob = torch.sigmoid(mask_logits) * mask2d
        tgt = mask_tgt * mask2d
        intersection = (mask_prob * tgt).sum(dim=(-1, -2))
        union = mask_prob.sum(dim=(-1, -2)) + tgt.sum(dim=(-1, -2))
        dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        dice_mask = valid[:, None, :].to(mask_logits.dtype) * time_w[None, :, None].to(mask_logits.dtype)
        loss_dice = (dice * dice_mask).sum() / dice_mask.sum().clamp_min(1e-6)
        loss_mask = 0.5 * loss_mask_bce + 0.5 * loss_dice

        loss_lpips = pred_rgb.new_tensor(0.0)
        total = (
            cfg.rgb_weight * loss_rgb
            + cfg.state_weight * loss_state
            + cfg.collision_weight * loss_collision
            + cfg.mask_weight * loss_mask
            + cfg.lpips_weight * loss_lpips
        )

        return {
            "loss": total,
            "loss_rgb": loss_rgb,
            "loss_state": loss_state,
            "loss_collision": loss_collision,
            "loss_mask": loss_mask,
            "loss_lpips": loss_lpips,
            "collision_pos_rate": collision_pos_rate,
        }
