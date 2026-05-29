"""Multi-task loss for physics video prediction.

Losses:
  1. RGB: L1 + MSE on predicted vs target frames
  2. State: Smooth L1 on predicted vs target physics states
  3. Collision: Focal BCE on predicted vs target collision labels
  4. Mask: BCE + Dice on predicted vs target segmentation masks

Masking rules:
  - Padding objects excluded from all losses
  - Static objects excluded from state loss
  - Static-static pairs excluded from collision loss
  - Empty masks / no valid objects → return 0 (avoid NaN)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def compute_collision_labels(
    force_matrix: torch.Tensor,  # [B, T, N, N, 3]
    threshold: float = 1e-6,
    force_std: float = 50.0,
) -> torch.Tensor:
    """Compute collision labels from force matrix.

    Collision = ||force|| > threshold (in original, un-normalized force space).
    Since force_matrix is normalized by force_std, threshold should also be normalized.
    """
    # Norm of force vector (in normalized space)
    force_norm = force_matrix.norm(dim=-1)  # [B, T, N, N]
    # Threshold in normalized space
    norm_threshold = threshold / force_std
    return (force_norm > norm_threshold).float()


class PhysicsLoss(nn.Module):
    """Combined multi-task loss."""

    def __init__(
        self,
        history_length: int = 12,
        predict_length: int = 12,
        state_dim: int = 16,
        rgb_weight: float = 1.0,
        state_weight: float = 0.5,
        collision_weight: float = 0.5,
        mask_weight: float = 0.3,
        collision_threshold: float = 1e-6,
        force_std: float = 50.0,
        state_component_weights: Optional[Tuple[float, ...]] = (
            1.0, 1.0, 1.0,       # position
            0.5, 0.5, 0.5, 0.5,  # quaternion
            0.5, 0.5, 0.5,       # velocity
            0.25, 0.25, 0.25,    # angular velocity
            0.25, 0.25, 0.25,    # force
        ),
    ):
        super().__init__()
        self.history_length = history_length
        self.predict_length = predict_length
        self.rgb_weight = rgb_weight
        self.state_weight = state_weight
        self.collision_weight = collision_weight
        self.mask_weight = mask_weight
        self.collision_threshold = collision_threshold
        self.force_std = force_std

        if state_component_weights is not None:
            self.register_buffer(
                'state_comp_w',
                torch.tensor(state_component_weights, dtype=torch.float32)
            )
        else:
            self.state_comp_w = None

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred: model output dict with keys:
                state_pred, collision_logits, pair_mask, mask_logits, mask_prob, rgb_pred
            batch: data dict with keys:
                rgb, mask, obj_attrs, dyn_state, force_matrix, valid_mask

        Returns:
            dict with total_loss and individual loss components
        """
        Th = self.history_length
        Tp = self.predict_length

        # Extract targets (future frames only)
        rgb_tgt = batch['rgb'][:, Th:Th+Tp]            # [B, Tp, 3, H, W]
        state_tgt = batch['dyn_state'][:, Th:Th+Tp]    # [B, Tp, N, 16]
        mask_tgt = batch['mask'][:, Th:Th+Tp]           # [B, Tp, N, H, W]
        force_tgt = batch['force_matrix'][:, Th:Th+Tp]  # [B, Tp, N, N, 3]
        valid_mask = batch['valid_mask']                 # [B, N]
        obj_attrs = batch['obj_attrs']                   # [B, N, 14]

        B, Tp_actual, N = state_tgt.shape[:3]
        Tp = min(Tp, Tp_actual)

        # Get static flags from attrs (index 8)
        static_flag = obj_attrs[..., 8] > 0.5  # [B, N]
        dynamic_mask = valid_mask & ~static_flag  # [B, N]

        # ===== 1. RGB Loss =====
        rgb_pred = pred['rgb_pred'][:, :Tp]  # [B, Tp, 3, H, W]
        rgb_loss_l1 = F.l1_loss(rgb_pred, rgb_tgt, reduction='mean')
        rgb_loss_mse = F.mse_loss(rgb_pred, rgb_tgt, reduction='mean')
        rgb_loss = rgb_loss_l1 + 0.5 * rgb_loss_mse

        # ===== 2. State Loss (only on dynamic, valid objects) =====
        state_pred = pred['state_pred'][:, :Tp]  # [B, Tp, N, 16]
        state_mask = dynamic_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, N, 1]
        n_state = state_mask.sum().clamp(min=1)

        state_diff = F.smooth_l1_loss(state_pred, state_tgt, reduction='none')  # [B, Tp, N, 16]

        # Apply per-component weights
        if self.state_comp_w is not None:
            state_diff = state_diff * self.state_comp_w.view(1, 1, 1, -1)

        state_loss = (state_diff * state_mask).sum() / n_state

        # ===== 3. Collision Loss =====
        collision_logits = pred['collision_logits'][:, :Tp]  # [B, Tp, N, N]
        pair_mask = pred['pair_mask'][:, :Tp]  # [B, Tp, N, N]

        collision_labels = compute_collision_labels(
            force_tgt, self.collision_threshold, self.force_std
        )  # [B, Tp, N, N]

        # Focal BCE
        pos_weight = (1 - collision_labels).mean() / collision_labels.mean().clamp(min=1e-6)
        bce = F.binary_cross_entropy_with_logits(
            collision_logits, collision_labels, reduction='none'
        )
        prob = torch.sigmoid(collision_logits)
        p_t = prob * collision_labels + (1 - prob) * (1 - collision_labels)
        focal_weight = (1 - p_t) ** 2.0
        collision_loss_raw = focal_weight * bce

        # Apply pair mask
        n_collision = pair_mask.sum().clamp(min=1)
        collision_loss = (collision_loss_raw * pair_mask).sum() / n_collision

        # ===== 4. Mask Loss =====
        mask_logits = pred['mask_logits'][:, :Tp]  # [B, Tp, N, H, W]
        mask_mask = valid_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, N, 1, 1]
        n_mask = mask_mask.sum().clamp(min=1)

        # BCE
        bce_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_tgt, reduction='none')
        bce_loss = (bce_loss * mask_mask).sum() / n_mask

        # Dice
        mask_prob = torch.sigmoid(mask_logits)
        intersection = (mask_prob * mask_tgt * mask_mask).sum()
        union = ((mask_prob + mask_tgt) * mask_mask).sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

        mask_loss = bce_loss + dice_loss

        # ===== Total =====
        total_loss = (
            self.rgb_weight * rgb_loss +
            self.state_weight * state_loss +
            self.collision_weight * collision_loss +
            self.mask_weight * mask_loss
        )

        return {
            'total_loss': total_loss,
            'rgb_loss': rgb_loss.detach(),
            'state_loss': state_loss.detach(),
            'collision_loss': collision_loss.detach(),
            'mask_loss': mask_loss.detach(),
        }
