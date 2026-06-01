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
        vel_scale: float = 10.0,
        use_uncertainty_weighting: bool = True,
        ssim_weight: float = 0.1,
        lpips_weight: float = 0.0,
        energy_weight: float = 0.01,
        collision_effect_weight: float = 0.05,
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
        self.vel_scale = vel_scale
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.energy_weight = energy_weight
        self.collision_effect_weight = collision_effect_weight
        self.__dict__['_lpips_fn'] = None

        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(4))
        else:
            self.register_parameter('log_vars', None)

        if state_component_weights is not None:
            self.register_buffer(
                'state_comp_w',
                torch.tensor(state_component_weights, dtype=torch.float32)
            )
        else:
            self.state_comp_w = None

    @staticmethod
    def _rgb_target_to_unit_range(rgb: torch.Tensor) -> torch.Tensor:
        """Convert normalized dataset RGB back to [0, 1] for decoder outputs."""
        return (rgb * 0.5 + 0.5).clamp(0, 1)

    @staticmethod
    def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Small dependency-free SSIM loss over video frames."""
        pred_flat = pred.flatten(0, 1)
        target_flat = target.flatten(0, 1)
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2

        mu_x = F.avg_pool2d(pred_flat, 7, stride=1, padding=3)
        mu_y = F.avg_pool2d(target_flat, 7, stride=1, padding=3)
        sigma_x = F.avg_pool2d(pred_flat * pred_flat, 7, stride=1, padding=3) - mu_x * mu_x
        sigma_y = F.avg_pool2d(target_flat * target_flat, 7, stride=1, padding=3) - mu_y * mu_y
        sigma_xy = F.avg_pool2d(pred_flat * target_flat, 7, stride=1, padding=3) - mu_x * mu_y

        numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
        denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
        ssim = (numerator / denominator.clamp(min=1e-6)).clamp(-1, 1)
        return ((1 - ssim) * 0.5).mean()

    def _get_lpips(self):
        """Lazily load LPIPS without registering its weights in checkpoints."""
        if self.__dict__['_lpips_fn'] is not None:
            return self.__dict__['_lpips_fn']
        try:
            import lpips as lpips_lib
        except ImportError as exc:
            raise RuntimeError(
                "lpips_weight > 0 requires the optional `lpips` package. "
                "Install it with `pip install lpips` or set lpips_weight=0."
            ) from exc
        lpips_fn = lpips_lib.LPIPS(net='alex', verbose=False)
        lpips_fn.eval()
        for param in lpips_fn.parameters():
            param.requires_grad = False
        self.__dict__['_lpips_fn'] = lpips_fn
        return lpips_fn

    def _lpips_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Deep perceptual RGB loss; pred/target are expected in [0, 1]."""
        if self.lpips_weight <= 0:
            return pred.new_tensor(0.0)
        lpips_fn = self._get_lpips()
        if next(lpips_fn.parameters()).device != pred.device:
            lpips_fn = lpips_fn.to(pred.device)
            self.__dict__['_lpips_fn'] = lpips_fn
        B, Tp = pred.shape[:2]
        pred_flat = pred.reshape(B * Tp, 3, pred.shape[-2], pred.shape[-1]).float()
        target_flat = target.reshape(B * Tp, 3, target.shape[-2], target.shape[-1]).float()
        return lpips_fn(pred_flat * 2 - 1, target_flat * 2 - 1).mean()

    def _energy_conservation_loss(
        self,
        state_pred: torch.Tensor,
        obj_attrs: torch.Tensor,
        dynamic_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Penalize sharp kinetic-energy drift across predicted future frames."""
        if state_pred.shape[1] <= 1:
            return state_pred.new_tensor(0.0)
        vel = state_pred[..., 7:10] * self.vel_scale
        mass = obj_attrs[..., 7:8].unsqueeze(1).clamp(min=0.0)
        kinetic_energy = 0.5 * mass * vel.pow(2).sum(dim=-1, keepdim=True)
        energy_diff = (kinetic_energy[:, 1:] - kinetic_energy[:, :-1]).abs()
        mask = dynamic_mask.unsqueeze(1).unsqueeze(-1).float()
        mask = mask.expand(-1, energy_diff.shape[1], -1, -1)
        denom = mask.sum().clamp(min=1.0)
        return (energy_diff * mask).sum() / denom

    @staticmethod
    def _collision_effect_loss(
        collision_effect: Optional[torch.Tensor],
        force_tgt: torch.Tensor,
        collision_labels: torch.Tensor,
        pair_mask: torch.Tensor,
        Tp: int,
    ) -> torch.Tensor:
        """Supervise collision direction/strength only on active collision pairs."""
        if collision_effect is None:
            return force_tgt.new_tensor(0.0)
        effect_pred = collision_effect[:, :Tp]
        effect_mask = (collision_labels * pair_mask).unsqueeze(-1)
        effect_diff = F.smooth_l1_loss(effect_pred, force_tgt, reduction='none')
        denom = effect_mask.expand_as(effect_diff).sum().clamp(min=1.0)
        return (effect_diff * effect_mask).sum() / denom

    def _combine_losses(
        self,
        rgb_loss: torch.Tensor,
        state_loss: torch.Tensor,
        collision_loss: torch.Tensor,
        mask_loss: torch.Tensor,
    ) -> torch.Tensor:
        losses = [rgb_loss, state_loss, collision_loss, mask_loss]
        manual_weights = [self.rgb_weight, self.state_weight, self.collision_weight, self.mask_weight]
        if not self.use_uncertainty_weighting:
            return sum(w * loss for w, loss in zip(manual_weights, losses))

        total = rgb_loss.new_tensor(0.0)
        for idx, (loss, base_weight) in enumerate(zip(losses, manual_weights)):
            log_var = self.log_vars[idx].clamp(min=-6.0, max=6.0)
            weighted = base_weight * (torch.exp(-log_var) * loss + log_var)
            total = total + weighted
        return total

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
        rgb_tgt = self._rgb_target_to_unit_range(batch['rgb'][:, Th:Th+Tp])  # [B, Tp, 3, H, W]
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
        ssim_loss = self._ssim_loss(rgb_pred, rgb_tgt) if self.ssim_weight > 0 else rgb_loss_l1.new_tensor(0.0)
        lpips_loss = self._lpips_loss(rgb_pred, rgb_tgt)
        rgb_loss = (
            rgb_loss_l1
            + 0.5 * rgb_loss_mse
            + self.ssim_weight * ssim_loss
            + self.lpips_weight * lpips_loss
        )

        # ===== 2. State Loss (only on dynamic, valid objects) =====
        state_pred = pred['state_pred'][:, :Tp]  # [B, Tp, N, 16]
        state_mask = dynamic_mask.unsqueeze(1).unsqueeze(-1).float()  # [B, 1, N, 1]
        state_diff = F.smooth_l1_loss(state_pred, state_tgt, reduction='none')  # [B, Tp, N, 16]

        # Apply per-component weights
        if self.state_comp_w is not None:
            state_diff = state_diff * self.state_comp_w.view(1, 1, 1, -1)

        n_state = state_mask.expand_as(state_diff).sum().clamp(min=1)
        state_loss = (state_diff * state_mask).sum() / n_state

        # ===== 3. Collision Loss =====
        collision_logits = pred['collision_logits'][:, :Tp]  # [B, Tp, N, N]
        pair_mask = pred['pair_mask'][:, :Tp]  # [B, Tp, N, N]

        collision_labels = compute_collision_labels(
            force_tgt, self.collision_threshold, self.force_std
        )  # [B, Tp, N, N]

        # Focal BCE with positive-class reweighting over valid pairs only.
        pos = (collision_labels * pair_mask).sum()
        neg = ((1 - collision_labels) * pair_mask).sum()
        pos_weight = (neg / pos.clamp(min=1.0)).clamp(min=1.0, max=10.0)
        bce = F.binary_cross_entropy_with_logits(
            collision_logits, collision_labels, reduction='none', pos_weight=pos_weight
        )
        prob = torch.sigmoid(collision_logits)
        p_t = prob * collision_labels + (1 - prob) * (1 - collision_labels)
        focal_weight = (1 - p_t) ** 2.0
        collision_loss_raw = focal_weight * bce

        # Apply pair mask
        n_collision = pair_mask.sum().clamp(min=1)
        collision_loss = (collision_loss_raw * pair_mask).sum() / n_collision
        effect_loss = self._collision_effect_loss(
            pred.get('collision_effect'),
            force_tgt,
            collision_labels,
            pair_mask,
            Tp,
        )

        # ===== 4. Mask Loss =====
        mask_logits = pred['mask_logits'][:, :Tp]  # [B, Tp, N, H, W]
        mask_mask = valid_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()  # [B, 1, N, 1, 1]
        # BCE
        bce_loss = F.binary_cross_entropy_with_logits(mask_logits, mask_tgt, reduction='none')
        n_mask = mask_mask.expand_as(bce_loss).sum().clamp(min=1)
        bce_loss = (bce_loss * mask_mask).sum() / n_mask

        # Dice
        mask_prob = torch.sigmoid(mask_logits)
        intersection = (mask_prob * mask_tgt * mask_mask).sum()
        union = ((mask_prob + mask_tgt) * mask_mask).sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

        mask_loss = bce_loss + dice_loss
        energy_loss = self._energy_conservation_loss(state_pred, obj_attrs, dynamic_mask)

        # ===== Total =====
        total_loss = self._combine_losses(rgb_loss, state_loss, collision_loss, mask_loss)
        total_loss = (
            total_loss
            + self.energy_weight * energy_loss
            + self.collision_effect_weight * effect_loss
        )

        return {
            'total_loss': total_loss,
            'rgb_loss': rgb_loss.detach(),
            'ssim_loss': ssim_loss.detach(),
            'lpips_loss': lpips_loss.detach(),
            'state_loss': state_loss.detach(),
            'collision_loss': collision_loss.detach(),
            'effect_loss': effect_loss.detach(),
            'mask_loss': mask_loss.detach(),
            'energy_loss': energy_loss.detach(),
        }
