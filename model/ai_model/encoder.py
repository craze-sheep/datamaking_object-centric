"""Dual-stream physics-aware object encoder.

Architecture:
  1. Visual stream: lightweight CNN → ROI pooling with object masks → per-object visual features
  2. Physics stream: embed static attributes + dynamic state → per-object physics features
  3. Fusion: concat + project → per-object tokens

Key differences from baseline:
  - Separate physics encoding branch (baseline bakes everything into one token)
  - Explicit mass/friction/geometry encoding (baseline uses raw features)
  - Position-velocity normalization with domain knowledge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class VisualEncoder(nn.Module):
    """Lightweight CNN that produces a feature map from RGB input."""

    def __init__(self, in_channels: int = 3, channels: Tuple[int, ...] = (32, 64, 128, 128)):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers.extend([
                nn.Conv2d(c_in, c_out, 3, stride=2, padding=1),
                nn.GroupNorm(min(8, c_out), c_out),
                nn.GELU(),
            ])
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_channels = c_in
        self.stride = 2 ** len(channels)  # total downsampling factor

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: [B, T, 3, H, W]
        Returns:
            feat: [B, T, C, H', W'] where H'=H/stride, W'=W/stride
        """
        B, T, C, H, W = rgb.shape
        x = rgb.reshape(B * T, C, H, W)
        feat = self.net(x)
        _, C_out, Hf, Wf = feat.shape
        return feat.reshape(B, T, C_out, Hf, Wf)


class MaskedROIPooler(nn.Module):
    """Pool per-object features from feature map using segmentation masks.

    Uses masked average pooling (not bbox ROI pooling) since we have pixel-level masks.
    This is more precise than bbox-based ROI pooling for irregular shapes.
    """

    def __init__(self, feat_channels: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(feat_channels, out_dim)

    def forward(
        self,
        feat_map: torch.Tensor,   # [B*T, C, Hf, Wf]
        masks: torch.Tensor,       # [B*T, N, H, W]
        valid_mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Returns:
            tokens: [B*T, N, out_dim]
        """
        BT, C, Hf, Wf = feat_map.shape
        _, N, H, W = masks.shape
        device = feat_map.device

        # Downsample masks to feature map resolution
        # masks: [BT, N, H, W] -> [BT*N, 1, H, W]
        masks_flat = masks.reshape(BT * N, 1, H, W)
        masks_ds = F.adaptive_avg_pool2d(masks_flat, (Hf, Wf))  # [BT*N, 1, Hf, Wf]
        masks_ds = masks_ds.reshape(BT, N, Hf, Wf)  # [BT, N, Hf, Wf]

        # Masked average pooling
        # feat_map: [BT, C, Hf, Wf] -> [BT, 1, C, Hf, Wf]
        # masks_ds: [BT, N, Hf, Wf] -> [BT, N, 1, Hf, Wf]
        mask_sum = masks_ds.sum(dim=(-2, -1)).clamp(min=1e-6)  # [BT, N]
        weighted = feat_map.unsqueeze(1) * masks_ds.unsqueeze(2)  # [BT, N, C, Hf, Wf]
        pooled = weighted.sum(dim=(-2, -1)) / mask_sum.unsqueeze(-1)  # [BT, N, C]

        # Handle valid_mask: zero out padding objects
        # valid_mask: [B, N] -> expand to [B, 1, N, 1] -> [BT, N, 1]
        B = BT // (BT // valid_mask.shape[0]) if valid_mask.shape[0] != BT else BT
        # Simpler: just use it per-sample
        T = BT // valid_mask.shape[0]
        vm = valid_mask.unsqueeze(1).expand(-1, T, -1).reshape(BT, N)  # [BT, N]
        pooled = pooled * vm.unsqueeze(-1).float()

        # Project
        tokens = self.proj(pooled)  # [BT, N, out_dim]
        return tokens


class PhysicsEncoder(nn.Module):
    """Encode static attributes + dynamic state into physics features.

    Encodes:
      - Static: object type, mass, geometry (size/radius), material (friction, restitution)
      - Dynamic: position, velocity, quaternion, angular velocity, force
    """

    def __init__(
        self,
        attr_dim: int = 15,
        state_dim: int = 16,
        attr_embed_dim: int = 64,
        state_embed_dim: int = 64,
        out_dim: int = 128,
    ):
        super().__init__()
        # Static attribute encoder
        self.attr_net = nn.Sequential(
            nn.Linear(attr_dim, attr_embed_dim),
            nn.LayerNorm(attr_embed_dim),
            nn.GELU(),
            nn.Linear(attr_embed_dim, attr_embed_dim),
            nn.LayerNorm(attr_embed_dim),
            nn.GELU(),
        )

        # Dynamic state encoder. The four derived features give the MLP
        # direct access to common mechanics quantities instead of forcing it
        # to rediscover them from mass, position, and velocity.
        self.derived_dim = 4
        self.state_net = nn.Sequential(
            nn.Linear(state_dim + self.derived_dim, state_embed_dim),
            nn.LayerNorm(state_embed_dim),
            nn.GELU(),
            nn.Linear(state_embed_dim, state_embed_dim),
            nn.LayerNorm(state_embed_dim),
            nn.GELU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(attr_embed_dim + state_embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    @staticmethod
    def _derived_physics_features(
        obj_attrs: torch.Tensor,
        dyn_state: torch.Tensor,
    ) -> torch.Tensor:
        """Return KE, PE, momentum magnitude, and angular momentum proxy."""
        B, T, N, _ = dyn_state.shape
        vel = dyn_state[..., 7:10]
        angvel = dyn_state[..., 10:13]
        height = dyn_state[..., 1:2]
        mass = obj_attrs[..., 7:8].unsqueeze(1).expand(B, T, N, 1)

        kinetic_energy = 0.5 * mass * vel.pow(2).sum(dim=-1, keepdim=True)
        potential_energy = mass * 9.8 * height
        momentum = mass * vel.norm(dim=-1, keepdim=True)
        angular_momentum = mass * angvel.norm(dim=-1, keepdim=True)
        return torch.cat(
            [kinetic_energy, potential_energy, momentum, angular_momentum],
            dim=-1,
        )

    def forward(
        self,
        obj_attrs: torch.Tensor,  # [B, N, attr_dim]
        dyn_state: torch.Tensor,  # [B, T, N, state_dim]
    ) -> torch.Tensor:
        """
        Returns:
            physics_tokens: [B, T, N, out_dim]
        """
        B, T, N, _ = dyn_state.shape

        # Static attributes: same for all timesteps
        attr_feat = self.attr_net(obj_attrs)  # [B, N, attr_embed_dim]
        attr_feat = attr_feat.unsqueeze(1).expand(-1, T, -1, -1)  # [B, T, N, attr_embed_dim]

        derived = self._derived_physics_features(obj_attrs, dyn_state)
        state_input = torch.cat([dyn_state, derived], dim=-1)
        state_feat = self.state_net(state_input)  # [B, T, N, state_embed_dim]

        # Fuse
        physics_tokens = self.fusion(torch.cat([attr_feat, state_feat], dim=-1))
        return physics_tokens  # [B, T, N, out_dim]


class PhysicsObjectEncoder(nn.Module):
    """Combined dual-stream encoder: visual + physics → fused object tokens."""

    def __init__(
        self,
        image_size: int = 128,
        cnn_channels: Tuple[int, ...] = (32, 64, 128, 128),
        visual_out_dim: int = 128,
        attr_dim: int = 15,
        state_dim: int = 16,
        attr_embed_dim: int = 64,
        state_embed_dim: int = 64,
        physics_out_dim: int = 128,
        fused_dim: int = 128,
    ):
        super().__init__()
        self.visual_encoder = VisualEncoder(3, cnn_channels)
        self.roi_pooler = MaskedROIPooler(self.visual_encoder.out_channels, visual_out_dim)
        self.physics_encoder = PhysicsEncoder(
            attr_dim, state_dim, attr_embed_dim, state_embed_dim, physics_out_dim
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(visual_out_dim + physics_out_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.GELU(),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys rgb, mask, obj_attrs, dyn_state, force_matrix, valid_mask
        Returns:
            dict with:
              tokens: [B, T, N, fused_dim] - per-object per-timestep tokens
              visual_feat: [B, T, N, visual_out_dim]
              physics_feat: [B, T, N, physics_out_dim]
              force_matrix: [B, T, N, N, 3] - pass-through
              valid_mask: [B, N] - pass-through
              obj_attrs: [B, N, attr_dim] - pass-through
        """
        rgb = batch['rgb']           # [B, T, 3, H, W]
        mask = batch['mask']         # [B, T, N, H, W]
        obj_attrs = batch['obj_attrs']  # [B, N, attr_dim]
        dyn_state = batch['dyn_state']  # [B, T, N, state_dim]
        force_matrix = batch['force_matrix']  # [B, T, N, N, 3]
        valid_mask = batch['valid_mask']  # [B, N]

        B, T, N = dyn_state.shape[:3]

        # Visual stream
        feat_map = self.visual_encoder(rgb)  # [B, T, C, Hf, Wf]
        BT = B * T
        feat_map_flat = feat_map.reshape(BT, *feat_map.shape[2:])
        mask_flat = mask.reshape(BT, N, *mask.shape[3:])
        vm_expanded = valid_mask.unsqueeze(1).expand(-1, T, -1).reshape(BT, N)
        visual_tokens = self.roi_pooler(feat_map_flat, mask_flat, vm_expanded)  # [BT, N, visual_out_dim]
        visual_tokens = visual_tokens.reshape(B, T, N, -1)

        # Physics stream
        physics_tokens = self.physics_encoder(obj_attrs, dyn_state)  # [B, T, N, physics_out_dim]

        # Fuse
        fused = self.fusion_proj(torch.cat([visual_tokens, physics_tokens], dim=-1))

        # Zero out invalid objects
        fused = fused * valid_mask.unsqueeze(1).unsqueeze(-1).float()

        return {
            'tokens': fused,           # [B, T, N, fused_dim]
            'visual_feat': visual_tokens,
            'physics_feat': physics_tokens,
            'force_matrix': force_matrix,
            'valid_mask': valid_mask,
            'obj_attrs': obj_attrs,
        }
