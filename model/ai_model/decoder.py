"""Multi-head output decoder.

Decodes future tokens into:
  1. State predictions: position, velocity, quaternion, angular velocity, force
  2. Collision predictions: pairwise collision logits
  3. Mask predictions: per-object segmentation masks
  4. RGB predictions: composited scene images

Key design choices:
  - State head: simple MLP (physics state is low-dimensional)
  - Collision head: pairwise MLP on concatenated features
  - Mask head: spatial decoder from per-object tokens
  - RGB head: composite predicted masks with learned appearance features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


def _best_group(channels: int, preferred: int = 8) -> int:
    """Find largest divisor of `channels` that is <= preferred."""
    if channels <= 0:
        return 1
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class StateHead(nn.Module):
    """Predict future physics states from tokens.

    Outputs same 16-dim state vector as input:
      position[3], quaternion[4], velocity[3], angular_velocity[3], force[3]
    """

    def __init__(self, input_dim: int, hidden_dim: int, state_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [..., D] -> state: [..., 16]"""
        return self.net(tokens)


class CollisionHead(nn.Module):
    """Predict pairwise collision logits.

    Input: concatenated pair features [node_i, node_j, |node_i - node_j|]
    Output: collision logit per pair
    """

    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        # Input: node_i(D) + node_j(D) + |node_i - node_j|(D) = 3D
        self.net = nn.Sequential(
            nn.Linear(node_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, Tp, N, D]
        Returns:
            logits: [B, Tp, N, N]
        """
        B, Tp, N, D = tokens.shape
        node_i = tokens.unsqueeze(3).expand(-1, -1, -1, N, -1)  # [B, Tp, N, N, D]
        node_j = tokens.unsqueeze(2).expand(-1, -1, N, -1, -1)  # [B, Tp, N, N, D]
        diff = (node_i - node_j).abs()  # [B, Tp, N, N, D]
        pair_input = torch.cat([node_i, node_j, diff], dim=-1)  # [B, Tp, N, N, 3D]
        logits = self.net(pair_input).squeeze(-1)  # [B, Tp, N, N]
        return logits


class MaskHead(nn.Module):
    """Predict per-object segmentation masks from tokens.

    Architecture: token → Linear → reshape to spatial → upsample conv stack → mask
    """

    def __init__(self, token_dim: int, base_channels: int, image_size: int = 128):
        super().__init__()
        self.image_size = image_size
        self.init_size = image_size // 16  # e.g., 128/16 = 8

        self.proj = nn.Linear(token_dim, base_channels * self.init_size * self.init_size)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.GroupNorm(_best_group(base_channels // 2), base_channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, stride=2, padding=1),
            nn.GroupNorm(_best_group(base_channels // 4), base_channels // 4),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels // 4, 1, 4, stride=2, padding=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, Tp, N, D]
        Returns:
            mask_logits: [B, Tp, N, H, W]
        """
        B, Tp, N, D = tokens.shape
        x = self.proj(tokens)  # [B, Tp, N, C*s*s]
        x = x.reshape(B * Tp * N, -1, self.init_size, self.init_size)
        x = self.decoder(x)  # [B*Tp*N, 1, H, W]
        _, _, H, W = x.shape
        # Center crop or pad to image_size if needed
        if H != self.image_size or W != self.image_size:
            x = F.adaptive_avg_pool2d(x, (self.image_size, self.image_size))
        return x.reshape(B, Tp, N, self.image_size, self.image_size)


class RGBDecoder(nn.Module):
    """Composite predicted masks with learned per-object appearance.

    Instead of generating full RGB from scratch (expensive), we:
      1. Learn a per-object appearance embedding from history RGB
      2. Predict soft masks for future frames
      3. Composite: RGB = sum_i(mask_i * appearance_i) + background
    """

    def __init__(self, token_dim: int, base_channels: int, image_size: int = 128):
        super().__init__()
        self.image_size = image_size
        self.init_size = image_size // 16

        # Predict per-object appearance color from token
        self.appearance_head = nn.Sequential(
            nn.Linear(token_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # RGB color per object
            nn.Sigmoid(),
        )

        # Background decoder (from scene-level average token)
        self.bg_proj = nn.Linear(token_dim, base_channels * self.init_size * self.init_size)
        self.bg_decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, stride=2, padding=1),
            nn.GroupNorm(_best_group(base_channels // 2), base_channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, stride=2, padding=1),
            nn.GroupNorm(_best_group(base_channels // 4), base_channels // 4),
            nn.GELU(),
            nn.ConvTranspose2d(base_channels // 4, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        tokens: torch.Tensor,     # [B, Tp, N, D]
        mask_prob: torch.Tensor,   # [B, Tp, N, H, W]
        valid_mask: torch.Tensor,  # [B, N]
    ) -> torch.Tensor:
        """
        Returns:
            rgb_pred: [B, Tp, 3, H, W]
        """
        B, Tp, N, D = tokens.shape
        H = W = self.image_size

        # Per-object appearance: [B, Tp, N, 3]
        appearance = self.appearance_head(tokens)

        # Apply valid_mask to mask_prob to exclude padding objects
        mask_prob_valid = mask_prob * valid_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).float()

        # Composite: mask * appearance per object, then sum
        appearance_map = appearance.unsqueeze(-1).unsqueeze(-1)  # [B, Tp, N, 3, 1, 1]
        mask_map = mask_prob_valid.unsqueeze(3)  # [B, Tp, N, 1, H, W]
        composited = (appearance_map * mask_map).sum(dim=2)  # [B, Tp, 3, H, W]

        # Background from mean token
        scene_token = tokens.mean(dim=2)  # [B, Tp, D]
        bg = self.bg_proj(scene_token)  # [B, Tp, C*s*s]
        bg = bg.reshape(B * Tp, -1, self.init_size, self.init_size)
        bg = self.bg_decoder(bg)  # [B*Tp, 3, H, W]
        bg = bg.reshape(B, Tp, 3, H, W)

        # Composite: foreground + background * (1 - total_mask)
        total_mask = mask_prob.sum(dim=2).clamp(0, 1)  # [B, Tp, H, W]
        total_mask = total_mask.unsqueeze(2)  # [B, Tp, 1, H, W]
        rgb_pred = composited + bg * (1 - total_mask)

        return rgb_pred.clamp(0, 1)


class MultiHeadDecoder(nn.Module):
    """Combines all output heads."""

    def __init__(
        self,
        token_dim: int = 128,
        state_dim: int = 16,
        state_decoder_hidden: int = 256,
        collision_hidden: int = 64,
        mask_base_channels: int = 32,
        rgb_base_channels: int = 32,
        image_size: int = 128,
    ):
        super().__init__()
        self.state_head = StateHead(token_dim, state_decoder_hidden, state_dim)
        self.collision_head = CollisionHead(token_dim, collision_hidden)
        self.mask_head = MaskHead(token_dim, mask_base_channels, image_size)
        self.rgb_decoder = RGBDecoder(token_dim, rgb_base_channels, image_size)

    def forward(
        self,
        future_tokens: torch.Tensor,  # [B, Tp, N, D]
        valid_mask: torch.Tensor,      # [B, N]
        obj_attrs: torch.Tensor = None, # [B, N, attr_dim] optional, for static flag
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            state_pred: [B, Tp, N, 16]
            collision_logits: [B, Tp, N, N]
            mask_logits: [B, Tp, N, H, W]
            rgb_pred: [B, Tp, 3, H, W]
        """
        state_pred = self.state_head(future_tokens)
        collision_logits = self.collision_head(future_tokens)
        mask_logits = self.mask_head(future_tokens)
        mask_prob = torch.sigmoid(mask_logits)

        # Zero out invalid objects in mask predictions (use large negative for sigmoid→0)
        invalid = ~valid_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, 1, N, 1, 1]
        mask_logits = mask_logits.masked_fill(invalid, -10.0)
        mask_prob = torch.sigmoid(mask_logits)

        rgb_pred = self.rgb_decoder(future_tokens, mask_prob, valid_mask)

        # Pair mask for collision (exclude self-edges, padding, and static-static pairs)
        B, Tp, N, _ = future_tokens.shape
        pair_mask = valid_mask.unsqueeze(1).unsqueeze(2) * valid_mask.unsqueeze(1).unsqueeze(3)
        eye = torch.eye(N, device=future_tokens.device).unsqueeze(0).unsqueeze(0)
        pair_mask = pair_mask.float() * (1 - eye)

        # Exclude static-static pairs if obj_attrs provided
        if obj_attrs is not None:
            static_flag = (obj_attrs[..., 8] > 0.5)  # [B, N]
            static_pair = static_flag.unsqueeze(1).unsqueeze(2) & static_flag.unsqueeze(1).unsqueeze(3)
            pair_mask = pair_mask * (~static_pair).float()

        return {
            'state_pred': state_pred,
            'collision_logits': collision_logits,
            'pair_mask': pair_mask,
            'mask_logits': mask_logits,
            'mask_prob': mask_prob,
            'rgb_pred': rgb_pred,
        }
