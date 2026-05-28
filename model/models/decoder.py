"""Output decoder heads for physics video prediction."""
try:
    from config import DecoderConfig, STATE_POS_SLICE, STATE_VEL_SLICE
except ImportError:  # pragma: no cover
    from ..config import DecoderConfig, STATE_POS_SLICE, STATE_VEL_SLICE  # type: ignore

import torch
from torch import nn
from torch.nn import functional as F


class StateHead(nn.Module):
    def __init__(self, token_dim=256, state_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.SiLU(),
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, state_dim),
        )

    def forward(self, token):
        return self.net(token)


class SpatialHead(nn.Module):
    def __init__(self, in_dim, out_channels, base_channels=64, activation=None):
        super().__init__()
        self.base_channels = base_channels
        self.seed = nn.Sequential(
            nn.Linear(in_dim, base_channels * 8 * 8),
            nn.SiLU(),
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels, base_channels // 2, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels // 2, base_channels // 4, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(base_channels // 4, max(base_channels // 8, 1), 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(max(base_channels // 8, 1), out_channels, 1),
        )
        self.activation = activation

    def forward(self, x):
        leading = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        seed = self.seed(flat).reshape(-1, self.base_channels, 8, 8)
        out = self.conv(seed)
        if self.activation == "tanh":
            out = torch.tanh(out)
        return out.reshape(*leading, out.shape[-3], out.shape[-2], out.shape[-1])


class CollisionHead(nn.Module):
    def __init__(self, token_dim=256, hidden_dim=256):
        super().__init__()
        in_dim = 2 * token_dim + 3 + 3 + 2
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, token, state, valid_mask, static_flag):
        b, tp, n, d = token.shape
        z_i = token[:, :, :, None, :].expand(b, tp, n, n, d)
        z_j = token[:, :, None, :, :].expand(b, tp, n, n, d)
        pos = state[..., STATE_POS_SLICE]
        vel = state[..., STATE_VEL_SLICE]
        rel_pos = pos[:, :, None, :, :] - pos[:, :, :, None, :]
        rel_vel = vel[:, :, None, :, :] - vel[:, :, :, None, :]
        static = static_flag.to(device=token.device, dtype=token.dtype)
        static_i = static[:, None, :, None].expand(b, tp, n, n)[..., None]
        static_j = static[:, None, None, :].expand(b, tp, n, n)[..., None]
        pair_feat = torch.cat([z_i, z_j, rel_pos, rel_vel, static_i, static_j], dim=-1)
        logits = self.net(pair_feat).squeeze(-1)
        valid = valid_mask.to(device=token.device, dtype=torch.bool)
        pair_mask = valid[:, :, None] & valid[:, None, :]
        not_self = ~torch.eye(n, dtype=torch.bool, device=token.device)[None, :, :]
        pair_mask = (pair_mask & not_self)[:, None].expand(b, tp, n, n)
        logits = logits * pair_mask.to(logits.dtype)
        return logits, pair_mask


def compose_rgb(object_rgb, mask_prob, valid_mask, background):
    valid = valid_mask.to(device=object_rgb.device, dtype=object_rgb.dtype)
    mask_prob = mask_prob * valid[:, None, :, None, None]
    mask_sum = mask_prob.sum(dim=2, keepdim=True)
    obj_alpha_2d = mask_prob / mask_sum.clamp_min(1e-6)
    coverage = mask_sum.clamp(max=1.0)
    obj_alpha_2d = obj_alpha_2d * coverage
    bg_alpha = 1.0 - coverage.squeeze(2)
    obj_alpha = obj_alpha_2d.unsqueeze(3)
    bg_alpha_ch = bg_alpha.unsqueeze(2)
    rgb = (obj_alpha * object_rgb).sum(dim=2) + bg_alpha_ch * background[None, None]
    return rgb, obj_alpha, bg_alpha_ch


class OutputDecoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or DecoderConfig()
        cfg = self.config
        spatial_in = cfg.token_dim + 6
        self.state_head = StateHead(cfg.token_dim, cfg.state_dim)
        self.mask_head = SpatialHead(spatial_in, 1, cfg.mask_base_channels, activation=None)
        self.rgb_head = SpatialHead(spatial_in, 3, cfg.rgb_base_channels, activation="tanh")
        self.collision_head = CollisionHead(cfg.token_dim, cfg.token_dim)
        self.background = nn.Parameter(torch.zeros(3, cfg.image_size, cfg.image_size))

    def _spatial_input(self, token, state):
        pos_vel = torch.cat([state[..., STATE_POS_SLICE], state[..., STATE_VEL_SLICE]], dim=-1)
        return torch.cat([token, pos_vel], dim=-1)

    def _normalize_dynamic_quaternion(self, state, valid_mask, static_flag):
        quat = state[..., 3:7]
        normalized = F.normalize(quat, dim=-1, eps=1e-6)
        dynamic = (valid_mask & (~static_flag)).to(device=state.device, dtype=torch.bool)
        state = state.clone()
        state[..., 3:7] = torch.where(dynamic[:, None, :, None], normalized, quat)
        return state

    def forward(self, future_token, valid_mask, static_flag, last_state, last_mask=None):
        cfg = self.config
        valid = valid_mask.to(device=future_token.device, dtype=torch.bool)
        static = static_flag.to(device=future_token.device, dtype=torch.bool) & valid
        last_state = last_state.to(future_token.device)

        raw_delta = self.state_head(future_token)
        dynamic_state = last_state[:, None, :, :] + raw_delta
        base = last_state[:, None, :, :].expand_as(dynamic_state)
        state = torch.where(static[:, None, :, None], base, dynamic_state)
        state = self._normalize_dynamic_quaternion(state, valid, static)
        state = state * valid[:, None, :, None].to(state.dtype)

        spatial_input = self._spatial_input(future_token, state)
        mask_logits = self.mask_head(spatial_input).squeeze(-3)
        mask_logits = mask_logits * valid[:, None, :, None, None].to(mask_logits.dtype)
        if cfg.static_mask_strategy != "predict" and last_mask is not None:
            static_mask = static[:, None, :, None, None]
            copied = last_mask.to(mask_logits.device)[:, None].expand_as(mask_logits)
            mask_logits = torch.where(static_mask, copied, mask_logits)
        mask_prob = torch.sigmoid(mask_logits) * valid[:, None, :, None, None].to(mask_logits.dtype)

        object_rgb = self.rgb_head(spatial_input)
        rgb, _, _ = compose_rgb(object_rgb, mask_prob, valid, self.background)

        collision_logits, pair_mask = self.collision_head(future_token, state, valid, static)

        return {
            "rgb": rgb,
            "state": state,
            "collision_logits": collision_logits,
            "pair_mask": pair_mask,
            "mask_logits": mask_logits,
            "mask_prob": mask_prob,
            "depth": None,
        }
