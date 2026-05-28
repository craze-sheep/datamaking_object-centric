"""Input encoder for physics video prediction."""
try:
    from config import EncoderConfig, ATTR_STATIC_INDEX
except ImportError:  # pragma: no cover
    from ..config import EncoderConfig, ATTR_STATIC_INDEX

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        groups = min(8, out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class LightCNNBackbone(nn.Module):
    """Small CNN producing 32x32 feature maps from 128x128 RGB."""

    def __init__(self, out_channels=256):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 32, stride=1),
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, out_channels, stride=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


class MaskROIPool(nn.Module):
    """Masked average pool feature maps into object features."""

    def __init__(self, visual_dim=256, feature_channels=256, feature_map_size=32, eps=1e-6):
        super().__init__()
        self.feature_map_size = feature_map_size
        self.eps = eps
        self.proj = nn.Sequential(
            nn.Linear(feature_channels, visual_dim),
            nn.LayerNorm(visual_dim),
        )

    def forward(self, feat_map, mask, valid_mask):
        b, t, c, h, w = feat_map.shape
        n = mask.shape[2]
        mask_flat = mask.reshape(b * t * n, 1, mask.shape[-2], mask.shape[-1]).to(
            device=feat_map.device, dtype=feat_map.dtype
        )
        resized = F.interpolate(
            mask_flat,
            size=(h, w),
            mode="nearest",
        ).reshape(b, t, n, h, w)

        weighted = feat_map[:, :, None] * resized[:, :, :, None]
        area = resized.sum(dim=(-1, -2))
        pooled = weighted.sum(dim=(-1, -2)) / area.clamp_min(self.eps)[..., None]
        pooled = torch.where(area[..., None] > self.eps, pooled, torch.zeros_like(pooled))
        pooled = self.proj(pooled)
        valid = valid_mask.to(device=feat_map.device, dtype=torch.bool)
        pooled = pooled * valid[:, None, :, None].to(pooled.dtype)
        return pooled, resized


class AttrEncoder(nn.Module):
    def __init__(self, attr_dim=14, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(attr_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, attrs):
        return self.net(attrs)


class StateEncoder(nn.Module):
    def __init__(self, state_dim=16, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, state):
        return self.net(state)


class TokenFusion(nn.Module):
    def __init__(self, visual_dim=256, physical_dim=128, token_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(visual_dim + physical_dim, token_dim)
        self.norm1 = nn.LayerNorm(token_dim)
        self.fc2 = nn.Linear(token_dim, token_dim)
        self.norm2 = nn.LayerNorm(token_dim)
        self.act = nn.SiLU()

    def forward(self, visual_feat, physical_feat):
        x = torch.cat([visual_feat, physical_feat], dim=-1)
        h = self.norm1(self.act(self.fc1(x)))
        return self.norm2(h + self.fc2(self.act(h)))


class InputEncoder(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or EncoderConfig()
        cfg = self.config
        self.backbone = LightCNNBackbone(out_channels=cfg.visual_dim)
        self.roi_pool = MaskROIPool(
            visual_dim=cfg.visual_dim,
            feature_channels=cfg.visual_dim,
            feature_map_size=cfg.feature_map_size,
        )
        self.attr_encoder = AttrEncoder(cfg.attr_dim, cfg.attr_embed_dim)
        self.state_encoder = StateEncoder(cfg.state_dim, cfg.state_embed_dim)
        self.physical_projector = nn.Sequential(
            nn.Linear(cfg.attr_embed_dim + cfg.state_embed_dim, cfg.physical_dim),
            nn.SiLU(),
            nn.LayerNorm(cfg.physical_dim),
        )
        self.fusion = TokenFusion(cfg.visual_dim, cfg.physical_dim, cfg.token_dim)
        self.time_embedding = nn.Embedding(cfg.history_length, cfg.token_dim)
        self.object_embedding = nn.Embedding(cfg.max_objects, cfg.token_dim)
        self.view_embedding = nn.Embedding(cfg.num_views, cfg.token_dim)

    def forward(self, batch):
        cfg = self.config
        rgb = batch["rgb"][:, : cfg.history_length]
        mask = batch["mask"][:, : cfg.history_length]
        state_hist = batch["dyn_state"][:, : cfg.history_length]
        force_hist = batch["force_matrix"][:, : cfg.history_length]
        attrs = batch["obj_attrs"]
        valid_mask = batch["valid_mask"].to(device=rgb.device, dtype=torch.bool)

        b, th, _, h, w = rgb.shape
        feat = self.backbone(rgb.reshape(b * th, 3, h, w)).reshape(
            b, th, cfg.visual_dim, cfg.feature_map_size, cfg.feature_map_size
        )
        visual_feat, resized_mask = self.roi_pool(feat, mask, valid_mask)

        attr_feat = self.attr_encoder(attrs.to(rgb.device))
        state_feat = self.state_encoder(state_hist.to(rgb.device))
        attr_feat_t = attr_feat[:, None].expand(b, th, cfg.max_objects, cfg.attr_embed_dim)
        physical_feat = self.physical_projector(torch.cat([attr_feat_t, state_feat], dim=-1))

        token = self.fusion(visual_feat, physical_feat)
        time_ids = torch.arange(th, device=rgb.device)
        obj_ids = torch.arange(cfg.max_objects, device=rgb.device)
        token = token + self.time_embedding(time_ids)[None, :, None, :]
        token = token + self.object_embedding(obj_ids)[None, None, :, :]
        view_id = batch.get("view_id")
        if view_id is None:
            view_id = torch.zeros(b, dtype=torch.long, device=rgb.device)
        token = token + self.view_embedding(view_id.to(rgb.device))[:, None, None, :]

        static_raw = attrs[..., ATTR_STATIC_INDEX].to(rgb.device) > 0.5
        static_flag = valid_mask & static_raw
        dynamic_mask = valid_mask & (~static_raw)
        valid_float = valid_mask[:, None, :, None].to(token.dtype)
        token = token * valid_float
        visual_feat = visual_feat * valid_float
        physical_feat = physical_feat * valid_float

        return {
            "object_token": token,
            "visual_feat": visual_feat,
            "physical_feat": physical_feat,
            "state_hist": state_hist,
            "force_hist": force_hist,
            "static_flag": static_flag,
            "dynamic_mask": dynamic_mask,
            "valid_mask": valid_mask,
            "resized_mask": resized_mask,
        }
