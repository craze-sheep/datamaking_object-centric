"""Interaction network modules for object-level message passing."""
try:
    from config import (
        InteractionConfig,
        STATE_POS_SLICE,
        STATE_VEL_SLICE,
    )
except ImportError:  # pragma: no cover
    from ..config import (  # type: ignore
        InteractionConfig,
        STATE_POS_SLICE,
        STATE_VEL_SLICE,
    )

import torch
from torch import nn


class EdgeFeatureBuilder(nn.Module):
    """Build dense directed pairwise edge features.

    Edge layout: [F_ij, F_ji, rel_pos, rel_vel, static_i, static_j,
    distance, force_norm] -> 16 dims.
    """

    def __init__(self, edge_dim=16, exclude_self_edges=True, eps=1e-6):
        super().__init__()
        if edge_dim != 16:
            raise ValueError("EdgeFeatureBuilder currently produces exactly 16 features")
        self.edge_dim = edge_dim
        self.exclude_self_edges = exclude_self_edges
        self.eps = eps

    def forward(self, state_hist, force_hist, valid_mask, static_flag):
        assert state_hist.shape[:3] == force_hist.shape[:3]
        assert force_hist.shape[3] == state_hist.shape[2]
        b, th, n, _ = state_hist.shape
        device = state_hist.device
        dtype = state_hist.dtype

        valid = valid_mask.to(device=device, dtype=torch.bool)
        static = (static_flag.to(device=device, dtype=torch.bool) & valid)

        pos = state_hist[..., STATE_POS_SLICE]
        vel = state_hist[..., STATE_VEL_SLICE]
        rel_pos = pos[:, :, None, :, :] - pos[:, :, :, None, :]
        rel_vel = vel[:, :, None, :, :] - vel[:, :, :, None, :]
        dist = torch.linalg.norm(rel_pos, dim=-1, keepdim=True)
        force_norm = torch.linalg.norm(force_hist, dim=-1, keepdim=True)

        f_ij = force_hist
        f_ji = force_hist.transpose(2, 3)
        static_i = static[:, None, :, None].expand(b, th, n, n)[..., None].to(dtype)
        static_j = static[:, None, None, :].expand(b, th, n, n)[..., None].to(dtype)

        edge_feat = torch.cat(
            [f_ij, f_ji, rel_pos, rel_vel, static_i, static_j, dist, force_norm],
            dim=-1,
        )

        valid_i = valid[:, :, None]
        valid_j = valid[:, None, :]
        edge_mask = valid_i & valid_j
        if self.exclude_self_edges:
            not_self = ~torch.eye(n, dtype=torch.bool, device=device)[None, :, :]
            edge_mask = edge_mask & not_self
        edge_mask = edge_mask[:, None].expand(b, th, n, n)
        edge_feat = edge_feat * edge_mask[..., None].to(dtype)
        return edge_feat, edge_mask


class MessagePassingLayer(nn.Module):
    """One dense message passing layer."""

    def __init__(self, token_dim=256, edge_dim=16, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * token_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * token_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, token_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, token, edge_feat, edge_mask, valid_mask):
        b, th, n, d = token.shape
        z_i = token[:, :, :, None, :].expand(b, th, n, n, d)
        z_j = token[:, :, None, :, :].expand(b, th, n, n, d)
        edge_input = torch.cat([z_i, z_j, edge_feat], dim=-1)
        messages = self.edge_mlp(edge_input)
        messages = messages * edge_mask[..., None].to(messages.dtype)

        degree = edge_mask.to(messages.dtype).sum(dim=2).clamp_min(1.0)
        agg = messages.sum(dim=2) / degree[..., None]
        delta = self.node_mlp(torch.cat([token, agg], dim=-1))
        out = self.norm(token + self.dropout(delta))
        out = out * valid_mask[:, None, :, None].to(out.dtype)
        return out, messages, agg


class InteractionModule(nn.Module):
    """Stacked dense interaction network."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or InteractionConfig()
        cfg = self.config
        self.edge_builder = EdgeFeatureBuilder(
            edge_dim=cfg.edge_dim,
            exclude_self_edges=cfg.exclude_self_edges,
        )
        self.layers = nn.ModuleList(
            [
                MessagePassingLayer(
                    token_dim=cfg.token_dim,
                    edge_dim=cfg.edge_dim,
                    hidden_dim=cfg.hidden_dim,
                    dropout=cfg.dropout,
                )
                for _ in range(cfg.num_layers)
            ]
        )

    def forward(self, object_token, state_hist, force_hist, valid_mask, static_flag):
        assert object_token.shape[:3] == state_hist.shape[:3]
        assert force_hist.shape[:4] == state_hist.shape[:3] + (state_hist.shape[2],)

        valid = valid_mask.to(device=object_token.device, dtype=torch.bool)
        static = static_flag.to(device=object_token.device, dtype=torch.bool) & valid
        edge_feat, edge_mask = self.edge_builder(
            state_hist.to(object_token.device),
            force_hist.to(object_token.device),
            valid,
            static,
        )

        token = object_token * valid[:, None, :, None].to(object_token.dtype)
        messages = object_token.new_zeros(*edge_mask.shape, object_token.shape[-1])
        agg = object_token.new_zeros(*object_token.shape)
        for layer in self.layers:
            token, messages, agg = layer(token, edge_feat, edge_mask, valid)

        return {
            "inter_token": token,
            "edge_feat": edge_feat,
            "edge_mask": edge_mask,
            "messages": messages,
            "agg_message": agg,
            "valid_mask": valid,
            "static_flag": static,
        }
