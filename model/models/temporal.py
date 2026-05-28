"""Temporal object-token prediction module."""
try:
    from config import TemporalConfig
except ImportError:  # pragma: no cover
    from ..config import TemporalConfig  # type: ignore

import torch
from torch import nn


class TemporalPredictor(nn.Module):
    """Non-autoregressive Transformer encoder-decoder predictor."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or TemporalConfig()
        cfg = self.config
        self.time_embedding = nn.Embedding(cfg.history_length + cfg.predict_length, cfg.token_dim)
        self.object_embedding = nn.Embedding(cfg.max_objects, cfg.token_dim)
        self.future_base_query = nn.Parameter(torch.zeros(cfg.predict_length, cfg.max_objects, cfg.token_dim))
        nn.init.normal_(self.future_base_query, mean=0.0, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=cfg.batch_first,
            norm_first=cfg.norm_first,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.token_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=cfg.batch_first,
            norm_first=cfg.norm_first,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_decoder_layers)
        self.output_norm = nn.LayerNorm(cfg.token_dim)

    def _memory_padding_mask(self, valid_mask, batch_size, device):
        cfg = self.config
        valid = valid_mask.to(device=device, dtype=torch.bool)
        safe_valid = valid.clone()
        all_invalid = ~safe_valid.any(dim=1)
        if all_invalid.any():
            safe_valid[all_invalid, 0] = True
        valid_time_obj = safe_valid[:, None, :].expand(
            batch_size, cfg.history_length, cfg.max_objects
        )
        return ~valid_time_obj.reshape(batch_size, cfg.history_length * cfg.max_objects)

    def forward(self, inter_token, valid_mask):
        cfg = self.config
        b, th, n, d = inter_token.shape
        assert th == cfg.history_length
        assert n == cfg.max_objects
        assert d == cfg.token_dim

        device = inter_token.device
        valid = valid_mask.to(device=device, dtype=torch.bool)
        time_ids = torch.arange(cfg.history_length, device=device)
        obj_ids = torch.arange(cfg.max_objects, device=device)
        x = inter_token + self.time_embedding(time_ids)[None, :, None, :]
        x = x + self.object_embedding(obj_ids)[None, None, :, :]
        x = x * valid[:, None, :, None].to(x.dtype)
        x_flat = x.reshape(b, cfg.history_length * cfg.max_objects, cfg.token_dim)
        padding_mask = self._memory_padding_mask(valid, b, device)

        memory = self.encoder(x_flat, src_key_padding_mask=padding_mask)

        future_time_ids = torch.arange(cfg.history_length, cfg.history_length + cfg.predict_length, device=device)
        future_query = self.future_base_query
        future_query = future_query + self.time_embedding(future_time_ids)[:, None, :]
        future_query = future_query + self.object_embedding(obj_ids)[None, :, :]
        future_query = future_query.reshape(cfg.predict_length * cfg.max_objects, cfg.token_dim)
        future_query = future_query.unsqueeze(0).expand(b, -1, -1)

        future_flat = self.decoder(
            future_query,
            memory,
            memory_key_padding_mask=padding_mask,
            tgt_mask=None,
            tgt_key_padding_mask=None,
        )
        future_token = self.output_norm(future_flat).reshape(b, cfg.predict_length, cfg.max_objects, cfg.token_dim)
        future_token = future_token * valid[:, None, :, None].to(future_token.dtype)

        return {
            "future_token": future_token,
            "memory": memory,
            "future_query": future_query,
            "token_padding_mask": padding_mask,
            "valid_mask": valid,
        }
