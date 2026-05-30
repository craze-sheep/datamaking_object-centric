"""GRU-based temporal prediction module.

Key design:
  - Process history frames through a per-object GRU to learn temporal dynamics
  - Use the final hidden state to initialize autoregressive future prediction
  - Per-object GRU (shared weights) with object masking

Key differences from baseline:
  - Baseline uses Transformer encoder-decoder for temporal prediction
  - We use GRU which is simpler, more memory-efficient, and has explicit recurrence
  - GRU naturally captures sequential dynamics (physics is inherently sequential)
  - We predict future autoregressively (each step conditioned on previous prediction)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


def _valid_num_heads(hidden_dim: int, requested: int) -> int:
    """Return a head count that divides hidden_dim."""
    for heads in range(min(requested, hidden_dim), 0, -1):
        if hidden_dim % heads == 0:
            return heads
    return 1


class TemporalGRU(nn.Module):
    """Per-object GRU for temporal state prediction.

    Strategy:
      1. Encode history: run GRU over [T_h, N, D] → get final hidden state per object
      2. Decode future: initialize GRU with history final state, run for T_p steps
      3. Each step produces a future token that can be decoded to state/collision/mask
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_attention: bool = True,
        attention_heads: int = 4,
        attention_dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention

        # Project once, then run a hidden_dim-sized GRU. This keeps decode inputs
        # compatible even when fused_dim and gru_hidden_dim diverge.
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        heads = _valid_num_heads(hidden_dim, attention_heads)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=attention_dropout,
            batch_first=True,
        ) if use_attention else None
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.attn_norm = nn.LayerNorm(hidden_dim)

    def encode_history(
        self,
        tokens: torch.Tensor,    # [B, Th, N, D]
        valid_mask: torch.Tensor, # [B, N]
    ) -> torch.Tensor:
        """
        Encode history sequence into final hidden state.

        Returns:
            h_last: [B, N, num_layers, hidden_dim] - final hidden state per object per layer
        """
        B, Th, N, D = tokens.shape

        # Reshape: process each object independently
        # [B, Th, N, D] -> [B*N, Th, D]
        tokens_flat = tokens.permute(0, 2, 1, 3).reshape(B * N, Th, D)

        # Run GRU (with input projection if dims differ)
        x_proj = self.input_proj(tokens_flat)
        gru_out, h_last = self.gru(x_proj)
        # h_last: [num_layers, B*N, hidden_dim]

        if self.temporal_attn is not None:
            attn_out, _ = self.temporal_attn(gru_out, gru_out, gru_out, need_weights=False)
            enhanced = self.attn_norm(gru_out + self.attn_dropout(attn_out))
            h_last = h_last.clone()
            # Only the top recurrent layer seeds decoding with global history
            # attention; lower layers keep the raw recurrent state.
            h_last[-1] = enhanced[:, -1]

        # Reshape back: [num_layers, B*N, H] -> [B, N, num_layers, H]
        h_last = h_last.permute(1, 0, 2).reshape(B, N, self.num_layers, self.hidden_dim)

        # Zero out invalid objects
        h_last = h_last * valid_mask.unsqueeze(-1).unsqueeze(-1).float()

        return h_last

    def decode_future(
        self,
        initial_token: torch.Tensor,  # [B, N, D] - last observed token
        h_init: torch.Tensor,          # [B, N, num_layers, hidden_dim]
        predict_length: int,
        valid_mask: torch.Tensor,      # [B, N]
        teacher_tokens: Optional[torch.Tensor] = None,  # [B, Tp, N, D]
        teacher_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Autoregressively decode future tokens.

        Returns:
            future_tokens: [B, Tp, N, hidden_dim]
        """
        B, N, D = initial_token.shape

        # Prepare initial hidden state: [num_layers, B*N, hidden_dim]
        h = h_init.permute(2, 0, 1, 3).reshape(self.num_layers, B * N, self.hidden_dim)

        # Prepare initial input: [B*N, 1, D] with projection
        x = self.input_proj(initial_token.reshape(B * N, 1, D))
        teacher_mask = None
        if teacher_tokens is not None and teacher_ratio > 0.0:
            if teacher_ratio >= 1.0:
                teacher_mask = torch.ones(B, 1, 1, device=initial_token.device, dtype=torch.bool)
            else:
                teacher_mask = torch.rand(B, 1, 1, device=initial_token.device) < teacher_ratio
            teacher_mask = teacher_mask.expand(B, N, 1).reshape(B * N, 1, 1)

        future_tokens = []
        for t in range(predict_length):
            # GRU step
            out, h = self.gru(x, h)  # out: [B*N, 1, hidden_dim]
            future_tokens.append(out)

            if teacher_mask is not None and t < teacher_tokens.shape[1]:
                teacher = teacher_tokens[:, t].reshape(B * N, 1, -1)
                teacher = self.input_proj(teacher)
                x = torch.where(teacher_mask, teacher, out)
            else:
                # Next input is the output (autoregressive)
                x = out

        # Stack: [B*N, Tp, hidden_dim] -> [B, Tp, N, hidden_dim]
        future = torch.cat(future_tokens, dim=1)  # [B*N, Tp, hidden_dim]
        future = future.reshape(B, N, predict_length, self.hidden_dim)
        future = future.permute(0, 2, 1, 3)  # [B, Tp, N, hidden_dim]

        # Zero out invalid objects
        future = future * valid_mask.unsqueeze(1).unsqueeze(-1).float()

        return future

    def forward(
        self,
        tokens: torch.Tensor,       # [B, T, N, D] - full sequence (history + maybe future)
        valid_mask: torch.Tensor,    # [B, N]
        history_length: int,
        predict_length: int,
        teacher_tokens: Optional[torch.Tensor] = None,
        teacher_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward: encode history, then decode future.

        Args:
            tokens: [B, T, N, D] where T >= history_length
            valid_mask: [B, N]
            history_length: number of history frames (Th)
            predict_length: number of future frames to predict (Tp)

        Returns:
            dict with:
              future_tokens: [B, Tp, N, hidden_dim]
              h_last: [B, N, num_layers, hidden_dim]
        """
        # Extract history
        history = tokens[:, :history_length]  # [B, Th, N, D]

        # Encode history
        h_last = self.encode_history(history, valid_mask)  # [B, N, num_layers, hidden_dim]

        # Use last history frame as initial input for decoding
        last_token = history[:, -1]  # [B, N, D]

        # Decode future
        future_tokens = self.decode_future(
            last_token,
            h_last,
            predict_length,
            valid_mask,
            teacher_tokens=teacher_tokens,
            teacher_ratio=teacher_ratio,
        )

        return {
            'future_tokens': future_tokens,  # [B, Tp, N, hidden_dim]
            'h_last': h_last,
        }
