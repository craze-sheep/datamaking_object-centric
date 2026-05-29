"""PhysicsObjectGraphPredictor: End-to-end model.

Architecture:
  1. PhysicsObjectEncoder (dual-stream: visual CNN + physics embedding)
  2. ForceAwareGNN (interaction modeling with force-aware edges)
  3. TemporalGRU (GRU-based temporal prediction)
  4. MultiHeadDecoder (state, collision, mask, RGB)

Data flow:
  Input batch → Encoder → GNN (per-frame interaction) → GRU (temporal) → Decoder → predictions
"""
import torch
import torch.nn as nn
from typing import Dict

from config import AIModelConfig
from encoder import PhysicsObjectEncoder
from interaction import ForceAwareGNN
from temporal import TemporalGRU
from decoder import MultiHeadDecoder
from loss import PhysicsLoss


class PhysicsObjectGraphPredictor(nn.Module):
    """Physics-aware object-centric video prediction with graph neural networks."""

    def __init__(self, config: AIModelConfig):
        super().__init__()
        self.config = config

        # 1. Dual-stream encoder
        self.encoder = PhysicsObjectEncoder(
            image_size=config.image_size,
            cnn_channels=config.cnn_channels,
            visual_out_dim=config.visual_out_dim,
            attr_dim=config.attr_dim,
            state_dim=config.state_dim,
            attr_embed_dim=config.attr_embed_dim,
            state_embed_dim=config.state_embed_dim,
            physics_out_dim=config.physics_out_dim,
            fused_dim=config.fused_dim,
        )

        # 2. Force-aware GNN interaction
        self.interaction = ForceAwareGNN(
            node_dim=config.fused_dim,
            edge_dim=config.gnn_edge_dim,
            hidden_dim=config.gnn_hidden_dim,
            num_layers=config.gnn_layers,
            dropout=config.gnn_dropout,
        )

        # 3. Temporal GRU
        self.temporal = TemporalGRU(
            input_dim=config.fused_dim,
            hidden_dim=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
        )

        # 4. Multi-head decoder
        self.decoder = MultiHeadDecoder(
            token_dim=config.gru_hidden_dim,
            state_dim=config.state_dim,
            state_decoder_hidden=config.state_decoder_hidden,
            collision_hidden=config.collision_hidden,
            mask_base_channels=config.mask_base_channels,
            rgb_base_channels=config.rgb_base_channels,
            image_size=config.image_size,
        )

        # 5. Loss
        self.loss_fn = PhysicsLoss(
            history_length=config.history_length,
            predict_length=config.predict_length,
            state_dim=config.state_dim,
            rgb_weight=config.rgb_weight,
            state_weight=config.state_weight,
            collision_weight=config.collision_weight,
            mask_weight=config.mask_weight,
            collision_threshold=config.collision_threshold,
            force_std=config.force_matrix_scale,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode → interact → temporal predict → decode.

        Args:
            batch: dict with rgb, mask, obj_attrs, dyn_state, force_matrix, valid_mask

        Returns:
            dict with predictions: state_pred, collision_logits, pair_mask,
                                   mask_logits, mask_prob, rgb_pred
        """
        # 1. Encode: visual + physics → per-object tokens
        enc_out = self.encoder(batch)  # tokens: [B, T, N, D]

        # 2. Interact: GNN message passing per frame
        tokens = enc_out['tokens']  # [B, T, N, D]
        interacted = self.interaction(
            tokens,
            batch['dyn_state'],     # raw state for edge features
            batch['force_matrix'],  # force matrix as edge features
            batch['valid_mask'],
        )  # [B, T, N, D]

        # 3. Temporal prediction: GRU encode history → decode future
        temporal_out = self.temporal(
            interacted,
            batch['valid_mask'],
            self.config.history_length,
            self.config.predict_length,
        )
        future_tokens = temporal_out['future_tokens']  # [B, Tp, N, hidden_dim]

        # 4. Decode: future tokens → state, collision, mask, RGB
        pred = self.decoder(future_tokens, batch['valid_mask'])

        return pred

    def compute_loss(
        self,
        pred: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss."""
        return self.loss_fn(pred, batch)

    def forward_with_loss(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward + loss in one call (convenient for training)."""
        pred = self.forward(batch)
        loss_dict = self.compute_loss(pred, batch)
        loss_dict['pred'] = pred
        return loss_dict
