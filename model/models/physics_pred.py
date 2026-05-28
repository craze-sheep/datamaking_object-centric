"""End-to-end model wrapper for physics video prediction."""
try:
    from config import (
        ModelConfig,
        EncoderConfig,
        InteractionConfig,
        TemporalConfig,
        DecoderConfig,
        LossConfig,
    )
    from models.encoder import InputEncoder
    from models.interaction import InteractionModule
    from models.temporal import TemporalPredictor
    from models.decoder import OutputDecoder
    from models.loss import PhysicsPredictionLoss
except ImportError:  # pragma: no cover
    from ..config import (  # type: ignore
        ModelConfig,
        EncoderConfig,
        InteractionConfig,
        TemporalConfig,
        DecoderConfig,
        LossConfig,
    )
    from .encoder import InputEncoder  # type: ignore
    from .interaction import InteractionModule  # type: ignore
    from .temporal import TemporalPredictor  # type: ignore
    from .decoder import OutputDecoder  # type: ignore
    from .loss import PhysicsPredictionLoss  # type: ignore

from torch import nn


class PhysicsVideoPredictor(nn.Module):
    """Compose T5-T9 modules into one predictor."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config
        self.encoder = InputEncoder(
            EncoderConfig(
                image_size=cfg.image_size,
                history_length=cfg.history_length,
                max_objects=cfg.max_objects,
                attr_dim=cfg.attr_dim,
                state_dim=cfg.state_dim,
                visual_dim=cfg.visual_dim,
                physical_dim=cfg.physical_dim,
                token_dim=cfg.token_dim,
                use_dinov2=cfg.use_dinov2,
                use_scene_embedding=cfg.use_scene_embedding,
                use_view_embedding=cfg.use_view_embedding,
                num_views=cfg.num_views,
            )
        )
        self.interaction = InteractionModule(
            InteractionConfig(
                token_dim=cfg.token_dim,
                state_dim=cfg.state_dim,
                edge_dim=cfg.edge_dim,
                hidden_dim=cfg.token_dim,
            )
        )
        heads = 4 if cfg.token_dim % 4 == 0 else 1
        self.temporal = TemporalPredictor(
            TemporalConfig(
                token_dim=cfg.token_dim,
                history_length=cfg.history_length,
                predict_length=cfg.predict_length,
                max_objects=cfg.max_objects,
                num_encoder_layers=1,
                num_decoder_layers=1,
                num_heads=heads,
                ffn_dim=max(cfg.token_dim * 2, 32),
            )
        )
        base_channels = max(4, min(64, cfg.token_dim // 2))
        self.decoder = OutputDecoder(
            DecoderConfig(
                token_dim=cfg.token_dim,
                state_dim=cfg.state_dim,
                image_size=cfg.image_size,
                max_objects=cfg.max_objects,
                predict_length=cfg.predict_length,
                mask_base_channels=base_channels,
                rgb_base_channels=base_channels,
            )
        )
        self.loss_fn = PhysicsPredictionLoss(
            LossConfig(history_length=cfg.history_length, predict_length=cfg.predict_length, state_dim=cfg.state_dim)
        )

    def forward(self, batch):
        encoder_out = self.encoder(batch)
        interaction_out = self.interaction(
            object_token=encoder_out["object_token"],
            state_hist=encoder_out["state_hist"],
            force_hist=encoder_out["force_hist"],
            valid_mask=encoder_out["valid_mask"],
            static_flag=encoder_out["static_flag"],
        )
        temporal_out = self.temporal(interaction_out["inter_token"], encoder_out["valid_mask"])
        last_state = encoder_out["state_hist"][:, -1]
        last_mask = batch.get("mask", None)
        if last_mask is not None:
            last_mask = last_mask[:, self.config.history_length - 1]
        pred = self.decoder(
            future_token=temporal_out["future_token"],
            valid_mask=encoder_out["valid_mask"],
            static_flag=encoder_out["static_flag"],
            last_state=last_state,
            last_mask=last_mask,
        )
        pred["valid_mask"] = encoder_out["valid_mask"]
        pred["static_flag"] = encoder_out["static_flag"]
        pred["dynamic_mask"] = encoder_out["dynamic_mask"]
        return pred

    def compute_loss(self, batch):
        pred = self.forward(batch)
        return self.loss_fn(pred, batch, pred["valid_mask"], pred["static_flag"])
