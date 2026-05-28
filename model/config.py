"""Shared configuration for the physics video prediction model."""
from dataclasses import dataclass
from typing import Optional, Tuple


ATTR_STATIC_INDEX = 8
STATE_POS_SLICE = slice(0, 3)
STATE_QUAT_SLICE = slice(3, 7)
STATE_VEL_SLICE = slice(7, 10)
STATE_ANGVEL_SLICE = slice(10, 13)
STATE_FORCE_SLICE = slice(13, 16)


@dataclass
class ModelConfig:
    """Default model and training dimensions."""

    image_size: int = 128
    history_length: int = 12
    predict_length: int = 12
    max_objects: int = 7
    attr_dim: int = 14
    state_dim: int = 16
    token_dim: int = 256
    visual_dim: int = 256
    physical_dim: int = 128
    edge_dim: int = 16
    num_views: int = 1
    use_dinov2: bool = False
    use_scene_embedding: bool = False
    use_view_embedding: bool = True

    @classmethod
    def minimal_12gb(cls):
        return cls(
            history_length=12,
            predict_length=12,
            max_objects=7,
            token_dim=128,
            visual_dim=128,
            physical_dim=128,
            edge_dim=16,
        )

    @classmethod
    def tiny_test(cls, token_dim=64, max_objects=3, history_length=2, predict_length=3):
        return cls(
            history_length=history_length,
            predict_length=predict_length,
            max_objects=max_objects,
            token_dim=token_dim,
            visual_dim=token_dim,
            physical_dim=token_dim,
            edge_dim=16,
        )

    @classmethod
    def minimal(cls, token_dim=64, max_objects=3, history_length=2, predict_length=3):
        return cls.tiny_test(token_dim, max_objects, history_length, predict_length)


@dataclass
class EncoderConfig:
    image_size: int = 128
    history_length: int = 12
    max_objects: int = 7
    attr_dim: int = 14
    state_dim: int = 16
    visual_dim: int = 256
    attr_embed_dim: int = 64
    state_embed_dim: int = 128
    physical_dim: int = 128
    token_dim: int = 256
    feature_map_size: int = 32
    use_dinov2: bool = False
    use_scene_embedding: bool = False
    use_view_embedding: bool = True
    num_views: int = 1


@dataclass
class InteractionConfig:
    token_dim: int = 256
    state_dim: int = 16
    force_dim: int = 3
    edge_dim: int = 16
    num_layers: int = 2
    hidden_dim: int = 256
    dropout: float = 0.0
    aggregation: str = "mean"
    exclude_self_edges: bool = True


@dataclass
class TemporalConfig:
    token_dim: int = 256
    history_length: int = 12
    predict_length: int = 12
    max_objects: int = 7
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4
    ffn_dim: int = 1024
    dropout: float = 0.0
    norm_first: bool = True
    batch_first: bool = True


@dataclass
class DecoderConfig:
    token_dim: int = 256
    state_dim: int = 16
    image_size: int = 128
    max_objects: int = 7
    predict_length: int = 12
    mask_base_channels: int = 64
    rgb_base_channels: int = 64
    use_mask_head: bool = True
    use_depth_head: bool = False
    rgb_output_activation: str = "tanh"
    static_mask_strategy: str = "predict"


@dataclass
class LossConfig:
    history_length: int = 12
    predict_length: int = 12
    state_dim: int = 16
    rgb_weight: float = 1.0
    state_weight: float = 0.1
    collision_weight: float = 0.5
    mask_weight: float = 0.1
    lpips_weight: float = 0.0
    collision_threshold: float = 1e-6
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    force_mean: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    force_std: Tuple[float, float, float] = (50.0, 50.0, 50.0)
    state_component_weights: Tuple[float, ...] = (
        1.0, 1.0, 1.0,
        0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5,
        0.25, 0.25, 0.25,
        0.25, 0.25, 0.25,
    )
    future_step_weights: Optional[Tuple[float, ...]] = None
