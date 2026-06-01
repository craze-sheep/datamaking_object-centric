"""Configuration for PhysicsObjectGraphPredictor.

Key differences from baseline PhysicsVideoPredictor:
  - GRU-based temporal modeling (vs Transformer)
  - Force-aware GNN interaction (vs dense pairwise MLP)
  - Physics-informed inductive biases (explicit mass/friction encoding)
  - Dual-stream visual+physics encoder
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class AIModelConfig:
    """Master config for the physics-object-graph model."""

    # --- Data ---
    image_size: int = 128
    history_length: int = 12
    predict_length: int = 12
    max_objects: int = 7
    attr_dim: int = 14       # 3(size)+4(friction)+1(mass)+1(restitution)+1(static)+4(type)
    state_dim: int = 16      # 3(pos)+4(quat)+3(vel)+3(angvel)+3(force)
    force_dim: int = 3

    # --- Visual encoder (lightweight CNN) ---
    cnn_channels: Tuple[int, ...] = (32, 64, 128, 128)
    visual_out_dim: int = 128  # per-object visual feature dim after ROI pool

    # --- Physics encoder ---
    attr_embed_dim: int = 64
    state_embed_dim: int = 64
    physics_out_dim: int = 128  # per-object physics feature dim

    # --- Fusion ---
    fused_dim: int = 128  # final per-object token dim (= visual_out + physics_out projected)

    # --- Interaction (GNN) ---
    gnn_layers: int = 3
    gnn_hidden_dim: int = 128
    gnn_edge_dim: int = 64
    gnn_dropout: float = 0.0
    gnn_use_attention: bool = True
    gnn_use_residual: bool = True

    # --- Temporal (GRU) ---
    gru_hidden_dim: int = 128
    gru_num_layers: int = 2
    use_temporal_attention: bool = True
    temporal_attention_heads: int = 4
    temporal_attention_dropout: float = 0.0
    # Disabled until a decoder-feedback teacher path exists. Feeding future
    # encoder tokens leaks RGB/mask/force targets into training.
    use_scheduled_sampling: bool = False
    scheduled_sampling_start: float = 1.0
    scheduled_sampling_end: float = 0.0

    # --- State decoder ---
    state_decoder_hidden: int = 256

    # --- Collision decoder ---
    collision_hidden: int = 64

    # --- RGB/Mask decoder ---
    mask_base_channels: int = 32
    rgb_base_channels: int = 32

    # --- Training ---
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 10
    gradient_clip: float = 1.0
    use_amp: bool = True
    gradient_accumulation_steps: int = 1

    # --- Loss weights ---
    rgb_weight: float = 1.0
    state_weight: float = 0.5
    collision_weight: float = 0.5
    mask_weight: float = 0.3
    use_uncertainty_weighting: bool = True
    ssim_weight: float = 0.1
    lpips_weight: float = 0.0
    energy_weight: float = 0.01
    collision_effect_weight: float = 0.05

    # --- Normalization ---
    # State normalization scales (rough estimates from data)
    pos_scale: float = 5.0
    vel_scale: float = 10.0
    angvel_scale: float = 5.0
    force_scale: float = 50.0
    # Force matrix normalization
    force_matrix_scale: float = 50.0

    # --- Collision threshold ---
    collision_threshold: float = 1e-6  # in original force magnitude (N)

    @classmethod
    def tiny(cls):
        """Config for smoke testing."""
        return cls(
            cnn_channels=(16, 32, 32),
            visual_out_dim=32,
            attr_embed_dim=16,
            state_embed_dim=16,
            physics_out_dim=32,
            fused_dim=32,
            gnn_layers=1,
            gnn_hidden_dim=32,
            gnn_edge_dim=16,
            gru_hidden_dim=32,
            gru_num_layers=1,
            state_decoder_hidden=64,
            collision_hidden=16,
            mask_base_channels=16,
            rgb_base_channels=16,
            batch_size=2,
            history_length=4,
            predict_length=4,
            max_objects=3,
        )

    @classmethod
    def small_8gb(cls):
        """Config for 8GB GPU, conservative."""
        return cls(
            cnn_channels=(32, 64, 128),
            visual_out_dim=96,
            attr_embed_dim=48,
            state_embed_dim=48,
            physics_out_dim=96,
            fused_dim=96,
            gnn_layers=3,
            gnn_hidden_dim=96,
            gnn_edge_dim=48,
            gru_hidden_dim=96,
            gru_num_layers=2,
            state_decoder_hidden=192,
            collision_hidden=48,
            mask_base_channels=24,
            rgb_base_channels=24,
            batch_size=4,
        )
