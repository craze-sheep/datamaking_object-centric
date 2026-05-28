"""Tests for encoder modules."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_batch(batch_size=2, total_length=24, max_objects=7):
    attrs = torch.randn(batch_size, max_objects, 14)
    attrs[..., 8] = 0.0
    attrs[:, 0, 8] = 1.0  # first object static
    valid_mask = torch.ones(batch_size, max_objects, dtype=torch.bool)
    valid_mask[:, -2:] = False
    return {
        'rgb': torch.randn(batch_size, total_length, 3, 128, 128),
        'mask': torch.rand(batch_size, total_length, max_objects, 128, 128),
        'obj_attrs': attrs,
        'dyn_state': torch.randn(batch_size, total_length, max_objects, 16),
        'force_matrix': torch.randn(batch_size, total_length, max_objects, max_objects, 3),
        'valid_mask': valid_mask,
    }


def test_light_cnn_outputs_expected_feature_shape():
    from models.encoder import LightCNNBackbone

    model = LightCNNBackbone(out_channels=256)
    x = torch.randn(4, 3, 128, 128)

    y = model(x)

    assert y.shape == (4, 256, 32, 32)


def test_mask_roi_pool_outputs_object_features_without_nan():
    from models.encoder import MaskROIPool

    pool = MaskROIPool(visual_dim=16, feature_channels=8)
    feat = torch.randn(2, 3, 8, 32, 32)
    mask = torch.zeros(2, 3, 4, 128, 128)
    mask[:, :, 0, :64, :64] = 1.0
    valid_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    out, resized = pool(feat, mask, valid_mask)

    assert out.shape == (2, 3, 4, 16)
    assert resized.shape == (2, 3, 4, 32, 32)
    assert torch.isfinite(out).all()
    assert torch.all(out[:, :, 2:] == 0)


def test_input_encoder_forward_shapes_and_history_slice():
    from models.encoder import EncoderConfig, InputEncoder

    cfg = EncoderConfig(token_dim=64, visual_dim=32, physical_dim=32, state_embed_dim=32)
    model = InputEncoder(cfg)
    batch = make_batch()

    out = model(batch)

    assert out['object_token'].shape == (2, 12, 7, 64)
    assert out['visual_feat'].shape == (2, 12, 7, 32)
    assert out['physical_feat'].shape == (2, 12, 7, 32)
    assert out['state_hist'].shape == (2, 12, 7, 16)
    assert out['force_hist'].shape == (2, 12, 7, 7, 3)
    assert out['resized_mask'].shape == (2, 12, 7, 32, 32)


def test_padding_objects_are_zeroed():
    from models.encoder import EncoderConfig, InputEncoder

    cfg = EncoderConfig(token_dim=64, visual_dim=32, physical_dim=32, state_embed_dim=32)
    model = InputEncoder(cfg)
    batch = make_batch()

    out = model(batch)

    assert torch.all(out['object_token'][:, :, -2:] == 0)
    assert torch.all(out['visual_feat'][:, :, -2:] == 0)
    assert torch.all(out['physical_feat'][:, :, -2:] == 0)


def test_static_and_dynamic_masks_from_attrs():
    from models.encoder import EncoderConfig, InputEncoder

    cfg = EncoderConfig(token_dim=64, visual_dim=32, physical_dim=32, state_embed_dim=32)
    model = InputEncoder(cfg)
    batch = make_batch()

    out = model(batch)

    assert out['static_flag'][:, 0].all()
    assert not out['dynamic_mask'][:, 0].any()
    assert not out['static_flag'][:, -1].any()
    assert not out['dynamic_mask'][:, -1].any()


def test_missing_view_id_defaults_to_zero():
    from models.encoder import EncoderConfig, InputEncoder

    cfg = EncoderConfig(token_dim=64, visual_dim=32, physical_dim=32, state_embed_dim=32)
    model = InputEncoder(cfg)
    batch = make_batch()

    out = model(batch)

    assert out['object_token'].shape == (2, 12, 7, 64)


def test_gradients_flow_through_trainable_encoder():
    from models.encoder import EncoderConfig, InputEncoder

    cfg = EncoderConfig(token_dim=32, visual_dim=16, physical_dim=16, state_embed_dim=16)
    model = InputEncoder(cfg)
    batch = make_batch(batch_size=1)

    loss = model(batch)['object_token'].sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_encoder_exports_force_and_state_for_interaction():
    from models.encoder import EncoderConfig, InputEncoder

    cfg = EncoderConfig(token_dim=64, visual_dim=32, physical_dim=32, state_embed_dim=32)
    model = InputEncoder(cfg)
    batch = make_batch()

    out = model(batch)

    assert torch.equal(out['state_hist'], batch['dyn_state'][:, :12])
    assert torch.equal(out['force_hist'], batch['force_matrix'][:, :12])
