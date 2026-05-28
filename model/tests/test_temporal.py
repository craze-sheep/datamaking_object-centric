"""Tests for temporal prediction module."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_temporal_predictor_forward_shapes():
    from models.temporal import TemporalConfig, TemporalPredictor

    cfg = TemporalConfig(token_dim=32, history_length=4, predict_length=5, max_objects=3, num_encoder_layers=1, num_decoder_layers=1, num_heads=4, ffn_dim=64)
    model = TemporalPredictor(cfg)
    inter_token = torch.randn(2, 4, 3, 32)
    valid = torch.tensor([[True, True, False], [True, False, False]])

    out = model(inter_token, valid)

    assert out['future_token'].shape == (2, 5, 3, 32)
    assert out['memory'].shape == (2, 12, 32)
    assert out['future_query'].shape == (2, 15, 32)
    assert out['token_padding_mask'].shape == (2, 12)


def test_padding_objects_are_zeroed_in_future_tokens():
    from models.temporal import TemporalConfig, TemporalPredictor

    cfg = TemporalConfig(token_dim=16, history_length=3, predict_length=2, max_objects=4, num_encoder_layers=1, num_decoder_layers=1, num_heads=4, ffn_dim=32)
    model = TemporalPredictor(cfg)
    inter_token = torch.randn(1, 3, 4, 16)
    valid = torch.tensor([[True, False, True, False]])

    out = model(inter_token, valid)

    assert torch.all(out['future_token'][:, :, 1] == 0)
    assert torch.all(out['future_token'][:, :, 3] == 0)


def test_memory_key_padding_mask_uses_valid_mask():
    from models.temporal import TemporalConfig, TemporalPredictor

    cfg = TemporalConfig(token_dim=16, history_length=2, predict_length=1, max_objects=3, num_encoder_layers=1, num_decoder_layers=1, num_heads=4, ffn_dim=32)
    model = TemporalPredictor(cfg)
    inter_token = torch.randn(1, 2, 3, 16)
    valid = torch.tensor([[True, False, True]])

    out = model(inter_token, valid)

    assert torch.equal(out['token_padding_mask'], torch.tensor([[False, True, False, False, True, False]]))


def test_valid_mask_is_returned_for_decoder():
    from models.temporal import TemporalConfig, TemporalPredictor

    cfg = TemporalConfig(token_dim=16, history_length=2, predict_length=1, max_objects=3, num_encoder_layers=1, num_decoder_layers=1, num_heads=4, ffn_dim=32)
    model = TemporalPredictor(cfg)
    inter_token = torch.randn(1, 2, 3, 16)
    valid = torch.tensor([[True, False, True]])

    out = model(inter_token, valid)

    assert torch.equal(out['valid_mask'], valid)


def test_all_invalid_mask_outputs_zero_without_nan():
    from models.temporal import TemporalConfig, TemporalPredictor

    cfg = TemporalConfig(token_dim=16, history_length=2, predict_length=2, max_objects=3, num_encoder_layers=1, num_decoder_layers=1, num_heads=4, ffn_dim=32)
    model = TemporalPredictor(cfg)
    inter_token = torch.randn(1, 2, 3, 16)
    valid = torch.zeros(1, 3, dtype=torch.bool)

    out = model(inter_token, valid)

    assert torch.isfinite(out['memory']).all()
    assert torch.isfinite(out['future_token']).all()
    assert torch.all(out['future_token'] == 0)


def test_gradients_flow_through_temporal_predictor():
    from models.temporal import TemporalConfig, TemporalPredictor

    cfg = TemporalConfig(token_dim=16, history_length=2, predict_length=2, max_objects=2, num_encoder_layers=1, num_decoder_layers=1, num_heads=4, ffn_dim=32)
    model = TemporalPredictor(cfg)
    inter_token = torch.randn(1, 2, 2, 16, requires_grad=True)
    valid = torch.ones(1, 2, dtype=torch.bool)

    loss = model(inter_token, valid)['future_token'].sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
