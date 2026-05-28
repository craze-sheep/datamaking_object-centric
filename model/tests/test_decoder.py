"""Tests for output decoder."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_inputs(batch_size=2, tp=3, n=4, d=32):
    token = torch.randn(batch_size, tp, n, d)
    valid = torch.ones(batch_size, n, dtype=torch.bool)
    valid[:, -1] = False
    static = torch.zeros(batch_size, n, dtype=torch.bool)
    static[:, 0] = True
    last_state = torch.randn(batch_size, n, 16)
    last_state[:, :, 3:7] = torch.nn.functional.normalize(last_state[:, :, 3:7], dim=-1)
    return token, valid, static, last_state


def test_decoder_forward_full_output_keys_and_shapes():
    from models.decoder import DecoderConfig, OutputDecoder

    cfg = DecoderConfig(token_dim=32, predict_length=3, max_objects=4, mask_base_channels=8, rgb_base_channels=8)
    decoder = OutputDecoder(cfg)
    token, valid, static, last_state = make_inputs()

    out = decoder(token, valid, static, last_state)

    assert out['rgb'].shape == (2, 3, 3, 128, 128)
    assert out['state'].shape == (2, 3, 4, 16)
    assert out['collision_logits'].shape == (2, 3, 4, 4)
    assert out['pair_mask'].shape == (2, 3, 4, 4)
    assert out['mask_logits'].shape == (2, 3, 4, 128, 128)
    assert out['mask_prob'].shape == (2, 3, 4, 128, 128)


def test_static_objects_copy_last_state():
    from models.decoder import DecoderConfig, OutputDecoder

    cfg = DecoderConfig(token_dim=16, predict_length=2, max_objects=3, mask_base_channels=4, rgb_base_channels=4)
    decoder = OutputDecoder(cfg)
    token, valid, static, last_state = make_inputs(batch_size=1, tp=2, n=3, d=16)

    out = decoder(token, valid, static, last_state)

    assert torch.allclose(out['state'][0, :, 0], last_state[0, 0].expand(2, 16))


def test_padding_object_outputs_are_zero():
    from models.decoder import DecoderConfig, OutputDecoder

    cfg = DecoderConfig(token_dim=16, predict_length=2, max_objects=3, mask_base_channels=4, rgb_base_channels=4)
    decoder = OutputDecoder(cfg)
    token, valid, static, last_state = make_inputs(batch_size=1, tp=2, n=3, d=16)
    valid[0, -1] = False

    out = decoder(token, valid, static, last_state)

    assert torch.all(out['state'][0, :, -1] == 0)
    assert torch.all(out['mask_logits'][0, :, -1] == 0)
    assert torch.all(out['mask_prob'][0, :, -1] == 0)


def test_collision_pair_mask_excludes_self_and_padding():
    from models.decoder import DecoderConfig, OutputDecoder

    cfg = DecoderConfig(token_dim=16, predict_length=2, max_objects=3, mask_base_channels=4, rgb_base_channels=4)
    decoder = OutputDecoder(cfg)
    token, valid, static, last_state = make_inputs(batch_size=1, tp=2, n=3, d=16)
    valid[0, -1] = False

    out = decoder(token, valid, static, last_state)
    pair_mask = out['pair_mask']

    assert not torch.diagonal(pair_mask, dim1=2, dim2=3).any()
    assert not pair_mask[:, :, -1, :].any()
    assert not pair_mask[:, :, :, -1].any()


def test_rgb_output_range_is_minus_one_to_one():
    from models.decoder import DecoderConfig, OutputDecoder

    cfg = DecoderConfig(token_dim=16, predict_length=1, max_objects=2, mask_base_channels=4, rgb_base_channels=4)
    decoder = OutputDecoder(cfg)
    token, valid, static, last_state = make_inputs(batch_size=1, tp=1, n=2, d=16)

    rgb = decoder(token, valid, static, last_state)['rgb']

    assert rgb.min() >= -1.0001
    assert rgb.max() <= 1.0001


def test_dynamic_quaternion_is_unit_normalized():
    from models.decoder import DecoderConfig, OutputDecoder

    cfg = DecoderConfig(token_dim=16, predict_length=2, max_objects=3, mask_base_channels=4, rgb_base_channels=4)
    decoder = OutputDecoder(cfg)
    token, valid, static, last_state = make_inputs(batch_size=1, tp=2, n=3, d=16)

    state = decoder(token, valid, static, last_state)['state']
    quat_norm = state[0, :, 1, 3:7].norm(dim=-1)

    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-4)


def test_rgb_compose_no_nan_when_masks_are_zero():
    from models.decoder import compose_rgb

    object_rgb = torch.randn(1, 2, 3, 3, 128, 128)
    mask_prob = torch.zeros(1, 2, 3, 128, 128)
    valid = torch.ones(1, 3, dtype=torch.bool)
    background = torch.zeros(3, 128, 128)

    rgb, obj_alpha, bg_alpha = compose_rgb(object_rgb, mask_prob, valid, background)

    assert torch.isfinite(rgb).all()
    assert torch.all(obj_alpha == 0)
    assert torch.all(bg_alpha == 1)


def test_depth_key_is_none_by_default():
    from models.decoder import DecoderConfig, OutputDecoder

    cfg = DecoderConfig(token_dim=16, predict_length=1, max_objects=2, mask_base_channels=4, rgb_base_channels=4)
    decoder = OutputDecoder(cfg)
    token, valid, static, last_state = make_inputs(batch_size=1, tp=1, n=2, d=16)

    out = decoder(token, valid, static, last_state)

    assert out['depth'] is None
