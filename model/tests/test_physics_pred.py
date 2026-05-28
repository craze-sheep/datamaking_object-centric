"""Tests for the end-to-end physics video predictor wrapper."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_batch(batch_size=2, total_length=5, n=3):
    attrs = torch.zeros(batch_size, n, 14)
    attrs[:, 0, 8] = 1.0
    valid = torch.ones(batch_size, n, dtype=torch.bool)
    valid[:, -1] = False
    return {
        'rgb': torch.randn(batch_size, total_length, 3, 128, 128),
        'mask': torch.randint(0, 2, (batch_size, total_length, n, 128, 128)).float(),
        'obj_attrs': attrs,
        'dyn_state': torch.randn(batch_size, total_length, n, 16),
        'force_matrix': torch.randn(batch_size, total_length, n, n, 3),
        'valid_mask': valid,
        'scene_id': torch.zeros(batch_size, dtype=torch.long),
        'sample_id': torch.arange(batch_size),
    }


def test_physics_video_predictor_forward_shapes():
    from config import ModelConfig
    from models.physics_pred import PhysicsVideoPredictor

    cfg = ModelConfig.tiny_test(token_dim=16, max_objects=3, history_length=2, predict_length=3)
    model = PhysicsVideoPredictor(cfg)
    batch = make_batch(total_length=5, n=3)

    out = model(batch)

    assert out['rgb'].shape == (2, 3, 3, 128, 128)
    assert out['state'].shape == (2, 3, 3, 16)
    assert out['collision_logits'].shape == (2, 3, 3, 3)
    assert out['mask_logits'].shape == (2, 3, 3, 128, 128)
    assert torch.equal(out['valid_mask'], batch['valid_mask'])


def test_physics_video_predictor_compute_loss():
    from config import ModelConfig
    from models.physics_pred import PhysicsVideoPredictor

    cfg = ModelConfig.tiny_test(token_dim=16, max_objects=3, history_length=2, predict_length=3)
    model = PhysicsVideoPredictor(cfg)
    batch = make_batch(total_length=5, n=3)

    loss = model.compute_loss(batch)

    assert 'loss' in loss
    assert torch.isfinite(loss['loss'])


def test_future_targets_do_not_change_forward_when_history_same():
    from config import ModelConfig
    from models.physics_pred import PhysicsVideoPredictor

    torch.manual_seed(0)
    cfg = ModelConfig.tiny_test(token_dim=16, max_objects=3, history_length=2, predict_length=3)
    model = PhysicsVideoPredictor(cfg)
    model.eval()
    batch1 = make_batch(total_length=5, n=3)
    batch2 = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch1.items()}
    batch2['rgb'][:, 2:] = torch.randn_like(batch2['rgb'][:, 2:])
    batch2['mask'][:, 2:] = torch.randn_like(batch2['mask'][:, 2:])
    batch2['dyn_state'][:, 2:] = torch.randn_like(batch2['dyn_state'][:, 2:])
    batch2['force_matrix'][:, 2:] = torch.randn_like(batch2['force_matrix'][:, 2:])

    with torch.no_grad():
        out1 = model(batch1)
        out2 = model(batch2)

    assert torch.allclose(out1['rgb'], out2['rgb'])
    assert torch.allclose(out1['state'], out2['state'])
