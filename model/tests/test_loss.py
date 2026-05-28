"""Tests for multi-task prediction losses."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_pred_batch(batch_size=2, th=2, tp=3, n=4):
    pred = {
        'rgb': torch.zeros(batch_size, tp, 3, 128, 128),
        'state': torch.zeros(batch_size, tp, n, 16),
        'collision_logits': torch.zeros(batch_size, tp, n, n),
        'pair_mask': torch.ones(batch_size, tp, n, n, dtype=torch.bool),
        'mask_logits': torch.zeros(batch_size, tp, n, 128, 128),
        'mask_prob': torch.full((batch_size, tp, n, 128, 128), 0.5),
    }
    eye = torch.eye(n, dtype=torch.bool)[None, None]
    pred['pair_mask'] = pred['pair_mask'] & ~eye
    attrs = torch.zeros(batch_size, n, 14)
    attrs[:, 0, 8] = 1.0
    valid = torch.ones(batch_size, n, dtype=torch.bool)
    valid[:, -1] = False
    batch = {
        'rgb': torch.zeros(batch_size, th + tp, 3, 128, 128),
        'dyn_state': torch.zeros(batch_size, th + tp, n, 16),
        'force_matrix': torch.zeros(batch_size, th + tp, n, n, 3),
        'mask': torch.zeros(batch_size, th + tp, n, 128, 128),
        'obj_attrs': attrs,
        'valid_mask': valid,
    }
    return pred, batch, valid, attrs[..., 8] > 0.5


def test_loss_returns_all_components():
    from models.loss import LossConfig, PhysicsPredictionLoss

    pred, batch, valid, static = make_pred_batch()
    loss_fn = PhysicsPredictionLoss(LossConfig(history_length=2, predict_length=3))

    out = loss_fn(pred, batch, valid, static)

    for key in ['loss', 'loss_rgb', 'loss_state', 'loss_collision', 'loss_mask', 'loss_lpips', 'collision_pos_rate']:
        assert key in out
        assert torch.isfinite(out[key])


def test_state_loss_masks_static_and_padding_objects():
    from models.loss import LossConfig, PhysicsPredictionLoss

    pred, batch, valid, static = make_pred_batch(batch_size=1)
    pred['state'][:, :, 0] = 1000.0
    pred['state'][:, :, -1] = 1000.0
    loss_fn = PhysicsPredictionLoss(LossConfig(history_length=2, predict_length=3))

    out = loss_fn(pred, batch, valid, static)

    assert out['loss_state'].item() == 0.0


def test_collision_labels_use_denormalized_force():
    from models.loss import build_collision_labels

    force = torch.zeros(1, 1, 2, 2, 3)
    force[..., 0] = 0.1

    labels = build_collision_labels(force, force_mean=(0, 0, 0), force_std=(50, 50, 50), threshold=1.0)

    assert labels.any()


def test_collision_loss_uses_pair_mask():
    from models.loss import LossConfig, PhysicsPredictionLoss

    pred, batch, valid, static = make_pred_batch(batch_size=1)
    pred['collision_logits'][:] = 1000.0
    pred['pair_mask'][:] = False
    loss_fn = PhysicsPredictionLoss(LossConfig(history_length=2, predict_length=3))

    out = loss_fn(pred, batch, valid, static)

    assert out['loss_collision'].item() == 0.0
    assert torch.isfinite(out['loss'])


def test_mask_loss_uses_valid_object_mask():
    from models.loss import LossConfig, PhysicsPredictionLoss

    pred, batch, valid, static = make_pred_batch(batch_size=1)
    pred['mask_logits'][:, :, -1] = 1000.0
    loss_fn = PhysicsPredictionLoss(LossConfig(history_length=2, predict_length=3))

    out = loss_fn(pred, batch, valid, static)

    assert torch.isfinite(out['loss_mask'])


def test_future_step_weights_apply_time_dimension():
    from models.loss import LossConfig, PhysicsPredictionLoss

    pred, batch, valid, static = make_pred_batch(batch_size=1)
    pred['rgb'][:, 0] = 1.0
    pred['rgb'][:, 1] = 2.0
    cfg1 = LossConfig(history_length=2, predict_length=3, future_step_weights=(1.0, 1.0, 1.0))
    cfg2 = LossConfig(history_length=2, predict_length=3, future_step_weights=(10.0, 1.0, 1.0))

    loss1 = PhysicsPredictionLoss(cfg1)(pred, batch, valid, static)['loss_rgb']
    loss2 = PhysicsPredictionLoss(cfg2)(pred, batch, valid, static)['loss_rgb']

    assert loss1 != loss2


def test_zero_valid_dynamic_objects_no_nan():
    from models.loss import LossConfig, PhysicsPredictionLoss

    pred, batch, valid, static = make_pred_batch(batch_size=1)
    static[:] = True
    loss_fn = PhysicsPredictionLoss(LossConfig(history_length=2, predict_length=3))

    out = loss_fn(pred, batch, valid, static)

    assert out['loss_state'].item() == 0.0
    assert torch.isfinite(out['loss'])


def test_zero_pair_mask_no_nan():
    from models.loss import LossConfig, PhysicsPredictionLoss

    pred, batch, valid, static = make_pred_batch(batch_size=1)
    pred['pair_mask'][:] = False
    out = PhysicsPredictionLoss(LossConfig(history_length=2, predict_length=3))(pred, batch, valid, static)

    assert out['loss_collision'].item() == 0.0
    assert torch.isfinite(out['loss'])


def test_lpips_disabled_by_default():
    from models.loss import LossConfig, PhysicsPredictionLoss

    loss_fn = PhysicsPredictionLoss(LossConfig(history_length=2, predict_length=3))

    assert loss_fn.lpips is None


def test_total_loss_weighted_sum():
    from models.loss import LossConfig, PhysicsPredictionLoss

    pred, batch, valid, static = make_pred_batch()
    cfg = LossConfig(history_length=2, predict_length=3)
    out = PhysicsPredictionLoss(cfg)(pred, batch, valid, static)

    expected = (
        cfg.rgb_weight * out['loss_rgb']
        + cfg.state_weight * out['loss_state']
        + cfg.collision_weight * out['loss_collision']
        + cfg.mask_weight * out['loss_mask']
        + cfg.lpips_weight * out['loss_lpips']
    )
    assert torch.allclose(out['loss'], expected)
