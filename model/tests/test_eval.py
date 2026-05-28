"""Tests for evaluation metrics and evaluation loop."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_video_metrics_for_identical_frames_are_perfect():
    from eval import compute_video_metrics

    pred = torch.zeros(2, 3, 3, 16, 16)
    target = pred.clone()

    metrics = compute_video_metrics(pred, target, value_range=2.0)

    assert metrics["mse"] == 0.0
    assert metrics["psnr"] == float("inf")
    assert abs(metrics["ssim"] - 1.0) < 1e-6


def test_video_metrics_known_mse_and_psnr():
    from eval import compute_video_metrics

    pred = torch.ones(1, 1, 3, 4, 4)
    target = torch.zeros_like(pred)

    metrics = compute_video_metrics(pred, target, value_range=2.0)

    assert abs(metrics["mse"] - 1.0) < 1e-6
    assert abs(metrics["psnr"] - 6.0205999) < 1e-4
    assert 0.0 <= metrics["ssim"] <= 1.0


def test_state_metrics_use_valid_dynamic_mask():
    from eval import compute_state_metrics

    pred = torch.zeros(1, 2, 3, 16)
    target = torch.zeros_like(pred)
    pred[:, :, 0, 0:3] = 2.0       # dynamic valid contributes
    pred[:, :, 1, 0:3] = 100.0     # static ignored
    pred[:, :, 2, 0:3] = 100.0     # padding ignored
    valid = torch.tensor([[True, True, False]])
    static = torch.tensor([[False, True, False]])

    metrics = compute_state_metrics(pred, target, valid, static)

    assert abs(metrics["position_mse"] - 4.0) < 1e-6
    assert metrics["velocity_mse"] == 0.0
    assert metrics["state_mse"] > 0.0


def test_state_metrics_no_dynamic_objects_are_zero_not_nan():
    from eval import compute_state_metrics

    pred = torch.randn(1, 2, 2, 16)
    target = torch.zeros_like(pred)
    valid = torch.tensor([[True, False]])
    static = torch.tensor([[True, False]])

    metrics = compute_state_metrics(pred, target, valid, static)

    assert metrics["state_mse"] == 0.0
    assert metrics["position_mse"] == 0.0
    assert metrics["velocity_mse"] == 0.0


def test_collision_metrics_use_denormalized_force_and_pair_mask():
    from eval import compute_collision_metrics

    logits = torch.tensor([[[[0.0, 10.0], [-10.0, 10.0]]]])
    force = torch.zeros(1, 1, 2, 2, 3)
    force[0, 0, 0, 1, 0] = 0.1  # denorm 5.0 with std=50 -> positive
    pair_mask = torch.tensor([[[[False, True], [True, False]]]])

    metrics = compute_collision_metrics(logits, force, pair_mask, force_std=(50.0, 50.0, 50.0), threshold=1e-6)

    assert metrics["collision_tp"] == 1
    assert metrics["collision_fp"] == 0
    assert metrics["collision_fn"] == 0
    assert abs(metrics["collision_precision"] - 1.0) < 1e-6
    assert abs(metrics["collision_recall"] - 1.0) < 1e-6
    assert abs(metrics["collision_f1"] - 1.0) < 1e-6


def test_collision_metrics_empty_pair_mask_are_zero_not_nan():
    from eval import compute_collision_metrics

    logits = torch.randn(1, 2, 3, 3)
    force = torch.randn(1, 2, 3, 3, 3)
    pair_mask = torch.zeros(1, 2, 3, 3, dtype=torch.bool)

    metrics = compute_collision_metrics(logits, force, pair_mask)

    assert metrics["collision_precision"] == 0.0
    assert metrics["collision_recall"] == 0.0
    assert metrics["collision_f1"] == 0.0
    assert metrics["collision_tp"] == 0


def test_group_metrics_by_scene_prefixes_keys():
    from eval import group_metrics_by_scene

    per_sample = [
        {"scene_id": 0, "mse": 1.0, "collision_f1": 0.5},
        {"scene_id": 0, "mse": 3.0, "collision_f1": 1.0},
        {"scene_id": 1, "mse": 10.0, "collision_f1": 0.0},
    ]

    grouped = group_metrics_by_scene(per_sample)

    assert grouped["scene_0/mse"] == 2.0
    assert grouped["scene_0/collision_f1"] == 0.75
    assert grouped["scene_1/mse"] == 10.0


def test_evaluate_model_runs_with_mock_loader():
    from eval import evaluate_model

    class DummyModel(torch.nn.Module):
        def forward(self, batch):
            b, total, _, h, w = batch["rgb"].shape
            n = batch["mask"].shape[2]
            tp = 2
            return {
                "rgb": batch["rgb"][:, -tp:].clone(),
                "state": batch["dyn_state"][:, -tp:].clone(),
                "collision_logits": torch.ones(b, tp, n, n, device=batch["rgb"].device),
                "pair_mask": torch.ones(b, tp, n, n, dtype=torch.bool, device=batch["rgb"].device),
                "mask_logits": torch.zeros(b, tp, n, h, w, device=batch["rgb"].device),
                "mask_prob": torch.zeros(b, tp, n, h, w, device=batch["rgb"].device),
                "valid_mask": batch["valid_mask"],
                "static_flag": batch["obj_attrs"][..., 8] > 0.5,
            }

    batch = {
        "rgb": torch.zeros(2, 4, 3, 16, 16),
        "mask": torch.zeros(2, 4, 2, 16, 16),
        "obj_attrs": torch.zeros(2, 2, 14),
        "dyn_state": torch.zeros(2, 4, 2, 16),
        "force_matrix": torch.zeros(2, 4, 2, 2, 3),
        "valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "scene_id": torch.tensor([0, 1]),
    }

    metrics = evaluate_model(DummyModel(), [batch], history_length=2, predict_length=2)

    assert metrics["mse"] == 0.0
    assert metrics["state_mse"] == 0.0
    assert "scene_0/mse" in metrics
    assert "scene_1/mse" in metrics


def test_evaluate_model_scene_metrics_are_per_sample_not_batch_copied():
    from eval import evaluate_model

    class SceneModel(torch.nn.Module):
        def forward(self, batch):
            b, total, _, h, w = batch["rgb"].shape
            n = batch["mask"].shape[2]
            tp = 2
            rgb = batch["rgb"][:, -tp:].clone()
            rgb[1] = 1.0
            return {
                "rgb": rgb,
                "state": batch["dyn_state"][:, -tp:].clone(),
                "collision_logits": torch.zeros(b, tp, n, n),
                "pair_mask": torch.zeros(b, tp, n, n, dtype=torch.bool),
                "mask_logits": torch.zeros(b, tp, n, h, w),
                "mask_prob": torch.zeros(b, tp, n, h, w),
                "valid_mask": batch["valid_mask"],
                "static_flag": batch["obj_attrs"][..., 8] > 0.5,
            }

    batch = {
        "rgb": torch.zeros(2, 4, 3, 8, 8),
        "mask": torch.zeros(2, 4, 1, 8, 8),
        "obj_attrs": torch.zeros(2, 1, 14),
        "dyn_state": torch.zeros(2, 4, 1, 16),
        "force_matrix": torch.zeros(2, 4, 1, 1, 3),
        "valid_mask": torch.ones(2, 1, dtype=torch.bool),
        "scene_id": torch.tensor([0, 1]),
    }

    metrics = evaluate_model(SceneModel(), [batch], history_length=2, predict_length=2)

    assert metrics["scene_0/mse"] == 0.0
    assert metrics["scene_1/mse"] == 1.0


def test_evaluate_model_filters_static_static_collision_pairs():
    from eval import evaluate_model

    class CollisionModel(torch.nn.Module):
        def forward(self, batch):
            b, _, _, h, w = batch["rgb"].shape
            n = batch["mask"].shape[2]
            tp = 1
            logits = torch.zeros(b, tp, n, n)
            logits[:, :, 0, 1] = 10.0  # static-static false positive if not filtered
            return {
                "rgb": batch["rgb"][:, -tp:].clone(),
                "state": batch["dyn_state"][:, -tp:].clone(),
                "collision_logits": logits,
                "pair_mask": torch.ones(b, tp, n, n, dtype=torch.bool),
                "mask_logits": torch.zeros(b, tp, n, h, w),
                "mask_prob": torch.zeros(b, tp, n, h, w),
                "valid_mask": batch["valid_mask"],
                "static_flag": batch["obj_attrs"][..., 8] > 0.5,
            }

    batch = {
        "rgb": torch.zeros(1, 3, 3, 8, 8),
        "mask": torch.zeros(1, 3, 2, 8, 8),
        "obj_attrs": torch.zeros(1, 2, 14),
        "dyn_state": torch.zeros(1, 3, 2, 16),
        "force_matrix": torch.zeros(1, 3, 2, 2, 3),
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "scene_id": torch.tensor([0]),
    }
    batch["obj_attrs"][..., 8] = 1.0

    metrics = evaluate_model(CollisionModel(), [batch], history_length=2, predict_length=1)

    assert metrics["collision_fp"] == 0
    assert metrics["collision_f1"] == 0.0
