"""Tests for inference and visualization utilities."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_batch(batch_size=2, total_length=4, n=2, h=16, w=16):
    return {
        "rgb": torch.zeros(batch_size, total_length, 3, h, w),
        "mask": torch.zeros(batch_size, total_length, n, h, w),
        "obj_attrs": torch.zeros(batch_size, n, 14),
        "dyn_state": torch.zeros(batch_size, total_length, n, 16),
        "force_matrix": torch.zeros(batch_size, total_length, n, n, 3),
        "valid_mask": torch.ones(batch_size, n, dtype=torch.bool),
        "scene_id": torch.arange(batch_size),
        "sample_id": torch.arange(batch_size),
    }


def test_run_inference_returns_prediction_and_metrics_on_device():
    from inference import run_inference

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch):
            b, _, _, h, w = batch["rgb"].shape
            n = batch["mask"].shape[2]
            tp = 2
            return {
                "rgb": batch["rgb"][:, -tp:].clone() + self.weight * 0,
                "state": batch["dyn_state"][:, -tp:].clone(),
                "collision_logits": torch.zeros(b, tp, n, n, device=batch["rgb"].device),
                "pair_mask": torch.ones(b, tp, n, n, dtype=torch.bool, device=batch["rgb"].device),
                "mask_logits": torch.zeros(b, tp, n, h, w, device=batch["rgb"].device),
                "mask_prob": torch.zeros(b, tp, n, h, w, device=batch["rgb"].device),
                "valid_mask": batch["valid_mask"],
                "static_flag": batch["obj_attrs"][..., 8] > 0.5,
            }

    result = run_inference(DummyModel(), make_batch(), history_length=2, predict_length=2, device="cpu")

    assert set(["pred", "metrics", "batch"]).issubset(result.keys())
    assert result["pred"]["rgb"].shape == (2, 2, 3, 16, 16)
    assert result["pred"]["state"].shape == (2, 2, 2, 16)
    assert result["metrics"]["mse"] == 0.0


def test_load_model_from_checkpoint_restores_state_dict(tmp_path):
    from config import ModelConfig
    from inference import load_model_from_checkpoint
    from models.physics_pred import PhysicsVideoPredictor

    cfg = ModelConfig.tiny_test(token_dim=16, max_objects=2, history_length=2, predict_length=2)
    model = PhysicsVideoPredictor(cfg)
    ckpt_path = tmp_path / "model.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)

    loaded = load_model_from_checkpoint(ckpt_path, device="cpu")

    assert isinstance(loaded, PhysicsVideoPredictor)
    assert loaded.config.history_length == 2
    assert loaded.config.predict_length == 2


def test_save_prediction_outputs_writes_pt_file(tmp_path):
    from inference import save_prediction_outputs

    pred = {"rgb": torch.zeros(1, 2, 3, 8, 8), "state": torch.zeros(1, 2, 1, 16)}
    metrics = {"mse": 0.0}
    out_path = save_prediction_outputs(pred, metrics, tmp_path / "pred.pt")

    loaded = torch.load(out_path, map_location="cpu")
    assert Path(out_path).exists()
    assert loaded["pred"]["rgb"].shape == (1, 2, 3, 8, 8)
    assert loaded["metrics"]["mse"] == 0.0


def test_visualization_generates_comparison_image(tmp_path):
    from visualization import save_prediction_comparison

    pred_rgb = torch.zeros(1, 2, 3, 16, 16)
    target_rgb = torch.ones(1, 2, 3, 16, 16)
    out_path = save_prediction_comparison(pred_rgb, target_rgb, tmp_path / "comparison.png")

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0


def test_visualization_generates_state_curve_image(tmp_path):
    from visualization import save_state_curves

    pred_state = torch.zeros(1, 3, 2, 16)
    target_state = torch.ones(1, 3, 2, 16)
    valid_mask = torch.tensor([[True, False]])
    out_path = save_state_curves(pred_state, target_state, valid_mask, tmp_path / "state.png")

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0


def test_visualization_generates_collision_confusion_image(tmp_path):
    from visualization import save_collision_confusion

    logits = torch.tensor([[[[0.0, 10.0], [-10.0, 0.0]]]])
    force = torch.zeros(1, 1, 2, 2, 3)
    force[0, 0, 0, 1, 0] = 1.0
    pair_mask = torch.tensor([[[[False, True], [True, False]]]])
    out_path = save_collision_confusion(logits, force, pair_mask, tmp_path / "collision.png", force_std=(1.0, 1.0, 1.0))

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 0
