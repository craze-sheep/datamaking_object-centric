"""Tests for model configuration."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_model_config_defaults():
    from config import ModelConfig, ATTR_STATIC_INDEX, STATE_POS_SLICE, STATE_VEL_SLICE

    cfg = ModelConfig()

    assert cfg.history_length == 12
    assert cfg.predict_length == 12
    assert cfg.max_objects == 7
    assert cfg.token_dim == 256
    assert cfg.edge_dim == 16
    min_cfg = ModelConfig.minimal_12gb()
    assert min_cfg.history_length == 12
    assert min_cfg.predict_length == 12
    assert min_cfg.max_objects == 7
    assert min_cfg.token_dim == 128
    tiny_cfg = ModelConfig.tiny_test(token_dim=16, max_objects=3, history_length=2, predict_length=3)
    assert tiny_cfg.history_length == 2
    assert tiny_cfg.predict_length == 3
    assert tiny_cfg.max_objects == 3
    assert tiny_cfg.token_dim == 16
    assert ATTR_STATIC_INDEX == 8
    assert STATE_POS_SLICE == slice(0, 3)
    assert STATE_VEL_SLICE == slice(7, 10)
