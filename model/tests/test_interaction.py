"""Tests for interaction modules."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_inputs(batch_size=2, th=3, n=4, d=32):
    object_token = torch.randn(batch_size, th, n, d)
    state = torch.zeros(batch_size, th, n, 16)
    state[..., 0:3] = torch.arange(n).float()[None, None, :, None]
    state[..., 7:10] = torch.arange(n).float()[None, None, :, None] * 0.1
    force = torch.randn(batch_size, th, n, n, 3)
    valid = torch.ones(batch_size, n, dtype=torch.bool)
    valid[:, -1] = False
    static = torch.zeros(batch_size, n, dtype=torch.bool)
    static[:, 0] = True
    return object_token, state, force, valid, static


def test_edge_feature_builder_shapes():
    from models.interaction import EdgeFeatureBuilder

    _, state, force, valid, static = make_inputs()
    builder = EdgeFeatureBuilder(edge_dim=16)

    edge_feat, edge_mask = builder(state, force, valid, static)

    assert edge_feat.shape == (2, 3, 4, 4, 16)
    assert edge_mask.shape == (2, 3, 4, 4)


def test_edge_mask_excludes_padding_and_self_edges():
    from models.interaction import EdgeFeatureBuilder

    _, state, force, valid, static = make_inputs()
    edge_feat, edge_mask = EdgeFeatureBuilder()(state, force, valid, static)

    assert not edge_mask[:, :, -1, :].any()
    assert not edge_mask[:, :, :, -1].any()
    diagonal = torch.diagonal(edge_mask, dim1=2, dim2=3)
    assert not diagonal.any()
    assert torch.all(edge_feat[~edge_mask] == 0)


def test_edge_features_include_bidirectional_force():
    from models.interaction import EdgeFeatureBuilder

    _, state, force, valid, static = make_inputs(batch_size=1, th=1, n=3)
    force.zero_()
    force[0, 0, 0, 1] = torch.tensor([1.0, 2.0, 3.0])
    force[0, 0, 1, 0] = torch.tensor([4.0, 5.0, 6.0])

    edge_feat, _ = EdgeFeatureBuilder()(state, force, valid, static)

    assert torch.equal(edge_feat[0, 0, 0, 1, 0:3], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(edge_feat[0, 0, 0, 1, 3:6], torch.tensor([4.0, 5.0, 6.0]))


def test_relative_position_and_velocity_computation():
    from models.interaction import EdgeFeatureBuilder

    _, state, force, valid, static = make_inputs(batch_size=1, th=1, n=3)
    edge_feat, _ = EdgeFeatureBuilder()(state, force, valid, static)

    assert torch.equal(edge_feat[0, 0, 0, 1, 6:9], torch.tensor([1.0, 1.0, 1.0]))
    assert torch.allclose(edge_feat[0, 0, 0, 1, 9:12], torch.tensor([0.1, 0.1, 0.1]))


def test_message_passing_output_shape_and_padding_zero():
    from models.interaction import InteractionConfig, InteractionModule

    object_token, state, force, valid, static = make_inputs(d=32)
    module = InteractionModule(InteractionConfig(token_dim=32, hidden_dim=32, num_layers=1))

    out = module(object_token, state, force, valid, static)

    assert out['inter_token'].shape == object_token.shape
    assert torch.all(out['inter_token'][:, :, -1] == 0)


def test_static_objects_can_send_messages():
    from models.interaction import EdgeFeatureBuilder

    _, state, force, valid, static = make_inputs(batch_size=1, th=1, n=3)
    _, edge_mask = EdgeFeatureBuilder()(state, force, valid, static)

    assert edge_mask[0, 0, 0, 1]
    assert edge_mask[0, 0, 1, 0]


def test_gradients_flow_through_interaction_module():
    from models.interaction import InteractionConfig, InteractionModule

    object_token, state, force, valid, static = make_inputs(batch_size=1, d=16)
    object_token.requires_grad_()
    module = InteractionModule(InteractionConfig(token_dim=16, hidden_dim=16, num_layers=1))

    loss = module(object_token, state, force, valid, static)['inter_token'].sum()
    loss.backward()

    grads = [p.grad for p in module.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_zero_force_edges_are_kept():
    from models.interaction import EdgeFeatureBuilder

    _, state, force, valid, static = make_inputs(batch_size=1, th=1, n=3)
    force.zero_()
    _, edge_mask = EdgeFeatureBuilder()(state, force, valid, static)

    assert edge_mask[0, 0, 0, 1]


def test_edge_dim_matches_config():
    from models.interaction import EdgeFeatureBuilder, InteractionConfig

    _, state, force, valid, static = make_inputs()
    cfg = InteractionConfig(edge_dim=16)
    edge_feat, _ = EdgeFeatureBuilder(edge_dim=cfg.edge_dim)(state, force, valid, static)

    assert cfg.edge_dim == edge_feat.shape[-1] == 16


def test_all_invalid_mask_outputs_zero_without_nan():
    from models.interaction import InteractionConfig, InteractionModule

    object_token, state, force, valid, static = make_inputs(d=16)
    valid[:] = False
    module = InteractionModule(InteractionConfig(token_dim=16, hidden_dim=16, num_layers=1))

    out = module(object_token, state, force, valid, static)

    assert torch.isfinite(out['inter_token']).all()
    assert torch.all(out['inter_token'] == 0)


def test_valid_mask_is_returned_for_temporal_module():
    from models.interaction import InteractionConfig, InteractionModule

    object_token, state, force, valid, static = make_inputs(d=16)
    module = InteractionModule(InteractionConfig(token_dim=16, hidden_dim=16, num_layers=1))

    out = module(object_token, state, force, valid, static)

    assert torch.equal(out['valid_mask'], valid)
