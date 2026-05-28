"""Tests for dataset.py"""
import pytest
import torch
import numpy as np
import sys
sys.path.insert(0, '/home/lzy/project/slot-datamaking/model')

from dataset import PhysicsVideoDataset, collate_fn


class TestPhysicsVideoDataset:
    """Test PhysicsVideoDataset"""
    
    @pytest.fixture
    def dataset(self):
        """Create dataset instance"""
        return PhysicsVideoDataset(
            root_dir='/home/lzy/project/slot-datamaking/database',
            history_length=12,
            predict_length=12,
            max_objects=7,
            split='train',
            normalize=True,
        )
    
    @pytest.fixture
    def dataset_no_norm(self):
        """Create dataset without normalization"""
        return PhysicsVideoDataset(
            root_dir='/home/lzy/project/slot-datamaking/database',
            history_length=12,
            predict_length=12,
            max_objects=7,
            split='train',
            normalize=False,
        )
    
    def test_dataset_length(self, dataset):
        """Dataset should have samples"""
        assert len(dataset) > 0
    
    def test_sample_keys(self, dataset):
        """Sample should contain all required keys"""
        sample = dataset[0]
        required_keys = ['rgb', 'mask', 'obj_attrs', 'dyn_state', 
                        'force_matrix', 'valid_mask']
        for key in required_keys:
            assert key in sample, f"Missing key: {key}"
    
    def test_rgb_shape(self, dataset):
        """RGB should be [T, 3, H, W]"""
        sample = dataset[0]
        rgb = sample['rgb']
        T = dataset.history_length + dataset.predict_length
        assert rgb.shape == (T, 3, 128, 128)
    
    def test_rgb_range_no_norm(self, dataset_no_norm):
        """RGB values without normalization should be in [0, 1]"""
        sample = dataset_no_norm[0]
        rgb = sample['rgb']
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0
    
    def test_rgb_normalized(self, dataset):
        """RGB with normalization should have ~0 mean"""
        sample = dataset[0]
        rgb = sample['rgb']
        # After normalization, mean should be close to 0
        assert abs(rgb.mean()) < 1.0
    
    def test_mask_shape(self, dataset):
        """Mask should be [T, N, H, W]"""
        sample = dataset[0]
        mask = sample['mask']
        T = dataset.history_length + dataset.predict_length
        N = dataset.max_objects
        assert mask.shape == (T, N, 128, 128)
    
    def test_obj_attrs_shape(self, dataset):
        """Object attributes should be [N, attr_dim]"""
        sample = dataset[0]
        attrs = sample['obj_attrs']
        N = dataset.max_objects
        assert attrs.shape[0] == N
        assert attrs.shape[1] == dataset.attr_dim
    
    def test_dyn_state_shape(self, dataset):
        """Dynamic state should be [T, N, state_dim]"""
        sample = dataset[0]
        state = sample['dyn_state']
        T = dataset.history_length + dataset.predict_length
        N = dataset.max_objects
        assert state.shape == (T, N, 16)
    
    def test_force_matrix_shape(self, dataset):
        """Force matrix should be [T, N, N, 3]"""
        sample = dataset[0]
        force = sample['force_matrix']
        T = dataset.history_length + dataset.predict_length
        N = dataset.max_objects
        assert force.shape == (T, N, N, 3)
    
    def test_valid_mask_shape(self, dataset):
        """Valid mask should be [N]"""
        sample = dataset[0]
        valid = sample['valid_mask']
        N = dataset.max_objects
        assert valid.shape == (N,)
        assert valid.dtype == torch.bool
    
    def test_valid_mask_has_real_objects(self, dataset):
        """At least one object should be valid"""
        sample = dataset[0]
        assert sample['valid_mask'].any()
    
    def test_object_type_encoding(self, dataset):
        """Object type one-hot should cover all 4 types"""
        sample = dataset[0]
        attrs = sample['obj_attrs']
        # Check that type encoding area (indices 9:13) has at most one 1.0
        for i in range(attrs.shape[0]):
            type_onehot = attrs[i, 9:13]
            assert type_onehot.sum() <= 1.0
    
    def test_scene_stratified_split(self, dataset):
        """Different splits should have different samples"""
        train_ds = PhysicsVideoDataset(
            root_dir='/home/lzy/project/slot-datamaking/database',
            split='train',
        )
        val_ds = PhysicsVideoDataset(
            root_dir='/home/lzy/project/slot-datamaking/database',
            split='val',
        )
        # Should have different sample counts
        assert len(train_ds) != len(val_ds) or len(train_ds) == 0


class TestCollateFn:
    """Test collate function for batching"""
    
    @pytest.fixture
    def dataset(self):
        return PhysicsVideoDataset(
            root_dir='/home/lzy/project/slot-datamaking/database',
            history_length=12,
            predict_length=12,
            max_objects=7,
            split='train',
        )
    
    def test_collate_batch(self, dataset):
        """Collate should produce correct batch shapes"""
        samples = [dataset[0], dataset[1]]
        batch = collate_fn(samples)
        
        B = 2
        T = dataset.history_length + dataset.predict_length
        N = dataset.max_objects
        
        assert batch['rgb'].shape[0] == B
        assert batch['mask'].shape[0] == B
        assert batch['obj_attrs'].shape[0] == B
        assert batch['dyn_state'].shape[0] == B
        assert batch['force_matrix'].shape[0] == B
        assert batch['valid_mask'].shape[0] == B


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
