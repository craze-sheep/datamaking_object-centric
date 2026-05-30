"""Dataset adapter for AI model.

Reuses the existing PhysicsVideoDataset from the baseline code.
"""
import sys
import os
import importlib.util

# Load the parent model/dataset.py directly by path to avoid name collision
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dataset_path = os.path.join(_parent_dir, 'dataset.py')

_spec = importlib.util.spec_from_file_location('_baseline_dataset', _dataset_path)
_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_baseline)

PhysicsVideoDataset = _baseline.PhysicsVideoDataset
collate_fn = _baseline.collate_fn

from torch.utils.data import DataLoader


def create_dataloader(
    root_dir: str,
    split: str = 'train',
    batch_size: int = 4,
    history_length: int = 12,
    predict_length: int = 12,
    max_objects: int = 7,
    num_workers: int = 0,
    stride: int = 6,
    normalize: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for the given split."""
    dataset = PhysicsVideoDataset(
        root_dir=root_dir,
        history_length=history_length,
        predict_length=predict_length,
        max_objects=max_objects,
        split=split,
        stride=stride,
        normalize=normalize,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=(split == 'train'),
    )


def get_norm_stats(dataloader):
    """Extract normalization stats from the dataloader's dataset."""
    ds = dataloader.dataset
    return {
        'rgb_mean': ds.rgb_mean,
        'rgb_std': ds.rgb_std,
        'state_mean': ds.state_mean,
        'state_std': ds.state_std,
        'force_mean': ds.force_mean,
        'force_std': ds.force_std,
    }
