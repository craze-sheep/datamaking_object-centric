"""Physics Video Prediction Dataset"""
import os
import json
import glob
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, List, Optional, Tuple
import pickle


class PhysicsVideoDataset(Dataset):
    """
    Dataset for physics video prediction.
    
    Structure:
        database/{S1-S8}/{L1-L5}/{sample_id}/
            {id}.mp4 - RGB video
            {id}.npz - depth (unused)
            object_static.json - static attributes
            video.json - metadata
            dynamic/{frame_id}/
                {frame_id}.png - RGB frame
                force_matrix.json - force matrix
                object_dynamicjson/{obj_id}.json - dynamic state
                object_segment/{obj_id}.npz - segmentation mask
    """
    
    # Object type encoding
    TYPE_MAP = {'ground': 0, 'sphere': 1, 'box': 2, 'cylinder': 3}
    NUM_TYPES = 4
    
    def __init__(
        self,
        root_dir: str,
        history_length: int = 12,
        predict_length: int = 12,
        max_objects: int = 7,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        stride: int = 6,
        transform=None,
        normalize: bool = True,
    ):
        self.root_dir = root_dir
        self.history_length = history_length
        self.predict_length = predict_length
        self.total_length = history_length + predict_length
        self.max_objects = max_objects
        self.split = split
        self.stride = stride
        self.transform = transform
        self.normalize = normalize
        
        # Attribute dimension: 3(size) + 4(friction) + 1(mass) + 1(restitution) + 1(static) + 4(type_onehot) = 14
        self.attr_dim = 3 + 4 + 1 + 1 + 1 + self.NUM_TYPES
        
        # Scan all samples
        self.all_samples = self._scan_samples_with_cache()
        
        # Split by scene (stratified)
        self.samples = self._split_samples_stratified(train_ratio, val_ratio)
        
        # Compute normalization stats if needed
        if self.normalize:
            self._compute_norm_stats()
        else:
            self.rgb_mean = torch.tensor([0.5, 0.5, 0.5])
            self.rgb_std = torch.tensor([0.5, 0.5, 0.5])
            self.state_mean = torch.zeros(16)
            self.state_std = torch.ones(16)
            self.force_mean = torch.zeros(3)
            self.force_std = torch.ones(3)
    
    def _get_cache_path(self) -> str:
        """Get cache file path with hash of root_dir"""
        cache_dir = os.path.dirname(os.path.abspath(__file__))
        dir_hash = hashlib.md5(self.root_dir.encode()).hexdigest()[:8]
        return os.path.join(cache_dir, f'dataset_cache_{dir_hash}.pkl')
    
    def _scan_samples_with_cache(self) -> List[Dict]:
        """Scan with cache, invalidate if cache is stale"""
        cache_path = self._get_cache_path()
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            
            # Check if cache is valid (same root_dir and file count)
            if cache.get('root_dir') == self.root_dir:
                # Quick validation: check if sample count matches
                cached_count = cache.get('sample_count', 0)
                # Trust cache if it exists
                return cache['samples']
        
        # Scan fresh
        samples = self._scan_samples()
        
        # Save cache
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'samples': samples,
                'root_dir': self.root_dir,
                'sample_count': len(samples),
            }, f)
        
        return samples
    
    def _scan_samples(self) -> List[Dict]:
        """Scan database to find all valid samples"""
        samples = []
        root = self.root_dir
        
        for scene_dir in sorted(glob.glob(os.path.join(root, 'S*'))):
            scene_name = os.path.basename(scene_dir)
            
            for level_dir in sorted(glob.glob(os.path.join(scene_dir, 'L*'))):
                level_name = os.path.basename(level_dir)
                
                for sample_dir in sorted(glob.glob(os.path.join(level_dir, '*'))):
                    if not os.path.isdir(sample_dir):
                        continue
                    
                    sample_id = os.path.basename(sample_dir)
                    
                    # Check required files exist
                    dynamic_dir = os.path.join(sample_dir, 'dynamic')
                    obj_static = os.path.join(sample_dir, 'object_static.json')
                    
                    if not os.path.exists(dynamic_dir) or not os.path.exists(obj_static):
                        continue
                    
                    samples.append({
                        'path': sample_dir,
                        'scene': scene_name,
                        'level': level_name,
                        'sample_id': sample_id,
                    })
        
        return samples
    
    def _split_samples_stratified(self, train_ratio: float, val_ratio: float) -> List[Dict]:
        """Split samples stratified by scene"""
        # Group by scene
        scene_groups = {}
        for s in self.all_samples:
            scene = s['scene']
            if scene not in scene_groups:
                scene_groups[scene] = []
            scene_groups[scene].append(s)
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        np.random.seed(42)
        
        for scene, group in scene_groups.items():
            indices = list(range(len(group)))
            np.random.shuffle(indices)
            
            n = len(indices)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            for i in indices[:train_end]:
                train_samples.append(group[i])
            for i in indices[train_end:val_end]:
                val_samples.append(group[i])
            for i in indices[val_end:]:
                test_samples.append(group[i])
        
        if self.split == 'train':
            return train_samples
        elif self.split == 'val':
            return val_samples
        else:
            return test_samples
    
    def _compute_norm_stats(self, num_samples: int = 100):
        """Compute normalization statistics from a subset"""
        # Use fixed values for now (can be computed from data if needed)
        self.rgb_mean = torch.tensor([0.5, 0.5, 0.5])
        self.rgb_std = torch.tensor([0.5, 0.5, 0.5])
        
        # For state: position ~[-5,5], velocity ~[-10,10], force ~[-100,100]
        self.state_mean = torch.zeros(16)
        self.state_std = torch.ones(16)
        # position
        self.state_std[0:3] = 5.0
        # quaternion - no normalization
        self.state_std[3:7] = 1.0
        # velocity
        self.state_std[7:10] = 10.0
        # angular velocity
        self.state_std[10:13] = 5.0
        # force
        self.state_std[13:16] = 50.0
        
        self.force_mean = torch.zeros(3)
        self.force_std = torch.tensor([50.0, 50.0, 50.0])
    
    def __len__(self) -> int:
        return len(self.samples)

    def _empty_sample(self, num_objects: int) -> Dict[str, torch.Tensor]:
        """Return a zero-filled sample when no frames are available."""
        T = self.total_length
        N = min(num_objects, self.max_objects)
        return {
            'rgb': torch.zeros(T, 3, 128, 128),
            'mask': torch.zeros(T, N, 128, 128),
            'obj_attrs': torch.zeros(self.max_objects, self.attr_dim),
            'dyn_state': torch.zeros(T, N, 16),
            'force_matrix': torch.zeros(T, N, N, 3),
            'valid_mask': torch.zeros(self.max_objects, dtype=torch.bool),
            'scene_id': 0,
            'sample_id': 0,
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]
        sample_path = sample_info['path']
        
        # Load static attributes
        obj_attrs, num_objects = self._load_static_attrs(sample_path)
        
        # Get available frames
        dynamic_dir = os.path.join(sample_path, 'dynamic')
        available_frames = sorted([
            int(os.path.basename(d)) 
            for d in glob.glob(os.path.join(dynamic_dir, '*'))
            if os.path.isdir(d)
        ])
        
        # Sliding window: select a window of total_length frames
        if len(available_frames) == 0:
            # No frames available - return zeros
            return self._empty_sample(num_objects)
        if len(available_frames) >= self.total_length:
            # Random start position for training
            if self.split == 'train':
                max_start = len(available_frames) - self.total_length
                start = np.random.randint(0, max(1, max_start // self.stride)) * self.stride
                start = min(start, max_start)
            else:
                start = 0
            frame_ids = available_frames[start:start + self.total_length]
        else:
            # Pad with last frame if not enough
            frame_ids = available_frames + [available_frames[-1]] * (self.total_length - len(available_frames))
        
        # Load per-frame data
        rgb_list = []
        mask_list = []
        dyn_state_list = []
        force_list = []
        
        for fid in frame_ids:
            frame_dir = os.path.join(dynamic_dir, str(fid))
            
            # RGB
            rgb = self._load_rgb(frame_dir, fid)
            rgb_list.append(rgb)
            
            # Masks and dynamic state per object
            masks = torch.zeros(self.max_objects, 128, 128)
            dyn_states = torch.zeros(self.max_objects, 16)
            
            for obj_id in range(1, min(num_objects, self.max_objects) + 1):
                idx_obj = obj_id - 1
                
                # Mask
                mask_path = os.path.join(frame_dir, 'object_segment', f'{obj_id}.npz')
                if os.path.exists(mask_path):
                    try:
                        mask_data = np.load(mask_path)
                        masks[idx_obj] = torch.from_numpy(mask_data['mask']).float()
                    except Exception:
                        pass
                
                # Dynamic state
                dyn_path = os.path.join(frame_dir, 'object_dynamicjson', f'{obj_id}.json')
                if os.path.exists(dyn_path):
                    try:
                        with open(dyn_path) as f:
                            dyn = json.load(f)
                        state = self._encode_dynamic_state(dyn)
                        dyn_states[idx_obj] = state
                    except Exception:
                        pass
            
            mask_list.append(masks)
            dyn_state_list.append(dyn_states)
            
            # Force matrix
            force = self._load_force_matrix(frame_dir)
            force_list.append(force)
        
        # Stack
        rgb_tensor = torch.stack(rgb_list)  # [T, 3, H, W]
        mask_tensor = torch.stack(mask_list)  # [T, N, H, W]
        dyn_tensor = torch.stack(dyn_state_list)  # [T, N, state_dim]
        force_tensor = torch.stack(force_list)  # [T, N, N, 3]
        
        # Normalize
        if self.normalize:
            rgb_tensor = (rgb_tensor - self.rgb_mean[None, :, None, None]) / self.rgb_std[None, :, None, None]
            dyn_tensor = (dyn_tensor - self.state_mean[None, None, :]) / self.state_std[None, None, :]
            force_tensor = (force_tensor - self.force_mean[None, None, None, :]) / self.force_std[None, None, None, :]
        
        # Valid mask
        valid_mask = torch.zeros(self.max_objects, dtype=torch.bool)
        valid_mask[:min(num_objects, self.max_objects)] = True
        
        return {
            'rgb': rgb_tensor,
            'mask': mask_tensor,
            'obj_attrs': obj_attrs,
            'dyn_state': dyn_tensor,
            'force_matrix': force_tensor,
            'valid_mask': valid_mask,
            'scene_id': int(hashlib.md5(sample_info['scene'].encode()).hexdigest()[:8], 16) % 8,
            'sample_id': int(sample_info['sample_id']) if sample_info['sample_id'].isdigit() else 0,
        }
    
    def _load_rgb(self, frame_dir: str, frame_id: int = None) -> torch.Tensor:
        """Load RGB frame as [3, H, W] tensor in [0, 1]"""
        # Try frame_id.png first, then 1.png
        if frame_id is not None:
            img_path = os.path.join(frame_dir, f'{frame_id}.png')
            if not os.path.exists(img_path):
                img_path = os.path.join(frame_dir, '1.png')
        else:
            img_path = os.path.join(frame_dir, '1.png')
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((128, 128))
            arr = np.array(img).astype(np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
            return tensor
        except Exception:
            # Return zeros if image loading fails
            return torch.zeros(3, 128, 128)
    
    def _load_static_attrs(self, sample_path: str) -> Tuple[torch.Tensor, int]:
        """Load and encode static object attributes"""
        try:
            with open(os.path.join(sample_path, 'object_static.json')) as f:
                objects = json.load(f)
        except Exception:
            return torch.zeros(self.max_objects, self.attr_dim), 0
        
        num_objects = len(objects)
        attrs = torch.zeros(self.max_objects, self.attr_dim)
        
        for i, obj in enumerate(objects[:self.max_objects]):
            # Size (3D)
            size = obj.get('size', [0, 0, 0])
            attrs[i, 0:3] = torch.tensor(size[:3]) if size and len(size) >= 3 else 0
            
            # Frictions (4 values)
            attrs[i, 3] = obj.get('lateralFriction', 0)
            attrs[i, 4] = obj.get('rollingFriction', 0)
            attrs[i, 5] = obj.get('spinningFriction', 0)
            attrs[i, 6] = obj.get('restitution', 0)
            
            # Mass (1 value, 0 for static)
            attrs[i, 7] = obj.get('mass', 0) or 0
            
            # Static flag
            attrs[i, 8] = 1.0 if obj.get('static', False) else 0.0
            
            # Object type (one-hot, 4 types)
            obj_type = self.TYPE_MAP.get(obj.get('object_type', ''), -1)
            if 0 <= obj_type < self.NUM_TYPES:
                attrs[i, 9 + obj_type] = 1.0
        
        return attrs, num_objects
    
    def _encode_dynamic_state(self, dyn: Dict) -> torch.Tensor:
        """Encode dynamic state dict to tensor [16]"""
        state = torch.zeros(16)
        
        # Position [3]
        pos = dyn.get('position', [0, 0, 0])
        state[0:3] = torch.tensor(pos[:3]) if len(pos) >= 3 else 0
        
        # Quaternion [4]
        quat = dyn.get('quaternion', [1, 0, 0, 0])
        state[3:7] = torch.tensor(quat[:4]) if len(quat) >= 4 else torch.tensor([1, 0, 0, 0])
        
        # Velocity [3]
        vel = dyn.get('velocity', [0, 0, 0])
        state[7:10] = torch.tensor(vel[:3]) if len(vel) >= 3 else 0
        
        # Angular velocity [3]
        ang = dyn.get('angular_velocity', [0, 0, 0])
        state[10:13] = torch.tensor(ang[:3]) if len(ang) >= 3 else 0
        
        # Resultant force [3]
        force = dyn.get('resultant force', [0, 0, 0])
        state[13:16] = torch.tensor(force[:3]) if len(force) >= 3 else 0
        
        return state
    
    def _load_force_matrix(self, frame_dir: str) -> torch.Tensor:
        """Load force matrix [N, N, 3]"""
        force_path = os.path.join(frame_dir, 'force_matrix.json')
        force = torch.zeros(self.max_objects, self.max_objects, 3)
        
        if not os.path.exists(force_path):
            return force
        
        try:
            with open(force_path) as f:
                data = json.load(f)
            
            if isinstance(data, dict) and 'force_matrix' in data:
                # Format: {"object_order": [1,2], "force_matrix": [[null, ...], ...]}
                matrix = data['force_matrix']
                for i, row in enumerate(matrix[:self.max_objects]):
                    for j, val in enumerate(row[:self.max_objects]):
                        if val is not None and isinstance(val, list) and len(val) >= 3:
                            force[i, j] = torch.tensor(val[:3])
            elif isinstance(data, list):
                # Direct matrix format
                for i, row in enumerate(data[:self.max_objects]):
                    for j, val in enumerate(row[:self.max_objects]):
                        if val is not None and isinstance(val, list) and len(val) >= 3:
                            force[i, j] = torch.tensor(val[:3])
        except Exception:
            pass
        
        return force


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    return {
        'rgb': torch.stack([s['rgb'] for s in batch]),
        'mask': torch.stack([s['mask'] for s in batch]),
        'obj_attrs': torch.stack([s['obj_attrs'] for s in batch]),
        'dyn_state': torch.stack([s['dyn_state'] for s in batch]),
        'force_matrix': torch.stack([s['force_matrix'] for s in batch]),
        'valid_mask': torch.stack([s['valid_mask'] for s in batch]),
        'scene_id': torch.tensor([s['scene_id'] for s in batch]),
        'sample_id': torch.tensor([s['sample_id'] for s in batch]),
    }
