"""
SignAvatars Dataset Adapter for BSL Translation Project

Supports all 5 tasks:
1. Sign Language Motion Generation
2. 3D Human Pose Estimation & Shape Recovery
3. 3D Hand Pose Estimation & Shape Recovery
4. 3D Holistic Pose Estimation & Shape Recovery
5. Sign Language Recognition

Author: Oke Iyanuoluwa Enoch
"""

import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SMPLXParams:
    """SMPL-X parameter container"""
    global_orient: np.ndarray  # (T, 3)
    body_pose: np.ndarray      # (T, 63) - 21 joints x 3
    left_hand_pose: np.ndarray # (T, 45) - 15 joints x 3
    right_hand_pose: np.ndarray # (T, 45) - 15 joints x 3
    jaw_pose: np.ndarray       # (T, 3)
    expression: np.ndarray     # (T, 10)
    betas: np.ndarray          # (T, 10) - shape params
    transl: np.ndarray         # (T, 3) - translation
    
    @property
    def num_frames(self) -> int:
        return self.global_orient.shape[0]
    
    def to_tensor(self, device: str = 'cpu') -> 'SMPLXParams':
        """Convert all arrays to torch tensors"""
        return SMPLXParams(
            global_orient=torch.from_numpy(self.global_orient).float().to(device),
            body_pose=torch.from_numpy(self.body_pose).float().to(device),
            left_hand_pose=torch.from_numpy(self.left_hand_pose).float().to(device),
            right_hand_pose=torch.from_numpy(self.right_hand_pose).float().to(device),
            jaw_pose=torch.from_numpy(self.jaw_pose).float().to(device),
            expression=torch.from_numpy(self.expression).float().to(device),
            betas=torch.from_numpy(self.betas).float().to(device),
            transl=torch.from_numpy(self.transl).float().to(device),
        )


@dataclass 
class SignSample:
    """Container for a single sign language sample"""
    sample_id: str
    dataset: str  # 'wlasl', 'phoenix', 'how2sign', 'hamnosys'
    num_frames: int
    smplx_params: SMPLXParams
    keypoints_2d: np.ndarray        # (T, 106, 3)
    pred_keypoints_2d: np.ndarray   # (T, 106, 2)
    left_hand_valid: np.ndarray     # (T,)
    right_hand_valid: np.ndarray    # (T,)
    valid_indices: np.ndarray       # (T,)
    camera_focal: np.ndarray        # (T, 2)
    camera_princpt: np.ndarray      # (T, 2)
    
    # Optional metadata
    gloss: Optional[str] = None
    text: Optional[str] = None


class SignAvatarsAdapter:
    """
    Adapter for SignAvatars dataset supporting multiple sign language datasets
    and all 5 benchmark tasks.
    
    Datasets:
    - WLASL: American Sign Language (1000 samples)
    - PHOENIX: German Sign Language (596 samples)
    - How2Sign: American Sign Language (61593 samples)
    - HamNoSys: Universal notation (11504 samples)
    """
    
    # SMPL-X parameter indices (total 182)
    SMPLX_INDICES = {
        'global_orient': (0, 3),
        'body_pose': (3, 66),       # 21 joints x 3 = 63
        'left_hand_pose': (66, 111), # 15 joints x 3 = 45
        'right_hand_pose': (111, 156), # 15 joints x 3 = 45
        'jaw_pose': (156, 159),
        'expression': (159, 169),
        'betas': (169, 179),
        'transl': (179, 182),
    }
    
    # 2D keypoint indices (106 total)
    KEYPOINT_INDICES = {
        'body': (0, 25),           # 25 body keypoints
        'left_hand': (25, 46),     # 21 hand keypoints  
        'right_hand': (46, 67),    # 21 hand keypoints
        'face': (67, 106),         # 39 face keypoints
    }
    
    def __init__(self, data_root: str = "data/signavatars"):
        """
        Initialize the adapter.
        
        Args:
            data_root: Root directory containing signavatars data
        """
        self.data_root = Path(data_root)
        self.datasets = {}
        self._discover_datasets()
        
    def _discover_datasets(self):
        """Discover available datasets"""
        dataset_paths = {
            'wlasl': self.data_root / 'wlasl',
            'phoenix': self.data_root / 'phoenix', 
            'how2sign': self.data_root / 'how2sign',
            'hamnosys': self.data_root / 'hamnosys',
        }
        
        for name, path in dataset_paths.items():
            if path.exists():
                pkl_files = list(path.rglob('*.pkl'))
                self.datasets[name] = {
                    'path': path,
                    'files': pkl_files,
                    'count': len(pkl_files)
                }
                logger.info(f"Found {name}: {len(pkl_files)} files")
            else:
                logger.warning(f"Dataset {name} not found at {path}")
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        return list(self.datasets.keys())
    
    def get_dataset_info(self) -> Dict:
        """Get info about all datasets"""
        return {
            name: {
                'count': info['count'],
                'path': str(info['path'])
            }
            for name, info in self.datasets.items()
        }
    
    def _parse_smplx(self, smplx_array: np.ndarray) -> SMPLXParams:
        """Parse raw SMPL-X array (T, 182) into structured params"""
        idx = self.SMPLX_INDICES
        return SMPLXParams(
            global_orient=smplx_array[:, idx['global_orient'][0]:idx['global_orient'][1]],
            body_pose=smplx_array[:, idx['body_pose'][0]:idx['body_pose'][1]],
            left_hand_pose=smplx_array[:, idx['left_hand_pose'][0]:idx['left_hand_pose'][1]],
            right_hand_pose=smplx_array[:, idx['right_hand_pose'][0]:idx['right_hand_pose'][1]],
            jaw_pose=smplx_array[:, idx['jaw_pose'][0]:idx['jaw_pose'][1]],
            expression=smplx_array[:, idx['expression'][0]:idx['expression'][1]],
            betas=smplx_array[:, idx['betas'][0]:idx['betas'][1]],
            transl=smplx_array[:, idx['transl'][0]:idx['transl'][1]],
        )
    
    def _to_numpy(self, data) -> np.ndarray:
        """Convert torch tensor or numpy array to numpy"""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return np.array(data)
    
    def load_sample(self, filepath: Union[str, Path], dataset: str = None) -> SignSample:
        """
        Load a single sample from pickle file.
        
        Args:
            filepath: Path to pickle file
            dataset: Dataset name (auto-detected if None)
            
        Returns:
            SignSample object
        """
        filepath = Path(filepath)
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle PHOENIX nested structure
        if dataset == 'phoenix' or (isinstance(data, dict) and len(data) == 1 
                                     and isinstance(list(data.values())[0], dict)):
            video_key = list(data.keys())[0]
            data = data[video_key]
            sample_id = video_key
        else:
            sample_id = filepath.stem
        
        # Auto-detect dataset
        if dataset is None:
            for ds_name, ds_info in self.datasets.items():
                if ds_info['path'] in filepath.parents or str(ds_info['path']) in str(filepath):
                    dataset = ds_name
                    break
            if dataset is None:
                dataset = 'unknown'
        
        # Parse SMPL-X parameters
        smplx_raw = self._to_numpy(data['smplx'])
        smplx_params = self._parse_smplx(smplx_raw)
        
        # Get 2D keypoints
        keypoints_2d = self._to_numpy(data.get('2d', np.zeros((smplx_raw.shape[0], 106, 3))))
        pred_2d = self._to_numpy(data.get('pred_2d', keypoints_2d[:, :, :2]))
        
        return SignSample(
            sample_id=sample_id,
            dataset=dataset,
            num_frames=smplx_raw.shape[0],
            smplx_params=smplx_params,
            keypoints_2d=keypoints_2d,
            pred_keypoints_2d=pred_2d,
            left_hand_valid=self._to_numpy(data['left_valid']),
            right_hand_valid=self._to_numpy(data['right_valid']),
            valid_indices=self._to_numpy(data['total_valid_index']),
            camera_focal=self._to_numpy(data['focal']),
            camera_princpt=self._to_numpy(data['princpt']),
        )
    
    def load_dataset(self, dataset_name: str, max_samples: int = None) -> List[SignSample]:
        """
        Load all samples from a dataset.
        
        Args:
            dataset_name: Name of dataset ('wlasl', 'phoenix', 'how2sign', 'hamnosys')
            max_samples: Maximum number of samples to load (None = all)
            
        Returns:
            List of SignSample objects
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not found. Available: {list(self.datasets.keys())}")
        
        files = self.datasets[dataset_name]['files']
        if max_samples:
            files = files[:max_samples]
        
        samples = []
        for f in files:
            try:
                sample = self.load_sample(f, dataset=dataset_name)
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        
        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples
    
    # =========================================================================
    # TASK 1: Sign Language Motion Generation
    # =========================================================================
    
    def get_motion_sequence(self, sample: SignSample) -> Dict[str, np.ndarray]:
        """
        Extract motion sequence for sign language generation.
        
        Returns dict with body and hand motion parameters.
        """
        params = sample.smplx_params
        return {
            'body_pose': params.body_pose,           # (T, 63)
            'left_hand_pose': params.left_hand_pose, # (T, 45)
            'right_hand_pose': params.right_hand_pose, # (T, 45)
            'global_orient': params.global_orient,   # (T, 3)
            'transl': params.transl,                 # (T, 3)
            'num_frames': sample.num_frames,
        }
    
    def motion_to_smplx_params(self, motion: Dict[str, np.ndarray], 
                                betas: np.ndarray = None) -> Dict[str, torch.Tensor]:
        """
        Convert motion dict to SMPL-X compatible parameters for rendering.
        """
        T = motion['body_pose'].shape[0]
        if betas is None:
            betas = np.zeros((T, 10))
        
        return {
            'global_orient': torch.from_numpy(motion['global_orient']).float(),
            'body_pose': torch.from_numpy(motion['body_pose']).float(),
            'left_hand_pose': torch.from_numpy(motion['left_hand_pose']).float(),
            'right_hand_pose': torch.from_numpy(motion['right_hand_pose']).float(),
            'betas': torch.from_numpy(betas).float(),
            'transl': torch.from_numpy(motion['transl']).float(),
        }
    
    # =========================================================================
    # TASK 2: 3D Human Pose Estimation & Shape Recovery
    # =========================================================================
    
    def get_body_pose(self, sample: SignSample) -> Dict[str, np.ndarray]:
        """
        Extract body pose for human pose estimation.
        
        Returns SMPL body parameters (excludes hands).
        """
        params = sample.smplx_params
        return {
            'body_pose': params.body_pose,       # (T, 63) - 21 joints
            'global_orient': params.global_orient, # (T, 3)
            'betas': params.betas,               # (T, 10) - shape
            'transl': params.transl,             # (T, 3)
            'keypoints_2d': sample.keypoints_2d[:, :25, :], # Body keypoints
        }
    
    def get_body_shape(self, sample: SignSample) -> np.ndarray:
        """Get body shape parameters (betas)"""
        return sample.smplx_params.betas  # (T, 10)
    
    # =========================================================================
    # TASK 3: 3D Hand Pose Estimation & Shape Recovery  
    # =========================================================================
    
    def get_hand_pose(self, sample: SignSample, hand: str = 'both') -> Dict[str, np.ndarray]:
        """
        Extract hand pose for hand pose estimation.
        
        Args:
            sample: SignSample object
            hand: 'left', 'right', or 'both'
            
        Returns:
            Dict with MANO-compatible hand parameters
        """
        params = sample.smplx_params
        kp_idx = self.KEYPOINT_INDICES
        
        result = {}
        
        if hand in ['left', 'both']:
            result['left'] = {
                'hand_pose': params.left_hand_pose,  # (T, 45)
                'valid': sample.left_hand_valid,
                'keypoints_2d': sample.keypoints_2d[:, kp_idx['left_hand'][0]:kp_idx['left_hand'][1], :],
            }
        
        if hand in ['right', 'both']:
            result['right'] = {
                'hand_pose': params.right_hand_pose,  # (T, 45)
                'valid': sample.right_hand_valid,
                'keypoints_2d': sample.keypoints_2d[:, kp_idx['right_hand'][0]:kp_idx['right_hand'][1], :],
            }
        
        return result
    
    def get_mano_params(self, sample: SignSample, hand: str) -> Dict[str, np.ndarray]:
        """
        Get MANO-compatible parameters for a specific hand.
        
        Args:
            hand: 'left' or 'right'
        """
        params = sample.smplx_params
        hand_pose = params.left_hand_pose if hand == 'left' else params.right_hand_pose
        
        # MANO uses first 3 dims as global orient, rest as hand pose
        return {
            'global_orient': hand_pose[:, :3],   # (T, 3)
            'hand_pose': hand_pose[:, 3:],       # (T, 42) - 14 joints
            'betas': np.zeros((sample.num_frames, 10)),  # MANO shape
        }
    
    # =========================================================================
    # TASK 4: 3D Holistic Pose Estimation & Shape Recovery
    # =========================================================================
    
    def get_holistic_pose(self, sample: SignSample) -> Dict[str, np.ndarray]:
        """
        Extract holistic pose (body + hands + face) for full-body estimation.
        """
        params = sample.smplx_params
        kp = self.KEYPOINT_INDICES
        
        return {
            # Body
            'body_pose': params.body_pose,
            'global_orient': params.global_orient,
            'betas': params.betas,
            'transl': params.transl,
            # Hands
            'left_hand_pose': params.left_hand_pose,
            'right_hand_pose': params.right_hand_pose,
            'left_hand_valid': sample.left_hand_valid,
            'right_hand_valid': sample.right_hand_valid,
            # Face
            'jaw_pose': params.jaw_pose,
            'expression': params.expression,
            # 2D keypoints by region
            'body_kp2d': sample.keypoints_2d[:, kp['body'][0]:kp['body'][1], :],
            'left_hand_kp2d': sample.keypoints_2d[:, kp['left_hand'][0]:kp['left_hand'][1], :],
            'right_hand_kp2d': sample.keypoints_2d[:, kp['right_hand'][0]:kp['right_hand'][1], :],
            'face_kp2d': sample.keypoints_2d[:, kp['face'][0]:kp['face'][1], :],
        }
    
    # =========================================================================
    # TASK 5: Sign Language Recognition
    # =========================================================================
    
    def get_recognition_features(self, sample: SignSample, 
                                  include_face: bool = True) -> np.ndarray:
        """
        Extract features for sign language recognition.
        
        Concatenates pose features into a single feature vector per frame.
        
        Returns:
            Feature array of shape (T, feature_dim)
        """
        params = sample.smplx_params
        
        features = [
            params.body_pose,        # (T, 63)
            params.left_hand_pose,   # (T, 45)
            params.right_hand_pose,  # (T, 45)
            params.global_orient,    # (T, 3)
        ]
        
        if include_face:
            features.extend([
                params.jaw_pose,     # (T, 3)
                params.expression,   # (T, 10)
            ])
        
        return np.concatenate(features, axis=1)  # (T, 156 or 169)
    
    def get_pose_sequence_for_recognition(self, sample: SignSample) -> Dict[str, np.ndarray]:
        """
        Get pose sequence formatted for recognition models.
        
        Returns dict compatible with transformer/RNN inputs.
        """
        return {
            'features': self.get_recognition_features(sample),
            'keypoints_2d': sample.pred_keypoints_2d,  # (T, 106, 2)
            'hand_validity': np.stack([
                sample.left_hand_valid, 
                sample.right_hand_valid
            ], axis=1),  # (T, 2)
            'num_frames': sample.num_frames,
            'sample_id': sample.sample_id,
        }
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def create_dataloader(self, dataset_name: str, batch_size: int = 32,
                          task: str = 'recognition', **kwargs):
        """
        Create a PyTorch DataLoader for training.
        
        Args:
            dataset_name: Dataset to load
            batch_size: Batch size
            task: One of 'motion', 'body_pose', 'hand_pose', 'holistic', 'recognition'
        """
        from torch.utils.data import Dataset, DataLoader
        
        class SignAvatarsDataset(Dataset):
            def __init__(self, adapter, dataset_name, task, max_samples=None):
                self.adapter = adapter
                self.task = task
                self.samples = adapter.load_dataset(dataset_name, max_samples)
                
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                
                if self.task == 'motion':
                    return self.adapter.get_motion_sequence(sample)
                elif self.task == 'body_pose':
                    return self.adapter.get_body_pose(sample)
                elif self.task == 'hand_pose':
                    return self.adapter.get_hand_pose(sample)
                elif self.task == 'holistic':
                    return self.adapter.get_holistic_pose(sample)
                elif self.task == 'recognition':
                    return self.adapter.get_pose_sequence_for_recognition(sample)
                else:
                    raise ValueError(f"Unknown task: {self.task}")
        
        dataset = SignAvatarsDataset(self, dataset_name, task, kwargs.get('max_samples'))
        return DataLoader(dataset, batch_size=batch_size, shuffle=kwargs.get('shuffle', True))
    
    def visualize_sample(self, sample: SignSample, frame_idx: int = 0):
        """Visualize a single frame's 2D keypoints (requires matplotlib)"""
        import matplotlib.pyplot as plt
        
        kp = sample.keypoints_2d[frame_idx]  # (106, 3)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot body
        body = kp[:25]
        ax.scatter(body[:, 0], -body[:, 1], c='blue', s=20, label='Body')
        
        # Plot hands
        left_hand = kp[25:46]
        right_hand = kp[46:67]
        ax.scatter(left_hand[:, 0], -left_hand[:, 1], c='red', s=15, label='Left Hand')
        ax.scatter(right_hand[:, 0], -right_hand[:, 1], c='green', s=15, label='Right Hand')
        
        # Plot face
        face = kp[67:106]
        ax.scatter(face[:, 0], -face[:, 1], c='orange', s=10, label='Face')
        
        ax.set_title(f"{sample.sample_id} - Frame {frame_idx}")
        ax.legend()
        ax.axis('equal')
        plt.tight_layout()
        
        return fig


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Initialize adapter
    adapter = SignAvatarsAdapter("data/signavatars")
    
    # Print dataset info
    print("\n" + "="*60)
    print("SignAvatars Adapter - Dataset Summary")
    print("="*60)
    for name, info in adapter.get_dataset_info().items():
        print(f"  {name}: {info['count']} samples")
    
    # Load a sample from each dataset
    print("\n" + "="*60)
    print("Sample Loading Test")
    print("="*60)
    
    for dataset_name in adapter.get_available_datasets():
        files = adapter.datasets[dataset_name]['files']
        if files:
            sample = adapter.load_sample(files[0], dataset=dataset_name)
            print(f"\n{dataset_name.upper()}:")
            print(f"  Sample ID: {sample.sample_id}")
            print(f"  Frames: {sample.num_frames}")
            print(f"  Body pose shape: {sample.smplx_params.body_pose.shape}")
            print(f"  Left hand valid: {sample.left_hand_valid.sum()}/{len(sample.left_hand_valid)}")
            
            # Test feature extraction
            features = adapter.get_recognition_features(sample)
            print(f"  Recognition features: {features.shape}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
