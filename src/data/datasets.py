#!/usr/bin/env python3
"""
PyTorch Dataset classes for BOBSL sign language recognition.

Provides datasets for:
- Isolated sign classification (single sign per sample)
- Continuous sign recognition (sequence of signs per video)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from .annotation_parser import BOBSLAnnotationParser, SignInstance, VideoAnnotation


@dataclass
class Vocabulary:
    """Vocabulary for mapping between glosses and indices."""
    
    gloss_to_idx: Dict[str, int]
    idx_to_gloss: Dict[int, str]
    
    def __init__(self, gloss_to_idx: Dict[str, int] = None):
        """
        Initialize vocabulary.
        
        Args:
            gloss_to_idx: Dictionary mapping gloss strings to indices.
                         If None, creates empty vocabulary with special tokens.
        """
        if gloss_to_idx is None:
            self.gloss_to_idx = {
                '<pad>': 0,
                '<unk>': 1,
                '<sos>': 2,
                '<eos>': 3
            }
        else:
            self.gloss_to_idx = gloss_to_idx
        
        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}
    
    def add_gloss(self, gloss: str) -> int:
        """Add a gloss to vocabulary and return its index."""
        if gloss not in self.gloss_to_idx:
            idx = len(self.gloss_to_idx)
            self.gloss_to_idx[gloss] = idx
            self.idx_to_gloss[idx] = gloss
        return self.gloss_to_idx[gloss]
    
    def encode(self, gloss: str) -> int:
        """Convert gloss to index."""
        return self.gloss_to_idx.get(gloss, self.gloss_to_idx.get('<unk>', 1))
    
    def decode(self, idx: int) -> str:
        """Convert index to gloss."""
        return self.idx_to_gloss.get(idx, '<unk>')
    
    def encode_sequence(self, glosses: List[str], add_eos: bool = False) -> List[int]:
        """Convert list of glosses to list of indices."""
        indices = [self.encode(g) for g in glosses]
        if add_eos:
            indices.append(self.gloss_to_idx.get('<eos>', 3))
        return indices
    
    def decode_sequence(self, indices: List[int]) -> List[str]:
        """Convert list of indices to list of glosses."""
        return [self.decode(i) for i in indices]
    
    def __len__(self) -> int:
        return len(self.gloss_to_idx)
    
    def save(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        data = {
            'gloss_to_idx': self.gloss_to_idx,
            'size': len(self)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Vocabulary':
        """Load vocabulary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls(data['gloss_to_idx'])
    
    @classmethod
    def from_parser(cls, parser: BOBSLAnnotationParser) -> 'Vocabulary':
        """Create vocabulary from parsed annotations."""
        vocab = cls()
        for gloss in parser.vocabulary.keys():
            vocab.add_gloss(gloss)
        return vocab


class IsolatedSignDataset(Dataset):
    """
    Dataset for isolated sign classification.
    
    Each sample contains features for a single sign instance with its label.
    Features are extracted from a temporal window around the sign's timestamp.
    """
    
    def __init__(
        self,
        parser: BOBSLAnnotationParser,
        vocabulary: Vocabulary,
        feature_type: str = 'swin',
        window_seconds: float = 2.0,
        fps: float = 25.0,
        max_frames: int = 64,
        feature_dim: int = 768,
    ):
        self.parser = parser
        self.vocabulary = vocabulary
        self.feature_type = feature_type
        self.window_seconds = window_seconds
        self.fps = fps
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        
        # Get training data with features
        self.samples = parser.to_training_format(
            feature_type=feature_type,
            only_with_features=True
        )
        
        print(f"IsolatedSignDataset initialized: {len(self.samples)} samples")
    
    def _extract_window(
        self, 
        features: np.ndarray, 
        global_time: float
    ) -> np.ndarray:
        """
        Extract temporal window of features around timestamp.
        
        Args:
            features: Full video features (T, D)
            global_time: Timestamp in seconds
            
        Returns:
            Windowed features (max_frames, D)
        """
        total_frames = features.shape[0]
        
        # Convert time to frame index
        center_frame = int(global_time * self.fps)
        half_window = int(self.window_seconds * self.fps / 2)
        
        # Calculate window bounds
        start_frame = max(0, center_frame - half_window)
        end_frame = min(total_frames, center_frame + half_window)
        
        # Extract window
        window = features[start_frame:end_frame]
        
        # Convert to float32 after extraction (memory efficient)
        if window.dtype == np.float16:
            window = window.astype(np.float32)
        
        # Pad or truncate to max_frames
        if len(window) < self.max_frames:
            padding = np.zeros((self.max_frames - len(window), self.feature_dim), 
                              dtype=np.float32)
            window = np.concatenate([window, padding], axis=0)
        elif len(window) > self.max_frames:
            # Uniform sampling
            indices = np.linspace(0, len(window) - 1, self.max_frames, dtype=int)
            window = window[indices]
        
        return window
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Load features directly (no caching - too memory intensive)
        features = np.load(sample['feature_path'], mmap_mode='r')
        
        # Extract temporal window (converts to float32 internally)
        window = self._extract_window(features, sample['global_time'])
        
        # Check for NaN
        if np.isnan(window).any():
            window = np.nan_to_num(window, nan=0.0)
        
        # Get label
        label = self.vocabulary.encode(sample['gloss'])
        
        return {
            'features': torch.from_numpy(window).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'gloss': sample['gloss'],
            'video_id': sample['video_id'],
            'confidence': sample['confidence'],
        }
    
    def clear_cache(self) -> None:
        """No-op for compatibility."""
        pass


class ContinuousSignDataset(Dataset):
    """
    Dataset for continuous sign recognition.
    
    Each sample contains full video features and the sequence of signs
    that appear in that video, sorted by time.
    """
    
    def __init__(
        self,
        parser: BOBSLAnnotationParser,
        vocabulary: Vocabulary,
        feature_type: str = 'swin',
        max_frames: int = 1024,
        max_glosses: int = 100,
        feature_dim: int = 768,
    ):
        self.parser = parser
        self.vocabulary = vocabulary
        self.feature_type = feature_type
        self.max_frames = max_frames
        self.max_glosses = max_glosses
        self.feature_dim = feature_dim
        
        # Get videos with features
        self.video_ids = parser.get_available_videos(feature_type)
        
        print(f"ContinuousSignDataset initialized: {len(self.video_ids)} videos")
    
    def __len__(self) -> int:
        return len(self.video_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id = self.video_ids[idx]
        video_ann = self.parser.videos[video_id]
        
        # Load features with memory mapping
        feature_path = self.parser.get_feature_path(video_id, self.feature_type)
        features = np.load(feature_path, mmap_mode='r')
        seq_length = features.shape[0]
        
        # Truncate or pad features
        if seq_length > self.max_frames:
            indices = np.linspace(0, seq_length - 1, self.max_frames, dtype=int)
            features = np.array(features[indices])
            seq_length = self.max_frames
        else:
            features = np.array(features)
            if seq_length < self.max_frames:
                padding = np.zeros((self.max_frames - seq_length, self.feature_dim),
                                  dtype=np.float32)
                features = np.concatenate([features, padding], axis=0)
        
        # Convert to float32
        if features.dtype == np.float16:
            features = features.astype(np.float32)
        
        # Get gloss sequence (sorted by time)
        gloss_sequence = video_ann.gloss_sequence[:self.max_glosses]
        gloss_indices = self.vocabulary.encode_sequence(gloss_sequence, add_eos=True)
        gloss_length = len(gloss_indices)
        
        # Pad gloss sequence
        if len(gloss_indices) < self.max_glosses + 1:
            gloss_indices.extend([0] * (self.max_glosses + 1 - len(gloss_indices)))
        
        return {
            'features': torch.from_numpy(features).float(),
            'seq_length': torch.tensor(seq_length, dtype=torch.long),
            'gloss_indices': torch.tensor(gloss_indices, dtype=torch.long),
            'gloss_length': torch.tensor(gloss_length, dtype=torch.long),
            'video_id': video_id,
        }
    
    def clear_cache(self) -> None:
        """No-op for compatibility."""
        pass


class PooledSignDataset(Dataset):
    """
    Dataset for isolated sign classification using pooled features.
    
    Instead of extracting a temporal window, this pools all features
    for each video and associates them with the most confident sign.
    """
    
    def __init__(
        self,
        parser: BOBSLAnnotationParser,
        vocabulary: Vocabulary,
        feature_type: str = 'swin',
        pooling: str = 'mean',
    ):
        self.parser = parser
        self.vocabulary = vocabulary
        self.feature_type = feature_type
        self.pooling = pooling
        
        # Build samples and filter out bad ones
        self.samples = self._build_samples()
        
        print(f"PooledSignDataset initialized: {len(self.samples)} samples")
    
    def _build_samples(self) -> List[Dict]:
        """Build unique (video, gloss) samples, filtering out NaN features."""
        samples = []
        nan_count = 0
        
        for video_id in self.parser.get_available_videos(self.feature_type):
            video_ann = self.parser.videos[video_id]
            feature_path = self.parser.get_feature_path(video_id, self.feature_type)
            
            # Load and check features once per video
            features = np.load(feature_path)
            if features.dtype == np.float16:
                features = features.astype(np.float32)
            
            # Skip videos with NaN features
            if np.isnan(features).any():
                nan_count += 1
                continue
            
            # Pool features once per video
            if self.pooling == 'mean':
                pooled = features.mean(axis=0)
            elif self.pooling == 'max':
                pooled = features.max(axis=0)
            else:
                pooled = features.mean(axis=0)
            
            # Check pooled features for NaN
            if np.isnan(pooled).any():
                nan_count += 1
                continue
            
            # Group signs by gloss, keep highest confidence
            gloss_best = {}
            for sign in video_ann.signs:
                if sign.gloss not in gloss_best:
                    gloss_best[sign.gloss] = sign
                elif sign.confidence > gloss_best[sign.gloss].confidence:
                    gloss_best[sign.gloss] = sign
            
            for gloss, sign in gloss_best.items():
                samples.append({
                    'video_id': video_id,
                    'gloss': gloss,
                    'confidence': sign.confidence,
                    'pooled_features': pooled,  # Store pre-pooled features
                })
        
        if nan_count > 0:
            print(f"  Filtered out {nan_count} videos with NaN features")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Use pre-pooled features
        pooled = sample['pooled_features']
        label = self.vocabulary.encode(sample['gloss'])
        
        return {
            'features': torch.from_numpy(pooled).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'gloss': sample['gloss'],
            'video_id': sample['video_id'],
        }


def create_data_splits(
    parser: BOBSLAnnotationParser,
    vocabulary: Vocabulary,
    dataset_type: str = 'isolated',
    feature_type: str = 'swin',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_state: int = 42,
    **dataset_kwargs
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train/val/test dataset splits.
    
    Args:
        parser: BOBSLAnnotationParser instance
        vocabulary: Vocabulary instance
        dataset_type: 'isolated', 'continuous', or 'pooled'
        feature_type: 'swin' or 'i3d'
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed for reproducibility
        **dataset_kwargs: Additional arguments for dataset class
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Select dataset class
    if dataset_type == 'isolated':
        DatasetClass = IsolatedSignDataset
    elif dataset_type == 'continuous':
        DatasetClass = ContinuousSignDataset
    elif dataset_type == 'pooled':
        DatasetClass = PooledSignDataset
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create full dataset
    full_dataset = DatasetClass(
        parser=parser,
        vocabulary=vocabulary,
        feature_type=feature_type,
        **dataset_kwargs
    )
    
    # Create indices
    n_samples = len(full_dataset)
    indices = list(range(n_samples))
    
    # Split indices
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=random_state
    )
    
    val_size = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_size,
        random_state=random_state
    )
    
    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)
    
    print(f"Data splits: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoader instances for train/val/test datasets.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, test_loader


def collate_fn_isolated(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for isolated sign dataset."""
    features = torch.stack([item['features'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    
    return {
        'features': features,
        'labels': labels,
        'glosses': [item['gloss'] for item in batch],
        'video_ids': [item['video_id'] for item in batch],
    }


def collate_fn_continuous(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for continuous sign dataset."""
    features = torch.stack([item['features'] for item in batch])
    seq_lengths = torch.stack([item['seq_length'] for item in batch])
    gloss_indices = torch.stack([item['gloss_indices'] for item in batch])
    gloss_lengths = torch.stack([item['gloss_length'] for item in batch])
    
    return {
        'features': features,
        'seq_lengths': seq_lengths,
        'gloss_indices': gloss_indices,
        'gloss_lengths': gloss_lengths,
        'video_ids': [item['video_id'] for item in batch],
    }