"""
BSL Sign Language Recognition Model

Recognizes BSL signs from pose sequences using a Transformer classifier.

Architecture:
    Pose Sequence → Linear Embedding → Transformer Encoder → Gloss Classification

Training:
    python scripts/train_recognition.py --epochs 50

Inference:
    from sign_recognition import SignRecognizer
    recognizer = SignRecognizer.load("models/sign_recognizer.pt")
    gloss = recognizer.recognize(pose_sequence)
"""

import os
import json
import math
import random
import argparse
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


# ============================================================
# Data Loading
# ============================================================

@dataclass
class PoseSample:
    """Single pose sample for training."""
    gloss: str
    poses: np.ndarray  # Shape: (num_frames, num_keypoints, 3)
    num_frames: int


class BSLPoseDataset(Dataset):
    """Dataset for BSL pose sequences."""
    
    # Keypoint dimensions
    POSE_KEYPOINTS = 33
    HAND_KEYPOINTS = 21
    FACE_KEYPOINTS = 11  # Subset we extract
    
    def __init__(
        self,
        data_dir: str,
        additional_data_dirs: Optional[List[str]] = None,
        split: str = "train",
        max_frames: int = 64,
        min_samples_per_gloss: int = 5,
        gloss_to_idx: Dict[str, int] = None,
        augment: bool = False,
        combine_splits: bool = False,
        max_classes: int = None,
        normalize_poses: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to primary extracted poses
            additional_data_dirs: Optional additional pose directories (same format)
            split: 'train', 'val', or 'test'
            max_frames: Maximum sequence length (pad/truncate)
            min_samples_per_gloss: Minimum samples required for a gloss
            gloss_to_idx: Gloss vocabulary mapping (build if None)
            augment: Apply data augmentation
            combine_splits: Combine all splits and reshuffle (better for this dataset)
            max_classes: Limit to top N most frequent glosses (for better accuracy)
            normalize_poses: Normalize keypoints per frame before flattening
        """
        self.data_dirs = [Path(data_dir)]
        if additional_data_dirs:
            self.data_dirs.extend(Path(p) for p in additional_data_dirs if p)
        self.data_dirs = [p for p in self.data_dirs if p.exists()]
        if not self.data_dirs:
            raise FileNotFoundError(f"No valid data directories found. Primary={data_dir} additional={additional_data_dirs}")

        self.data_dir = self.data_dirs[0]
        self.max_frames = max_frames
        self.min_samples = min_samples_per_gloss
        self.augment = augment
        self.split = split
        self.max_classes = max_classes
        self.normalize_poses = normalize_poses
        
        # Load samples
        if combine_splits:
            self.samples = self._load_all_splits()
        else:
            self.samples = self._load_samples(split)
        
        # Build or use provided vocabulary
        if gloss_to_idx is None:
            self.gloss_to_idx = self._build_vocabulary()
        else:
            self.gloss_to_idx = gloss_to_idx
            # Filter samples to only include known glosses
            self.samples = [s for s in self.samples if s['gloss'] in self.gloss_to_idx]
        
        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}
        
        print(
            f"Loaded {len(self.samples)} samples, {len(self.gloss_to_idx)} classes "
            f"from {len(self.data_dirs)} data source(s)"
        )
    
    def _load_all_splits(self) -> List[Dict]:
        """Load from all splits and reshuffle deterministically."""
        all_samples = []
        for data_root in self.data_dirs:
            cache_file = data_root / "sample_index.json"

            if cache_file.exists():
                print(f"Loading from cache: {cache_file}")
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                for item in cached:
                    file_path = Path(item.get('file', ''))
                    if not file_path.is_absolute():
                        item['file'] = str((data_root / file_path).resolve())
                    if not Path(item['file']).exists():
                        continue
                    item['source'] = str(data_root)
                    all_samples.append(item)
                print(f"Loaded {len(cached)} samples from cache")
            else:
                for split_name in ['train', 'val', 'test']:
                    all_samples.extend(self._load_samples(split_name, data_root))

        print(f"Total samples loaded: {len(all_samples)}")

        # Deterministic split based on stable hash of filename (reproducible across runs)
        def get_split(sample: Dict) -> str:
            file_key = sample['file'].replace('\\', '/').encode('utf-8')
            h = int(hashlib.md5(file_key).hexdigest(), 16) % 100
            if h < 80:
                return "train"
            if h < 90:
                return "val"
            return "test"
        
        # Filter to requested split
        split_samples = [s for s in all_samples if get_split(s) == self.split]
        
        # Shuffle within split (with fixed seed for reproducibility)
        rng = random.Random(42)
        rng.shuffle(split_samples)
        
        return split_samples
    
    def _load_samples(self, split: str, data_root: Optional[Path] = None) -> List[Dict]:
        """Load all pose samples from directory."""
        samples = []
        if data_root is None:
            roots = self.data_dirs
        else:
            roots = [data_root]

        for root in roots:
            split_dir = root / split

            if not split_dir.exists():
                print(f"Warning: {split_dir} not found")
                continue

            json_files = list(split_dir.glob("*.json"))
            print(f"Loading {split} from {root.name}: {len(json_files)} files...")

            for json_file in tqdm(json_files, desc=f"  {split} [{root.name}]"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    samples.append({
                        'gloss': data['gloss'],
                        'file': str(json_file),
                        'num_frames': data['num_frames'],
                        'source': str(root)
                    })
                except Exception:
                    continue

        return samples
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build gloss vocabulary from samples."""
        gloss_counts = Counter(s['gloss'] for s in self.samples)
        
        # Filter by minimum samples
        valid_glosses = [(g, c) for g, c in gloss_counts.items() if c >= self.min_samples]
        
        # Sort by frequency and limit if max_classes is set
        valid_glosses = sorted(valid_glosses, key=lambda x: -x[1])
        
        if hasattr(self, 'max_classes') and self.max_classes:
            valid_glosses = valid_glosses[:self.max_classes]
        
        valid_glosses = sorted([g for g, c in valid_glosses])
        
        return {g: i for i, g in enumerate(valid_glosses)}
    
    def _load_pose_data(self, file_path: str) -> np.ndarray:
        """Load and process pose data from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        frames = []
        for pose_frame in data['poses']:
            frame_data = []
            
            # Body pose (33 keypoints)
            if pose_frame.get('pose'):
                frame_data.extend(pose_frame['pose'])
            else:
                frame_data.extend([(0, 0, 0)] * self.POSE_KEYPOINTS)
            
            # Left hand (21 keypoints)
            if pose_frame.get('left_hand'):
                frame_data.extend(pose_frame['left_hand'])
            else:
                frame_data.extend([(0, 0, 0)] * self.HAND_KEYPOINTS)
            
            # Right hand (21 keypoints)
            if pose_frame.get('right_hand'):
                frame_data.extend(pose_frame['right_hand'])
            else:
                frame_data.extend([(0, 0, 0)] * self.HAND_KEYPOINTS)
            
            frames.append(frame_data)
        
        return np.array(frames, dtype=np.float32)
    
    def _augment_poses(self, poses: np.ndarray) -> np.ndarray:
        """Apply data augmentation to poses."""
        # Random scaling (simulate distance variation)
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            poses = poses * scale
        
        # Random horizontal flip (mirror)
        if random.random() < 0.5:
            # Flip x coordinates
            poses[:, :, 0] = 1.0 - poses[:, :, 0]
            # Swap left and right hands
            # pose: 0-32, left_hand: 33-53, right_hand: 54-74
            left_hand = poses[:, 33:54, :].copy()
            right_hand = poses[:, 54:75, :].copy()
            poses[:, 33:54, :] = right_hand
            poses[:, 54:75, :] = left_hand
        
        # Random noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.01, poses.shape).astype(np.float32)
            poses = poses + noise
        
        # Random temporal jitter (speed variation)
        if random.random() < 0.3:
            speed = random.uniform(0.8, 1.2)
            num_frames = int(poses.shape[0] * speed)
            if num_frames > 1:
                indices = np.linspace(0, poses.shape[0] - 1, num_frames).astype(int)
                poses = poses[indices]
        
        return poses

    def _normalize_pose_sequence(self, poses: np.ndarray) -> np.ndarray:
        """Normalize keypoints per frame to reduce camera-scale/offset variance."""
        poses = poses.copy()
        for f in range(poses.shape[0]):
            frame = poses[f]
            valid = np.linalg.norm(frame, axis=1) > 1e-6
            if not valid.any():
                continue

            # Shoulder anchors from body pose, plus hand wrists if available.
            anchor_indices = [11, 12, 33, 54]
            anchor_points = [frame[i] for i in anchor_indices if i < frame.shape[0] and valid[i]]
            if anchor_points:
                center = np.mean(anchor_points, axis=0)
            else:
                center = np.mean(frame[valid], axis=0)

            scale = None
            if 11 < frame.shape[0] and 12 < frame.shape[0] and valid[11] and valid[12]:
                scale = float(np.linalg.norm(frame[11, :2] - frame[12, :2]))
            if scale is None or scale < 1e-4:
                d = np.linalg.norm(frame[valid, :2] - center[:2], axis=1)
                scale = float(np.mean(d)) if d.size else 1.0
            scale = max(scale, 1e-3)

            frame_norm = (frame - center) / scale
            frame_norm[~valid] = 0.0
            poses[f] = frame_norm

        return poses

    def get_class_counts(self) -> Dict[str, int]:
        """Return gloss counts for current dataset samples."""
        return dict(Counter(s['gloss'] for s in self.samples if s['gloss'] in self.gloss_to_idx))

    def get_sample_weights(self, power: float = 0.5) -> List[float]:
        """Return sample-level weights using inverse-frequency weighting."""
        class_counts = self.get_class_counts()
        weights = []
        for sample in self.samples:
            count = class_counts.get(sample['gloss'], 1)
            weights.append(float(1.0 / (count ** power)))
        return weights
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load pose data
        poses = self._load_pose_data(sample['file'])
        
        # Apply augmentation during training
        if self.augment and self.split == 'train':
            poses = self._augment_poses(poses)

        # Normalize poses for better cross-domain robustness
        if self.normalize_poses:
            poses = self._normalize_pose_sequence(poses)
        
        # Pad or truncate to max_frames
        num_frames = poses.shape[0]
        if num_frames > self.max_frames:
            # Random crop during training, center crop during eval
            if self.augment:
                start = random.randint(0, num_frames - self.max_frames)
            else:
                start = (num_frames - self.max_frames) // 2
            poses = poses[start:start + self.max_frames]
            mask = torch.ones(self.max_frames, dtype=torch.bool)
        elif num_frames < self.max_frames:
            # Pad with zeros
            pad_size = self.max_frames - num_frames
            poses = np.pad(poses, ((0, pad_size), (0, 0), (0, 0)), mode='constant')
            mask = torch.zeros(self.max_frames, dtype=torch.bool)
            mask[:num_frames] = True
        else:
            mask = torch.ones(self.max_frames, dtype=torch.bool)
        
        # Flatten keypoints: (frames, keypoints, 3) -> (frames, keypoints*3)
        poses = poses.reshape(self.max_frames, -1)
        
        # Get label
        label = self.gloss_to_idx.get(sample['gloss'], 0)
        
        return {
            'poses': torch.tensor(poses, dtype=torch.float32),
            'mask': mask,
            'label': torch.tensor(label, dtype=torch.long),
            'gloss': sample['gloss']
        }


# ============================================================
# Model Architecture
# ============================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


class SignRecognitionModel(nn.Module):
    """
    Transformer-based sign language recognition model.
    
    Input: Pose sequence (batch, frames, keypoints*3)
    Output: Gloss logits (batch, num_classes)
    """
    
    def __init__(
        self,
        input_dim: int = 225,  # (33+21+21) * 3
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        num_classes: int = 100,
        dropout: float = 0.1,
        max_frames: int = 64
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_frames = max_frames
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_frames, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, poses: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            poses: (batch, frames, input_dim)
            mask: (batch, frames) - True for valid frames
            
        Returns:
            logits: (batch, num_classes)
        """
        # Project input
        x = self.input_proj(poses)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create attention mask (True = ignore)
        if mask is not None:
            src_key_padding_mask = ~mask
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling over valid frames
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


# ============================================================
# Training
# ============================================================

class SignRecognitionTrainer:
    """Training loop for sign recognition model."""
    
    def __init__(
        self,
        model: SignRecognitionModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        class_weights: Optional[torch.Tensor] = None,
        class_priors: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        about_idx: Optional[int] = None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.about_idx = about_idx
        self.class_priors = class_priors.detach().cpu() if class_priors is not None else None
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_weights = class_weights.to(device) if class_weights is not None else None
        try:
            self.criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=label_smoothing)
        except TypeError:
            # Fallback for older torch versions
            self.criterion = nn.CrossEntropyLoss(weight=loss_weights)
            if label_smoothing > 0:
                print("Warning: label_smoothing not supported by this torch version, continuing without it.")
        
        self.best_val_acc = 0.0
        self.best_val_macro_f1 = -1.0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_macro_f1': [],
            'val_about_fp_rate': []
        }

    @staticmethod
    def _macro_f1(y_true: List[int], y_pred: List[int]) -> float:
        """Compute macro F1 without external dependencies."""
        classes = sorted(set(y_true) | set(y_pred))
        if not classes:
            return 0.0
        f1_scores = []
        for cls in classes:
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p == cls)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != cls and p == cls)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == cls and p != cls)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            f1_scores.append(f1)
        return float(np.mean(f1_scores))
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            poses = batch['poses'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(poses, mask)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(self.train_loader), correct / total
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        if len(self.val_loader) == 0:
            return {
                "loss": 0.0,
                "acc": 0.0,
                "macro_f1": 0.0,
                "about_fp_rate": 0.0,
            }

        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        y_true: List[int] = []
        y_pred: List[int] = []
        
        for batch in self.val_loader:
            poses = batch['poses'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(poses, mask)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

        macro_f1 = self._macro_f1(y_true, y_pred)
        about_fp_rate = 0.0
        if self.about_idx is not None:
            non_about = sum(1 for t in y_true if t != self.about_idx)
            about_fp = sum(1 for t, p in zip(y_true, y_pred) if t != self.about_idx and p == self.about_idx)
            about_fp_rate = about_fp / max(non_about, 1)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'acc': correct / total,
            'macro_f1': macro_f1,
            'about_fp_rate': about_fp_rate
        }
    
    def train(self, epochs: int, save_dir: str):
        """Full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_metrics = self.validate()
            scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_macro_f1'].append(val_metrics['macro_f1'])
            self.history['val_about_fp_rate'].append(val_metrics['about_fp_rate'])
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(
                f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['acc']:.4f}, "
                f"Macro-F1: {val_metrics['macro_f1']:.4f}, ABOUT-FP: {val_metrics['about_fp_rate']:.4f}"
            )
            
            # Save best model by macro-F1 to reduce dominant-class collapse.
            if val_metrics['macro_f1'] > self.best_val_macro_f1:
                self.best_val_macro_f1 = val_metrics['macro_f1']
                self.best_val_acc = val_metrics['acc']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_metrics['acc'],
                    'val_macro_f1': val_metrics['macro_f1'],
                    'val_about_fp_rate': val_metrics['about_fp_rate'],
                    'num_classes': self.model.num_classes,
                    'd_model': self.model.d_model,
                    'input_dim': self.model.input_dim,
                    'nhead': self.model.nhead,
                    'num_layers': self.model.num_layers,
                    'dim_feedforward': self.model.dim_feedforward,
                    'dropout': self.model.dropout,
                    'max_frames': self.model.max_frames,
                    'class_priors': self.class_priors.tolist() if self.class_priors is not None else None,
                }, save_dir / "best_model.pt")
                print(f"Saved best model (macro_f1: {val_metrics['macro_f1']:.4f})")
        
        # Save final model and history
        torch.save(self.model.state_dict(), save_dir / "final_model.pt")
        with open(save_dir / "history.json", 'w') as f:
            json.dump(self.history, f)
        
        return self.history


# ============================================================
# Inference
# ============================================================

class SignRecognizer:
    """Sign language recognition inference."""
    
    def __init__(
        self,
        model: SignRecognitionModel,
        gloss_to_idx: Dict[str, int],
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.gloss_to_idx = gloss_to_idx
        self.idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
    
    @classmethod
    def load(cls, checkpoint_path: str, vocab_path: str, device: str = "cuda"):
        """Load trained model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        with open(vocab_path, 'r') as f:
            gloss_to_idx = json.load(f)
        
        model = SignRecognitionModel(
            input_dim=checkpoint.get('input_dim', 225),
            d_model=checkpoint.get('d_model', 256),
            nhead=checkpoint.get('nhead', 8),
            num_layers=checkpoint.get('num_layers', 4),
            dim_feedforward=checkpoint.get('dim_feedforward', checkpoint.get('d_model', 256) * 4),
            num_classes=checkpoint.get('num_classes', len(gloss_to_idx)),
            dropout=checkpoint.get('dropout', 0.1),
            max_frames=checkpoint.get('max_frames', 64)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, gloss_to_idx, device)
    
    @torch.no_grad()
    def recognize(self, poses: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Recognize sign from pose sequence.
        
        Args:
            poses: (frames, keypoints, 3) or (frames, keypoints*3)
            top_k: Number of top predictions to return
            
        Returns:
            List of (gloss, confidence) tuples
        """
        # Reshape if needed
        if poses.ndim == 3:
            poses = poses.reshape(poses.shape[0], -1)

        if poses.ndim != 2:
            raise ValueError(f"Expected poses with shape (frames, features), got {poses.shape}")

        if poses.shape[0] == 0:
            poses = np.zeros((1, self.model.input_dim), dtype=np.float32)

        # Build validity mask before pad/truncate
        valid_mask = np.any(np.abs(poses) > 1e-6, axis=1)

        # Pad/truncate to model max frames
        max_frames = getattr(self.model, "max_frames", 64)
        if poses.shape[0] > max_frames:
            poses = poses[-max_frames:]
            valid_mask = valid_mask[-max_frames:]
        elif poses.shape[0] < max_frames:
            pad_size = max_frames - poses.shape[0]
            poses = np.pad(poses, ((0, pad_size), (0, 0)), mode='constant')
            valid_mask = np.pad(valid_mask, (0, pad_size), mode='constant', constant_values=False)
        
        # Convert to tensor
        poses_tensor = torch.tensor(poses, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        if not bool(mask_tensor.any()):
            return [("NO_SIGN", 1.0)]
        
        # Forward pass
        logits = self.model(poses_tensor, mask_tensor)
        probs = F.softmax(logits, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = probs.topk(top_k, dim=1)
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            gloss = self.idx_to_gloss.get(idx.item(), "UNKNOWN")
            results.append((gloss, prob.item()))
        
        return results


# ============================================================
# Main Training Script
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Train sign recognition model")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--extra-data-dir", type=str, default=None,
                       help="Optional extra pose dataset (e.g., data/poses_bsldict)")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--min-samples", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--combine-splits", action="store_true", default=True,
                       help="Combine all splits and reshuffle for better training")
    parser.add_argument("--no-combine", action="store_false", dest="combine_splits",
                       help="Use original train/val/test splits")
    parser.add_argument("--max-classes", type=int, default=None,
                       help="Limit to top N most frequent glosses (None = use all)")
    parser.add_argument("--use-balanced-sampler", action="store_true", dest="use_balanced_sampler",
                       help="Use class-balanced weighted random sampling")
    parser.add_argument("--no-balanced-sampler", action="store_false", dest="use_balanced_sampler",
                       help="Disable class-balanced sampling")
    parser.add_argument("--label-smoothing", type=float, default=0.05,
                       help="Cross-entropy label smoothing factor")
    parser.add_argument("--class-weight-power", type=float, default=0.5,
                       help="Class weighting power. Weight = 1 / (count^power)")
    parser.add_argument("--normalize-poses", action="store_true", dest="normalize_poses",
                       help="Normalize pose coordinates per frame")
    parser.add_argument("--no-normalize", action="store_false", dest="normalize_poses",
                       help="Disable pose normalization")
    parser.add_argument("--save-class-stats", action="store_true", dest="save_class_stats",
                       help="Save class counts and priors")
    parser.add_argument("--no-save-class-stats", action="store_false", dest="save_class_stats",
                       help="Do not save class stats")

    parser.set_defaults(use_balanced_sampler=True, normalize_poses=True, save_class_stats=True)
    
    args = parser.parse_args()
    
    # Default paths
    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    
    if args.data_dir is None:
        args.data_dir = project_root / "data" / "poses"

    if args.extra_data_dir is None:
        default_extra = project_root / "data" / "poses_bsldict"
        args.extra_data_dir = default_extra if default_extra.exists() else None
    
    if args.save_dir is None:
        args.save_dir = project_root / "models" / "sign_recognition"

    fixed_vocab_path = project_root / "models" / "sign_recognition" / "vocabulary.json"
    fixed_gloss_to_idx = None
    if args.max_classes is None and fixed_vocab_path.exists():
        with open(fixed_vocab_path, "r") as f:
            fixed_gloss_to_idx = json.load(f)
        print(f"Using fixed vocabulary: {fixed_vocab_path} ({len(fixed_gloss_to_idx)} classes)")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Combine splits: {args.combine_splits}")
    print(f"Balanced sampler: {args.use_balanced_sampler}")
    print(f"Normalize poses: {args.normalize_poses}")
    print(f"Extra data dir: {args.extra_data_dir}")
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Load datasets
    print("Loading training data...")
    additional_dirs = []
    if args.extra_data_dir is not None and Path(args.extra_data_dir).exists():
        additional_dirs.append(str(args.extra_data_dir))

    train_dataset = BSLPoseDataset(
        args.data_dir,
        additional_data_dirs=additional_dirs,
        split="train",
        max_frames=args.max_frames,
        min_samples_per_gloss=args.min_samples,
        gloss_to_idx=fixed_gloss_to_idx,
        augment=True,
        combine_splits=args.combine_splits,
        max_classes=args.max_classes,
        normalize_poses=args.normalize_poses
    )
    
    print("Loading validation data...")
    val_dataset = BSLPoseDataset(
        args.data_dir,
        additional_data_dirs=additional_dirs,
        split="val",
        max_frames=args.max_frames,
        gloss_to_idx=train_dataset.gloss_to_idx,
        augment=False,
        combine_splits=args.combine_splits,
        normalize_poses=args.normalize_poses
    )

    if len(train_dataset.gloss_to_idx) == 0:
        raise ValueError(
            "No classes available after filtering. Lower --min-samples or disable --max-classes limit."
        )
    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty.")
    if len(val_dataset) == 0:
        print("Warning: validation dataset is empty. Metrics and checkpoint selection may be unreliable.")
    
    # Save vocabulary
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "vocabulary.json", 'w') as f:
        json.dump(train_dataset.gloss_to_idx, f)

    # Compute class counts/weights/priors from train split.
    class_counts_gloss = train_dataset.get_class_counts()
    class_counts = np.zeros(len(train_dataset.gloss_to_idx), dtype=np.float32)
    for gloss, idx in train_dataset.gloss_to_idx.items():
        class_counts[idx] = float(max(class_counts_gloss.get(gloss, 0), 0))

    weight_counts = np.clip(class_counts, 1.0, None)
    raw_weights = 1.0 / np.power(weight_counts, args.class_weight_power)
    class_weights = raw_weights / max(np.mean(raw_weights), 1e-8)
    if class_counts.sum() <= 0:
        class_priors = np.ones_like(class_counts) / max(len(class_counts), 1)
    else:
        class_priors = class_counts / class_counts.sum()

    missing_classes = int(np.sum(class_counts == 0))
    if missing_classes > 0:
        print(f"Info: {missing_classes} class(es) have zero training samples in this run.")

    if args.save_class_stats:
        stats = {
            "num_classes": len(train_dataset.gloss_to_idx),
            "class_weight_power": args.class_weight_power,
            "class_counts": {gloss: int(class_counts_gloss.get(gloss, 0)) for gloss in train_dataset.gloss_to_idx},
            "class_priors": {gloss: float(class_priors[idx]) for gloss, idx in train_dataset.gloss_to_idx.items()},
        }
        with open(save_dir / "class_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved class stats to: {save_dir / 'class_stats.json'}")
    
    # Data loaders
    train_sampler = None
    if args.use_balanced_sampler:
        sample_weights = train_dataset.get_sample_weights(power=args.class_weight_power)
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False if train_sampler is not None else True,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    # Calculate input dimension
    input_dim = (33 + 21 + 21) * 3  # pose + left_hand + right_hand
    
    # Create model
    model = SignRecognitionModel(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=8,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        num_classes=len(train_dataset.gloss_to_idx),
        dropout=0.2,  # Increased dropout for regularization
        max_frames=args.max_frames
    )
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Classes: {len(train_dataset.gloss_to_idx)}")
    
    # Train
    trainer = SignRecognitionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        class_weights=torch.tensor(class_weights, dtype=torch.float32),
        class_priors=torch.tensor(class_priors, dtype=torch.float32),
        label_smoothing=args.label_smoothing,
        about_idx=train_dataset.gloss_to_idx.get("ABOUT")
    )
    
    trainer.train(epochs=args.epochs, save_dir=args.save_dir)
    
    print(
        f"\nTraining complete! Best val accuracy: {trainer.best_val_acc:.4f}, "
        f"Best macro-F1: {trainer.best_val_macro_f1:.4f}"
    )
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
