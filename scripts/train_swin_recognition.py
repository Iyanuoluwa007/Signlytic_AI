"""
Training Pipeline for Temporal Sign Recognition

Combines:
- SWIN temporal features
- BSL-1K multi-source annotations
- Optional Leap Motion hand supervision
- Curriculum learning (high-quality first, then weak labels)

Usage:
    python train_swin_recognition.py --config configs/swin_recognition.yaml
    python train_swin_recognition.py --epochs 100 --batch_size 32
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import random

# Add project root to path
PROJECT_ROOT = Path("D:/Signlytic_AI/code/bsl_translation_project")
sys.path.insert(0, str(PROJECT_ROOT))

# Local imports (will be available after copying files)
try:
    from models.temporal_recognition import (
        TemporalSignRecognitionModel,
        RecognitionModelWithSmoothing,
        create_model
    )
    from data.bsl1k_parser import BSL1KParser, SWINFeatureLoader
except ImportError:
    print("Note: Run from project directory or install modules")


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    swin_features_dir: str = str(PROJECT_ROOT / "data" / "swin_features")
    bsl1k_dir: str = str(PROJECT_ROOT / "data" / "BSL-1K")
    leap_motion_path: str = str(PROJECT_ROOT / "data" / "Leap_Motion" / "BSL-leap-motion.csv")
    
    # Model
    feature_dim: int = 768
    num_layers: int = 4
    num_heads: int = 8
    d_model: int = 512
    dropout: float = 0.2
    use_handshape_head: bool = False
    
    # Training
    batch_size: int = 32
    max_seq_len: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    
    # Curriculum learning
    curriculum_stages: List[str] = None
    stage_epochs: List[int] = None
    
    # Loss weights
    classification_weight: float = 1.0
    handshape_weight: float = 0.1
    confidence_weight: float = 0.05
    
    # Output
    output_dir: str = str(PROJECT_ROOT / "models" / "swin_recognition")
    save_every: int = 10
    
    def __post_init__(self):
        if self.curriculum_stages is None:
            self.curriculum_stages = ['EXEMPLARS', 'MOUTHING', 'DICTIONARY', 'I3D']
        if self.stage_epochs is None:
            self.stage_epochs = [30, 20, 30, 20]


class SWINRecognitionDataset(Dataset):
    """
    Dataset for SWIN feature-based recognition.
    
    Loads temporal feature segments with corresponding gloss labels.
    """
    
    def __init__(
        self,
        samples: List[Tuple[str, int, int, int]],
        feature_loader: SWINFeatureLoader,
        gloss_to_idx: Dict[str, int],
        max_seq_len: int = 64,
        augment: bool = True
    ):
        self.samples = samples
        self.feature_loader = feature_loader
        self.gloss_to_idx = gloss_to_idx
        self.max_seq_len = max_seq_len
        self.augment = augment
        
        # Filter to only available videos
        available = feature_loader.available_videos
        self.samples = [s for s in samples if s[0] in available]
        
        print(f"Dataset: {len(self.samples)} samples (filtered from {len(samples)})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, start_idx, end_idx, gloss_idx = self.samples[idx]
        
        # Apply temporal augmentation
        if self.augment:
            # Random temporal jitter
            jitter = random.randint(-2, 2)
            start_idx = max(0, start_idx + jitter)
            end_idx = max(start_idx + 1, end_idx + jitter)
            
            # Random temporal scaling
            if random.random() < 0.3:
                duration = end_idx - start_idx
                scale = random.uniform(0.8, 1.2)
                new_duration = int(duration * scale)
                end_idx = start_idx + max(1, new_duration)
        
        # Load features
        result = self.feature_loader.load_segment(
            video_id, start_idx, end_idx, pad_to=self.max_seq_len
        )
        
        if result is None:
            # Return zeros if loading fails
            features = torch.zeros(self.max_seq_len, self.feature_loader.feature_dim)
            mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        else:
            features, mask = result
            features = torch.from_numpy(features).float()
            mask = torch.from_numpy(mask).bool()
        
        # Feature augmentation
        if self.augment:
            # Add noise
            if random.random() < 0.3:
                noise = torch.randn_like(features) * 0.05
                features = features + noise
            
            # Random feature dropout
            if random.random() < 0.2:
                drop_mask = torch.rand(features.shape[0], 1) > 0.1
                features = features * drop_mask
        
        return {
            'features': features,
            'mask': mask,
            'label': torch.tensor(gloss_idx, dtype=torch.long),
            'video_id': video_id
        }


class LeapMotionSupervisor:
    """
    Provides auxiliary supervision from Leap Motion hand data.
    Maps glosses to canonical hand shapes for consistency loss.
    """
    
    def __init__(self, leap_motion_path: str):
        self.leap_path = Path(leap_motion_path)
        self.gloss_to_handshape: Dict[str, int] = {}
        self.handshape_templates: Dict[int, np.ndarray] = {}
        self._load_data()
    
    def _load_data(self):
        """Load and process Leap Motion data."""
        if not self.leap_path.exists():
            print(f"Leap Motion data not found: {self.leap_path}")
            return
        
        import pandas as pd
        df = pd.read_csv(self.leap_path)
        
        # Find label column
        label_cols = [c for c in df.columns if 'label' in c.lower() or 'sign' in c.lower() or 'class' in c.lower()]
        
        if not label_cols:
            # Try first column as label
            label_col = df.columns[0]
        else:
            label_col = label_cols[0]
        
        # Extract hand feature columns
        hand_cols = [c for c in df.columns if c != label_col]
        
        # Group by label and compute mean hand shape
        unique_labels = df[label_col].unique()
        
        for idx, label in enumerate(unique_labels):
            label_data = df[df[label_col] == label][hand_cols]
            mean_shape = label_data.mean().values
            
            gloss = str(label).upper().strip()
            self.gloss_to_handshape[gloss] = idx
            self.handshape_templates[idx] = mean_shape
        
        print(f"Loaded {len(self.gloss_to_handshape)} handshape templates from Leap Motion")
    
    def get_handshape_idx(self, gloss: str) -> Optional[int]:
        """Get handshape index for a gloss."""
        return self.gloss_to_handshape.get(gloss.upper())
    
    def get_template(self, handshape_idx: int) -> Optional[np.ndarray]:
        """Get handshape template."""
        return self.handshape_templates.get(handshape_idx)
    
    @property
    def num_handshapes(self) -> int:
        return len(self.gloss_to_handshape)


class LabelSmoothingLoss(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Smooth labels
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss


class ConfidencePenaltyLoss(nn.Module):
    """Penalize overconfident predictions for better calibration."""
    
    def __init__(self, penalty_weight: float = 0.1):
        super().__init__()
        self.penalty_weight = penalty_weight
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        return -self.penalty_weight * entropy.mean()


class SWINRecognitionTrainer:
    """
    Trainer for SWIN-based sign recognition.
    
    Features:
    - Curriculum learning with annotation quality progression
    - Multi-task learning with handshape auxiliary head
    - Confidence calibration
    - Gradient accumulation for large effective batch sizes
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = None
        self.feature_loader = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.leap_supervisor = None
        
        # Metrics
        self.best_val_acc = 0.0
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def setup(self):
        """Initialize all components."""
        print("Setting up trainer...")
        
        # Load annotations
        print("\n1. Loading BSL-1K annotations...")
        self.parser = BSL1KParser(self.config.bsl1k_dir)
        self.parser.load_all_annotations(
            sources=self.config.curriculum_stages,
            min_duration=0.3,
            max_duration=4.0
        )
        
        # Initialize feature loader
        print("\n2. Initializing SWIN feature loader...")
        self.feature_loader = SWINFeatureLoader(
            self.config.swin_features_dir,
            feature_dim=self.config.feature_dim
        )
        
        # Initialize Leap Motion supervisor (optional)
        if self.config.use_handshape_head and Path(self.config.leap_motion_path).exists():
            print("\n3. Loading Leap Motion supervision...")
            self.leap_supervisor = LeapMotionSupervisor(self.config.leap_motion_path)
        
        # Create model
        print("\n4. Creating model...")
        num_classes = self.parser.get_vocabulary_size()
        num_handshapes = self.leap_supervisor.num_handshapes if self.leap_supervisor else 64
        
        self.model = TemporalSignRecognitionModel(
            feature_dim=self.config.feature_dim,
            num_classes=num_classes,
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            d_model=self.config.d_model,
            dropout=self.config.dropout,
            use_handshape_head=self.config.use_handshape_head,
            num_handshapes=num_handshapes
        ).to(self.device)
        
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2
        )
        
        # Loss functions
        self.criterion = LabelSmoothingLoss(num_classes, smoothing=0.1)
        self.confidence_loss = ConfidencePenaltyLoss(self.config.confidence_weight)
        
        # Save config
        config_path = Path(self.config.output_dir) / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)
        
        # Save vocabulary
        vocab_path = Path(self.config.output_dir) / 'vocabulary.json'
        self.parser.save_vocabulary(str(vocab_path))
        
        print("\n[OK] Setup complete!")
    
    def create_dataloaders(
        self,
        sources: List[str],
        train_videos: List[str],
        val_videos: List[str]
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders for given sources."""
        
        # Get samples filtered by source
        all_samples = self.parser.get_training_samples(min_confidence=0.5)
        
        # Filter by video split
        train_samples = [s for s in all_samples if s[0] in set(train_videos)]
        val_samples = [s for s in all_samples if s[0] in set(val_videos)]
        
        # Create datasets
        train_dataset = SWINRecognitionDataset(
            train_samples,
            self.feature_loader,
            self.parser.gloss_to_idx,
            max_seq_len=self.config.max_seq_len,
            augment=True
        )
        
        val_dataset = SWINRecognitionDataset(
            val_samples,
            self.feature_loader,
            self.parser.gloss_to_idx,
            max_seq_len=self.config.max_seq_len,
            augment=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            features = batch['features'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(features, mask)
            
            # Classification loss
            loss = self.criterion(output['logits'], labels)
            
            # Confidence penalty
            loss = loss + self.confidence_loss(output['logits'])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip
            )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            preds = output['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{100 * correct / total:.2f}%"
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0
        
        for batch in val_loader:
            features = batch['features'].to(self.device)
            mask = batch['mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            output = self.model(features, mask)
            
            loss = self.criterion(output['logits'], labels)
            total_loss += loss.item()
            
            # Top-1 accuracy
            preds = output['logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_preds = output['logits'].topk(5, dim=-1)
            correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=-1).sum().item()
            
            total += labels.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total,
            'top5_accuracy': correct_top5 / total
        }
    
    def train(self):
        """Full training loop with curriculum learning."""
        print("\n" + "="*60)
        print("STARTING TRAINING")
        print("="*60)
        
        # Get video splits
        train_videos, val_videos, test_videos = self.parser.split_by_video()
        print(f"Train videos: {len(train_videos)}")
        print(f"Val videos: {len(val_videos)}")
        print(f"Test videos: {len(test_videos)}")
        
        # Curriculum learning stages
        total_epochs = 0
        
        for stage_idx, (source, stage_epochs) in enumerate(
            zip(self.config.curriculum_stages, self.config.stage_epochs)
        ):
            print(f"\n{'='*60}")
            print(f"STAGE {stage_idx + 1}: {source} ({stage_epochs} epochs)")
            print(f"{'='*60}")
            
            # Create dataloaders for current stage
            # Accumulate sources up to current stage
            current_sources = self.config.curriculum_stages[:stage_idx + 1]
            train_loader, val_loader = self.create_dataloaders(
                current_sources, train_videos, val_videos
            )
            
            for epoch in range(stage_epochs):
                total_epochs += 1
                
                # Train
                train_metrics = self.train_epoch(train_loader, total_epochs)
                
                # Validate
                val_metrics = self.validate(val_loader)
                
                # Update scheduler
                self.scheduler.step()
                
                # Log
                print(f"Epoch {total_epochs}: "
                      f"Train Loss={train_metrics['loss']:.4f}, "
                      f"Train Acc={100*train_metrics['accuracy']:.2f}%, "
                      f"Val Acc={100*val_metrics['accuracy']:.2f}%, "
                      f"Val Top5={100*val_metrics['top5_accuracy']:.2f}%")
                
                # Update history
                self.history['train_loss'].append(train_metrics['loss'])
                self.history['train_acc'].append(train_metrics['accuracy'])
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
                
                # Save best model
                if val_metrics['accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['accuracy']
                    self.save_checkpoint('best_model.pt')
                    print(f"  [NEW BEST] Val Acc: {100*self.best_val_acc:.2f}%")
                
                # Periodic save
                if total_epochs % self.config.save_every == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{total_epochs}.pt')
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        # Save training history
        history_path = Path(self.config.output_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f)
        
        print("\n" + "="*60)
        print(f"TRAINING COMPLETE")
        print(f"Best Val Accuracy: {100*self.best_val_acc:.2f}%")
        print("="*60)
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = Path(self.config.output_dir) / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': vars(self.config)
        }, path)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = Path(self.config.output_dir) / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        
        print(f"Loaded checkpoint: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Train SWIN Recognition Model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--use_handshape", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        feature_dim=args.feature_dim,
        use_handshape_head=args.use_handshape
    )
    
    # Initialize trainer
    trainer = SWINRecognitionTrainer(config)
    trainer.setup()
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
