"""
=============================================================================
SIGNLYTIC AI - SIGNAVATARS TRAINING PIPELINE (v2)
=============================================================================

Uses actual gloss labels from WLASL annotations.

Author: Oke Iyanuoluwa Enoch
=============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import logging
import random
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.enhanced_models import (
    PoseSignRecognizer,
    MultiModalFusion, 
    SignMotionGenerator,
    ContinuousSignRecognizer,
    HandPoseRefiner,
    EnhancedBSLRecognizer,
)
from src.motion.signavatars_adapter import SignAvatarsAdapter


# =============================================================================
# DATASETS WITH GLOSS LABELS
# =============================================================================

class PoseRecognitionDataset(Dataset):
    """
    Dataset for pose-based sign recognition using SignAvatars + WLASL glosses.
    """
    def __init__(
        self,
        data_root: str = "data/signavatars",
        dataset_name: str = "wlasl",
        split: str = "train",
        max_samples: int = None,
        max_seq_len: int = 200,
        augment: bool = True,
        min_samples_per_class: int = 2,
    ):
        self.adapter = SignAvatarsAdapter(data_root)
        self.max_seq_len = max_seq_len
        self.augment = augment and (split == "train")
        
        # Load gloss mapping
        gloss_mapping_path = Path(data_root) / "wlasl" / "video_to_gloss.json"
        if gloss_mapping_path.exists():
            with open(gloss_mapping_path, 'r') as f:
                self.video_to_gloss = json.load(f)
            logger.info(f"Loaded {len(self.video_to_gloss)} video-to-gloss mappings")
        else:
            raise FileNotFoundError(f"Gloss mapping not found at {gloss_mapping_path}. Run download_wlasl_labels.py first.")
        
        # Load all samples
        all_samples = self.adapter.load_dataset(dataset_name, max_samples)
        
        # Map samples to glosses
        samples_with_gloss = []
        for sample in all_samples:
            gloss = self.video_to_gloss.get(sample.sample_id)
            if gloss:
                sample.gloss = gloss
                samples_with_gloss.append(sample)
        
        logger.info(f"Samples with gloss labels: {len(samples_with_gloss)}/{len(all_samples)}")
        
        # Count samples per gloss
        gloss_counts = {}
        for sample in samples_with_gloss:
            gloss_counts[sample.gloss] = gloss_counts.get(sample.gloss, 0) + 1
        
        # Filter glosses with minimum samples
        valid_glosses = {g for g, c in gloss_counts.items() if c >= min_samples_per_class}
        samples_with_gloss = [s for s in samples_with_gloss if s.gloss in valid_glosses]
        
        logger.info(f"Glosses with >= {min_samples_per_class} samples: {len(valid_glosses)}")
        
        # Create label mapping
        unique_glosses = sorted(valid_glosses)
        self.label_map = {gloss: i for i, gloss in enumerate(unique_glosses)}
        self.idx_to_label = {i: gloss for gloss, i in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        # Split data (stratified by gloss)
        random.seed(42)
        
        # Group by gloss
        gloss_to_samples = {}
        for sample in samples_with_gloss:
            if sample.gloss not in gloss_to_samples:
                gloss_to_samples[sample.gloss] = []
            gloss_to_samples[sample.gloss].append(sample)
        
        # Stratified split
        train_samples, val_samples, test_samples = [], [], []
        for gloss, samples in gloss_to_samples.items():
            random.shuffle(samples)
            n = len(samples)
            n_train = max(1, int(0.7 * n))
            n_val = max(1, int(0.15 * n))
            
            train_samples.extend(samples[:n_train])
            val_samples.extend(samples[n_train:n_train + n_val])
            test_samples.extend(samples[n_train + n_val:])
        
        if split == "train":
            self.samples = train_samples
        elif split == "val":
            self.samples = val_samples
        else:
            self.samples = test_samples
        
        random.shuffle(self.samples)
        
        logger.info(f"Split '{split}': {len(self.samples)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def _augment(self, features: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        if not self.augment:
            return features
        
        # Random scaling
        if random.random() < 0.5:
            scale = 1.0 + random.uniform(-0.1, 0.1)
            features = features * scale
        
        # Random noise
        if random.random() < 0.3:
            noise = np.random.randn(*features.shape) * 0.02
            features = features + noise
        
        # Random temporal shift
        if random.random() < 0.3 and features.shape[0] > 10:
            shift = random.randint(1, 3)
            if random.random() < 0.5:
                features = features[shift:]
            else:
                features = features[:-shift]
        
        # Random temporal scaling (speed up/slow down)
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            new_len = int(features.shape[0] * scale)
            if new_len > 5:
                indices = np.linspace(0, features.shape[0] - 1, new_len).astype(int)
                features = features[indices]
        
        return features.astype(np.float32)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get recognition features
        features = self.adapter.get_recognition_features(sample)
        features = self._augment(features)
        
        # Pad/truncate
        T = features.shape[0]
        if T > self.max_seq_len:
            if self.augment:
                start = random.randint(0, T - self.max_seq_len)
            else:
                start = (T - self.max_seq_len) // 2
            features = features[start:start + self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=bool)
        else:
            pad_len = self.max_seq_len - T
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            mask = np.zeros(self.max_seq_len, dtype=bool)
            mask[:T] = True
        
        label = self.label_map[sample.gloss]
        
        return {
            'poses': torch.from_numpy(features).float(),
            'mask': torch.from_numpy(mask),
            'label': torch.tensor(label, dtype=torch.long),
            'gloss': sample.gloss,
        }


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_pose_recognizer(
    data_root: str = "data/signavatars",
    dataset_name: str = "wlasl",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "models/pose_recognition",
    min_samples_per_class: int = 2,
):
    """Train pose-based sign recognizer with gloss labels"""
    
    logger.info("="*70)
    logger.info("TRAINING POSE-BASED SIGN RECOGNIZER (with glosses)")
    logger.info("="*70)
    
    # Create datasets
    train_dataset = PoseRecognitionDataset(
        data_root, dataset_name, "train", 
        min_samples_per_class=min_samples_per_class
    )
    val_dataset = PoseRecognitionDataset(
        data_root, dataset_name, "val",
        augment=False,
        min_samples_per_class=min_samples_per_class
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Create model
    model = PoseSignRecognizer(num_classes=train_dataset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Training loop
    best_acc = 0
    best_top5 = 0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            poses = batch['poses'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(poses, masks)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.shape[0]
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{train_correct/train_total*100:.1f}%'
            })
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_correct_top5 = 0
        
        with torch.no_grad():
            for batch in val_loader:
                poses = batch['poses'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(poses, masks)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]
                
                # Top-5
                k = min(5, logits.shape[-1])
                _, top5 = logits.topk(k, dim=-1)
                val_correct_top5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_acc_top5 = val_correct_top5 / val_total if val_total > 0 else 0
        
        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss={train_loss/len(train_loader):.4f}, "
            f"Train Acc={train_correct/train_total*100:.2f}%, "
            f"Val Top-1={val_acc*100:.2f}%, "
            f"Val Top-5={val_acc_top5*100:.2f}%"
        )
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_top5 = val_acc_top5
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_acc_top5': val_acc_top5,
                'num_classes': train_dataset.num_classes,
                'label_map': train_dataset.label_map,
                'idx_to_label': train_dataset.idx_to_label,
            }, save_path / 'best_model.pt')
            logger.info(f"  -> Saved best model (Top-1={val_acc*100:.2f}%, Top-5={val_acc_top5*100:.2f}%)")
        
        scheduler.step()
    
    logger.info("="*70)
    logger.info(f"Training complete!")
    logger.info(f"Best Val Top-1: {best_acc*100:.2f}%")
    logger.info(f"Best Val Top-5: {best_top5*100:.2f}%")
    logger.info("="*70)
    
    return model


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SignAvatars pose recognizer")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default="data/signavatars")
    parser.add_argument("--min_samples", type=int, default=2, help="Min samples per class")
    
    args = parser.parse_args()
    
    train_pose_recognizer(
        data_root=args.data_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        min_samples_per_class=args.min_samples,
    )
