"""
=============================================================================
SIGNLYTIC AI - SIGNAVATARS TRAINING PIPELINE
=============================================================================

Complete training scripts for all enhanced models using SignAvatars data.

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

# Import our models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
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
# DATASETS
# =============================================================================

class PoseRecognitionDataset(Dataset):
    """
    Dataset for pose-based sign recognition using SignAvatars.
    """
    def __init__(
        self,
        data_root: str = "data/signavatars",
        dataset_name: str = "wlasl",
        split: str = "train",
        max_samples: int = None,
        max_seq_len: int = 200,
        augment: bool = True,
    ):
        self.adapter = SignAvatarsAdapter(data_root)
        self.max_seq_len = max_seq_len
        self.augment = augment and (split == "train")
        
        # Load all samples
        all_samples = self.adapter.load_dataset(dataset_name, max_samples)
        
        # Create label mapping from sample IDs
        unique_ids = sorted(set(s.sample_id for s in all_samples))
        self.label_map = {sid: i for i, sid in enumerate(unique_ids)}
        self.num_classes = len(self.label_map)
        
        # Split data
        random.seed(42)
        random.shuffle(all_samples)
        
        n = len(all_samples)
        if split == "train":
            self.samples = all_samples[:int(0.8 * n)]
        elif split == "val":
            self.samples = all_samples[int(0.8 * n):int(0.9 * n)]
        else:  # test
            self.samples = all_samples[int(0.9 * n):]
        
        logger.info(f"Loaded {len(self.samples)} samples for {split} split ({self.num_classes} classes)")
    
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
            noise = np.random.randn(*features.shape) * 0.01
            features = features + noise
        
        # Random temporal shift (drop first/last frames)
        if random.random() < 0.3 and features.shape[0] > 10:
            shift = random.randint(1, 3)
            if random.random() < 0.5:
                features = features[shift:]
            else:
                features = features[:-shift]
        
        return features
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get recognition features
        features = self.adapter.get_recognition_features(sample)  # (T, 169)
        features = self._augment(features)
        
        # Pad/truncate to max_seq_len
        T = features.shape[0]
        if T > self.max_seq_len:
            # Random crop during training, center crop otherwise
            if self.augment:
                start = random.randint(0, T - self.max_seq_len)
            else:
                start = (T - self.max_seq_len) // 2
            features = features[start:start + self.max_seq_len]
            mask = np.ones(self.max_seq_len, dtype=bool)
        else:
            # Pad
            pad_len = self.max_seq_len - T
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
            mask = np.zeros(self.max_seq_len, dtype=bool)
            mask[:T] = True
        
        label = self.label_map[sample.sample_id]
        
        return {
            'poses': torch.from_numpy(features).float(),
            'mask': torch.from_numpy(mask),
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': sample.sample_id,
        }


class MotionGenerationDataset(Dataset):
    """
    Dataset for text/gloss to motion generation.
    """
    def __init__(
        self,
        data_root: str = "data/signavatars",
        gloss_vocab_path: str = "data/gloss_vocabulary.json",
        dataset_name: str = "wlasl",
        split: str = "train",
        max_samples: int = None,
        max_motion_len: int = 200,
        max_text_len: int = 50,
    ):
        self.adapter = SignAvatarsAdapter(data_root)
        self.max_motion_len = max_motion_len
        self.max_text_len = max_text_len
        
        # Load samples
        all_samples = self.adapter.load_dataset(dataset_name, max_samples)
        
        # Build or load vocabulary
        if Path(gloss_vocab_path).exists():
            with open(gloss_vocab_path, 'r') as f:
                self.vocab = json.load(f)
        else:
            # Create vocabulary from sample IDs (glosses)
            glosses = sorted(set(s.sample_id.upper() for s in all_samples))
            self.vocab = {
                '<PAD>': 0,
                '<BOS>': 1,
                '<EOS>': 2,
                '<UNK>': 3,
            }
            for i, g in enumerate(glosses):
                self.vocab[g] = i + 4
            
            # Save vocabulary
            Path(gloss_vocab_path).parent.mkdir(parents=True, exist_ok=True)
            with open(gloss_vocab_path, 'w') as f:
                json.dump(self.vocab, f)
        
        self.vocab_size = len(self.vocab)
        
        # Split
        random.seed(42)
        random.shuffle(all_samples)
        n = len(all_samples)
        if split == "train":
            self.samples = all_samples[:int(0.8 * n)]
        elif split == "val":
            self.samples = all_samples[int(0.8 * n):int(0.9 * n)]
        else:
            self.samples = all_samples[int(0.9 * n):]
        
        logger.info(f"Motion dataset: {len(self.samples)} samples, vocab size: {self.vocab_size}")
    
    def __len__(self):
        return len(self.samples)
    
    def tokenize(self, text: str) -> List[int]:
        """Convert text/gloss to token IDs"""
        tokens = [self.vocab.get('<BOS>', 1)]
        for word in text.upper().split():
            tokens.append(self.vocab.get(word, self.vocab.get('<UNK>', 3)))
        tokens.append(self.vocab.get('<EOS>', 2))
        return tokens
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get motion sequence
        motion = self.adapter.get_motion_sequence(sample)
        pose_seq = np.concatenate([
            motion['body_pose'],
            motion['left_hand_pose'],
            motion['right_hand_pose'],
            motion['global_orient'],
            np.zeros((motion['num_frames'], 13)),  # Pad to 169
        ], axis=1)[:, :169]
        
        # Pad/truncate motion
        T = pose_seq.shape[0]
        if T > self.max_motion_len:
            pose_seq = pose_seq[:self.max_motion_len]
        else:
            pad_len = self.max_motion_len - T
            pose_seq = np.pad(pose_seq, ((0, pad_len), (0, 0)), mode='constant')
        
        # Tokenize gloss (using sample_id as gloss)
        tokens = self.tokenize(sample.sample_id)
        if len(tokens) > self.max_text_len:
            tokens = tokens[:self.max_text_len]
        else:
            tokens = tokens + [0] * (self.max_text_len - len(tokens))
        
        return {
            'text_ids': torch.tensor(tokens, dtype=torch.long),
            'motion': torch.from_numpy(pose_seq).float(),
            'motion_length': torch.tensor(min(T, self.max_motion_len), dtype=torch.long),
        }


class ContinuousSignDataset(Dataset):
    """
    Dataset for continuous sign recognition (PHOENIX).
    """
    def __init__(
        self,
        data_root: str = "data/signavatars",
        split: str = "train",
        max_seq_len: int = 512,
    ):
        self.adapter = SignAvatarsAdapter(data_root)
        self.max_seq_len = max_seq_len
        
        # Load PHOENIX dataset (has longer sequences)
        all_samples = self.adapter.load_dataset("phoenix")
        
        # For PHOENIX, we'd need actual gloss annotations
        # For now, we'll simulate with sample IDs
        self.gloss_vocab = {}
        for sample in all_samples:
            # Split video name into pseudo-glosses
            parts = sample.sample_id.replace('-', '_').split('_')
            for part in parts:
                if part not in self.gloss_vocab:
                    self.gloss_vocab[part] = len(self.gloss_vocab)
        
        self.num_classes = len(self.gloss_vocab)
        
        # Split
        random.seed(42)
        random.shuffle(all_samples)
        n = len(all_samples)
        if split == "train":
            self.samples = all_samples[:int(0.8 * n)]
        elif split == "val":
            self.samples = all_samples[int(0.8 * n):int(0.9 * n)]
        else:
            self.samples = all_samples[int(0.9 * n):]
        
        logger.info(f"Continuous dataset: {len(self.samples)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        features = self.adapter.get_recognition_features(sample)
        
        # Pad/truncate
        T = features.shape[0]
        if T > self.max_seq_len:
            features = features[:self.max_seq_len]
            T = self.max_seq_len
        else:
            pad_len = self.max_seq_len - T
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
        
        # Create pseudo-targets from sample ID
        parts = sample.sample_id.replace('-', '_').split('_')
        targets = [self.gloss_vocab.get(p, 0) for p in parts if p in self.gloss_vocab]
        target_len = len(targets)
        
        # Pad targets
        max_target_len = 50
        if len(targets) > max_target_len:
            targets = targets[:max_target_len]
        else:
            targets = targets + [0] * (max_target_len - len(targets))
        
        return {
            'poses': torch.from_numpy(features).float(),
            'targets': torch.tensor(targets, dtype=torch.long),
            'input_length': torch.tensor(T, dtype=torch.long),
            'target_length': torch.tensor(target_len, dtype=torch.long),
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_pose_recognizer(
    data_root: str = "data/signavatars",
    dataset_name: str = "wlasl",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "models/pose_recognition",
):
    """Train pose-based sign recognizer"""
    
    logger.info("="*70)
    logger.info("TRAINING POSE-BASED SIGN RECOGNIZER")
    logger.info("="*70)
    
    # Create datasets
    train_dataset = PoseRecognitionDataset(data_root, dataset_name, "train")
    val_dataset = PoseRecognitionDataset(data_root, dataset_name, "val", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = PoseSignRecognizer(num_classes=train_dataset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0
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
                _, top5 = logits.topk(5, dim=-1)
                val_correct_top5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        
        val_acc = val_correct / val_total
        val_acc_top5 = val_correct_top5 / val_total
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                   f"Train Acc={train_correct/train_total*100:.2f}%, "
                   f"Val Acc={val_acc*100:.2f}%, Val Top-5={val_acc_top5*100:.2f}%")
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_acc_top5': val_acc_top5,
                'num_classes': train_dataset.num_classes,
                'label_map': train_dataset.label_map,
            }, save_path / 'best_model.pt')
            logger.info(f"  -> Saved new best model (acc={val_acc*100:.2f}%)")
        
        scheduler.step()
    
    logger.info(f"Training complete. Best validation accuracy: {best_acc*100:.2f}%")
    return model


def train_motion_generator(
    data_root: str = "data/signavatars",
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "models/motion_generation",
):
    """Train sign motion generator"""
    
    logger.info("="*70)
    logger.info("TRAINING SIGN MOTION GENERATOR")
    logger.info("="*70)
    
    # Create datasets
    train_dataset = MotionGenerationDataset(data_root, split="train")
    val_dataset = MotionGenerationDataset(data_root, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = SignMotionGenerator(vocab_size=train_dataset.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    
    # Training
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            text_ids = batch['text_ids'].to(device)
            motion = batch['motion'].to(device)
            
            optimizer.zero_grad()
            output = model(text_ids, motion)
            loss = criterion(output['motion'], motion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch['text_ids'].to(device)
                motion = batch['motion'].to(device)
                output = model(text_ids, motion)
                val_loss += criterion(output['motion'], motion).item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'vocab_size': train_dataset.vocab_size,
                'vocab': train_dataset.vocab,
            }, save_path / 'best_model.pt')
            logger.info(f"  -> Saved new best model (loss={val_loss:.4f})")
        
        scheduler.step()
    
    return model


def train_continuous_recognizer(
    data_root: str = "data/signavatars",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "models/continuous_recognition",
):
    """Train CTC-based continuous recognizer"""
    
    logger.info("="*70)
    logger.info("TRAINING CONTINUOUS SIGN RECOGNIZER (CTC)")
    logger.info("="*70)
    
    train_dataset = ContinuousSignDataset(data_root, "train")
    val_dataset = ContinuousSignDataset(data_root, "val")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = ContinuousSignRecognizer(num_classes=train_dataset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            poses = batch['poses'].to(device)
            targets = batch['targets'].to(device)
            input_lengths = batch['input_length'].to(device)
            target_lengths = batch['target_length'].to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(poses, targets, input_lengths, target_lengths)
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                poses = batch['poses'].to(device)
                targets = batch['targets'].to(device)
                input_lengths = batch['input_length'].to(device)
                target_lengths = batch['target_length'].to(device)
                
                loss = model.compute_loss(poses, targets, input_lengths, target_lengths)
                if not torch.isnan(loss):
                    val_loss += loss.item()
        
        val_loss /= len(val_loader)
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'num_classes': train_dataset.num_classes,
            }, save_path / 'best_model.pt')
        
        scheduler.step()
    
    return model


# =============================================================================
# QUICK TRAINING SCRIPT
# =============================================================================

def quick_train_all(
    data_root: str = "data/signavatars",
    epochs: int = 10,
    device: str = "cuda",
):
    """Quick training of all models for testing"""
    
    logger.info("="*70)
    logger.info("QUICK TRAINING ALL MODELS (10 epochs each)")
    logger.info("="*70)
    
    results = {}
    
    # 1. Pose Recognition
    try:
        logger.info("\n[1/3] Training Pose Recognizer...")
        model = train_pose_recognizer(
            data_root=data_root,
            epochs=epochs,
            batch_size=32,
            device=device,
        )
        results['pose_recognition'] = "SUCCESS"
    except Exception as e:
        logger.error(f"Pose recognition failed: {e}")
        results['pose_recognition'] = f"FAILED: {e}"
    
    # 2. Motion Generation
    try:
        logger.info("\n[2/3] Training Motion Generator...")
        model = train_motion_generator(
            data_root=data_root,
            epochs=epochs,
            batch_size=16,
            device=device,
        )
        results['motion_generation'] = "SUCCESS"
    except Exception as e:
        logger.error(f"Motion generation failed: {e}")
        results['motion_generation'] = f"FAILED: {e}"
    
    # 3. Continuous Recognition
    try:
        logger.info("\n[3/3] Training Continuous Recognizer...")
        model = train_continuous_recognizer(
            data_root=data_root,
            epochs=epochs,
            batch_size=8,
            device=device,
        )
        results['continuous_recognition'] = "SUCCESS"
    except Exception as e:
        logger.error(f"Continuous recognition failed: {e}")
        results['continuous_recognition'] = f"FAILED: {e}"
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)
    for task, status in results.items():
        logger.info(f"  {task}: {status}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train SignAvatars models")
    parser.add_argument("--task", type=str, default="pose", 
                       choices=["pose", "motion", "ctc", "all", "quick"],
                       help="Which task to train")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_root", type=str, default="data/signavatars")
    
    args = parser.parse_args()
    
    if args.task == "pose":
        train_pose_recognizer(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    elif args.task == "motion":
        train_motion_generator(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    elif args.task == "ctc":
        train_continuous_recognizer(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    elif args.task == "quick":
        quick_train_all(
            data_root=args.data_root,
            epochs=10,
            device=args.device,
        )
    elif args.task == "all":
        quick_train_all(
            data_root=args.data_root,
            epochs=args.epochs,
            device=args.device,
        )

