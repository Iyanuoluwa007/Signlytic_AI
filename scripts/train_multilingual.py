"""
=============================================================================
SIGNLYTIC AI - MULTI-LINGUAL POSE RECOGNIZER
=============================================================================

Combines WLASL (ASL) + French (LSF) for expanded training.

Author: Oke Iyanuoluwa Enoch
=============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from pathlib import Path
import pickle
import json
from tqdm import tqdm
from typing import Dict, List, Optional
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.enhanced_models import PoseSignRecognizer
from src.motion.signavatars_adapter import SignAvatarsAdapter


class MultiLingualPoseDataset(Dataset):
    """
    Combined dataset for WLASL (ASL) + French (LSF) signs.
    """
    def __init__(
        self,
        data_root: str = "data/signavatars",
        split: str = "train",
        max_seq_len: int = 200,
        augment: bool = True,
        min_samples_per_class: int = 2,
    ):
        self.data_root = Path(data_root)
        self.max_seq_len = max_seq_len
        self.augment = augment and (split == "train")
        self.adapter = SignAvatarsAdapter(data_root)
        
        all_samples = []
        
        # 1. Load WLASL
        logger.info("Loading WLASL (ASL)...")
        wlasl_mapping_path = self.data_root / "wlasl" / "video_to_gloss.json"
        with open(wlasl_mapping_path, 'r') as f:
            wlasl_labels = json.load(f)
        
        wlasl_samples = self.adapter.load_dataset("wlasl")
        for sample in wlasl_samples:
            if sample.sample_id in wlasl_labels:
                sample.gloss = f"ASL:{wlasl_labels[sample.sample_id].upper()}"
                all_samples.append(sample)
        logger.info(f"  WLASL: {len([s for s in all_samples if s.gloss.startswith('ASL:')])} samples")
        
        # 2. Load French LSF
        logger.info("Loading French (LSF)...")
        french_mapping_path = self.data_root / "hamnosys" / "french_gloss_mapping.json"
        if french_mapping_path.exists():
            with open(french_mapping_path, 'r') as f:
                french_labels = json.load(f)
            
            hamnosys_files = list((self.data_root / "hamnosys").rglob("*.pkl"))
            for f in hamnosys_files:
                if str(f) in french_labels:
                    try:
                        sample = self.adapter.load_sample(f, dataset="hamnosys")
                        sample.gloss = f"LSF:{french_labels[str(f)].upper()}"
                        all_samples.append(sample)
                    except Exception as e:
                        pass
        
        lsf_count = len([s for s in all_samples if s.gloss.startswith('LSF:')])
        logger.info(f"  French LSF: {lsf_count} samples")
        
        # 3. Count and filter classes
        gloss_counts = {}
        for sample in all_samples:
            gloss_counts[sample.gloss] = gloss_counts.get(sample.gloss, 0) + 1
        
        valid_glosses = {g for g, c in gloss_counts.items() if c >= min_samples_per_class}
        all_samples = [s for s in all_samples if s.gloss in valid_glosses]
        
        logger.info(f"Total samples with >= {min_samples_per_class} per class: {len(all_samples)}")
        logger.info(f"Total valid classes: {len(valid_glosses)}")
        
        # 4. Create label mapping
        unique_glosses = sorted(valid_glosses)
        self.label_map = {gloss: i for i, gloss in enumerate(unique_glosses)}
        self.idx_to_label = {i: gloss for gloss, i in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        # 5. Stratified split
        random.seed(42)
        gloss_to_samples = {}
        for sample in all_samples:
            if sample.gloss not in gloss_to_samples:
                gloss_to_samples[sample.gloss] = []
            gloss_to_samples[sample.gloss].append(sample)
        
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
        
        # Count by language
        asl_count = len([s for s in self.samples if s.gloss.startswith('ASL:')])
        lsf_count = len([s for s in self.samples if s.gloss.startswith('LSF:')])
        logger.info(f"Split '{split}': {len(self.samples)} samples ({asl_count} ASL, {lsf_count} LSF), {self.num_classes} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def _augment(self, features: np.ndarray) -> np.ndarray:
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
        
        # Random temporal scaling
        if random.random() < 0.3:
            scale = random.uniform(0.8, 1.2)
            new_len = int(features.shape[0] * scale)
            if new_len > 5:
                indices = np.linspace(0, features.shape[0] - 1, new_len).astype(int)
                features = features[indices]
        
        return features.astype(np.float32)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = self.adapter.get_recognition_features(sample)
        features = self._augment(features)
        
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


def train_multilingual(
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "models/multilingual_pose",
):
    logger.info("="*70)
    logger.info("TRAINING MULTI-LINGUAL POSE RECOGNIZER (ASL + LSF)")
    logger.info("="*70)
    
    # Create datasets
    train_dataset = MultiLingualPoseDataset(split="train")
    val_dataset = MultiLingualPoseDataset(split="val", augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Create model
    model = PoseSignRecognizer(num_classes=train_dataset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    best_acc = 0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
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
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total*100:.1f}%'})
        
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
                
                k = min(5, logits.shape[-1])
                _, top5 = logits.topk(k, dim=-1)
                val_correct_top5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_acc_top5 = val_correct_top5 / val_total if val_total > 0 else 0
        
        logger.info(f"Epoch {epoch+1}: Train Acc={train_correct/train_total*100:.2f}%, Val Top-1={val_acc*100:.2f}%, Val Top-5={val_acc_top5*100:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'val_acc_top5': val_acc_top5,
                'num_classes': train_dataset.num_classes,
                'label_map': train_dataset.label_map,
                'idx_to_label': train_dataset.idx_to_label,
            }, save_path / 'best_model.pt')
            logger.info(f"  -> Saved best (Top-1={val_acc*100:.2f}%, Top-5={val_acc_top5*100:.2f}%)")
        
        scheduler.step()
    
    logger.info("="*70)
    logger.info(f"Training complete! Best Val Top-1: {best_acc*100:.2f}%")
    logger.info("="*70)
    
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train_multilingual(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
