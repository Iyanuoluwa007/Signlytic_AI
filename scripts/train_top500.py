"""
SWIN Recognition Training - Top 500 Classes

Focused training on most frequent glosses for better accuracy.
Expected: 20-40% Top-1, 50-70% Top-5

Usage:
    python scripts/train_top500.py --epochs 30 --batch_size 64
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import json
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import random
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from src.data.bsl1k_parser import BSL1KParser, get_swin_video_ids


# ============================================================
# Model
# ============================================================

class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_classes: int = 500
    ):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 256, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Small init for classifier
        nn.init.normal_(self.classifier.weight, std=0.01)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        T_pos = min(T, self.pos_encoding.size(1))
        x[:, :T_pos] = x[:, :T_pos] + self.pos_encoding[:, :T_pos]
        
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        x = self.norm(x)
        x = self.dropout(x)
        x = torch.clamp(x, -10, 10)
        
        return self.classifier(x)


# ============================================================
# Feature Manager
# ============================================================

class FeatureManager:
    def __init__(self, swin_dir: str, video_ids: Set[str]):
        self.swin_dir = Path(swin_dir)
        self.valid_ids = set()
        
        print(f"Validating {len(video_ids)} feature files...")
        for vid in tqdm(video_ids, desc="Checking", ncols=80, leave=False):
            if (self.swin_dir / f"{vid}.npy").exists():
                self.valid_ids.add(vid)
        print(f"Valid: {len(self.valid_ids)} / {len(video_ids)}")
    
    def get(self, video_id: str) -> Optional[np.ndarray]:
        if video_id not in self.valid_ids:
            return None
        try:
            return np.load(self.swin_dir / f"{video_id}.npy", mmap_mode='r')
        except:
            return None


# ============================================================
# Dataset
# ============================================================

class SWINDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int, int, int]],
        feature_manager: FeatureManager,
        max_seq_len: int = 64,
        augment: bool = True
    ):
        self.samples = samples
        self.fm = feature_manager
        self.max_seq_len = max_seq_len
        self.augment = augment
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        video_id, start_idx, end_idx, gloss_idx = self.samples[idx]
        
        features = self.fm.get(video_id)
        
        if features is None:
            return {
                'features': torch.zeros(self.max_seq_len, 768),
                'mask': torch.ones(self.max_seq_len, dtype=torch.bool),
                'label': torch.tensor(gloss_idx, dtype=torch.long)
            }
        
        T, D = features.shape
        start_idx = max(0, min(start_idx, T - 1))
        end_idx = max(start_idx + 1, min(end_idx, T))
        
        if self.augment:
            jitter = random.randint(-2, 2)
            start_idx = max(0, start_idx + jitter)
            end_idx = min(T, end_idx + jitter)
            if end_idx <= start_idx:
                end_idx = start_idx + 1
        
        segment = np.array(features[start_idx:end_idx], dtype=np.float32)
        segment = np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
        segment = np.clip(segment, -50, 50)
        
        seq_len = len(segment)
        
        if seq_len < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - seq_len, D), dtype=np.float32)
            segment = np.concatenate([segment, padding], axis=0)
            mask = np.concatenate([
                np.zeros(seq_len, dtype=bool),
                np.ones(self.max_seq_len - seq_len, dtype=bool)
            ])
        else:
            segment = segment[:self.max_seq_len]
            mask = np.zeros(self.max_seq_len, dtype=bool)
        
        return {
            'features': torch.from_numpy(segment),
            'mask': torch.from_numpy(mask),
            'label': torch.tensor(gloss_idx, dtype=torch.long)
        }


# ============================================================
# Training
# ============================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float,
    epoch: int
) -> Dict[str, float]:
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"E{epoch+1:02d} Train", ncols=100, leave=False)
    
    for batch in pbar:
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(features, mask)
        
        if torch.isnan(logits).any():
            continue
        
        loss = criterion(logits, labels)
        
        if torch.isnan(loss) or torch.isinf(loss):
            continue
        
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            optimizer.zero_grad()
            continue
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix_str(f"L={loss.item():.3f} A={100*correct/max(1,total):.1f}%")
    
    return {
        'loss': total_loss / max(1, len(train_loader)),
        'accuracy': correct / max(1, total)
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.eval()
    
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0
    
    for batch in tqdm(val_loader, desc="     Val ", ncols=100, leave=False):
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(features, mask)
        
        if torch.isnan(logits).any():
            continue
        
        loss = criterion(logits, labels)
        
        if not torch.isnan(loss):
            total_loss += loss.item()
        
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        
        _, top5_preds = logits.topk(min(5, logits.size(-1)), dim=-1)
        correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=-1).sum().item()
        total += labels.size(0)
    
    return {
        'loss': total_loss / max(1, len(val_loader)),
        'accuracy': correct / max(1, total),
        'top5_accuracy': correct_top5 / max(1, total)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--top_k", type=int, default=500, help="Number of top classes to use")
    parser.add_argument("--min_samples", type=int, default=50, help="Minimum samples per class")
    parser.add_argument("--no_balance", action="store_true", help="Disable balanced sampling")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    
    # Paths
    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    swin_dir = project_root / "data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32"
    bsl1k_dir = project_root / "data/BSL-1K"
    output_dir = project_root / "models/swin_recognition_top500"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all annotations
    print("\nLoading annotations...")
    bsl_parser = BSL1KParser(str(bsl1k_dir))
    bsl_parser.load_annotations()  # Load all sources
    
    # Get video IDs
    video_ids = get_swin_video_ids(str(swin_dir))
    feature_manager = FeatureManager(str(swin_dir), video_ids)
    
    # Get all samples using parser's method
    print("\nGetting all samples...")
    all_samples = bsl_parser.get_training_samples(
        min_confidence=0.5,
        available_videos=feature_manager.valid_ids
    )
    print(f"Total samples with valid videos: {len(all_samples):,}")
    
    # Count glosses (samples are tuples: video_id, start_idx, end_idx, gloss_idx)
    # Need to get gloss names from parser
    gloss_counts = Counter()
    for video_id, start_idx, end_idx, gloss_idx in all_samples:
        gloss = bsl_parser.idx_to_gloss.get(gloss_idx, f"UNK_{gloss_idx}")
        gloss_counts[gloss] += 1
    
    print(f"Unique glosses: {len(gloss_counts):,}")
    
    # Filter to top K with minimum samples
    top_glosses = [
        gloss for gloss, count in gloss_counts.most_common()
        if count >= args.min_samples
    ][:args.top_k]
    
    print(f"\nSelected {len(top_glosses)} classes with >= {args.min_samples} samples")
    
    if len(top_glosses) == 0:
        print("[ERROR] No classes found with minimum samples. Try lowering --min_samples")
        sys.exit(1)
    
    print(f"Sample counts: min={gloss_counts[top_glosses[-1]]}, max={gloss_counts[top_glosses[0]]}")
    
    # Create new vocabulary
    gloss_to_idx = {g: i for i, g in enumerate(top_glosses)}
    idx_to_gloss = {i: g for g, i in gloss_to_idx.items()}
    
    # Save vocabulary
    vocab_path = output_dir / "vocabulary.json"
    with open(vocab_path, 'w') as f:
        json.dump({
            'gloss_to_idx': gloss_to_idx,
            'idx_to_gloss': {str(k): v for k, v in idx_to_gloss.items()},
            'num_classes': len(top_glosses),
            'min_samples': args.min_samples
        }, f, indent=2)
    print(f"Saved vocabulary to: {vocab_path}")
    
    # Filter annotations and create samples with new indices
    # First, create mapping from old gloss to new index
    old_idx_to_gloss = bsl_parser.idx_to_gloss
    
    samples = []
    for video_id, start_idx, end_idx, old_gloss_idx in all_samples:
        gloss = old_idx_to_gloss.get(old_gloss_idx, None)
        if gloss and gloss in gloss_to_idx:
            new_gloss_idx = gloss_to_idx[gloss]
            samples.append((video_id, start_idx, end_idx, new_gloss_idx))
    
    print(f"Filtered samples: {len(samples):,}")
    
    if len(samples) == 0:
        print("[ERROR] No samples after filtering. Check vocabulary mapping.")
        sys.exit(1)
    
    # Split by video
    all_videos = list(set(s[0] for s in samples))
    random.seed(42)
    random.shuffle(all_videos)
    
    n_train = int(0.85 * len(all_videos))
    n_val = int(0.10 * len(all_videos))
    
    train_videos = set(all_videos[:n_train])
    val_videos = set(all_videos[n_train:n_train + n_val])
    test_videos = set(all_videos[n_train + n_val:])
    
    train_samples = [s for s in samples if s[0] in train_videos]
    val_samples = [s for s in samples if s[0] in val_videos]
    
    print(f"\nTrain: {len(train_samples):,} | Val: {len(val_samples):,}")
    
    # Class weights for balanced sampling (optional)
    if not args.no_balance:
        train_labels = [s[3] for s in train_samples]
        class_counts = Counter(train_labels)
        weights = [1.0 / class_counts[label] for label in train_labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        shuffle = False
        balance_str = "ON"
    else:
        sampler = None
        shuffle = True
        balance_str = "OFF"
    
    # Create datasets
    train_dataset = SWINDataset(train_samples, feature_manager, max_seq_len=64, augment=True)
    val_dataset = SWINDataset(val_samples, feature_manager, max_seq_len=64, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # Build model
    print("\nBuilding model...")
    model = TemporalTransformerEncoder(
        num_classes=len(top_glosses),
        dropout=args.dropout
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Classes: {len(top_glosses)}")
    
    # Optimizer with class-balanced loss
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/100)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"\nBatch: {args.batch_size} | LR: {args.lr} | Steps/epoch: {len(train_loader):,}")
    print(f"Gradient clip: 1.0 | Balanced sampling: {balance_str}")
    print("="*70)
    
    best_acc = 0.0
    best_top5 = 0.0
    patience = 10
    no_improve = 0
    
    for epoch in range(args.epochs):
        t0 = time.time()
        
        train_m = train_epoch(model, train_loader, optimizer, criterion, device, 1.0, epoch)
        val_m = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        elapsed = time.time() - t0
        
        is_best = val_m['accuracy'] > best_acc
        if is_best:
            best_acc = val_m['accuracy']
            best_top5 = val_m['top5_accuracy']
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_acc,
                'best_val_top5': best_top5,
                'num_classes': len(top_glosses),
                'gloss_to_idx': gloss_to_idx,
                'idx_to_gloss': idx_to_gloss,
            }, output_dir / "best_model.pt")
        else:
            no_improve += 1
        
        marker = " *BEST*" if is_best else ""
        print(f"E{epoch+1:02d} | {elapsed:.0f}s | "
              f"Train: L={train_m['loss']:.3f} A={100*train_m['accuracy']:.1f}% | "
              f"Val: {100*val_m['accuracy']:.1f}% T5={100*val_m['top5_accuracy']:.1f}%{marker}")
        
        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f"ckpt_e{epoch+1}.pt")
    
    print(f"\n{'='*70}")
    print(f"COMPLETE: Best {100*best_acc:.2f}% Top1, {100*best_top5:.2f}% Top5")
    print(f"Model saved to: {output_dir / 'best_model.pt'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
