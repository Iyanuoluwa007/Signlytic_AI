"""
SWIN Recognition Training Script V2 (ULTRA STABLE)

Key changes for stability:
- NO AMP (float32 only)
- Standard CrossEntropyLoss
- Very low learning rate
- Gradient monitoring
- Model weight checking

Usage:
    python scripts/train_v2.py --config configs/recognition_v2.yaml --batch_size 32
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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set
import random
import time
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

from src.data.bsl1k_parser import BSL1KParser, get_swin_video_ids


# ============================================================
# Model (with careful initialization)
# ============================================================

class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 256,  # Smaller for stability
        num_layers: int = 4,  # Fewer layers
        num_heads: int = 4,
        dropout: float = 0.3,
        num_classes: int = 1000
    ):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Smaller positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, 256, d_model))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,  # Smaller FFN
            dropout=dropout,
            activation='relu',  # ReLU more stable than GELU
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Very small initialization for classifier
        nn.init.normal_(self.classifier.weight, std=0.001)
        nn.init.zeros_(self.classifier.bias)
        
        # Initialize other layers
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and m is not self.classifier:
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
        
        # Mean pooling
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        x = self.norm(x)
        x = self.dropout(x)
        
        # Clamp before classifier for stability
        x = torch.clamp(x, -10, 10)
        
        return self.classifier(x)


def build_model(config: Dict) -> nn.Module:
    return TemporalTransformerEncoder(
        input_dim=config.get('feature_dim', 768),
        d_model=config.get('d_model', 256),
        num_layers=config.get('num_layers', 4),
        num_heads=config.get('num_heads', 4),
        dropout=config.get('dropout', 0.3),
        num_classes=config.get('num_classes', 1000)
    )


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
        
        # Convert to float32 and normalize
        segment = np.array(features[start_idx:end_idx], dtype=np.float32)
        
        # Handle bad values
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
# Training (NO AMP - pure float32)
# ============================================================

def check_model_health(model: nn.Module) -> bool:
    """Check if model weights contain NaN."""
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            return False
    return True


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gradient_clip: float,
    epoch: int,
    total_epochs: int
) -> Dict[str, float]:
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    skipped = 0
    
    pbar = tqdm(train_loader, desc=f"E{epoch+1:02d} Train", ncols=100, leave=False)
    
    for batch in pbar:
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward (float32, no autocast)
        logits = model(features, mask)
        
        # Check for NaN in output
        if torch.isnan(logits).any():
            skipped += 1
            continue
        
        loss = criterion(logits, labels)
        
        # Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            skipped += 1
            continue
        
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            skipped += 1
            optimizer.zero_grad()
            continue
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix_str(f"L={loss.item():.3f} A={100*correct/max(1,total):.1f}%")
    
    if skipped > 0:
        print(f"  [INFO] Skipped {skipped} batches")
    
    n_batches = len(train_loader) - skipped
    return {
        'loss': total_loss / max(1, n_batches),
        'accuracy': correct / max(1, total),
        'skipped': skipped
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


def run_training_phase(
    phase: int,
    config: Dict,
    model: nn.Module,
    device: torch.device,
    feature_manager: FeatureManager,
    resume_from: Optional[str] = None
) -> Dict:
    
    training_cfg = config.get("training", {})
    curriculum_cfg = training_cfg.get("curriculum", {})
    phase_cfg = curriculum_cfg.get(f"phase{phase}", {})
    
    if phase == 1:
        sources = phase_cfg.get("sources", ["EXEMPLARS", "MOUTHING"])
        epochs = phase_cfg.get("epochs", 40)
        min_confidence = phase_cfg.get("min_confidence", 0.7)
    else:
        sources = phase_cfg.get("sources", ["EXEMPLARS", "MOUTHING", "DICTIONARY", "I3D_PSEUDO_LABELS"])
        epochs = phase_cfg.get("epochs", 60)
        min_confidence = phase_cfg.get("min_confidence", 0.5)
    
    print(f"\n{'='*70}")
    print(f"PHASE {phase}: {sources}")
    print(f"Epochs: {epochs} | Min confidence: {min_confidence}")
    print(f"{'='*70}")
    
    paths = config.get("paths", {})
    bsl1k_dir = paths.get("bsl1k_dir", "D:/Signlytic_AI/code/bsl_translation_project/data/BSL-1K")
    output_dir = Path(paths.get("output_dir", "D:/Signlytic_AI/code/bsl_translation_project/models/swin_recognition"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parser = BSL1KParser(bsl1k_dir)
    parser.load_annotations(sources=sources)
    
    available_videos = feature_manager.valid_ids
    all_samples = parser.get_training_samples(min_confidence=min_confidence, available_videos=available_videos)
    
    train_videos, val_videos, _ = parser.split_by_video(train_ratio=0.85, val_ratio=0.15)
    train_set, val_set = set(train_videos), set(val_videos)
    
    train_samples = [s for s in all_samples if s[0] in train_set]
    val_samples = [s for s in all_samples if s[0] in val_set]
    print(f"Train: {len(train_samples):,} | Val: {len(val_samples):,}")
    
    # Use smaller batch for stability
    batch_size = training_cfg.get("batch_size", 32)
    max_seq_len = training_cfg.get("max_seq_len", 64)
    lr = training_cfg.get("learning_rate", 5e-5)
    gradient_clip = 0.5  # Tight clipping
    
    train_dataset = SWINDataset(train_samples, feature_manager, max_seq_len, augment=True)
    val_dataset = SWINDataset(val_samples, feature_manager, max_seq_len, augment=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size*2, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    
    # Simple optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1, eps=1e-8)
    
    # Simple scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr/100)
    
    # Standard cross entropy (most stable)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_acc = 0.0
    best_top5 = 0.0
    start_epoch = 0
    
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_epoch = ckpt.get('epoch', 0)
        best_acc = ckpt.get('best_val_acc', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    print(f"Batch: {batch_size} | LR: {lr} | Steps/epoch: {len(train_loader):,}")
    print(f"Gradient clip: {gradient_clip} | NO AMP (float32)\n")
    
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        
        # Check model health before training
        if not check_model_health(model):
            print(f"  [ERROR] Model has NaN weights! Stopping.")
            break
        
        train_m = train_epoch(model, train_loader, optimizer, criterion,
                              device, gradient_clip, epoch, epochs)
        
        # Check model health after training
        if not check_model_health(model):
            print(f"  [ERROR] Model corrupted during training! Stopping.")
            break
        
        val_m = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        elapsed = time.time() - t0
        
        is_best = val_m['accuracy'] > best_acc
        if is_best:
            best_acc = val_m['accuracy']
            best_top5 = val_m['top5_accuracy']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_acc,
                'best_val_top5': best_top5,
                'num_classes': parser.get_vocabulary_size(),
            }, output_dir / f"best_phase{phase}.pt")
        
        marker = " *BEST*" if is_best else ""
        print(f"E{epoch+1:02d} | {elapsed:.0f}s | "
              f"Train: L={train_m['loss']:.3f} A={100*train_m['accuracy']:.1f}% | "
              f"Val: {100*val_m['accuracy']:.1f}% T5={100*val_m['top5_accuracy']:.1f}%{marker}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f"ckpt_phase{phase}_e{epoch+1}.pt")
    
    print(f"\nPhase {phase}: Best {100*best_acc:.2f}% Top1, {100*best_top5:.2f}% Top5")
    return {'best_val_acc': best_acc, 'best_val_top5': best_top5}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recognition_v2.yaml")
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()
    
    cfg_path = Path(args.config)
    config = yaml.safe_load(open(cfg_path)) if cfg_path.exists() else {}
    
    if args.epochs:
        config.setdefault("training", {}).setdefault("curriculum", {}).setdefault("phase1", {})["epochs"] = args.epochs
        config["training"]["curriculum"].setdefault("phase2", {})["epochs"] = args.epochs
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("training", {})["learning_rate"] = args.lr
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory/1024**3:.1f}GB)")
    
    paths = config.get("paths", {})
    swin_dir = paths.get("swin_features", "D:/Signlytic_AI/code/bsl_translation_project/data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32")
    bsl1k_dir = paths.get("bsl1k_dir", "D:/Signlytic_AI/code/bsl_translation_project/data/BSL-1K")
    output_dir = Path(paths.get("output_dir", "D:/Signlytic_AI/code/bsl_translation_project/models/swin_recognition"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading vocabulary...")
    temp_parser = BSL1KParser(bsl1k_dir)
    temp_parser.load_annotations()
    num_classes = temp_parser.get_vocabulary_size()
    temp_parser.save_vocabulary(str(output_dir / "vocabulary.json"))
    
    print("\nInitializing feature manager...")
    video_ids = get_swin_video_ids(swin_dir)
    feature_manager = FeatureManager(swin_dir, video_ids)
    
    print("\nBuilding model...")
    model_cfg = config.get("model", {})
    model = build_model({**model_cfg, 'num_classes': num_classes}).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Classes: {num_classes:,}")
    
    if args.phase == 1:
        run_training_phase(1, config, model, device, feature_manager, args.resume)
    elif args.phase == 2:
        run_training_phase(2, config, model, device, feature_manager, args.resume)
    else:
        r1 = run_training_phase(1, config, model, device, feature_manager, args.resume)
        
        best_p1 = output_dir / "best_phase1.pt"
        if best_p1.exists():
            ckpt = torch.load(best_p1, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"\nLoaded Phase 1 best: {100*ckpt['best_val_acc']:.2f}%")
        
        r2 = run_training_phase(2, config, model, device, feature_manager)
        
        print(f"\n{'='*70}")
        print(f"COMPLETE: Phase1={100*r1['best_val_acc']:.1f}% Phase2={100*r2['best_val_acc']:.1f}%")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()