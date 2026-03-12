"""
Evaluate SWIN Recognition Model

Usage:
    python scripts/evaluate_model.py --model models/swin_recognition/best_phase2.pt
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import random

from src.data.bsl1k_parser import BSL1KParser, get_swin_video_ids


# ============================================================
# Model (same as training)
# ============================================================

class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.3,
        num_classes: int = 1000
    ):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 256, d_model))
        
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
# Feature Manager & Dataset
# ============================================================

class FeatureManager:
    def __init__(self, swin_dir: str, video_ids: Set[str]):
        self.swin_dir = Path(swin_dir)
        self.valid_ids = set()
        
        for vid in video_ids:
            if (self.swin_dir / f"{vid}.npy").exists():
                self.valid_ids.add(vid)
    
    def get(self, video_id: str) -> Optional[np.ndarray]:
        if video_id not in self.valid_ids:
            return None
        try:
            return np.load(self.swin_dir / f"{video_id}.npy", mmap_mode='r')
        except:
            return None


class SWINDataset(Dataset):
    def __init__(self, samples, feature_manager, max_seq_len=64):
        self.samples = samples
        self.fm = feature_manager
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_id, start_idx, end_idx, gloss_idx = self.samples[idx]
        
        features = self.fm.get(video_id)
        
        if features is None:
            return {
                'features': torch.zeros(self.max_seq_len, 768),
                'mask': torch.ones(self.max_seq_len, dtype=torch.bool),
                'label': torch.tensor(gloss_idx, dtype=torch.long),
                'video_id': video_id,
                'gloss_idx': gloss_idx
            }
        
        T, D = features.shape
        start_idx = max(0, min(start_idx, T - 1))
        end_idx = max(start_idx + 1, min(end_idx, T))
        
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
            'label': torch.tensor(gloss_idx, dtype=torch.long),
            'video_id': video_id,
            'gloss_idx': gloss_idx
        }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, idx_to_gloss):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_top5_correct = []
    all_probs = []
    
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    
    for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(features, mask)
        probs = torch.softmax(logits, dim=-1)
        
        preds = logits.argmax(dim=-1)
        _, top5_preds = logits.topk(5, dim=-1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        
        # Top-5 accuracy
        top5_correct = (top5_preds == labels.unsqueeze(1)).any(dim=-1)
        all_top5_correct.extend(top5_correct.cpu().numpy())
        
        # Per-class accuracy
        for pred, label in zip(preds.cpu().numpy(), labels.cpu().numpy()):
            per_class_total[label] += 1
            if pred == label:
                per_class_correct[label] += 1
    
    # Overall metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    top1_acc = (all_preds == all_labels).mean()
    top5_acc = np.mean(all_top5_correct)
    
    # Per-class accuracy
    per_class_acc = {}
    for cls_idx in per_class_total:
        acc = per_class_correct[cls_idx] / per_class_total[cls_idx]
        gloss = idx_to_gloss.get(cls_idx, f"CLASS_{cls_idx}")
        per_class_acc[gloss] = {
            'accuracy': acc,
            'correct': per_class_correct[cls_idx],
            'total': per_class_total[cls_idx]
        }
    
    return {
        'top1_accuracy': top1_acc,
        'top5_accuracy': top5_acc,
        'total_samples': len(all_labels),
        'per_class': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }


def print_results(results, idx_to_gloss, top_n=20):
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    
    print(f"\nOverall Metrics:")
    print(f"  Top-1 Accuracy: {100*results['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy: {100*results['top5_accuracy']:.2f}%")
    print(f"  Total Samples: {results['total_samples']:,}")
    
    # Best performing classes
    per_class = results['per_class']
    sorted_classes = sorted(
        per_class.items(),
        key=lambda x: (x[1]['accuracy'], x[1]['total']),
        reverse=True
    )
    
    print(f"\nTop {top_n} Best Performing Classes:")
    print(f"  {'Gloss':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("  " + "-"*52)
    for gloss, stats in sorted_classes[:top_n]:
        if stats['total'] >= 10:  # Only show classes with enough samples
            print(f"  {gloss:<20} {100*stats['accuracy']:>9.1f}% {stats['correct']:>10} {stats['total']:>10}")
    
    # Worst performing classes (with enough samples)
    worst_classes = [
        (g, s) for g, s in sorted_classes
        if s['total'] >= 50
    ][-top_n:]
    
    print(f"\nBottom {top_n} Performing Classes (min 50 samples):")
    print(f"  {'Gloss':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print("  " + "-"*52)
    for gloss, stats in worst_classes:
        print(f"  {gloss:<20} {100*stats['accuracy']:>9.1f}% {stats['correct']:>10} {stats['total']:>10}")
    
    # Confusion analysis
    print(f"\nClass Distribution:")
    total_classes = len(per_class)
    classes_with_correct = sum(1 for s in per_class.values() if s['correct'] > 0)
    print(f"  Total classes: {total_classes:,}")
    print(f"  Classes with at least 1 correct: {classes_with_correct:,} ({100*classes_with_correct/total_classes:.1f}%)")
    
    # Sample count distribution
    sample_counts = [s['total'] for s in per_class.values()]
    print(f"\nSamples per class:")
    print(f"  Min: {min(sample_counts)}")
    print(f"  Max: {max(sample_counts)}")
    print(f"  Mean: {np.mean(sample_counts):.1f}")
    print(f"  Median: {np.median(sample_counts):.1f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Paths
    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    swin_dir = project_root / "data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32"
    bsl1k_dir = project_root / "data/BSL-1K"
    vocab_path = project_root / "models/swin_recognition/vocabulary.json"
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    gloss_to_idx = vocab['gloss_to_idx']
    idx_to_gloss = {int(k): v for k, v in vocab['idx_to_gloss'].items()}
    print(f"Vocabulary size: {len(gloss_to_idx):,}")
    
    # Load model
    print(f"\nLoading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    num_classes = checkpoint.get('num_classes', len(gloss_to_idx))
    print(f"Model classes: {num_classes:,}")
    
    model = TemporalTransformerEncoder(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Best val accuracy from training: {100*checkpoint.get('best_val_acc', 0):.2f}%")
    print(f"Best top-5 accuracy: {100*checkpoint.get('best_val_top5', 0):.2f}%")
    
    # Load data
    print("\nLoading annotations...")
    parser_obj = BSL1KParser(str(bsl1k_dir))
    parser_obj.load_annotations()
    parser_obj.gloss_to_idx = gloss_to_idx  # Use same vocabulary
    parser_obj.idx_to_gloss = idx_to_gloss
    
    # Get features
    video_ids = get_swin_video_ids(str(swin_dir))
    feature_manager = FeatureManager(str(swin_dir), video_ids)
    print(f"Valid videos: {len(feature_manager.valid_ids)}")
    
    # Get samples for evaluation
    all_samples = parser_obj.get_training_samples(
        min_confidence=0.5,
        available_videos=feature_manager.valid_ids
    )
    
    # Split
    train_videos, val_videos, test_videos = parser_obj.split_by_video(
        train_ratio=0.85, val_ratio=0.10
    )
    
    if args.split == "train":
        eval_videos = set(train_videos)
    elif args.split == "val":
        eval_videos = set(val_videos)
    else:
        eval_videos = set(test_videos)
    
    eval_samples = [s for s in all_samples if s[0] in eval_videos]
    print(f"\n{args.split.upper()} samples: {len(eval_samples):,}")
    
    # Create dataloader
    dataset = SWINDataset(eval_samples, feature_manager)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    results = evaluate(model, dataloader, device, idx_to_gloss)
    
    # Print results
    print_results(results, idx_to_gloss)
    
    # Save results
    output_path = project_root / "models/swin_recognition/evaluation_results.json"
    save_results = {
        'model': args.model,
        'split': args.split,
        'top1_accuracy': float(results['top1_accuracy']),
        'top5_accuracy': float(results['top5_accuracy']),
        'total_samples': results['total_samples'],
        'num_classes': num_classes,
        'best_classes': [
            {'gloss': g, **s} 
            for g, s in sorted(results['per_class'].items(), key=lambda x: -x[1]['accuracy'])[:50]
            if s['total'] >= 10
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
