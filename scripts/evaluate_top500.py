"""
Comprehensive Evaluation for Top 500 SWIN Recognition Model

Outputs:
- Overall metrics (Top-1, Top-5, Top-10)
- Per-class accuracy breakdown
- Confusion analysis
- Best/worst performing classes
- Saves detailed results to JSON

Usage:
    python scripts/evaluate_top500.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Set
from collections import defaultdict, Counter
import random

from src.data.bsl1k_parser import BSL1KParser, get_swin_video_ids


# ============================================================
# Model (same architecture as training)
# ============================================================

class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.5,
        num_classes: int = 500
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
                'label': torch.tensor(gloss_idx, dtype=torch.long)
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
            'label': torch.tensor(gloss_idx, dtype=torch.long)
        }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, device, num_classes):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    for batch in tqdm(dataloader, desc="Evaluating", ncols=100):
        features = batch['features'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(features, mask)
        probs = torch.softmax(logits, dim=-1)
        
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    
    # Compute metrics
    top1_correct = (all_preds == all_labels).sum()
    
    # Top-5
    top5_preds = np.argsort(all_probs, axis=-1)[:, -5:]
    top5_correct = sum(label in preds for label, preds in zip(all_labels, top5_preds))
    
    # Top-10
    top10_preds = np.argsort(all_probs, axis=-1)[:, -10:]
    top10_correct = sum(label in preds for label, preds in zip(all_labels, top10_preds))
    
    total = len(all_labels)
    
    # Per-class metrics
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    per_class_top5 = defaultdict(int)
    
    for i in range(len(all_labels)):
        label = all_labels[i]
        pred = all_preds[i]
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1
        if label in top5_preds[i]:
            per_class_top5[label] += 1
    
    return {
        'top1_accuracy': top1_correct / total,
        'top5_accuracy': top5_correct / total,
        'top10_accuracy': top10_correct / total,
        'total_samples': total,
        'per_class_correct': dict(per_class_correct),
        'per_class_total': dict(per_class_total),
        'per_class_top5': dict(per_class_top5),
        'predictions': all_preds,
        'labels': all_labels
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Paths
    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    model_dir = project_root / "models/swin_recognition_top500"
    swin_dir = project_root / "data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32"
    bsl1k_dir = project_root / "data/BSL-1K"
    
    # Load vocabulary
    print("\nLoading vocabulary...")
    vocab_path = model_dir / "vocabulary.json"
    with open(vocab_path) as f:
        vocab = json.load(f)
    
    gloss_to_idx = vocab['gloss_to_idx']
    idx_to_gloss = {int(k): v for k, v in vocab['idx_to_gloss'].items()}
    num_classes = vocab['num_classes']
    print(f"Classes: {num_classes}")
    
    # Load model
    print("\nLoading model...")
    model_path = model_dir / "best_model.pt"
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = TemporalTransformerEncoder(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded: Top-1={100*checkpoint['best_val_acc']:.2f}%, Top-5={100*checkpoint['best_val_top5']:.2f}%")
    
    # Load data
    print("\nLoading data...")
    bsl_parser = BSL1KParser(str(bsl1k_dir))
    bsl_parser.load_annotations()
    
    video_ids = get_swin_video_ids(str(swin_dir))
    feature_manager = FeatureManager(str(swin_dir), video_ids)
    print(f"Valid videos: {len(feature_manager.valid_ids)}")
    
    # Get samples
    all_samples_raw = bsl_parser.get_training_samples(
        min_confidence=0.5,
        available_videos=feature_manager.valid_ids
    )
    
    # Filter to Top 500 vocabulary and remap indices
    old_idx_to_gloss = bsl_parser.idx_to_gloss
    samples = []
    for video_id, start_idx, end_idx, old_gloss_idx in all_samples_raw:
        gloss = old_idx_to_gloss.get(old_gloss_idx, None)
        if gloss and gloss in gloss_to_idx:
            new_gloss_idx = gloss_to_idx[gloss]
            samples.append((video_id, start_idx, end_idx, new_gloss_idx))
    
    print(f"Filtered samples: {len(samples):,}")
    
    # Split - use test set
    all_videos = list(set(s[0] for s in samples))
    random.seed(42)
    random.shuffle(all_videos)
    
    n_train = int(0.85 * len(all_videos))
    n_val = int(0.10 * len(all_videos))
    
    test_videos = set(all_videos[n_train + n_val:])
    test_samples = [s for s in samples if s[0] in test_videos]
    
    print(f"Test samples: {len(test_samples):,}")
    
    # Create dataloader
    dataset = SWINDataset(test_samples, feature_manager)
    dataloader = DataLoader(
        dataset, batch_size=128, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
    # Evaluate
    print("\nRunning evaluation...")
    results = evaluate(model, dataloader, device, num_classes)
    
    # Print results
    print("\n" + "="*70)
    print("EVALUATION RESULTS (Test Set)")
    print("="*70)
    
    print(f"\nOverall Metrics:")
    print(f"  Top-1 Accuracy:  {100*results['top1_accuracy']:.2f}%")
    print(f"  Top-5 Accuracy:  {100*results['top5_accuracy']:.2f}%")
    print(f"  Top-10 Accuracy: {100*results['top10_accuracy']:.2f}%")
    print(f"  Total Samples:   {results['total_samples']:,}")
    print(f"  Random Chance:   {100/num_classes:.3f}%")
    print(f"  Improvement:     {results['top1_accuracy'] / (1/num_classes):.0f}x better than random")
    
    # Per-class analysis
    per_class_stats = []
    for cls_idx in results['per_class_total']:
        gloss = idx_to_gloss[cls_idx]
        total = results['per_class_total'][cls_idx]
        correct = results['per_class_correct'].get(cls_idx, 0)
        top5 = results['per_class_top5'].get(cls_idx, 0)
        per_class_stats.append({
            'gloss': gloss,
            'total': total,
            'correct': correct,
            'top5': top5,
            'accuracy': correct / total if total > 0 else 0,
            'top5_accuracy': top5 / total if total > 0 else 0
        })
    
    # Sort by accuracy (descending)
    per_class_stats.sort(key=lambda x: (-x['accuracy'], -x['total']))
    
    # Top 20 best
    print(f"\nTop 20 Best Performing Classes:")
    print(f"  {'Gloss':<20} {'Top-1':>8} {'Top-5':>8} {'Samples':>10}")
    print("  " + "-"*50)
    for stats in per_class_stats[:20]:
        print(f"  {stats['gloss']:<20} {100*stats['accuracy']:>7.1f}% {100*stats['top5_accuracy']:>7.1f}% {stats['total']:>10}")
    
    # Classes with >50% Top-5
    high_performers = [s for s in per_class_stats if s['top5_accuracy'] >= 0.5 and s['total'] >= 20]
    print(f"\nClasses with ≥50% Top-5 Accuracy (min 20 samples): {len(high_performers)}")
    
    # Classes with 0% accuracy
    zero_acc = [s for s in per_class_stats if s['accuracy'] == 0 and s['total'] >= 20]
    print(f"Classes with 0% Top-1 Accuracy (min 20 samples): {len(zero_acc)}")
    
    # Distribution
    accuracies = [s['accuracy'] for s in per_class_stats]
    print(f"\nAccuracy Distribution:")
    print(f"  Min:    {100*min(accuracies):.1f}%")
    print(f"  Max:    {100*max(accuracies):.1f}%")
    print(f"  Mean:   {100*np.mean(accuracies):.1f}%")
    print(f"  Median: {100*np.median(accuracies):.1f}%")
    
    # Save results
    output_path = model_dir / "evaluation_results.json"
    save_data = {
        'top1_accuracy': float(results['top1_accuracy']),
        'top5_accuracy': float(results['top5_accuracy']),
        'top10_accuracy': float(results['top10_accuracy']),
        'total_samples': results['total_samples'],
        'num_classes': num_classes,
        'random_chance': 1/num_classes,
        'improvement_over_random': results['top1_accuracy'] / (1/num_classes),
        'per_class': per_class_stats[:100]  # Top 100
    }
    
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
