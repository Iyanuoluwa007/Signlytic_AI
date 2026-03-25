"""
BSL Recognition Training - Memory Efficient Version
Loads features on-demand instead of all at once.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import Counter
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BSLDataset(Dataset):
    """BSL Dataset - loads features lazily to save memory."""
    
    def __init__(
        self,
        data_root: str = "data",
        split: str = "train",
        min_samples_per_class: int = 50,
        min_confidence: float = 0.7,
        window_frames: int = 32,
        max_classes: int = 500,
    ):
        self.data_root = Path(data_root)
        self.window_frames = window_frames
        self.feature_dim = 768
        
        # Feature directory
        self.features_dir = self.data_root / "processed" / "features" / "bobsl" / "v1.4" / "video_features" / "swin_v1" / "video-swin-s_c8697_16f_bs32"
        
        # Load training data mapping
        with open(self.data_root / "bsl1k_training_data.json", 'r') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        
        # Get available feature files
        available_videos = {f.stem for f in self.features_dir.glob("*.npy")}
        logger.info(f"Available feature files: {len(available_videos)}")
        
        # Collect all annotations with confidence filter
        all_samples = []
        for vid, annots in annotations.items():
            if vid not in available_videos:
                continue
            
            for annot in annots:
                if annot['prob'] >= min_confidence:
                    all_samples.append({
                        'video': vid,
                        'word': annot['word'].lower(),
                        'time': annot['time'],
                        'prob': annot['prob']
                    })
        
        logger.info(f"Total samples (conf >= {min_confidence}): {len(all_samples)}")
        
        # Filter by class frequency
        word_counts = Counter(s['word'] for s in all_samples)
        top_words = [w for w, c in word_counts.most_common(max_classes) if c >= min_samples_per_class]
        valid_words = set(top_words[:max_classes])
        
        all_samples = [s for s in all_samples if s['word'] in valid_words]
        logger.info(f"Filtered samples: {len(all_samples)}, Classes: {len(valid_words)}")
        
        # Create label mapping
        self.label_map = {w: i for i, w in enumerate(sorted(valid_words))}
        self.idx_to_label = {i: w for w, i in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        # Split data
        random.seed(42)
        random.shuffle(all_samples)
        
        n = len(all_samples)
        if split == "train":
            self.samples = all_samples[:int(0.8 * n)]
        elif split == "val":
            self.samples = all_samples[int(0.8 * n):int(0.9 * n)]
        else:
            self.samples = all_samples[int(0.9 * n):]
        
        logger.info(f"Split '{split}': {len(self.samples)} samples, {self.num_classes} classes")
        
        # Cache for recently loaded features
        self._cache = {}
        self._cache_size = 10
    
    def _load_features(self, vid):
        if vid in self._cache:
            return self._cache[vid]
        
        feat_path = self.features_dir / f"{vid}.npy"
        feat = np.load(feat_path)
        
        # Manage cache
        if len(self._cache) >= self._cache_size:
            self._cache.pop(next(iter(self._cache)))
        self._cache[vid] = feat
        
        return feat
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        feat = self._load_features(sample['video'])
        center = int(sample['time'] * 25)  # 25 fps
        
        # Extract window
        half = self.window_frames // 2
        start = max(0, center - half)
        end = min(len(feat), center + half)
        
        window = feat[start:end]
        
        # Pad if needed
        if len(window) < self.window_frames:
            pad = np.zeros((self.window_frames - len(window), self.feature_dim), dtype=np.float16)
            window = np.vstack([window, pad])
        elif len(window) > self.window_frames:
            window = window[:self.window_frames]
        
        mask = np.zeros(self.window_frames, dtype=bool)
        mask[:min(end - start, self.window_frames)] = True
        
        label = self.label_map[sample['word']]
        
        return {
            'features': torch.from_numpy(window.astype(np.float32)),
            'mask': torch.from_numpy(mask),
            'label': torch.tensor(label, dtype=torch.long),
            'word': sample['word'],
        }


class BSLRecognizer(nn.Module):
    def __init__(self, num_classes, input_dim=768, hidden_dim=512, num_layers=4, num_heads=8, dropout=0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :T, :]
        
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)
        
        if mask is not None:
            x = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        else:
            x = x.mean(dim=1)
        
        x = self.norm(x)
        return self.classifier(x)


def train_bsl_recognizer(epochs=30, batch_size=64, lr=1e-4, device="cuda", save_dir="models/bsl_recognition", max_classes=500, min_samples=50):
    logger.info("="*70)
    logger.info("TRAINING BSL RECOGNIZER (BRITISH SIGN LANGUAGE)")
    logger.info("="*70)
    
    train_dataset = BSLDataset(split="train", max_classes=max_classes, min_samples_per_class=min_samples)
    val_dataset = BSLDataset(split="val", max_classes=max_classes, min_samples_per_class=min_samples)
    
    val_dataset.label_map = train_dataset.label_map
    val_dataset.idx_to_label = train_dataset.idx_to_label
    val_dataset.num_classes = train_dataset.num_classes
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = BSLRecognizer(num_classes=train_dataset.num_classes).to(device)
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
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            features = batch['features'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(features, masks)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.shape[0]
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{train_correct/train_total*100:.1f}%'})
        
        model.eval()
        val_correct = 0
        val_total = 0
        val_correct_top5 = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(features, masks)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.shape[0]
                
                k = min(5, logits.shape[-1])
                _, top5 = logits.topk(k, dim=-1)
                val_correct_top5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
        
        val_acc = val_correct / val_total if val_total > 0 else 0
        val_acc_top5 = val_correct_top5 / val_total if val_total > 0 else 0
        
        logger.info(f"Epoch {epoch+1}: Train={train_correct/train_total*100:.2f}%, Val Top-1={val_acc*100:.2f}%, Top-5={val_acc_top5*100:.2f}%")
        
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_classes", type=int, default=500)
    parser.add_argument("--min_samples", type=int, default=50)
    args = parser.parse_args()
    
    train_bsl_recognizer(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        max_classes=args.max_classes, min_samples=args.min_samples,
    )
