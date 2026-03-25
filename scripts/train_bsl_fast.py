"""
BSL Recognition Training - OPTIMIZED VERSION
Combines: subsampling + large cache + video grouping
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from collections import Counter, defaultdict
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BSLDatasetFast(Dataset):
    """Optimized BSL Dataset with video grouping and large cache."""
    
    def __init__(
        self,
        data_root: str = "data",
        split: str = "train",
        min_samples_per_class: int = 50,
        min_confidence: float = 0.7,
        window_frames: int = 32,
        max_classes: int = 500,
        samples_per_epoch: int = 50000,  # Subsample for speed
        cache_size: int = 100,  # Cache more videos
    ):
        self.data_root = Path(data_root)
        self.window_frames = window_frames
        self.feature_dim = 768
        self.samples_per_epoch = samples_per_epoch
        
        self.features_dir = self.data_root / "processed" / "features" / "bobsl" / "v1.4" / "video_features" / "swin_v1" / "video-swin-s_c8697_16f_bs32"
        
        with open(self.data_root / "bsl1k_training_data.json", 'r') as f:
            data = json.load(f)
        
        annotations = data['annotations']
        available_videos = {f.stem for f in self.features_dir.glob("*.npy")}
        logger.info(f"Available videos: {len(available_videos)}")
        
        # Collect samples grouped by video
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
        
        logger.info(f"Total samples: {len(all_samples)}")
        
        # Filter by class frequency
        word_counts = Counter(s['word'] for s in all_samples)
        top_words = [w for w, c in word_counts.most_common(max_classes) if c >= min_samples_per_class]
        valid_words = set(top_words[:max_classes])
        
        all_samples = [s for s in all_samples if s['word'] in valid_words]
        logger.info(f"Filtered: {len(all_samples)} samples, {len(valid_words)} classes")
        
        # Create label mapping
        self.label_map = {w: i for i, w in enumerate(sorted(valid_words))}
        self.idx_to_label = {i: w for w, i in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        # Split
        random.seed(42)
        random.shuffle(all_samples)
        n = len(all_samples)
        
        if split == "train":
            self.all_samples = all_samples[:int(0.8 * n)]
        elif split == "val":
            self.all_samples = all_samples[int(0.8 * n):int(0.9 * n)]
        else:
            self.all_samples = all_samples[int(0.9 * n):]
        
        # Group samples by video for cache efficiency
        self.video_to_samples = defaultdict(list)
        for i, s in enumerate(self.all_samples):
            self.video_to_samples[s['video']].append(i)
        
        logger.info(f"Split '{split}': {len(self.all_samples)} total, {samples_per_epoch} per epoch")
        
        # LRU Cache with larger size
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size
        
        # Create epoch samples
        self._create_epoch_samples()
    
    def _create_epoch_samples(self):
        """Create samples for this epoch, grouped by video."""
        # Sample videos, then samples within videos
        videos = list(self.video_to_samples.keys())
        random.shuffle(videos)
        
        self.epoch_samples = []
        samples_needed = min(self.samples_per_epoch, len(self.all_samples))
        
        # Round-robin from videos to maximize cache hits
        video_idx = 0
        while len(self.epoch_samples) < samples_needed:
            vid = videos[video_idx % len(videos)]
            vid_samples = self.video_to_samples[vid]
            if vid_samples:
                idx = random.choice(vid_samples)
                self.epoch_samples.append(idx)
            video_idx += 1
        
        # Sort by video for better cache locality
        self.epoch_samples.sort(key=lambda i: self.all_samples[i]['video'])
    
    def _load_features(self, vid):
        if vid in self._cache:
            return self._cache[vid]
        
        feat_path = self.features_dir / f"{vid}.npy"
        feat = np.load(feat_path)
        
        # LRU eviction
        if len(self._cache) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            if oldest in self._cache:
                del self._cache[oldest]
        
        self._cache[vid] = feat
        self._cache_order.append(vid)
        return feat
    
    def __len__(self):
        return len(self.epoch_samples)
    
    def __getitem__(self, idx):
        sample_idx = self.epoch_samples[idx]
        sample = self.all_samples[sample_idx]
        
        feat = self._load_features(sample['video'])
        center = int(sample['time'] * 25)
        
        half = self.window_frames // 2
        start = max(0, center - half)
        end = min(len(feat), center + half)
        
        window = feat[start:end]
        
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
    
    def on_epoch_end(self):
        """Reshuffle samples for next epoch."""
        self._create_epoch_samples()


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
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
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
        
        return self.classifier(self.norm(x))


def train_bsl_fast(epochs=30, batch_size=128, lr=1e-4, device="cuda", 
                   save_dir="models/bsl_recognition", max_classes=100, 
                   samples_per_epoch=50000):
    
    logger.info("="*70)
    logger.info("TRAINING BSL RECOGNIZER - OPTIMIZED")
    logger.info("="*70)
    
    train_ds = BSLDatasetFast(split="train", max_classes=max_classes, samples_per_epoch=samples_per_epoch)
    val_ds = BSLDatasetFast(split="val", max_classes=max_classes, samples_per_epoch=10000)
    
    val_ds.label_map = train_ds.label_map
    val_ds.idx_to_label = train_ds.idx_to_label
    val_ds.num_classes = train_ds.num_classes
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = BSLRecognizer(num_classes=train_ds.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    logger.info(f"Train: {len(train_ds)} samples/epoch, Val: {len(val_ds)} samples")
    
    best_acc = 0
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0
        
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
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{train_correct/train_total*100:.1f}%'})
        
        # Validate
        model.eval()
        val_correct, val_total, val_top5 = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                masks = batch['mask'].to(device)
                labels = batch['label'].to(device)
                
                logits = model(features, masks)
                val_correct += (logits.argmax(-1) == labels).sum().item()
                val_total += labels.shape[0]
                _, top5 = logits.topk(5, dim=-1)
                val_top5 += (top5 == labels.unsqueeze(-1)).any(-1).sum().item()
        
        val_acc = val_correct / val_total
        val_acc5 = val_top5 / val_total
        
        logger.info(f"Epoch {epoch+1}: Train={train_correct/train_total*100:.1f}%, Val Top-1={val_acc*100:.1f}%, Top-5={val_acc5*100:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc, 'val_acc_top5': val_acc5,
                'num_classes': train_ds.num_classes,
                'label_map': train_ds.label_map,
                'idx_to_label': train_ds.idx_to_label,
            }, save_path / 'best_model.pt')
            logger.info(f"  -> Saved best!")
        
        scheduler.step()
        train_ds.on_epoch_end()
        val_ds.on_epoch_end()
    
    logger.info(f"Done! Best Val Top-1: {best_acc*100:.2f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_classes", type=int, default=100)
    parser.add_argument("--samples_per_epoch", type=int, default=50000)
    args = parser.parse_args()
    
    train_bsl_fast(epochs=args.epochs, batch_size=args.batch_size, 
                   max_classes=args.max_classes, samples_per_epoch=args.samples_per_epoch)
