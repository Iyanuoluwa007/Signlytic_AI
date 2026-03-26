"""
Train BSL recognizer on dictionary features.
This model will work for end-to-end video recognition.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BSLDictDataset(Dataset):
    """BSL Dictionary dataset with extracted SWIN features."""
    
    def __init__(self, features_dir: str = "data/bsl_dict_features", split: str = "train", augment: bool = True):
        self.features_dir = Path(features_dir)
        self.augment = augment and split == "train"
        
        # Load index
        with open(self.features_dir / "index.json", 'r') as f:
            all_glosses = json.load(f)
        
        # Load all features
        self.samples = []
        for gloss in all_glosses:
            feat_path = self.features_dir / f"{gloss}.npy"
            if feat_path.exists():
                self.samples.append((gloss, feat_path))
        
        # Create label mapping
        unique_glosses = sorted(set(g for g, _ in self.samples))
        self.label_map = {g: i for i, g in enumerate(unique_glosses)}
        self.idx_to_label = {i: g for g, i in self.label_map.items()}
        self.num_classes = len(self.label_map)
        
        # Split: 80% train, 10% val, 10% test
        random.seed(42)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)
        
        n = len(indices)
        if split == "train":
            self.indices = indices[:int(0.8 * n)]
        elif split == "val":
            self.indices = indices[int(0.8 * n):int(0.9 * n)]
        else:
            self.indices = indices[int(0.9 * n):]
        
        logger.info(f"BSL Dict {split}: {len(self.indices)} samples, {self.num_classes} classes")
    
    def __len__(self):
        return len(self.indices) * (10 if self.augment else 1)  # Augment 10x
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.indices)
        sample_idx = self.indices[real_idx]
        gloss, feat_path = self.samples[sample_idx]
        
        features = np.load(feat_path).squeeze()  # Shape: (768,)
        
        # Augment with noise
        if self.augment:
            noise = np.random.randn(*features.shape) * 0.1
            features = features + noise
        
        label = self.label_map[gloss]
        
        return {
            'features': torch.from_numpy(features.astype(np.float32)),
            'label': torch.tensor(label, dtype=torch.long),
            'gloss': gloss,
        }


class BSLDictRecognizer(nn.Module):
    """Simple MLP classifier for SWIN features."""
    
    def __init__(self, num_classes: int, input_dim: int = 768, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x):
        return self.classifier(x)


def train_bsl_dict():
    """Train BSL dictionary recognizer."""
    
    print("="*70)
    print("TRAINING BSL DICTIONARY RECOGNIZER")
    print("="*70)
    
    device = "cuda"
    epochs = 50
    batch_size = 64
    lr = 1e-3
    
    # Datasets
    train_ds = BSLDictDataset(split="train", augment=True)
    val_ds = BSLDictDataset(split="val", augment=False)
    test_ds = BSLDictDataset(split="test", augment=False)
    
    # Share label mapping
    val_ds.label_map = train_ds.label_map
    val_ds.idx_to_label = train_ds.idx_to_label
    test_ds.label_map = train_ds.label_map
    test_ds.idx_to_label = train_ds.idx_to_label
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = BSLDictRecognizer(num_classes=train_ds.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    print(f"Classes: {train_ds.num_classes}")
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    best_acc = 0
    save_dir = Path("models/bsl_dict_recognition")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        train_correct, train_total = 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
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
                labels = batch['label'].to(device)
                
                logits = model(features)
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
                'val_acc': val_acc,
                'val_acc_top5': val_acc5,
                'num_classes': train_ds.num_classes,
                'label_map': train_ds.label_map,
                'idx_to_label': train_ds.idx_to_label,
            }, save_dir / 'best_model.pt')
            logger.info(f"  -> Saved best!")
        
        scheduler.step()
    
    # Test evaluation
    print("\n" + "="*70)
    print("TEST SET EVALUATION")
    print("="*70)
    
    checkpoint = torch.load(save_dir / 'best_model.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct, test_total, test_top5 = 0, 0, 0
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(features)
            test_correct += (logits.argmax(-1) == labels).sum().item()
            test_total += labels.shape[0]
            
            _, top5 = logits.topk(5, dim=-1)
            test_top5 += (top5 == labels.unsqueeze(-1)).any(-1).sum().item()
    
    print(f"Test Top-1: {test_correct/test_total*100:.2f}%")
    print(f"Test Top-5: {test_top5/test_total*100:.2f}%")
    print("="*70)


if __name__ == "__main__":
    train_bsl_dict()
