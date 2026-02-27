#!/usr/bin/env python3
"""
Training script for BSL sign recognition models.
"""

import sys
from pathlib import Path
import json
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.annotation_parser import BOBSLAnnotationParser
from data.datasets import (
    Vocabulary,
    create_data_splits,
    create_dataloaders,
)
from models.classifier import MLPClassifier, TemporalMLPClassifier, count_parameters
from models.transformer import TransformerClassifier, TransformerClassifierWithCLS


class Trainer:
    """Training manager for sign recognition models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
    ):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        print(f"Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.1)
        )
        
        optimizer_name = config.get('optimizer', 'adamw')
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-4)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            self.optimizer = optim.SGD(
                model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )
        
        # Warmup scheduler for transformers
        self.warmup_epochs = config.get('warmup_epochs', 0)
        self.warmup_scheduler = None
        if self.warmup_epochs > 0 and 'transformer' in config.get('model', ''):
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=0.1, 
                end_factor=1.0, 
                total_iters=self.warmup_epochs
            )
        
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.best_epoch = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': [],
        }
        
        self.patience = config.get('patience', 10)
        self.patience_counter = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        nan_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch in pbar:
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Skip batches with NaN
            if torch.isnan(features).any():
                nan_batches += 1
                continue
            
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            # Skip if loss is NaN
            if torch.isnan(loss):
                nan_batches += 1
                continue
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
        
        if nan_batches > 0:
            print(f"  Warning: Skipped {nan_batches} batches with NaN")
        
        if total == 0:
            return float('nan'), 0.0
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float, Dict]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        correct_top5 = 0
        total = 0
        
        for batch in data_loader:
            features = batch['features'].to(self.device)
            labels = batch['label'].to(self.device)
            
            logits = self.model(features)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item() * features.size(0)
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            
            _, top5_pred = torch.topk(logits, k=5, dim=-1)
            correct_top5 += (top5_pred == labels.unsqueeze(-1)).any(dim=-1).sum().item()
            
            total += labels.size(0)
        
        metrics = {
            'top1_accuracy': correct / total,
            'top5_accuracy': correct_top5 / total,
            'loss': total_loss / total,
        }
        
        return total_loss / total, correct / total, metrics
    
    def train(self, num_epochs: int) -> Dict:
        """Full training loop."""
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            start_time = time.time()
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_metrics = self.evaluate(self.val_loader)
            
            # Update learning rate scheduler
            if self.warmup_scheduler is not None and self.current_epoch <= self.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {self.current_epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Top5: {val_metrics['top5_accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = self.current_epoch
                self.patience_counter = 0
                self.save_checkpoint('best_model.pt')
                print(f"  [New best model saved]")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {self.current_epoch}")
                break
        
        print(f"\nBest validation accuracy: {self.best_val_accuracy:.4f} at epoch {self.best_epoch}")
        return self.history
    
    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config,
            'history': self.history,
        }
        
        save_path = Path('outputs') / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        load_path = Path('outputs') / filename
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        self.history = checkpoint.get('history', self.history)
    
    def test(self) -> Dict:
        """Final evaluation on test set."""
        print("\n" + "=" * 70)
        print("TEST EVALUATION")
        print("=" * 70)
        
        self.load_checkpoint('best_model.pt')
        test_loss, test_acc, test_metrics = self.evaluate(self.test_loader)
        
        print(f"Test Loss:     {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-5:    {test_metrics['top5_accuracy']:.4f}")
        
        return test_metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BSL sign classifier')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--dataset', type=str, default='pooled', choices=['pooled', 'isolated'])
    parser.add_argument('--model', type=str, default='mlp', 
                       choices=['mlp', 'temporal_mlp', 'transformer', 'transformer_cls'])
    parser.add_argument('--d_model', type=int, default=256, help='Transformer hidden dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs for transformer')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BSL SIGN RECOGNITION TRAINING")
    print("=" * 70)
    
    config = {
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'label_smoothing': 0.1,
        'patience': 10,
        'hidden_dim': args.hidden_dim,
        'dropout': args.dropout,
        'model': args.model,
        'd_model': args.d_model,
        'nhead': args.nhead,
        'num_layers': args.num_layers,
        'warmup_epochs': args.warmup_epochs,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\nLoading data...")
    data_parser = BOBSLAnnotationParser(args.data_dir)
    data_parser.parse_isolated_signs()
    
    vocab = Vocabulary.from_parser(data_parser)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Force isolated dataset for transformer models
    dataset_type = args.dataset
    if args.model in ['transformer', 'transformer_cls', 'temporal_mlp']:
        dataset_type = 'isolated'
        print(f"Using isolated dataset for {args.model} model")
    
    # Create datasets
    print("\nCreating datasets...")
    train_data, val_data, test_data = create_data_splits(
        parser=data_parser,
        vocabulary=vocab,
        dataset_type=dataset_type,
        feature_type='swin',
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=args.batch_size,
        num_workers=0 if dataset_type == 'isolated' else 4,
    )
    
    # Create model
    print("\nCreating model...")
    if args.model == 'mlp':
        model = MLPClassifier(
            input_dim=768,
            hidden_dim=config['hidden_dim'],
            num_classes=len(vocab),
            dropout=config['dropout'],
        )
    elif args.model == 'temporal_mlp':
        model = TemporalMLPClassifier(
            input_dim=768,
            hidden_dim=config['hidden_dim'],
            num_classes=len(vocab),
            dropout=config['dropout'],
            pooling='attention',
        )
    elif args.model == 'transformer':
        model = TransformerClassifier(
            input_dim=768,
            num_classes=len(vocab),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['d_model'] * 4,
            dropout=config['dropout'],
            max_seq_len=128,
        )
    elif args.model == 'transformer_cls':
        model = TransformerClassifierWithCLS(
            input_dim=768,
            num_classes=len(vocab),
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_layers=config['num_layers'],
            dim_feedforward=config['d_model'] * 4,
            dropout=config['dropout'],
            max_seq_len=128,
        )
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"Model parameters: {num_params:,}")
    
    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
    )
    
    history = trainer.train(num_epochs=args.epochs)
    test_metrics = trainer.test()
    
    # Save results
    results = {
        'config': config,
        'history': history,
        'test_metrics': test_metrics,
        'best_epoch': trainer.best_epoch,
        'best_val_accuracy': trainer.best_val_accuracy,
    }
    
    results_path = Path('outputs') / f'training_results_{args.model}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()