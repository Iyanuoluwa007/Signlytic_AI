#!/usr/bin/env python3
"""
Quick test to verify model and training setup before full training.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from data.annotation_parser import BOBSLAnnotationParser
from data.datasets import Vocabulary, create_data_splits, create_dataloaders
from models.classifier import MLPClassifier, count_parameters


def test_training_setup(data_dir: str) -> None:
    """Verify training setup works correctly."""
    
    print("=" * 70)
    print("TRAINING SETUP TEST")
    print("=" * 70)
    
    # Device check
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load data
    print("\nLoading data...")
    parser = BOBSLAnnotationParser(data_dir)
    parser.parse_isolated_signs()
    
    vocab = Vocabulary.from_parser(parser)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset
    print("\nCreating dataset...")
    train_data, val_data, test_data = create_data_splits(
        parser=parser,
        vocabulary=vocab,
        dataset_type='pooled',
        feature_type='swin',
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=32,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    
    # Create model
    print("\nCreating model...")
    model = MLPClassifier(
        input_dim=768,
        hidden_dim=512,
        num_classes=len(vocab),
        dropout=0.3,
    )
    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Training test
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining test (2 batches)...")
    model.train()
    
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= 2:
            break
        
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(features)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        
        print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}, acc={acc.item():.4f}")
    
    # Evaluation test
    print("\nEvaluation test...")
    model.eval()
    
    with torch.no_grad():
        batch = next(iter(val_loader))
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(features)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        
        print(f"  Val batch: loss={loss.item():.4f}, acc={acc.item():.4f}")
    
    # Checkpoint test
    print("\nCheckpoint test...")
    save_path = Path('outputs') / 'test_checkpoint.pt'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({'model_state_dict': model.state_dict()}, save_path)
    print(f"  Saved: {save_path}")
    
    save_path.unlink()
    print("  Cleaned up")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)
    print("\nRun full training:")
    print("  python src/models/train.py --epochs 50 --batch_size 64")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test training setup')
    parser.add_argument('--data_dir', type=str, default='data/processed')
    
    args = parser.parse_args()
    test_training_setup(args.data_dir)


if __name__ == "__main__":
    main()
