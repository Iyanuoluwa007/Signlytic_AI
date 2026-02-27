#!/usr/bin/env python3
"""
Verify dataloader outputs are valid (no NaN, correct dtype).
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.annotation_parser import BOBSLAnnotationParser
from data.datasets import Vocabulary, PooledSignDataset, create_data_splits, create_dataloaders


def verify_dataloader(data_dir: str) -> None:
    """Check dataloader outputs for NaN and dtype issues."""
    
    print("=" * 70)
    print("DATALOADER VERIFICATION")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    parser = BOBSLAnnotationParser(data_dir)
    parser.parse_isolated_signs()
    
    vocab = Vocabulary.from_parser(parser)
    
    # Create dataset directly to check
    print("\nCreating dataset...")
    dataset = PooledSignDataset(
        parser=parser,
        vocabulary=vocab,
        feature_type='swin',
        pooling='mean',
    )
    
    # Check first sample
    print("\nChecking first sample...")
    sample = dataset[0]
    features = sample['features']
    
    print(f"  Features dtype: {features.dtype}")
    print(f"  Features shape: {features.shape}")
    print(f"  Features range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  Features mean: {features.mean():.4f}")
    print(f"  Features std: {features.std():.4f}")
    print(f"  Has NaN: {torch.isnan(features).any().item()}")
    print(f"  Has Inf: {torch.isinf(features).any().item()}")
    
    # Check multiple samples
    print("\nChecking 100 random samples...")
    nan_count = 0
    inf_count = 0
    
    indices = np.random.choice(len(dataset), min(100, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        features = sample['features']
        
        if torch.isnan(features).any():
            nan_count += 1
        if torch.isinf(features).any():
            inf_count += 1
    
    print(f"  Samples with NaN: {nan_count}/100")
    print(f"  Samples with Inf: {inf_count}/100")
    
    # Check dataloader batch
    print("\nChecking dataloader batch...")
    train_data, val_data, test_data = create_data_splits(
        parser=parser,
        vocabulary=vocab,
        dataset_type='pooled',
        feature_type='swin',
    )
    
    train_loader, _, _ = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=32,
        num_workers=0,
    )
    
    batch = next(iter(train_loader))
    features = batch['features']
    labels = batch['label']
    
    print(f"  Batch features dtype: {features.dtype}")
    print(f"  Batch features shape: {features.shape}")
    print(f"  Batch features range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"  Batch has NaN: {torch.isnan(features).any().item()}")
    print(f"  Batch has Inf: {torch.isinf(features).any().item()}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    from models.classifier import MLPClassifier
    
    model = MLPClassifier(input_dim=768, hidden_dim=512, num_classes=len(vocab))
    
    with torch.no_grad():
        logits = model(features)
    
    print(f"  Logits dtype: {logits.dtype}")
    print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"  Logits has NaN: {torch.isnan(logits).any().item()}")
    print(f"  Logits has Inf: {torch.isinf(logits).any().item()}")
    
    # Test loss
    print("\nTesting loss computation...")
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    
    print(f"  Loss value: {loss.item()}")
    print(f"  Loss is NaN: {torch.isnan(loss).item()}")
    
    print("\n" + "=" * 70)
    if torch.isnan(loss):
        print("PROBLEM: Loss is NaN")
        print("Check the feature values and model outputs above.")
    else:
        print("SUCCESS: All checks passed")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    args = parser.parse_args()
    
    verify_dataloader(args.data_dir)
