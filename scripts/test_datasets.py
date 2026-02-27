#!/usr/bin/env python3
"""
Test script for BOBSL datasets and dataloaders.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from data.annotation_parser import BOBSLAnnotationParser
from data.datasets import (
    Vocabulary,
    IsolatedSignDataset,
    ContinuousSignDataset,
    PooledSignDataset,
    create_data_splits,
    create_dataloaders,
)


def test_vocabulary(parser: BOBSLAnnotationParser) -> Vocabulary:
    """Test vocabulary creation."""
    print("\n" + "=" * 70)
    print("VOCABULARY TEST")
    print("=" * 70)
    
    vocab = Vocabulary.from_parser(parser)
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: <pad>={vocab.encode('<pad>')}, <unk>={vocab.encode('<unk>')}")
    
    # Test encode/decode
    test_glosses = list(parser.vocabulary.keys())[:5]
    print(f"\nEncode/decode test:")
    for gloss in test_glosses:
        idx = vocab.encode(gloss)
        decoded = vocab.decode(idx)
        print(f"  '{gloss}' -> {idx} -> '{decoded}'")
    
    # Save vocabulary
    vocab_path = 'data/processed/vocabulary.json'
    vocab.save(vocab_path)
    print(f"\nSaved vocabulary to {vocab_path}")
    
    return vocab


def test_isolated_dataset(parser: BOBSLAnnotationParser, vocab: Vocabulary) -> None:
    """Test IsolatedSignDataset."""
    print("\n" + "=" * 70)
    print("ISOLATED SIGN DATASET TEST")
    print("=" * 70)
    
    dataset = IsolatedSignDataset(
        parser=parser,
        vocabulary=vocab,
        feature_type='swin',
        window_seconds=2.0,
        max_frames=64,
        feature_dim=768,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  features shape: {sample['features'].shape}")
    print(f"  label: {sample['label'].item()} ({sample['gloss']})")
    print(f"  video_id: {sample['video_id']}")
    print(f"  confidence: {sample['confidence']:.3f}")
    
    # Test batch
    print("\nBatch test:")
    batch_indices = [0, 1, 2, 3]
    batch = [dataset[i] for i in batch_indices]
    features = torch.stack([b['features'] for b in batch])
    labels = torch.stack([b['label'] for b in batch])
    print(f"  Batch features: {features.shape}")
    print(f"  Batch labels: {labels.shape}")
    
    dataset.clear_cache()


def test_pooled_dataset(parser: BOBSLAnnotationParser, vocab: Vocabulary) -> None:
    """Test PooledSignDataset."""
    print("\n" + "=" * 70)
    print("POOLED SIGN DATASET TEST")
    print("=" * 70)
    
    dataset = PooledSignDataset(
        parser=parser,
        vocabulary=vocab,
        feature_type='swin',
        pooling='mean',
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  features shape: {sample['features'].shape}")
    print(f"  label: {sample['label'].item()} ({sample['gloss']})")
    print(f"  video_id: {sample['video_id']}")


def test_continuous_dataset(parser: BOBSLAnnotationParser, vocab: Vocabulary) -> None:
    """Test ContinuousSignDataset."""
    print("\n" + "=" * 70)
    print("CONTINUOUS SIGN DATASET TEST")
    print("=" * 70)
    
    dataset = ContinuousSignDataset(
        parser=parser,
        vocabulary=vocab,
        feature_type='swin',
        max_frames=1024,
        max_glosses=100,
        feature_dim=768,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get sample
    sample = dataset[0]
    print(f"\nSample item:")
    print(f"  features shape: {sample['features'].shape}")
    print(f"  seq_length: {sample['seq_length'].item()}")
    print(f"  gloss_indices shape: {sample['gloss_indices'].shape}")
    print(f"  gloss_length: {sample['gloss_length'].item()}")
    print(f"  video_id: {sample['video_id']}")
    
    # Decode gloss sequence
    gloss_len = sample['gloss_length'].item()
    gloss_indices = sample['gloss_indices'][:gloss_len].tolist()
    glosses = vocab.decode_sequence(gloss_indices)
    print(f"  First 10 glosses: {glosses[:10]}")
    
    dataset.clear_cache()


def test_data_splits(parser: BOBSLAnnotationParser, vocab: Vocabulary) -> None:
    """Test data splitting and dataloaders."""
    print("\n" + "=" * 70)
    print("DATA SPLITS AND DATALOADERS TEST")
    print("=" * 70)
    
    # Create splits using pooled dataset (faster for testing)
    train_data, val_data, test_data = create_data_splits(
        parser=parser,
        vocabulary=vocab,
        dataset_type='pooled',
        feature_type='swin',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=42,
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_data)}")
    print(f"  Val: {len(val_data)}")
    print(f"  Test: {len(test_data)}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data,
        batch_size=32,
        num_workers=0,  # Use 0 for testing
        pin_memory=False,
    )
    
    print(f"\nDataloader batches:")
    print(f"  Train: {len(train_loader)}")
    print(f"  Val: {len(val_loader)}")
    print(f"  Test: {len(test_loader)}")
    
    # Test iteration
    print("\nIteration test:")
    for batch_idx, batch in enumerate(train_loader):
        print(f"  Batch {batch_idx}:")
        print(f"    features: {batch['features'].shape}")
        print(f"    labels: {batch['label'].shape}")
        print(f"    unique labels: {batch['label'].unique().shape[0]}")
        if batch_idx >= 1:
            break


def main():
    import argparse
    
    arg_parser = argparse.ArgumentParser(description='Test BOBSL datasets')
    arg_parser.add_argument('--data_dir', type=str, default='data/processed',
                           help='Path to processed data directory')
    
    args = arg_parser.parse_args()
    
    print("=" * 70)
    print("BOBSL DATASET TESTS")
    print("=" * 70)
    
    # Initialize parser
    print(f"\nData directory: {args.data_dir}")
    parser = BOBSLAnnotationParser(args.data_dir)
    parser.parse_isolated_signs()
    
    # Run tests
    vocab = test_vocabulary(parser)
    test_pooled_dataset(parser, vocab)
    test_isolated_dataset(parser, vocab)
    test_continuous_dataset(parser, vocab)
    test_data_splits(parser, vocab)
    
    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
