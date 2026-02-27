#!/usr/bin/env python3
"""
Extract compressed feature files (.gz) from BOBSL dataset.

The BOBSL features are stored as gzipped numpy files and need to be
extracted before use. This script handles batch extraction.
"""

import gzip
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse


def extract_gz_file(gz_path: Path, keep_original: bool = False) -> bool:
    """
    Extract a single .gz file.
    
    Args:
        gz_path: Path to .gz file
        keep_original: Whether to keep the original .gz file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Determine output path (remove .gz extension)
        output_path = gz_path.with_suffix('')
        
        # Skip if already extracted
        if output_path.exists():
            return True
        
        # Extract
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove original if requested
        if not keep_original:
            gz_path.unlink()
        
        return True
        
    except Exception as e:
        print(f"Error extracting {gz_path}: {e}")
        return False


def extract_features(features_dir: str, feature_type: str = 'all', 
                     num_workers: int = 4, keep_original: bool = False) -> None:
    """
    Extract all compressed feature files.
    
    Args:
        features_dir: Path to features directory
        feature_type: 'i3d', 'swin', or 'all'
        num_workers: Number of parallel workers
        keep_original: Whether to keep original .gz files
    """
    features_dir = Path(features_dir)
    
    print("=" * 70)
    print("BOBSL FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Features directory: {features_dir}")
    print(f"Feature type: {feature_type}")
    print(f"Workers: {num_workers}")
    print(f"Keep original: {keep_original}")
    
    # Find all .gz files
    if feature_type == 'all':
        gz_files = list(features_dir.rglob("*.gz"))
    elif feature_type == 'i3d':
        gz_files = list((features_dir / "bobsl" / "v1.4" / "video_features" / "i3d").rglob("*.gz"))
    elif feature_type == 'swin':
        gz_files = list((features_dir / "bobsl" / "v1.4" / "video_features" / "swin_v1").rglob("*.gz"))
    else:
        print(f"Unknown feature type: {feature_type}")
        return
    
    print(f"\nFound {len(gz_files)} compressed files")
    
    if not gz_files:
        print("No .gz files found to extract.")
        return
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in gz_files)
    print(f"Total compressed size: {total_size / (1024**3):.2f} GB")
    
    # Extract files with progress bar
    print("\nExtracting files...")
    
    success_count = 0
    error_count = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(extract_gz_file, gz_path, keep_original): gz_path 
            for gz_path in gz_files
        }
        
        with tqdm(total=len(gz_files), unit='file') as pbar:
            for future in as_completed(futures):
                gz_path = futures[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        error_count += 1
                except Exception as e:
                    print(f"Error processing {gz_path}: {e}")
                    error_count += 1
                pbar.update(1)
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Successfully extracted: {success_count}")
    print(f"Errors: {error_count}")


def verify_extraction(features_dir: str) -> None:
    """Verify that features have been extracted correctly."""
    features_dir = Path(features_dir)
    
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Count file types
    gz_files = list(features_dir.rglob("*.gz"))
    npy_files = list(features_dir.rglob("*.npy"))
    
    print(f"Remaining .gz files: {len(gz_files)}")
    print(f"Extracted .npy files: {len(npy_files)}")
    
    # Sample a few files
    if npy_files:
        print("\nSample extracted files:")
        import numpy as np
        
        for npy_file in npy_files[:3]:
            try:
                data = np.load(npy_file)
                print(f"  {npy_file.name}: shape={data.shape}, dtype={data.dtype}")
            except Exception as e:
                print(f"  {npy_file.name}: Error loading - {e}")


def main():
    parser = argparse.ArgumentParser(description='Extract BOBSL feature files')
    parser.add_argument('--features_dir', type=str, default='data/processed/features',
                        help='Path to features directory')
    parser.add_argument('--type', type=str, default='all', choices=['all', 'i3d', 'swin'],
                        help='Feature type to extract')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers')
    parser.add_argument('--keep', action='store_true',
                        help='Keep original .gz files after extraction')
    parser.add_argument('--verify', action='store_true',
                        help='Only verify extraction status, do not extract')
    
    args = parser.parse_args()
    
    if args.verify:
        verify_extraction(args.features_dir)
    else:
        extract_features(
            features_dir=args.features_dir,
            feature_type=args.type,
            num_workers=args.workers,
            keep_original=args.keep
        )
        verify_extraction(args.features_dir)


if __name__ == "__main__":
    main()
