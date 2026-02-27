#!/usr/bin/env python3
"""
Diagnose feature data issues (NaN, Inf, dtype).
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.annotation_parser import BOBSLAnnotationParser


def diagnose_features(data_dir: str) -> None:
    """Check features for NaN, Inf, and dtype issues."""
    
    print("=" * 70)
    print("FEATURE DIAGNOSTICS")
    print("=" * 70)
    
    parser = BOBSLAnnotationParser(data_dir)
    
    # Get sample feature files
    swin_cache = parser._feature_cache.get('swin', {})
    print(f"\nSwin features: {len(swin_cache)} files")
    
    if not swin_cache:
        print("No features found!")
        return
    
    # Check first 10 files
    print("\nChecking first 10 feature files...")
    
    issues = {'nan': 0, 'inf': 0, 'float16': 0}
    
    for i, (video_id, path) in enumerate(list(swin_cache.items())[:10]):
        data = np.load(path)
        
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        is_float16 = data.dtype == np.float16
        
        if has_nan:
            issues['nan'] += 1
        if has_inf:
            issues['inf'] += 1
        if is_float16:
            issues['float16'] += 1
        
        print(f"\n  File {i+1}: {video_id}")
        print(f"    Shape: {data.shape}")
        print(f"    Dtype: {data.dtype}")
        print(f"    Range: [{data.min():.4f}, {data.max():.4f}]")
        print(f"    Mean: {data.mean():.4f}, Std: {data.std():.4f}")
        print(f"    Has NaN: {has_nan}, Has Inf: {has_inf}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files with NaN: {issues['nan']}/10")
    print(f"Files with Inf: {issues['inf']}/10")
    print(f"Files with float16: {issues['float16']}/10")
    
    if issues['float16'] > 0:
        print("\nISSUE DETECTED: Features are float16")
        print("This can cause NaN during training due to limited precision.")
        print("Solution: Convert to float32 in the dataset.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/processed')
    args = parser.parse_args()
    
    diagnose_features(args.data_dir)


if __name__ == "__main__":
    main()
