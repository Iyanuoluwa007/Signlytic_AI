"""
Inspect all BSL datasets to understand their structure before integration.

Datasets:
1. SWIN Features V1 - Temporal video embeddings
2. BSL-1K annotations - Weak supervision labels  
3. Leap Motion - Hand position supervision

Run this first to understand data formats before building the pipeline.
"""

import os
import json
import tarfile
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path("D:/Signlytic_AI/code/bsl_translation_project")


def inspect_leap_motion():
    """Inspect Leap Motion hand tracking data."""
    print("\n" + "="*60)
    print("LEAP MOTION DATASET")
    print("="*60)
    
    csv_path = PROJECT_ROOT / "data" / "Leap_Motion" / "BSL-leap-motion.csv"
    
    if not csv_path.exists():
        print(f"NOT FOUND: {csv_path}")
        return None
    
    print(f"Path: {csv_path}")
    print(f"Size: {csv_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"\nShape: {df.shape}")
    print(f"Columns ({len(df.columns)}):")
    
    # Show column groups
    cols = df.columns.tolist()
    print(f"  First 10: {cols[:10]}")
    print(f"  Last 10: {cols[-10:]}")
    
    # Sample data
    print(f"\nSample row:")
    print(df.iloc[0])
    
    # Check for gloss/label column
    label_cols = [c for c in cols if 'label' in c.lower() or 'gloss' in c.lower() or 'sign' in c.lower()]
    print(f"\nPotential label columns: {label_cols}")
    
    if label_cols:
        label_col = label_cols[0]
        unique_labels = df[label_col].nunique()
        print(f"  Unique labels in '{label_col}': {unique_labels}")
        print(f"  Sample labels: {df[label_col].value_counts().head(10).to_dict()}")
    
    return df


def inspect_bsl1k():
    """Inspect BSL-1K annotation structure."""
    print("\n" + "="*60)
    print("BSL-1K ANNOTATIONS")
    print("="*60)
    
    bsl1k_dir = PROJECT_ROOT / "data" / "BSL-1K"
    
    if not bsl1k_dir.exists():
        print(f"NOT FOUND: {bsl1k_dir}")
        return None
    
    print(f"Path: {bsl1k_dir}")
    
    # List subdirectories
    subdirs = [d for d in bsl1k_dir.iterdir() if d.is_dir()]
    print(f"\nSubdirectories ({len(subdirs)}):")
    
    all_annotations = {}
    
    for subdir in sorted(subdirs):
        print(f"\n--- {subdir.name} ---")
        
        # Count files by type
        files = list(subdir.rglob("*"))
        file_types = Counter(f.suffix for f in files if f.is_file())
        print(f"  Files: {len(files)}")
        print(f"  Types: {dict(file_types)}")
        
        # Try to load sample file
        json_files = list(subdir.glob("*.json"))
        csv_files = list(subdir.glob("*.csv"))
        txt_files = list(subdir.glob("*.txt"))
        
        if json_files:
            sample = json_files[0]
            print(f"  Sample JSON: {sample.name}")
            try:
                with open(sample, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    print(f"    Keys: {list(data.keys())[:10]}")
                    # Get first item
                    first_key = list(data.keys())[0]
                    print(f"    First item ({first_key}): {data[first_key]}")
                elif isinstance(data, list):
                    print(f"    List length: {len(data)}")
                    print(f"    First item: {data[0]}")
                all_annotations[subdir.name] = data
            except Exception as e:
                print(f"    Error loading: {e}")
        
        if csv_files:
            sample = csv_files[0]
            print(f"  Sample CSV: {sample.name}")
            try:
                df = pd.read_csv(sample, nrows=5)
                print(f"    Columns: {df.columns.tolist()}")
                print(f"    Shape: {df.shape}")
            except Exception as e:
                print(f"    Error loading: {e}")
        
        if txt_files:
            sample = txt_files[0]
            print(f"  Sample TXT: {sample.name}")
            try:
                with open(sample, 'r') as f:
                    lines = f.readlines()[:5]
                print(f"    First lines: {[l.strip() for l in lines]}")
            except Exception as e:
                print(f"    Error loading: {e}")
    
    return all_annotations


def inspect_swin_features():
    """Inspect SWIN feature archive structure."""
    print("\n" + "="*60)
    print("SWIN FEATURES V1")
    print("="*60)
    
    tar_path = PROJECT_ROOT / "data" / "raw" / "bobsl_v1_4_features_swin_v1_3.tar"
    
    if not tar_path.exists():
        print(f"NOT FOUND: {tar_path}")
        # Check if already extracted
        extract_dir = PROJECT_ROOT / "data" / "swin_features"
        if extract_dir.exists():
            print(f"Found extracted directory: {extract_dir}")
            inspect_extracted_swin(extract_dir)
        return None
    
    print(f"Path: {tar_path}")
    print(f"Size: {tar_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")
    
    # List contents without extracting
    print("\nInspecting archive contents (first 20 files)...")
    
    try:
        with tarfile.open(tar_path, 'r') as tar:
            members = tar.getmembers()[:50]
            
            print(f"Total members (sampled): {len(members)}")
            
            # Show structure
            dirs = set()
            files_by_ext = Counter()
            
            for m in members:
                if m.isdir():
                    dirs.add(m.name)
                else:
                    ext = Path(m.name).suffix
                    files_by_ext[ext] += 1
                    
            print(f"Directories: {sorted(dirs)[:10]}")
            print(f"File types: {dict(files_by_ext)}")
            
            # Try to extract and inspect one .npy file
            npy_members = [m for m in tar.getmembers() if m.name.endswith('.npy')]
            if npy_members:
                print(f"\nFound {len(npy_members)} .npy files")
                sample_member = npy_members[0]
                print(f"Sample: {sample_member.name}")
                
                # Extract to memory
                f = tar.extractfile(sample_member)
                if f:
                    data = np.load(f)
                    print(f"  Shape: {data.shape}")
                    print(f"  Dtype: {data.dtype}")
                    print(f"  Min/Max: {data.min():.4f} / {data.max():.4f}")
                    print(f"  Mean/Std: {data.mean():.4f} / {data.std():.4f}")
                    
                    if len(data.shape) == 2:
                        print(f"  Interpretation: T={data.shape[0]} frames, D={data.shape[1]} features")
    
    except Exception as e:
        print(f"Error inspecting archive: {e}")
    
    return tar_path


def inspect_extracted_swin(extract_dir):
    """Inspect already extracted SWIN features."""
    print(f"\nInspecting extracted features: {extract_dir}")
    
    npy_files = list(extract_dir.rglob("*.npy"))
    print(f"Total .npy files: {len(npy_files)}")
    
    if npy_files:
        sample = npy_files[0]
        print(f"\nSample: {sample.name}")
        data = np.load(sample)
        print(f"  Shape: {data.shape}")
        print(f"  Dtype: {data.dtype}")
        
        if len(data.shape) == 2:
            print(f"  T={data.shape[0]} frames, D={data.shape[1]} features")


def inspect_existing_poses():
    """Inspect existing extracted poses for comparison."""
    print("\n" + "="*60)
    print("EXISTING POSE DATA")
    print("="*60)
    
    poses_dir = PROJECT_ROOT / "data" / "poses"
    
    if not poses_dir.exists():
        print(f"NOT FOUND: {poses_dir}")
        return
    
    # Count files
    for split in ['train', 'val', 'test']:
        split_dir = poses_dir / split
        if split_dir.exists():
            files = list(split_dir.glob("*.json"))
            print(f"  {split}: {len(files)} files")


def main():
    print("="*60)
    print("BSL DATASET INSPECTION")
    print("="*60)
    print(f"Project root: {PROJECT_ROOT}")
    
    # 1. Leap Motion
    leap_df = inspect_leap_motion()
    
    # 2. BSL-1K
    bsl1k_data = inspect_bsl1k()
    
    # 3. SWIN Features
    swin_path = inspect_swin_features()
    
    # 4. Existing poses
    inspect_existing_poses()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("""
Next steps:
1. Extract SWIN features if not done
2. Parse BSL-1K annotations into training format
3. Integrate Leap Motion as auxiliary supervision
4. Build temporal recognition model
""")


if __name__ == "__main__":
    main()
