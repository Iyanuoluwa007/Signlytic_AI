"""
Inspect BOBSL annotation CSV format for AI training pipeline.
"""

import os
import csv
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path("D:/Signlytic_AI/code/bsl_translation_project")
ANNOTATIONS_DIR = PROJECT_ROOT / "data/processed/annotations/bobsl/v1.4/manual_annotations"


def inspect_csv(csv_path: Path):
    """Inspect a single CSV file."""
    print(f"\n{'='*60}")
    print(f"File: {csv_path.name}")
    print(f"Size: {csv_path.stat().st_size / 1024:.1f} KB")
    print('='*60)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        # Try to detect delimiter
        sample = f.read(2048)
        f.seek(0)
        
        # Check if it's tab or comma separated
        if '\t' in sample:
            reader = csv.reader(f, delimiter='\t')
        else:
            reader = csv.reader(f)
        
        rows = list(reader)
    
    print(f"Total rows: {len(rows)}")
    
    if rows:
        # Header
        header = rows[0]
        print(f"\nColumns ({len(header)}):")
        for i, col in enumerate(header):
            print(f"  [{i}] {col}")
        
        # Sample data rows
        print(f"\nSample rows:")
        for i, row in enumerate(rows[1:6]):  # First 5 data rows
            print(f"\n  Row {i+1}:")
            for j, (col, val) in enumerate(zip(header, row)):
                val_str = str(val)[:60] + "..." if len(str(val)) > 60 else val
                print(f"    {col}: {val_str}")
    
    return rows


def inspect_all_annotations():
    """Inspect all annotation types."""
    
    print("="*60)
    print("BOBSL ANNOTATION INSPECTION")
    print("="*60)
    
    # 1. Continuous sign sequences (CSLR)
    cslr_dir = ANNOTATIONS_DIR / "continuous_sign_sequences/cslr-raw"
    if cslr_dir.exists():
        print("\n" + "="*60)
        print("CONTINUOUS SIGN SEQUENCES (CSLR)")
        print("="*60)
        
        for split in ['train', 'val', 'test']:
            split_dir = cslr_dir / split
            if split_dir.exists():
                csv_files = list(split_dir.glob("*.csv"))
                print(f"\n{split}: {len(csv_files)} files")
                
                if csv_files:
                    # Inspect first file
                    inspect_csv(csv_files[0])
    
    # 2. Isolated signs
    isolated_dir = ANNOTATIONS_DIR / "isolated_signs"
    if isolated_dir.exists():
        print("\n" + "="*60)
        print("ISOLATED SIGNS")
        print("="*60)
        
        for item in isolated_dir.iterdir():
            if item.is_file():
                print(f"\nFile: {item.name}")
                if item.suffix == '.csv':
                    inspect_csv(item)
                    break
            elif item.is_dir():
                print(f"\nSubdir: {item.name}")
                sub_files = list(item.glob("*"))[:5]
                for sf in sub_files:
                    print(f"  {sf.name}")
    
    # 3. Fingerspelled signs
    finger_dir = ANNOTATIONS_DIR / "fingerspelled_signs"
    if finger_dir.exists():
        print("\n" + "="*60)
        print("FINGERSPELLED SIGNS")
        print("="*60)
        
        for item in list(finger_dir.iterdir())[:5]:
            print(f"  {item.name}")
            if item.is_file() and item.suffix == '.csv':
                inspect_csv(item)
                break
    
    # 4. Signing-aligned subtitles
    subs_dir = ANNOTATIONS_DIR / "signing_aligned_subtitles"
    if subs_dir.exists():
        print("\n" + "="*60)
        print("SIGNING-ALIGNED SUBTITLES")
        print("="*60)
        
        for item in list(subs_dir.iterdir())[:5]:
            print(f"  {item.name}")
            if item.is_file() and item.suffix == '.csv':
                inspect_csv(item)
                break


def count_training_data():
    """Count total training examples."""
    
    print("\n" + "="*60)
    print("TRAINING DATA STATISTICS")
    print("="*60)
    
    cslr_dir = ANNOTATIONS_DIR / "continuous_sign_sequences/cslr-raw"
    
    stats = {}
    all_glosses = []
    
    for split in ['train', 'val', 'test']:
        split_dir = cslr_dir / split
        if not split_dir.exists():
            continue
        
        csv_files = list(split_dir.glob("*.csv"))
        total_rows = 0
        
        for csv_file in csv_files:
            with open(csv_file, 'r', encoding='utf-8') as f:
                if '\t' in f.read(1024):
                    f.seek(0)
                    reader = csv.reader(f, delimiter='\t')
                else:
                    f.seek(0)
                    reader = csv.reader(f)
                
                rows = list(reader)
                total_rows += len(rows) - 1  # Exclude header
                
                # Extract glosses if column exists
                if rows:
                    header = rows[0]
                    if 'gloss' in [h.lower() for h in header]:
                        gloss_idx = [h.lower() for h in header].index('gloss')
                        for row in rows[1:]:
                            if len(row) > gloss_idx:
                                all_glosses.append(row[gloss_idx])
        
        stats[split] = {
            'files': len(csv_files),
            'rows': total_rows
        }
        print(f"\n{split}:")
        print(f"  Files: {len(csv_files)}")
        print(f"  Rows: {total_rows}")
    
    # Gloss statistics
    if all_glosses:
        unique_glosses = set(all_glosses)
        gloss_counts = Counter(all_glosses)
        
        print(f"\nGloss Statistics:")
        print(f"  Total annotations: {len(all_glosses)}")
        print(f"  Unique glosses: {len(unique_glosses)}")
        print(f"\n  Top 20 glosses:")
        for gloss, count in gloss_counts.most_common(20):
            print(f"    {gloss}: {count}")


def main():
    inspect_all_annotations()
    count_training_data()


if __name__ == "__main__":
    main()
