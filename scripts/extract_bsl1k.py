"""
Extract BSL-1K .tar.gz archives and verify data.

The BSL-1K annotations are stored as compressed archives.
This script extracts them and verifies the contents.

Run before training:
    python scripts/extract_bsl1k.py
"""

import tarfile
import gzip
import json
from pathlib import Path
from collections import Counter


def extract_tar_gz(filepath: Path, output_dir: Path) -> int:
    """Extract .tar.gz and return number of files extracted."""
    count = 0
    
    try:
        with tarfile.open(filepath, 'r:gz') as tar:
            members = tar.getmembers()
            
            for member in members:
                if member.isfile():
                    tar.extract(member, output_dir)
                    count += 1
            
        print(f"  [OK] Extracted {count} files from {filepath.name}")
        return count
    except Exception as e:
        print(f"  [FAIL] {filepath.name}: {e}")
        return 0


def inspect_extracted_json(json_path: Path) -> dict:
    """Inspect extracted JSON file structure."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        stats = {
            'type': type(data).__name__,
            'size': len(data) if hasattr(data, '__len__') else 0,
        }
        
        if isinstance(data, dict):
            stats['keys'] = list(data.keys())[:5]
            # Sample first entry
            first_key = list(data.keys())[0] if data else None
            if first_key:
                stats['sample_key'] = first_key
                stats['sample_value'] = data[first_key]
        elif isinstance(data, list) and data:
            stats['sample'] = data[0]
        
        return stats
    except Exception as e:
        return {'error': str(e)}


def main():
    print("="*60)
    print("BSL-1K ARCHIVE EXTRACTION")
    print("="*60)
    
    bsl1k_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/BSL-1K")
    
    if not bsl1k_dir.exists():
        print(f"[FAIL] BSL-1K directory not found: {bsl1k_dir}")
        return
    
    # Find all .tar.gz files
    tar_files = list(bsl1k_dir.rglob("*.tar.gz"))
    print(f"\nFound {len(tar_files)} .tar.gz archives:")
    
    for tar_file in tar_files:
        print(f"\n--- {tar_file.parent.name}/{tar_file.name} ---")
        print(f"  Size: {tar_file.stat().st_size / 1024 / 1024:.2f} MB")
        
        # Extract to same directory as tar file
        extract_tar_gz(tar_file, tar_file.parent)
    
    # Optional: Delete .tar.gz files after extraction
    response = input("\nDelete original .tar.gz files? (y/n): ")
    if response.lower() == 'y':
        for tar_file in tar_files:
            tar_file.unlink()
            print(f"  Deleted: {tar_file.name}")
    
    
    # Verify extracted content
    print("\n" + "="*60)
    print("VERIFYING EXTRACTED CONTENT")
    print("="*60)
    
    source_dirs = [
        "ISOLATED_SIGN-EXEMPLARS",
        "ISOLATED_SIGN-MOUTHING", 
        "ISOLATED_SIGN-DICTIONARY",
        "ISOLATED_SIGN-I3D_PSEUDO_LABELS"
    ]
    
    for source in source_dirs:
        source_dir = bsl1k_dir / source
        
        if not source_dir.exists():
            print(f"\n[SKIP] {source}: directory not found")
            continue
        
        print(f"\n--- {source} ---")
        
        # Count files by type
        files = list(source_dir.rglob("*"))
        file_types = Counter(f.suffix.lower() for f in files if f.is_file())
        print(f"  Files: {dict(file_types)}")
        
        # Inspect JSON files
        json_files = list(source_dir.glob("*.json"))
        if json_files:
            print(f"  JSON files: {len(json_files)}")
            
            # Inspect first one
            sample = json_files[0]
            print(f"  Sample: {sample.name}")
            
            stats = inspect_extracted_json(sample)
            for key, value in stats.items():
                print(f"    {key}: {value}")
    
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE")
    print("="*60)
    print("""
Next steps:
1. Run: python scripts/check_v2_integration.py
2. If checks pass, start training:
   python scripts/train_v2.py --config configs/recognition_v2_fixed.yaml
""")


if __name__ == "__main__":
    main()


