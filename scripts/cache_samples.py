"""
Pre-cache pose sample index to speed up training.

Run once:
    python scripts/cache_samples.py

Then training will be instant to start.
"""

import json
from pathlib import Path
from tqdm import tqdm

project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
data_dir = project_root / "data" / "poses"
cache_file = data_dir / "sample_index.json"

def build_cache():
    """Build sample index cache."""
    all_samples = []
    
    for split_name in ['train', 'val', 'test']:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            continue
        
        json_files = list(split_dir.glob("*.json"))
        print(f"Indexing {split_name}: {len(json_files)} files...")
        
        for json_file in tqdm(json_files, desc=f"  {split_name}"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                all_samples.append({
                    'gloss': data['gloss'],
                    'file': str(json_file),
                    'num_frames': data['num_frames']
                })
            except Exception as e:
                continue
    
    # Save cache
    with open(cache_file, 'w') as f:
        json.dump(all_samples, f)
    
    print(f"\nCached {len(all_samples)} samples to {cache_file}")
    print(f"Cache size: {cache_file.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    build_cache()
