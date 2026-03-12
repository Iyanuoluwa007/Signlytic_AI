"""
Debug BSL-1K pickle structure in detail.
"""

import pickle
from pathlib import Path
from pprint import pprint


def inspect_pickle_detailed(filepath: Path):
    """Detailed inspection of pickle file."""
    print(f"\n{'='*60}")
    print(f"FILE: {filepath}")
    print('='*60)
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Type: {type(data).__name__}")
    print(f"Keys: {list(data.keys())}")
    
    # Inspect each key
    for key in data.keys():
        value = data[key]
        print(f"\n--- Key: {key} ---")
        print(f"  Type: {type(value).__name__}")
        
        if isinstance(value, list):
            print(f"  Length: {len(value)}")
            if value:
                print(f"  First item type: {type(value[0]).__name__}")
                print(f"  First 3 items: {value[:3]}")
                
                # Check if nested
                if isinstance(value[0], (list, tuple)):
                    print(f"  First item length: {len(value[0])}")
                    print(f"  First item contents: {value[0]}")
        elif isinstance(value, dict):
            print(f"  Keys: {list(value.keys())[:5]}")
        else:
            print(f"  Value: {value}")


def main():
    bsl1k_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/BSL-1K")
    
    # Find all pickles
    pkl_files = list(bsl1k_dir.rglob("*.pkl"))
    
    print(f"Found {len(pkl_files)} pickle files:")
    for f in pkl_files:
        print(f"  {f.relative_to(bsl1k_dir)}")
    
    # Inspect first one in detail
    if pkl_files:
        inspect_pickle_detailed(pkl_files[0])
        
        # Also check second one if different structure
        if len(pkl_files) > 1:
            inspect_pickle_detailed(pkl_files[2])  # exemplars


if __name__ == "__main__":
    main()
