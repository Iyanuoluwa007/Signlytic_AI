"""
Inspect BSL-1K pickle files to understand their structure.

Run this to see what format the annotations are in.
"""

import pickle
from pathlib import Path
from pprint import pprint


def inspect_pickle(filepath: Path):
    """Inspect a pickle file."""
    print(f"\n{'='*60}")
    print(f"FILE: {filepath.name}")
    print(f"SIZE: {filepath.stat().st_size / 1024:.1f} KB")
    print('='*60)
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"TYPE: {type(data).__name__}")
        
        if isinstance(data, dict):
            print(f"KEYS: {len(data)} entries")
            keys = list(data.keys())
            print(f"Sample keys: {keys[:5]}")
            
            # Show first entry
            if keys:
                first_key = keys[0]
                first_value = data[first_key]
                print(f"\nFirst entry:")
                print(f"  Key: {first_key}")
                print(f"  Value type: {type(first_value).__name__}")
                
                if isinstance(first_value, list):
                    print(f"  Value length: {len(first_value)}")
                    if first_value:
                        print(f"  First item type: {type(first_value[0]).__name__}")
                        print(f"  First item: {first_value[0]}")
                elif isinstance(first_value, dict):
                    print(f"  Value keys: {list(first_value.keys())}")
                    pprint(first_value, depth=2)
                else:
                    print(f"  Value: {first_value}")
                    
        elif isinstance(data, list):
            print(f"LENGTH: {len(data)} items")
            if data:
                print(f"First item type: {type(data[0]).__name__}")
                print(f"First item:")
                pprint(data[0], depth=3)
                
        else:
            print(f"DATA: {data}")
            
    except Exception as e:
        print(f"ERROR: {e}")


def main():
    bsl1k_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/BSL-1K")
    
    print("INSPECTING BSL-1K PICKLE FILES")
    print("="*60)
    
    # Find all pickle files
    pkl_files = list(bsl1k_dir.rglob("*.pkl"))
    print(f"Found {len(pkl_files)} pickle files:\n")
    
    for pkl_file in pkl_files:
        print(f"  {pkl_file.parent.name}/{pkl_file.name}")
    
    # Inspect each
    for pkl_file in pkl_files:
        inspect_pickle(pkl_file)


if __name__ == "__main__":
    main()
