#!/usr/bin/env python3
"""
Explore the actual structure of feature directories.
"""

from pathlib import Path


def explore_features(base_dir: str) -> None:
    """Explore feature directory structure."""
    base_dir = Path(base_dir)
    features_base = base_dir / "features" / "bobsl" / "v1.4" / "video_features"
    
    print("=" * 70)
    print("FEATURE DIRECTORY STRUCTURE")
    print("=" * 70)
    print(f"Base: {features_base}")
    print(f"Exists: {features_base.exists()}")
    
    if not features_base.exists():
        print("Features base directory not found")
        return
    
    # List immediate children
    print("\n--- Top-level contents ---")
    for item in sorted(features_base.iterdir()):
        if item.is_dir():
            sub_count = len(list(item.iterdir()))
            print(f"  {item.name}/  ({sub_count} items)")
        else:
            print(f"  {item.name}  ({item.stat().st_size / 1024:.1f} KB)")
    
    # Explore Swin directory
    swin_dir = features_base / "swin_v1"
    if swin_dir.exists():
        print(f"\n--- swin_v1/ contents ---")
        for item in sorted(swin_dir.iterdir())[:10]:
            if item.is_dir():
                sub_count = len(list(item.iterdir()))
                print(f"  {item.name}/  ({sub_count} items)")
            else:
                print(f"  {item.name}  ({item.stat().st_size / 1024:.1f} KB)")
        
        # Go deeper
        for subdir in swin_dir.iterdir():
            if subdir.is_dir():
                print(f"\n--- swin_v1/{subdir.name}/ contents ---")
                items = list(subdir.iterdir())
                
                # Check if items are directories or files
                dirs = [i for i in items if i.is_dir()]
                files = [i for i in items if i.is_file()]
                
                print(f"  Directories: {len(dirs)}")
                print(f"  Files: {len(files)}")
                
                # Show sample directories
                if dirs:
                    print("\n  Sample directories:")
                    for d in dirs[:5]:
                        d_files = list(d.glob("*.npy"))
                        print(f"    {d.name}/  ({len(d_files)} .npy files)")
                        for f in d_files[:2]:
                            print(f"      {f.name}")
                
                # Show sample files
                if files:
                    print("\n  Sample files:")
                    for f in files[:5]:
                        print(f"    {f.name}  ({f.stat().st_size / 1024 / 1024:.1f} MB)")
                
                break
    
    # Explore I3D directory
    i3d_dir = features_base / "i3d"
    if i3d_dir.exists():
        print(f"\n--- i3d/ contents ---")
        for item in sorted(i3d_dir.iterdir())[:10]:
            if item.is_dir():
                sub_count = len(list(item.iterdir()))
                print(f"  {item.name}/  ({sub_count} items)")
            else:
                print(f"  {item.name}  ({item.stat().st_size / 1024:.1f} KB)")
    
    # Find all .npy files and show their locations
    print("\n--- All .npy file locations ---")
    npy_files = list(features_base.rglob("*.npy"))
    print(f"Total .npy files: {len(npy_files)}")
    
    if npy_files:
        # Group by parent directory
        parents = {}
        for f in npy_files:
            parent = f.parent
            if parent not in parents:
                parents[parent] = []
            parents[parent].append(f)
        
        print(f"\nUnique parent directories: {len(parents)}")
        print("\nSample locations:")
        for parent, files in list(parents.items())[:5]:
            rel_path = parent.relative_to(features_base)
            print(f"  {rel_path}/  ({len(files)} files)")
            for f in files[:2]:
                print(f"    {f.name}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Explore feature directory')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    
    args = parser.parse_args()
    explore_features(args.data_dir)
