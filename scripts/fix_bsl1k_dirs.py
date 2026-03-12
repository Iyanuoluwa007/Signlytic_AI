"""
Fix BSL-1K directory naming issues.

The "ISOLATED_SIGN- DICTIONARY" folder has an extra space.
This script renames it to the correct name.
"""

import os
import shutil
from pathlib import Path


def main():
    bsl1k_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/BSL-1K")
    
    print("="*60)
    print("BSL-1K DIRECTORY FIX")
    print("="*60)
    
    if not bsl1k_dir.exists():
        print(f"[FAIL] BSL-1K directory not found: {bsl1k_dir}")
        return
    
    # List current directories
    print("\nCurrent directories:")
    for item in sorted(bsl1k_dir.iterdir()):
        if item.is_dir():
            files = list(item.glob("*"))
            print(f"  {item.name}: {len(files)} files")
    
    # Check for directory with space issue
    bad_dir = bsl1k_dir / "ISOLATED_SIGN- DICTIONARY"
    good_dir = bsl1k_dir / "ISOLATED_SIGN-DICTIONARY"
    
    if bad_dir.exists():
        print(f"\n[FOUND] Directory with space: {bad_dir.name}")
        
        if good_dir.exists():
            print(f"[WARN] Target already exists: {good_dir.name}")
            # Merge files
            for f in bad_dir.iterdir():
                target = good_dir / f.name
                if not target.exists():
                    shutil.move(str(f), str(target))
                    print(f"  Moved: {f.name}")
            
            # Remove empty bad dir
            try:
                bad_dir.rmdir()
                print(f"[OK] Removed empty: {bad_dir.name}")
            except:
                print(f"[WARN] Could not remove: {bad_dir.name}")
        else:
            # Simple rename
            bad_dir.rename(good_dir)
            print(f"[OK] Renamed to: {good_dir.name}")
    else:
        print(f"\n[OK] No space issue found in directory names")
    
    # Verify
    print("\nVerified directories:")
    for item in sorted(bsl1k_dir.iterdir()):
        if item.is_dir():
            files = list(item.glob("*"))
            print(f"  {item.name}: {len(files)} files")


if __name__ == "__main__":
    main()
