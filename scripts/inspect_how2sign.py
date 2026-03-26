import pickle
from pathlib import Path
import json

print("="*70)
print("HOW2SIGN DATASET INSPECTION")
print("="*70)

# Check file names for label patterns
h2s_path = Path("data/signavatars/how2sign")
pkl_files = list(h2s_path.rglob("*.pkl"))

print(f"Total files: {len(pkl_files)}")
print(f"\nSample file names (first 20):")
for f in pkl_files[:20]:
    print(f"  {f.name}")

# Check if there's any pattern in filenames
print(f"\nAnalyzing filename patterns...")
prefixes = {}
for f in pkl_files[:1000]:
    # Get prefix before first underscore
    name = f.stem
    parts = name.split("_")
    if len(parts) > 1:
        prefix = parts[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

print(f"Unique prefixes in first 1000 files: {len(prefixes)}")
print(f"Top 10 prefixes:")
for p, c in sorted(prefixes.items(), key=lambda x: -x[1])[:10]:
    print(f"  {p}: {c}")

# Check subdirectories
print(f"\nSubdirectories:")
for d in h2s_path.iterdir():
    if d.is_dir():
        count = len(list(d.rglob("*.pkl")))
        print(f"  {d.name}: {count} files")

# Look for any label/annotation files
print(f"\nLooking for label files...")
for ext in ["*.json", "*.txt", "*.csv"]:
    for f in h2s_path.rglob(ext):
        print(f"  Found: {f}")
