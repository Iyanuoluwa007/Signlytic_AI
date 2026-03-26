import pickle
from pathlib import Path
import json
from collections import Counter

print("="*70)
print("HAMNOSYS DATASET INSPECTION")
print("="*70)

hamnosys_path = Path("data/signavatars/hamnosys")
pkl_files = list(hamnosys_path.rglob("*.pkl"))

print(f"Total files: {len(pkl_files)}")
print(f"\nSample file names:")
for f in pkl_files[:30]:
    print(f"  {f.name}")

# Analyze filename patterns - HamNoSys files often have language prefix
prefixes = Counter()
for f in pkl_files:
    name = f.stem
    # Pattern seems to be: language_number.pkl (e.g., pjm_3253.pkl)
    parts = name.split("_")
    if len(parts) >= 1:
        prefix = parts[0]
        prefixes[prefix] += 1

print(f"\nLanguage prefixes found:")
for prefix, count in prefixes.most_common(20):
    print(f"  {prefix}: {count} samples")

# Check if there's a label file in HamNoSys
print(f"\nLooking for annotation files...")
for ext in ["*.json", "*.txt", "*.csv"]:
    for f in hamnosys_path.rglob(ext):
        print(f"  Found: {f}")

# Check pickle content for any label info
print(f"\nInspecting pickle content for labels...")
sample = pkl_files[0]
with open(sample, 'rb') as f:
    data = pickle.load(f)
print(f"Keys in pickle: {list(data.keys())}")
