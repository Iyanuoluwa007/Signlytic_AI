import pickle
from pathlib import Path
import json
from collections import Counter
import re

print("="*70)
print("MULTI-DATASET TRAINING ANALYSIS")
print("="*70)

# 1. WLASL - already have labels (1000 files, 124 valid classes)
wlasl_path = Path("data/signavatars/wlasl/video_to_gloss.json")
with open(wlasl_path, 'r') as f:
    wlasl_labels = json.load(f)
    
wlasl_files = list(Path("data/signavatars/wlasl").rglob("*.pkl"))
wlasl_labeled = sum(1 for f in wlasl_files if f.stem in wlasl_labels)
print(f"\n1. WLASL:")
print(f"   Files: {len(wlasl_files)}")
print(f"   With labels: {wlasl_labeled}")
print(f"   Unique glosses: {len(set(wlasl_labels.values()))}")

# 2. HamNoSys - extract labels from filenames
hamnosys_path = Path("data/signavatars/hamnosys")
hamnosys_files = list(hamnosys_path.rglob("*.pkl"))

# Classify by type
pjm_files = []  # Polish - need external labels
gsl_files = []  # Greek - need external labels
french_files = []  # French - filename IS the gloss

for f in hamnosys_files:
    name = f.stem
    if name.startswith("pjm_"):
        pjm_files.append(f)
    elif name.startswith("gsl_"):
        gsl_files.append(f)
    elif name.isdigit():
        pass  # Skip numeric-only (unknown source)
    elif re.match(r'^[A-Z_]+$', name):
        # All caps = French gloss
        french_files.append((f, name.replace("_", " ").strip()))

print(f"\n2. HamNoSys:")
print(f"   Total: {len(hamnosys_files)}")
print(f"   Polish (pjm): {len(pjm_files)} - need labels")
print(f"   Greek (gsl): {len(gsl_files)} - need labels")
print(f"   French (gloss in filename): {len(french_files)} - USABLE")

# Show French glosses
french_glosses = Counter(g for _, g in french_files)
print(f"   French unique glosses: {len(french_glosses)}")
print(f"   Sample French glosses: {list(french_glosses.keys())[:15]}")

# 3. Combined training potential
print(f"\n3. COMBINED TRAINING POTENTIAL:")
print(f"   WLASL (ASL): 1000 samples, ~124 classes")
print(f"   French (LSF): {len(french_files)} samples, {len(french_glosses)} classes")
print(f"   TOTAL: {1000 + len(french_files)} samples")

# Save French mapping
french_mapping = {str(f): g for f, g in french_files}
output_path = Path("data/signavatars/hamnosys/french_gloss_mapping.json")
with open(output_path, 'w') as f:
    json.dump(french_mapping, f, indent=2)
print(f"\nSaved French mapping to {output_path}")

# 4. Try to find SignAvatars annotation files online
print(f"\n4. SIGNAVATARS OFFICIAL ANNOTATIONS:")
print(f"   SignAvatars provides SMPL-X params but labels come from original datasets")
print(f"   - WLASL: Using video_to_gloss.json (downloaded)")
print(f"   - How2Sign: Sentence-level (requires CTC training)")
print(f"   - HamNoSys: French subset usable, pjm/gsl need external files")
