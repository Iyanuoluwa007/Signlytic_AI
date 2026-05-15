# scripts/inspect_features.py
# Run: conda activate BSL && python scripts/inspect_features.py

import numpy as np
import json
import os
from pathlib import Path

BASE = Path(r"D:\Signlytic_AI\code\bsl_translation_project\data")

# 1. Check shape of a few .npy feature files
print("=== bsl_dict_features sample shapes ===")
feat_dir = BASE / "bsl_dict_features"
samples = list(feat_dir.glob("*.npy"))[:5]
for f in samples:
    arr = np.load(f, allow_pickle=True)
    print(f"  {f.name}: shape={arr.shape}, dtype={arr.dtype}")

# 2. Peek at index.json if it exists
index_path = feat_dir / "index.json"
if index_path.exists():
    with open(index_path) as f:
        idx = json.load(f)
    print(f"\n=== index.json keys: {list(idx.keys())[:10]}")
    # Show first entry
    first_key = list(idx.keys())[0]
    print(f"  First entry ({first_key}): {idx[first_key]}")

# 3. Check bsl_video_glosses.json for video paths
gloss_path = BASE / "bsl_video_glosses.json"
if gloss_path.exists():
    with open(gloss_path) as f:
        glosses = json.load(f)
    print(f"\n=== bsl_video_glosses.json ===")
    print(f"  Type: {type(glosses)}, Length: {len(glosses)}")
    # Show first entry
    if isinstance(glosses, dict):
        first_key = list(glosses.keys())[0]
        print(f"  First key: {first_key}")
        print(f"  First value: {glosses[first_key]}")
    elif isinstance(glosses, list):
        print(f"  First item: {glosses[0]}")

# 4. Scan all data subdirs for video files not in BOBSL
print("\n=== Video dirs (non-BOBSL) ===")
for d in BASE.iterdir():
    if d.is_dir() and "bobsl" not in d.name.lower():
        vids = list(d.rglob("*.mp4")) + list(d.rglob("*.avi")) + list(d.rglob("*.mov"))
        if vids:
            print(f"  {d.name}: {len(vids)} videos")
            print(f"    Sample: {vids[0].name}")