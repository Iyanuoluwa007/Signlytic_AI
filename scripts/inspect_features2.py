# scripts/inspect_features2.py
import numpy as np
import json
from pathlib import Path

BASE = Path(r"D:\Signlytic_AI\code\bsl_translation_project\data")

# 1. index.json — it's a list, show first few items
index_path = BASE / "bsl_dict_features" / "index.json"
with open(index_path) as f:
    idx = json.load(f)
print(f"=== bsl_dict_features/index.json ===")
print(f"  Type: {type(idx)}, Length: {len(idx)}")
print(f"  First 3 items: {idx[:3]}")

# 2. bsl_video_glosses.json
gloss_path = BASE / "bsl_video_glosses.json"
with open(gloss_path) as f:
    glosses = json.load(f)
print(f"\n=== bsl_video_glosses.json ===")
print(f"  Type: {type(glosses)}, Length: {len(glosses)}")
if isinstance(glosses, dict):
    keys = list(glosses.keys())
    print(f"  First 3 keys: {keys[:3]}")
    print(f"  First entry: {glosses[keys[0]]}")
elif isinstance(glosses, list):
    print(f"  First 3 items: {glosses[:3]}")

# 3. Scan ALL subdirs for videos (excluding BOBSL)
print(f"\n=== Non-BOBSL video directories ===")
for d in sorted(BASE.iterdir()):
    if not d.is_dir(): continue
    if "bobsl" in d.name.lower(): continue
    vids = list(d.rglob("*.mp4")) + list(d.rglob("*.avi")) + list(d.rglob("*.mov"))
    if vids:
        print(f"  {d.name}/  — {len(vids)} videos")
        for v in vids[:2]:
            print(f"    {v.relative_to(BASE)}")

# 4. Check if there's a BSL dict video folder specifically
print(f"\n=== Searching for 'bsl_dict' or 'dictionary' video folders ===")
for p in BASE.rglob("*"):
    if p.is_dir() and any(k in p.name.lower() for k in ["dict", "dictionary", "bsl_dict"]):
        vids = list(p.rglob("*.mp4")) + list(p.rglob("*.avi"))
        print(f"  {p.relative_to(BASE)}  — {len(vids)} videos")