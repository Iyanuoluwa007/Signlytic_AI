# scripts/inspect_pose_format.py
import json
from pathlib import Path

BASE = Path(r"D:\Signlytic_AI\code\bsl_translation_project\data")
poses_dir = BASE / "poses_bsldict"

# 1. Check summary.json
with open(poses_dir / "summary.json") as f:
    summary = json.load(f)
print("=== summary.json ===")
print(json.dumps(summary, indent=2)[:800])

# 2. Check sample_index.json
with open(poses_dir / "sample_index.json") as f:
    idx = json.load(f)
print(f"\n=== sample_index.json ===")
print(f"  Type: {type(idx)}, Length: {len(idx)}")
if isinstance(idx, list):
    print(f"  First 3: {idx[:3]}")
elif isinstance(idx, dict):
    keys = list(idx.keys())[:3]
    for k in keys:
        print(f"  {k}: {idx[k]}")

# 3. Inspect one pose JSON file
train_dir = poses_dir / "train"
sample_files = list(train_dir.glob("*.json"))
print(f"\n=== train/ has {len(sample_files)} files ===")
print(f"  First 3 filenames: {[f.name for f in sample_files[:3]]}")

with open(sample_files[0]) as f:
    pose = json.load(f)

print(f"\n=== Pose file structure: {sample_files[0].name} ===")
print(f"  Top-level keys: {list(pose.keys())}")

for k, v in pose.items():
    if isinstance(v, list):
        print(f"  {k}: list of {len(v)} items")
        if len(v) > 0:
            item = v[0]
            if isinstance(item, list):
                print(f"    item[0]: list of {len(item)} → first element: {item[0] if item else 'empty'}")
            elif isinstance(item, dict):
                print(f"    item[0] keys: {list(item.keys())}")
            else:
                print(f"    item[0]: {item}")
    elif isinstance(v, dict):
        print(f"  {k}: dict keys={list(v.keys())[:6]}")
    else:
        print(f"  {k}: {v}")

# 4. Check if filenames map to glosses
print(f"\n=== Filename → gloss mapping check ===")
for f in sample_files[:5]:
    # e.g. ABOUT_about_0.00.json
    parts = f.stem.split("_")
    print(f"  {f.name} → parts: {parts}")