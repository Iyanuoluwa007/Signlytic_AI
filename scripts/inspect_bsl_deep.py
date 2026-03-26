from pathlib import Path
import json
import pickle
import numpy as np

print("="*70)
print("BSL DATA DEEP INSPECTION")
print("="*70)

# 1. BSLDict - this has word mappings!
print("\n1. BSLDICT VOCABULARY:")
bsldict_pkl = Path("data/bsldict/bsldict/bsldict_v1.pkl")
with open(bsldict_pkl, 'rb') as f:
    bsldict = pickle.load(f)

print(f"   Keys: {list(bsldict.keys())}")
print(f"   Words count: {len(bsldict.get('words', []))}")
print(f"   Videos count: {len(bsldict.get('videos', []))}")

# Sample words
words = bsldict.get('words', [])
print(f"   Sample words: {words[:20]}")

# Video map
video_map_path = Path("data/bsldict/bsldict/bsldict_video_map.json")
with open(video_map_path, 'r') as f:
    video_map = json.load(f)
print(f"   Video map entries: {len(video_map)}")

# 2. BOBSL Features - pre-extracted SWIN features!
print("\n" + "="*70)
print("2. BOBSL SWIN FEATURES:")
bobsl_features = Path("data/processed/features/bobsl")
npy_files = list(bobsl_features.rglob("*.npy"))
print(f"   Total NPY files: {len(npy_files)}")

# Check shape of features
if npy_files:
    sample = np.load(npy_files[0])
    print(f"   Sample file: {npy_files[0].name}")
    print(f"   Shape: {sample.shape}")
    print(f"   Dtype: {sample.dtype}")

# 3. BSL-1K Labels
print("\n" + "="*70)
print("3. BSL-1K LABELS:")
bsl1k = Path("data/BSL-1K")
for subdir in bsl1k.iterdir():
    if subdir.is_dir():
        print(f"\n   {subdir.name}/")
        for bobsl_dir in subdir.iterdir():
            if bobsl_dir.is_dir():
                print(f"     {bobsl_dir.name}/")
                for f in bobsl_dir.iterdir():
                    print(f"       - {f.name} ({f.stat().st_size} bytes)")
                    
                    # Try to read
                    if f.suffix == '.csv':
                        with open(f, 'r') as fp:
                            lines = fp.readlines()
                        print(f"         Lines: {len(lines)}")
                        print(f"         Header: {lines[0].strip() if lines else 'empty'}")
                        print(f"         Sample: {lines[1].strip() if len(lines) > 1 else 'empty'}")
                    elif f.suffix == '.pkl':
                        with open(f, 'rb') as fp:
                            data = pickle.load(fp)
                        if isinstance(data, dict):
                            print(f"         Keys: {list(data.keys())[:5]}")
                        elif isinstance(data, list):
                            print(f"         Items: {len(data)}")

# 4. Check BSL videos
print("\n" + "="*70)
print("4. BSL VIDEOS:")
bsl_videos = Path("data/videos/bsl_signs")
if bsl_videos.exists():
    videos = list(bsl_videos.glob("*.mp4"))
    print(f"   MP4 files: {len(videos)}")
    print(f"   Sample names: {[v.stem for v in videos[:10]]}")

print("\n" + "="*70)
print("SUMMARY: WHAT WE CAN USE")
print("="*70)
