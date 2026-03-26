from pathlib import Path
import json
import numpy as np
from collections import Counter

print("="*70)
print("BSL TRAINING DATA SETUP")
print("="*70)

# 1. BSL Videos - filename IS the gloss
print("\n1. BSL VIDEOS (5203 files):")
bsl_videos = Path("data/videos/bsl_signs")
videos = list(bsl_videos.glob("*.mp4"))

# Extract glosses from filenames
gloss_to_videos = {}
for v in videos:
    gloss = v.stem.upper()
    if gloss not in gloss_to_videos:
        gloss_to_videos[gloss] = []
    gloss_to_videos[gloss].append(v)

print(f"   Total videos: {len(videos)}")
print(f"   Unique glosses: {len(gloss_to_videos)}")

# Count samples per gloss
gloss_counts = {g: len(v) for g, v in gloss_to_videos.items()}
count_dist = Counter(gloss_counts.values())
print(f"   Distribution of samples per gloss:")
for count, num_glosses in sorted(count_dist.items())[:10]:
    print(f"     {count} sample(s): {num_glosses} glosses")

# Glosses with multiple samples (better for training)
multi_sample = {g: c for g, c in gloss_counts.items() if c >= 2}
print(f"   Glosses with >= 2 samples: {len(multi_sample)}")

# Sample glosses
print(f"   Sample glosses: {list(gloss_to_videos.keys())[:20]}")

# 2. Check BOBSL Features format
print("\n" + "="*70)
print("2. BOBSL SWIN FEATURES:")
bobsl_features = Path("data/processed/features/bobsl")
npy_files = list(bobsl_features.rglob("*.npy"))

# Map feature file IDs
feature_ids = {f.stem for f in npy_files}
print(f"   Feature files: {len(feature_ids)}")

# Load one to check
sample = np.load(npy_files[0])
print(f"   Shape: {sample.shape} (frames x 768)")
print(f"   This is CONTINUOUS video features, not isolated signs")

# 3. Check if there's a mapping between videos and glosses
print("\n" + "="*70)
print("3. BOBSL METADATA:")
metadata_path = Path("data/processed/metadata/bobsl")
if metadata_path.exists():
    for f in metadata_path.iterdir():
        print(f"   - {f.name} ({f.stat().st_size} bytes)")

# 4. Create BSL training dataset from videos
print("\n" + "="*70)
print("4. BSL VIDEO TRAINING POTENTIAL:")
print(f"""
   We can train on BSL videos directly:
   - 5,203 videos with gloss labels (filename = gloss)
   - Need to extract features (SWIN or pose)
   
   Options:
   A) Extract SWIN features from BSL videos (GPU intensive)
   B) Extract pose with MediaPipe (faster, CPU)
   C) Use existing BOBSL features if we can map them
""")

# Save gloss mapping
output_path = Path("data/bsl_video_glosses.json")
mapping = {str(v): g for g, vids in gloss_to_videos.items() for v in vids}
with open(output_path, 'w') as f:
    json.dump(mapping, f, indent=2)
print(f"\nSaved gloss mapping to {output_path}")
print(f"Total video-to-gloss mappings: {len(mapping)}")
