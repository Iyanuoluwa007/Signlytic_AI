from pathlib import Path
import json

print("="*70)
print("DEBUGGING BSL DATA PATHS")
print("="*70)

# Check feature files
features_dir = Path("data/processed/features/bobsl")
feature_files = list(features_dir.rglob("*.npy"))
print(f"\n1. Feature files location: {features_dir}")
print(f"   Total NPY files: {len(feature_files)}")
if feature_files:
    print(f"   Sample paths:")
    for f in feature_files[:5]:
        print(f"     {f}")
    print(f"   Sample stems (IDs):")
    for f in feature_files[:5]:
        print(f"     {f.stem}")

# Check annotation video IDs
with open("data/bsl1k_training_data.json", 'r') as f:
    data = json.load(f)

videos = data['videos']
print(f"\n2. Annotation video IDs:")
print(f"   Total: {len(videos)}")
print(f"   Sample IDs:")
for v in videos[:5]:
    print(f"     {v}")

# Check for overlap
feature_ids = {f.stem for f in feature_files}
annotation_ids = set(videos)

overlap = feature_ids & annotation_ids
print(f"\n3. Overlap check:")
print(f"   Feature IDs: {len(feature_ids)}")
print(f"   Annotation IDs: {len(annotation_ids)}")
print(f"   Overlapping: {len(overlap)}")

if not overlap:
    print("\n   NO OVERLAP - checking format differences...")
    print(f"   Feature ID sample: {list(feature_ids)[:3]}")
    print(f"   Annotation ID sample: {list(annotation_ids)[:3]}")
