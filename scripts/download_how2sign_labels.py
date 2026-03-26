import urllib.request
import json
from pathlib import Path
import csv

print("="*70)
print("DOWNLOADING HOW2SIGN ANNOTATIONS")
print("="*70)

# How2Sign has sentence-level annotations
# The official dataset has CSV files with video_id -> sentence mappings

# Try to get annotations from How2Sign GitHub
urls = [
    ("https://raw.githubusercontent.com/how2sign/how2sign/main/how2sign_realigned_train.csv", "train"),
    ("https://raw.githubusercontent.com/how2sign/how2sign/main/how2sign_realigned_val.csv", "val"),
    ("https://raw.githubusercontent.com/how2sign/how2sign/main/how2sign_realigned_test.csv", "test"),
]

output_dir = Path("data/signavatars/how2sign/annotations")
output_dir.mkdir(parents=True, exist_ok=True)

for url, split in urls:
    try:
        output_path = output_dir / f"{split}.csv"
        print(f"Downloading {split}...")
        urllib.request.urlretrieve(url, output_path)
        
        # Count lines
        with open(output_path, 'r', encoding='utf-8') as f:
            lines = len(f.readlines()) - 1  # minus header
        print(f"  {split}: {lines} samples")
    except Exception as e:
        print(f"  {split}: Failed - {e}")

# If that fails, let's check if we can use the video IDs to extract glosses
print("\n" + "="*70)
print("ANALYZING SIGNAVATARS HOW2SIGN FILES")
print("="*70)

h2s_files = list(Path("data/signavatars/how2sign").rglob("*.pkl"))
print(f"Total SignAvatars files: {len(h2s_files)}")

# Extract video IDs from filenames
video_ids = set()
for f in h2s_files:
    # Format: videoID_segment-signer-rgb_front.pkl
    name = f.stem
    # Extract the video ID part
    parts = name.replace("-rgb_front", "").split("_")
    if len(parts) >= 2:
        video_id = parts[0]
        video_ids.add(video_id)

print(f"Unique video IDs: {len(video_ids)}")
print(f"Sample video IDs: {list(video_ids)[:10]}")

# Save video ID list
with open(output_dir / "video_ids.json", 'w') as f:
    json.dump(list(video_ids), f)
print(f"\nSaved video IDs to {output_dir / 'video_ids.json'}")
