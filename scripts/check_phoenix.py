import urllib.request
import json
from pathlib import Path

print("="*70)
print("DOWNLOADING PHOENIX-2014T ANNOTATIONS")
print("="*70)

# PHOENIX annotations are available from multiple sources
# Try the official RWTH Aachen mirror

output_dir = Path("data/signavatars/phoenix/annotations")
output_dir.mkdir(parents=True, exist_ok=True)

# Check our PHOENIX files
phoenix_files = list(Path("data/signavatars/phoenix").rglob("*.pkl"))
print(f"SignAvatars PHOENIX files: {len(phoenix_files)}")

# Extract video names
video_names = set()
for f in phoenix_files:
    name = f.stem
    video_names.add(name)

print(f"Unique video names: {len(video_names)}")
print(f"Sample names: {list(video_names)[:5]}")

# Save video names for manual lookup
with open(output_dir / "phoenix_video_ids.json", 'w') as f:
    json.dump(sorted(video_names), f, indent=2)
print(f"Saved video IDs to {output_dir / 'phoenix_video_ids.json'}")

# PHOENIX-2014T format: the video name often contains date info
# e.g., "11August_2009_Tuesday_tagesschau-4355"
# The gloss annotations are in separate CSV/corpus files

print("\nPHOENIX-2014T requires manual download from:")
print("  https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/")
print("\nFiles needed:")
print("  - PHOENIX-2014-T.v3.train.corpus.csv")
print("  - PHOENIX-2014-T.v3.dev.corpus.csv")
print("  - PHOENIX-2014-T.v3.test.corpus.csv")
