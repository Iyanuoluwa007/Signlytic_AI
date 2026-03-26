import sys
from pathlib import Path
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.motion.signavatars_adapter import SignAvatarsAdapter

# Load gloss mapping
mapping_path = Path("data/signavatars/wlasl/video_to_gloss.json")
with open(mapping_path, 'r') as f:
    video_to_gloss = json.load(f)

print(f"Loaded {len(video_to_gloss)} video-to-gloss mappings")

# Load samples
adapter = SignAvatarsAdapter("data/signavatars")
samples = adapter.load_dataset("wlasl", max_samples=100)

# Map to glosses
gloss_counts = {}
for sample in samples:
    gloss = video_to_gloss.get(sample.sample_id, "UNKNOWN")
    gloss_counts[gloss] = gloss_counts.get(gloss, 0) + 1

print(f"Unique glosses in 100 samples: {len(gloss_counts)}")
print("Top 10 glosses:")
for gloss, count in sorted(gloss_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {gloss}: {count} samples")
