from pathlib import Path
import torch

# Check what words are in the model's vocabulary
checkpoint = torch.load("models/bsl_recognition/best_model.pt", map_location='cpu', weights_only=False)
vocab = set(checkpoint['label_map'].keys())

# Find matching videos
video_dir = Path("data/videos/bsl_signs")
videos = list(video_dir.glob("*.mp4"))

matches = []
for v in videos:
    name = v.stem.lower()
    if name in vocab:
        matches.append(v)

print(f"Model vocabulary: {len(vocab)} words")
print(f"Total videos: {len(videos)}")
print(f"Matching videos: {len(matches)}")
print(f"\nSample matches: {[m.stem for m in matches[:20]]}")

# Save matches for testing
with open("data/matching_bsl_videos.txt", 'w') as f:
    for m in matches:
        f.write(f"{m}\n")
