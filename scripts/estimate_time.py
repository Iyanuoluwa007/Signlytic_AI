from pathlib import Path
videos = list(Path("data/videos/bsl_signs").glob("*.mp4"))
print(f"BSL Dictionary videos: {len(videos)}")
print(f"Estimated feature extraction: {len(videos) * 0.5 / 60:.0f} minutes")
print(f"Then training: ~15 minutes")
