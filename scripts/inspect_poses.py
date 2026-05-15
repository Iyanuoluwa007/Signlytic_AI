# scripts/inspect_poses.py
import os
from pathlib import Path

BASE = Path(r"D:\Signlytic_AI\code\bsl_translation_project\data")

# 1. What's in poses_bsldict?
poses_dir = BASE / "poses_bsldict"
print(f"=== poses_bsldict ===")
all_files = list(poses_dir.rglob("*"))
non_dirs = [f for f in all_files if f.is_file()]
print(f"  Total files: {len(non_dirs)}")
if non_dirs:
    # Show extensions breakdown
    from collections import Counter
    exts = Counter(f.suffix.lower() for f in non_dirs)
    print(f"  Extensions: {dict(exts)}")
    print(f"  First 3 files: {[str(f.relative_to(BASE)) for f in non_dirs[:3]]}")

# 2. What's in bsldict/bsldict?
bsldict_dir = BASE / "bsldict" / "bsldict"
print(f"\n=== bsldict/bsldict ===")
all_files2 = list(bsldict_dir.rglob("*"))
non_dirs2 = [f for f in all_files2 if f.is_file()]
print(f"  Total files: {len(non_dirs2)}")
if non_dirs2:
    from collections import Counter
    exts2 = Counter(f.suffix.lower() for f in non_dirs2)
    print(f"  Extensions: {dict(exts2)}")
    print(f"  First 3 files: {[str(f.relative_to(BASE)) for f in non_dirs2[:3]]}")

# 3. Peek at one BSL dict video to confirm it's usable
import cv2
sample_vid = BASE / "videos" / "bsl_signs" / "hello.mp4"
if not sample_vid.exists():
    # find any video
    sample_vid = next((BASE / "videos" / "bsl_signs").glob("*.mp4"))
cap = cv2.VideoCapture(str(sample_vid))
fps   = cap.get(cv2.CAP_PROP_FPS)
frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()
print(f"\n=== Sample video: {sample_vid.name} ===")
print(f"  {w}x{h}, {fps:.1f}fps, {frames} frames, ~{frames/fps:.1f}s")