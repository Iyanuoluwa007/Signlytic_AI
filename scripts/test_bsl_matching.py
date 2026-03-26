import sys
sys.path.insert(0, '.')

from pathlib import Path
from src.inference.bsl_video_recognizer import BSLVideoRecognizer
import random

print("="*70)
print("TESTING BSL ON MATCHING VOCABULARY")
print("="*70)

# Load matching videos
with open("data/matching_bsl_videos.txt", 'r') as f:
    videos = [Path(line.strip()) for line in f.readlines()]

print(f"Testing on {len(videos)} matching videos...")

# Initialize recognizer
recognizer = BSLVideoRecognizer()

# Test on random sample
random.seed(42)
sample = random.sample(videos, min(20, len(videos)))

correct_top1 = 0
correct_top5 = 0
total = 0

for video_path in sample:
    expected = video_path.stem.lower()
    
    try:
        results = recognizer.recognize(str(video_path), top_k=5)
        predictions = [r[0] for r in results]
        
        is_top1 = predictions[0] == expected
        is_top5 = expected in predictions
        
        correct_top1 += int(is_top1)
        correct_top5 += int(is_top5)
        total += 1
        
        status = "[OK]" if is_top1 else ("[TOP5]" if is_top5 else "[MISS]")
        print(f"{status} {expected} -> {predictions[0]} ({results[0][1]*100:.1f}%)")
        
    except Exception as e:
        print(f"[ERR] {expected}: {e}")

print(f"\n" + "="*70)
print(f"RESULTS: {total} videos tested")
print(f"  Top-1 Accuracy: {correct_top1}/{total} = {correct_top1/total*100:.1f}%")
print(f"  Top-5 Accuracy: {correct_top5}/{total} = {correct_top5/total*100:.1f}%")
print("="*70)
