from pathlib import Path

video_dir = Path("data/videos/bsl_signs")
videos = list(video_dir.glob("*.mp4"))

# Find common English words (not numbers)
common_words = ['hello', 'thank', 'please', 'help', 'good', 'bad', 'yes', 'no', 
                'name', 'what', 'where', 'when', 'why', 'how', 'love', 'happy',
                'sad', 'work', 'home', 'family', 'friend', 'eat', 'drink', 'sorry',
                'morning', 'afternoon', 'evening', 'night', 'today', 'tomorrow']

print("="*70)
print("RECOMMENDED TEST VIDEOS")
print("="*70)
print(f"\nThese videos exist in data/videos/bsl_signs/:")
print(f"(Filename = Expected BSL gloss)\n")

found = []
for word in common_words:
    video_path = video_dir / f"{word}.mp4"
    if video_path.exists():
        size_kb = video_path.stat().st_size / 1024
        found.append((word, size_kb))
        print(f"  {word}.mp4 ({size_kb:.0f} KB)")

print(f"\nTotal: {len(found)} test videos")
print(f"\nTo test in the app:")
print(f"1. Open http://127.0.0.1:7860")
print(f"2. Go to 'Direction 1: BSL to Speech'")
print(f"3. Upload one of these videos")
print(f"4. Click 'Recognize (SWIN - 5203 signs)'")
print(f"5. Expected output = filename (without .mp4)")
