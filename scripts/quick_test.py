import sys
sys.path.insert(0, '.')

from src.inference.bsl_dict_recognizer import BSLDictRecognizer
from pathlib import Path

print("="*70)
print("TESTING BSL RECOGNITION ON COMMON WORDS")
print("="*70)

recognizer = BSLDictRecognizer()

test_words = ['help', 'good', 'bad', 'yes', 'no', 'love', 'work', 'family', 'eat', 'drink']
video_dir = Path("data/videos/bsl_signs")

correct = 0
for word in test_words:
    video_path = video_dir / f"{word}.mp4"
    if video_path.exists():
        results = recognizer.recognize(str(video_path), top_k=3)
        pred = results[0][0]
        conf = results[0][1] * 100
        status = "[OK]" if pred == word else "[MISS]"
        correct += 1 if pred == word else 0
        print(f"{status} {word} -> {pred} ({conf:.0f}%)")

print(f"\nAccuracy: {correct}/{len(test_words)} = {correct/len(test_words)*100:.0f}%")
