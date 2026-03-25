import pickle
from pathlib import Path
from collections import Counter
import json

print("="*70)
print("BSL-1K DETAILED ANALYSIS")
print("="*70)

# Load dictionary spottings (most reliable)
dict_path = Path("data/BSL-1K/ISOLATED_SIGN-DICTIONARY/bobsl/v1.4/automatic_annotations/isolated_signs/dictionary/dictionary_spottings_v2.pkl")
with open(dict_path, 'rb') as f:
    data = pickle.load(f)

# Extract the parallel arrays
episode_names = data['episode_name']  # video filenames
annot_words = data['annot_word']       # BSL glosses
annot_times = data['annot_time']       # timestamps
annot_probs = data['annot_prob']       # confidence scores

print(f"Total spottings: {len(episode_names)}")

# Show sample
print("\nSample annotations:")
for i in range(5):
    print(f"  Video: {episode_names[i]}")
    print(f"  Word: {annot_words[i]}")
    print(f"  Time: {annot_times[i]}")
    print(f"  Prob: {annot_probs[i]}")
    print()

# Count unique glosses
unique_words = set(annot_words)
print(f"Unique BSL glosses: {len(unique_words)}")

# Word frequency
word_counts = Counter(annot_words)
print("\nTop 20 BSL glosses:")
for word, count in word_counts.most_common(20):
    print(f"  {word}: {count:,} instances")

# Count unique videos
unique_videos = set(episode_names)
print(f"\nUnique videos: {len(unique_videos)}")

# Check overlap with our BOBSL features
bobsl_features = Path("data/processed/features/bobsl")
feature_files = {f.stem for f in bobsl_features.rglob("*.npy")}
print(f"BOBSL feature files: {len(feature_files)}")

# Find matching videos
video_ids = {v.replace('.mp4', '') for v in unique_videos}
matching = video_ids & feature_files
print(f"Videos with features: {len(matching)}")

# Create training dataset mapping
print("\n" + "="*70)
print("CREATING BSL TRAINING DATASET")
print("="*70)

# Group by video and filter by confidence
video_to_annotations = {}
for i in range(len(episode_names)):
    vid = episode_names[i].replace('.mp4', '')
    word = annot_words[i]
    time = annot_times[i]
    prob = annot_probs[i]
    
    if vid not in video_to_annotations:
        video_to_annotations[vid] = []
    
    video_to_annotations[vid].append({
        'word': word,
        'time': time,
        'prob': prob
    })

# Filter to videos we have features for
usable_videos = {v: a for v, a in video_to_annotations.items() if v in feature_files}
print(f"Videos with features and labels: {len(usable_videos)}")

# Count total usable annotations
total_annotations = sum(len(a) for a in usable_videos.values())
print(f"Total usable annotations: {total_annotations:,}")

# Save mapping
output = {
    'videos': list(usable_videos.keys()),
    'annotations': usable_videos,
    'vocabulary': list(unique_words),
    'word_counts': dict(word_counts.most_common(1000))
}

output_path = Path("data/bsl1k_training_data.json")
with open(output_path, 'w') as f:
    json.dump(output, f)
print(f"\nSaved to {output_path}")

print("\n" + "="*70)
print("SUMMARY: BSL TRAINING DATA")
print("="*70)
print(f"  Videos with features: {len(usable_videos)}")
print(f"  Total annotations: {total_annotations:,}")
print(f"  Unique BSL glosses: {len(unique_words)}")
print(f"  Ready for training!")
