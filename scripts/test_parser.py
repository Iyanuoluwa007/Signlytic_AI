#!/usr/bin/env python3
"""
Test script for BOBSL annotation parser.
Verifies that annotations are parsed correctly and features are accessible.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.annotation_parser import BOBSLAnnotationParser


def test_parser(data_dir: str) -> None:
    """Run comprehensive parser tests."""
    print("=" * 70)
    print("BOBSL PARSER TEST")
    print("=" * 70)
    
    # Initialize parser (this also scans features)
    print(f"\nData directory: {data_dir}")
    parser = BOBSLAnnotationParser(data_dir)
    
    # Check directory structure
    print("\n--- Directory Structure Check ---")
    print(f"Annotations dir exists: {parser.annotations_dir.exists()}")
    print(f"Metadata dir exists: {parser.metadata_dir.exists()}")
    print(f"Features base exists: {parser.features_base.exists()}")
    print(f"Subtitles dir exists: {parser.subtitles_dir.exists()}")
    
    # Show detected feature directories
    print("\n--- Detected Feature Directories ---")
    print(f"I3D dir: {parser._feature_dirs.get('i3d', 'Not found')}")
    print(f"Swin dir: {parser._feature_dirs.get('swin', 'Not found')}")
    
    # List annotation files
    print("\n--- Annotation Files ---")
    if parser.annotations_dir.exists():
        for json_file in parser.annotations_dir.rglob("*.json"):
            size_kb = json_file.stat().st_size / 1024
            print(f"  {json_file.relative_to(parser.annotations_dir)} ({size_kb:.1f} KB)")
    else:
        print("  Annotations directory not found")
    
    # Parse isolated signs
    print("\n--- Parsing Isolated Signs ---")
    videos = parser.parse_isolated_signs()
    
    if not videos:
        print("  No videos parsed. Check annotation file format.")
        return
    
    print(f"  Parsed {len(videos)} videos")
    print(f"  Total sign instances: {len(parser.sign_instances)}")
    print(f"  Vocabulary size: {len(parser.vocabulary)}")
    
    # Show sample video
    print("\n--- Sample Video Annotation ---")
    sample_video_id = list(videos.keys())[0]
    sample_video = videos[sample_video_id]
    
    print(f"  Video ID: {sample_video.video_id}")
    print(f"  Number of signs: {len(sample_video.signs)}")
    print(f"  Gloss sequence (first 10): {sample_video.gloss_sequence[:10]}")
    
    print("\n  First 5 signs:")
    for sign in sorted(sample_video.signs, key=lambda s: s.global_time)[:5]:
        print(f"    {sign.gloss} @ {sign.global_time:.2f}s (conf: {sign.confidence:.3f})")
    
    # Check feature availability
    print("\n--- Feature Cache Status ---")
    stats = parser.get_statistics()
    print(f"  I3D features cached: {stats['i3d_features_available']}")
    print(f"  Swin features cached: {stats['swin_features_available']}")
    
    # Check overlap between annotations and features
    print("\n--- Annotation-Feature Overlap ---")
    available_swin = parser.get_available_videos('swin')
    available_i3d = parser.get_available_videos('i3d')
    
    print(f"  Videos with annotations: {len(videos)}")
    print(f"  Videos with Swin features AND annotations: {len(available_swin)}")
    print(f"  Videos with I3D features AND annotations: {len(available_i3d)}")
    
    # Show sample feature paths
    if available_swin:
        print("\n--- Sample Feature Paths (Swin) ---")
        for video_id in available_swin[:3]:
            path = parser.get_feature_path(video_id, 'swin')
            print(f"  {video_id}: {path}")
    
    # Statistics
    print("\n--- Statistics ---")
    print(f"  Total sign instances: {stats['total_sign_instances']}")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Vocabulary size: {stats['vocabulary_size']}")
    print(f"  Avg signs per video: {stats['avg_signs_per_video']:.2f}")
    
    print("\n  Top 10 glosses:")
    for gloss, count in stats['most_common_glosses'][:10]:
        print(f"    {gloss}: {count}")
    
    # Training format sample
    print("\n--- Training Data Format ---")
    training_data = parser.to_training_format(feature_type='swin')
    
    # Count samples with features
    with_features = sum(1 for s in training_data if s['has_features'])
    print(f"  Total samples: {len(training_data)}")
    print(f"  Samples with features: {with_features}")
    
    # Show samples with features
    print("\n  Sample entries WITH features:")
    featured_samples = [s for s in training_data if s['has_features']][:3]
    for sample in featured_samples:
        print(f"    video={sample['video_id']}, gloss={sample['gloss']}, has_features={sample['has_features']}")
        print(f"      path={sample['feature_path']}")
    
    # Training data with only featured samples
    print("\n--- Training Data (features only) ---")
    training_featured = parser.to_training_format(feature_type='swin', only_with_features=True)
    print(f"  Usable training samples: {len(training_featured)}")
    
    # Count unique glosses in featured samples
    featured_glosses = set(s['gloss'] for s in training_featured)
    print(f"  Unique glosses with features: {len(featured_glosses)}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test BOBSL annotation parser')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    
    args = parser.parse_args()
    test_parser(args.data_dir)


if __name__ == "__main__":
    main()