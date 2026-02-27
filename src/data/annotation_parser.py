#!/usr/bin/env python3
"""
BOBSL Dataset Annotation Parser

Parses the BOBSL v1.4 annotation format where data is organized by gloss (word)
rather than by video. Each gloss entry contains lists of video IDs, timestamps,
and confidence scores.
"""

import json
import gzip
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import csv


@dataclass
class SignInstance:
    """Single instance of a sign in a video."""
    gloss: str
    video_id: str
    global_time: float
    confidence: float
    anno_idx: int


@dataclass 
class VideoAnnotation:
    """All sign annotations for a single video."""
    video_id: str
    signs: List[SignInstance] = field(default_factory=list)
    
    @property
    def gloss_sequence(self) -> List[str]:
        """Return glosses sorted by time."""
        sorted_signs = sorted(self.signs, key=lambda s: s.global_time)
        return [s.gloss for s in sorted_signs]


class BOBSLAnnotationParser:
    """Parser for BOBSL v1.4 dataset annotations."""
    
    def __init__(self, base_dir: str):
        """
        Initialize parser with base directory.
        
        Args:
            base_dir: Path to processed data directory (e.g., 'data/processed')
        """
        self.base_dir = Path(base_dir)
        
        # Define paths based on BOBSL structure
        self.annotations_dir = self.base_dir / "annotations" / "bobsl" / "v1.4" / "manual_annotations"
        self.metadata_dir = self.base_dir / "metadata" / "bobsl" / "v1.4" / "original_data" / "metadata"
        self.subtitles_dir = self.base_dir / "subtitles" / "bobsl" / "v1.4" / "original_data" / "subtitles"
        self.features_base = self.base_dir / "features" / "bobsl" / "v1.4" / "video_features"
        
        # Storage for parsed data
        self.sign_instances: List[SignInstance] = []
        self.videos: Dict[str, VideoAnnotation] = {}
        self.vocabulary: Dict[str, int] = {}
        self.metadata: Dict = {}
        
        # Cache for feature paths
        self._feature_cache: Dict[str, Dict[str, Optional[Path]]] = {}
        self._feature_dirs: Dict[str, Optional[Path]] = {}
        self._scan_features()
    
    def _scan_features(self) -> None:
        """Scan feature directories and build lookup cache."""
        self._feature_cache = {'i3d': {}, 'swin': {}}
        self._feature_dirs = {'i3d': None, 'swin': None}
        
        if not self.features_base.exists():
            print(f"Features base not found: {self.features_base}")
            return
        
        # Find Swin features directory
        swin_base = self.features_base / "swin_v1"
        if swin_base.exists():
            for subdir in swin_base.iterdir():
                if subdir.is_dir():
                    self._feature_dirs['swin'] = subdir
                    break
        
        # Find I3D features directory (name varies)
        for item in self.features_base.iterdir():
            if item.is_dir() and item.name.startswith('i3d'):
                self._feature_dirs['i3d'] = item
                break
        
        # Scan Swin features (files directly in folder, named {video_id}.npy)
        if self._feature_dirs['swin'] and self._feature_dirs['swin'].exists():
            for npy_file in self._feature_dirs['swin'].glob("*.npy"):
                video_id = npy_file.stem  # filename without extension
                self._feature_cache['swin'][video_id] = npy_file
        
        # Scan I3D features (may be files or subdirectories)
        if self._feature_dirs['i3d'] and self._feature_dirs['i3d'].exists():
            i3d_dir = self._feature_dirs['i3d']
            
            # Check if files are directly in folder
            npy_files = list(i3d_dir.glob("*.npy"))
            if npy_files:
                for npy_file in npy_files:
                    video_id = npy_file.stem
                    self._feature_cache['i3d'][video_id] = npy_file
            else:
                # Check subdirectories
                for video_dir in i3d_dir.iterdir():
                    if video_dir.is_dir():
                        video_id = video_dir.name
                        sub_npy = list(video_dir.glob("*.npy"))
                        if sub_npy:
                            self._feature_cache['i3d'][video_id] = sub_npy[0]
        
        print(f"Feature cache built: I3D={len(self._feature_cache['i3d'])}, Swin={len(self._feature_cache['swin'])}")
        if self._feature_dirs['swin']:
            print(f"  Swin dir: {self._feature_dirs['swin']}")
        if self._feature_dirs['i3d']:
            print(f"  I3D dir: {self._feature_dirs['i3d']}")
    
    def parse_isolated_signs(self) -> Dict[str, VideoAnnotation]:
        """
        Parse isolated sign annotations from verified_dict_spottings.json.
        
        Returns:
            Dictionary mapping video_id to VideoAnnotation
        """
        isolated_dir = self.annotations_dir / "isolated_signs"
        
        if not isolated_dir.exists():
            print(f"Warning: Isolated signs directory not found at {isolated_dir}")
            return {}
        
        # Find and parse JSON files
        json_files = list(isolated_dir.glob("*.json"))
        print(f"Found {len(json_files)} JSON files in isolated_signs/")
        
        for json_file in json_files:
            print(f"Parsing: {json_file.name}")
            self._parse_spotting_file(json_file)
        
        return self.videos
    
    def _parse_spotting_file(self, filepath: Path) -> None:
        """Parse a spotting JSON file (verified_dict_spottings.json format)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle nested structure with 'test', 'train', 'val' keys
        for split_name, split_data in data.items():
            if not isinstance(split_data, dict):
                continue
                
            print(f"  Processing split: {split_name} ({len(split_data)} glosses)")
            
            for gloss, gloss_data in split_data.items():
                if not isinstance(gloss_data, dict):
                    continue
                
                # Extract parallel lists
                anno_indices = gloss_data.get('anno_idx', [])
                global_times = gloss_data.get('global_times', [])
                video_names = gloss_data.get('names', [])
                probabilities = gloss_data.get('probs', [])
                
                # Ensure all lists have same length
                n_instances = min(len(anno_indices), len(global_times), 
                                  len(video_names), len(probabilities))
                
                for i in range(n_instances):
                    instance = SignInstance(
                        gloss=gloss,
                        video_id=str(video_names[i]),
                        global_time=float(global_times[i]),
                        confidence=float(probabilities[i]),
                        anno_idx=int(anno_indices[i])
                    )
                    
                    self.sign_instances.append(instance)
                    
                    # Group by video
                    if instance.video_id not in self.videos:
                        self.videos[instance.video_id] = VideoAnnotation(
                            video_id=instance.video_id
                        )
                    self.videos[instance.video_id].signs.append(instance)
                    
                    # Build vocabulary
                    if gloss not in self.vocabulary:
                        self.vocabulary[gloss] = len(self.vocabulary)
    
    def parse_metadata(self) -> Dict:
        """Parse episode metadata from TSV file."""
        tsv_file = self.metadata_dir / "metadata_public_episodes.tsv"
        
        if not tsv_file.exists():
            print(f"Warning: Metadata file not found at {tsv_file}")
            return {}
        
        print(f"Parsing metadata: {tsv_file.name}")
        
        with open(tsv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                episode_id = row.get('episode_id', row.get('name', ''))
                if episode_id:
                    self.metadata[episode_id] = row
        
        print(f"  Loaded metadata for {len(self.metadata)} episodes")
        return self.metadata
    
    def parse_subset_mapping(self) -> Dict:
        """Parse subset to episode mapping."""
        json_file = self.metadata_dir / "subset2episode.json"
        
        if not json_file.exists():
            print(f"Warning: Subset mapping not found at {json_file}")
            return {}
        
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_feature_path(self, video_id: str, feature_type: str = 'swin') -> Optional[Path]:
        """
        Get path to feature file for a video.
        
        Args:
            video_id: Video identifier
            feature_type: 'i3d' or 'swin'
            
        Returns:
            Path to feature file or None if not found
        """
        if feature_type in self._feature_cache:
            return self._feature_cache[feature_type].get(video_id)
        return None
    
    def get_available_videos(self, feature_type: str = 'swin') -> List[str]:
        """Get list of video IDs that have both annotations and features."""
        available = []
        
        for video_id in self.videos.keys():
            if video_id in self._feature_cache.get(feature_type, {}):
                available.append(video_id)
        
        return available
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        gloss_counts = defaultdict(int)
        for instance in self.sign_instances:
            gloss_counts[instance.gloss] += 1
        
        signs_per_video = [len(v.signs) for v in self.videos.values()]
        
        return {
            'total_sign_instances': len(self.sign_instances),
            'total_videos': len(self.videos),
            'vocabulary_size': len(self.vocabulary),
            'avg_signs_per_video': sum(signs_per_video) / len(signs_per_video) if signs_per_video else 0,
            'max_signs_per_video': max(signs_per_video) if signs_per_video else 0,
            'min_signs_per_video': min(signs_per_video) if signs_per_video else 0,
            'most_common_glosses': sorted(gloss_counts.items(), key=lambda x: -x[1])[:20],
            'least_common_glosses': sorted(gloss_counts.items(), key=lambda x: x[1])[:20],
            'i3d_features_available': len(self._feature_cache.get('i3d', {})),
            'swin_features_available': len(self._feature_cache.get('swin', {})),
        }
    
    def to_training_format(self, feature_type: str = 'swin', 
                           only_with_features: bool = False) -> List[Dict]:
        """
        Convert parsed data to training format.
        
        Args:
            feature_type: 'i3d' or 'swin'
            only_with_features: If True, only return samples with available features
        
        Returns:
            List of dictionaries with video_id, gloss, features_path, etc.
        """
        training_data = []
        
        for instance in self.sign_instances:
            feature_path = self.get_feature_path(instance.video_id, feature_type)
            has_features = feature_path is not None
            
            if only_with_features and not has_features:
                continue
            
            training_data.append({
                'video_id': instance.video_id,
                'gloss': instance.gloss,
                'gloss_idx': self.vocabulary.get(instance.gloss, -1),
                'global_time': instance.global_time,
                'confidence': instance.confidence,
                'feature_path': str(feature_path) if feature_path else None,
                'has_features': has_features
            })
        
        return training_data
    
    def save_vocabulary(self, filepath: str) -> None:
        """Save vocabulary to JSON file."""
        vocab_data = {
            'gloss_to_idx': self.vocabulary,
            'idx_to_gloss': {v: k for k, v in self.vocabulary.items()},
            'size': len(self.vocabulary)
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2)
        
        print(f"Saved vocabulary ({len(self.vocabulary)} glosses) to {filepath}")


def main():
    """Test the annotation parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse BOBSL annotations')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--output', type=str, default='data/processed/vocabulary.json',
                        help='Output path for vocabulary')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BOBSL ANNOTATION PARSER")
    print("=" * 70)
    
    # Initialize parser
    bobsl_parser = BOBSLAnnotationParser(args.data_dir)
    
    # Parse annotations
    print("\nParsing isolated signs...")
    videos = bobsl_parser.parse_isolated_signs()
    
    # Parse metadata
    print("\nParsing metadata...")
    bobsl_parser.parse_metadata()
    
    # Get statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    
    stats = bobsl_parser.get_statistics()
    print(f"Total sign instances: {stats['total_sign_instances']}")
    print(f"Total videos: {stats['total_videos']}")
    print(f"Vocabulary size: {stats['vocabulary_size']}")
    print(f"Avg signs per video: {stats['avg_signs_per_video']:.2f}")
    print(f"Max signs per video: {stats['max_signs_per_video']}")
    print(f"Min signs per video: {stats['min_signs_per_video']}")
    print(f"I3D features available: {stats['i3d_features_available']}")
    print(f"Swin features available: {stats['swin_features_available']}")
    
    print("\nMost common glosses:")
    for gloss, count in stats['most_common_glosses'][:10]:
        print(f"  {gloss}: {count}")
    
    # Check feature availability
    print("\n" + "=" * 70)
    print("FEATURE AVAILABILITY")
    print("=" * 70)
    
    available_i3d = bobsl_parser.get_available_videos('i3d')
    available_swin = bobsl_parser.get_available_videos('swin')
    
    print(f"Videos with I3D features: {len(available_i3d)}")
    print(f"Videos with Swin features: {len(available_swin)}")
    
    # Save vocabulary
    bobsl_parser.save_vocabulary(args.output)
    
    # Show sample training data
    print("\n" + "=" * 70)
    print("SAMPLE TRAINING DATA")
    print("=" * 70)
    
    training_data = bobsl_parser.to_training_format(feature_type='swin')
    
    # Count samples with features
    with_features = sum(1 for s in training_data if s['has_features'])
    print(f"Total samples: {len(training_data)}")
    print(f"Samples with features: {with_features}")
    
    print("\nSample entries with features:")
    featured_samples = [s for s in training_data if s['has_features']][:3]
    for sample in featured_samples:
        print(f"  {sample}")
    
    print("\nParsing complete.")


if __name__ == "__main__":
    main()