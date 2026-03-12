"""
BSL-1K Parser - FIXED for actual BOBSL directory structure

Directory structure (actual):
    data/BSL-1K/
    ├── exemplars/exemplar_spottings.pkl
    ├── mouthing/mouthing_spottings_v2.pkl
    ├── dictionary/dictionary_spottings_v2.pkl
    └── i3d_pseudo_labels/i3d_pseudo_labels_spottings.pkl

Pickle format (columnar):
    {
        'episode_name': [video_ids...],
        'annot_word': [glosses...],
        'annot_time': [timestamps or (start,end) tuples...],
        'annot_prob': [confidence scores...],
    }
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import hashlib


@dataclass
class SignAnnotation:
    """Single sign annotation."""
    video_id: str
    gloss: str
    start_time: float
    end_time: float
    source: str
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_feature_indices(self, fps: float = 25.0, feature_stride: int = 4) -> Tuple[int, int]:
        """Convert time to SWIN feature indices."""
        feature_fps = fps / feature_stride
        start_idx = int(self.start_time * feature_fps)
        end_idx = int(self.end_time * feature_fps)
        return start_idx, max(end_idx, start_idx + 1)


class BSL1KParser:
    """Parser for BSL-1K annotations."""
    
    # Actual directory names in BOBSL (lowercase, no prefix)
    SOURCE_DIRS = {
        'EXEMPLARS': ['exemplars', 'ISOLATED_SIGN-EXEMPLARS'],
        'MOUTHING': ['mouthing', 'ISOLATED_SIGN-MOUTHING'],
        'DICTIONARY': ['dictionary', 'ISOLATED_SIGN-DICTIONARY'],
        'I3D_PSEUDO_LABELS': ['i3d_pseudo_labels', 'ISOLATED_SIGN-I3D_PSEUDO_LABELS'],
    }
    
    SOURCE_CONFIDENCE = {
        'EXEMPLARS': 1.0,
        'MOUTHING': 0.8,
        'DICTIONARY': 0.6,
        'I3D_PSEUDO_LABELS': 0.4,
    }
    
    # Default sign duration when only single timestamp is given
    DEFAULT_SIGN_DURATION = 0.5  # seconds
    
    def __init__(self, bsl1k_dir: str):
        self.bsl1k_dir = Path(bsl1k_dir)
        self.annotations: Dict[str, List[SignAnnotation]] = defaultdict(list)
        self.gloss_to_idx: Dict[str, int] = {}
        self.idx_to_gloss: Dict[int, str] = {}
        self.gloss_counts: Dict[str, int] = defaultdict(int)
        self.source_stats: Dict[str, int] = defaultdict(int)
    
    def _find_pickle_file(self, source: str) -> Optional[Path]:
        """Find the pickle file for a source."""
        for dir_name in self.SOURCE_DIRS.get(source, []):
            source_dir = self.bsl1k_dir / dir_name
            if source_dir.exists():
                pkl_files = list(source_dir.glob("*.pkl"))
                if pkl_files:
                    return pkl_files[0]
        
        # Fallback: search recursively
        source_lower = source.lower().replace('_', '')
        for pkl_file in self.bsl1k_dir.rglob("*.pkl"):
            if source_lower in pkl_file.stem.lower().replace('_', ''):
                return pkl_file
        
        return None
    
    def load_annotations(
        self,
        sources: Optional[List[str]] = None,
        min_duration: float = 0.1,
        max_duration: float = 10.0
    ) -> int:
        """Load annotations from all sources."""
        if sources is None:
            sources = ['EXEMPLARS', 'MOUTHING', 'DICTIONARY', 'I3D_PSEUDO_LABELS']
        
        total = 0
        
        for source in sources:
            count = self._load_source(source, min_duration, max_duration)
            self.source_stats[source] = count
            total += count
            print(f"  [{source}] Loaded {count:,} annotations")
        
        self._build_vocabulary()
        
        print(f"\nTotal: {total:,} annotations")
        print(f"Unique videos: {len(self.annotations):,}")
        print(f"Unique glosses: {len(self.gloss_counts):,}")
        print(f"Vocabulary size: {len(self.gloss_to_idx):,}")
        
        return total
    
    def _load_source(self, source: str, min_dur: float, max_dur: float) -> int:
        """Load annotations from a single source."""
        pkl_file = self._find_pickle_file(source)
        
        if pkl_file is None:
            print(f"  [WARN] No pickle file found for: {source}")
            return 0
        
        print(f"  [INFO] Loading: {pkl_file.relative_to(self.bsl1k_dir)}")
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"  [ERROR] Failed to load: {e}")
            return 0
        
        return self._parse_columnar_data(data, source, min_dur, max_dur)
    
    def _parse_columnar_data(self, data: Dict, source: str, min_dur: float, max_dur: float) -> int:
        """Parse columnar pickle data."""
        
        episode_names = data.get('episode_name', [])
        annot_words = data.get('annot_word', [])
        annot_times = data.get('annot_time', [])
        annot_probs = data.get('annot_prob', [])
        
        n = len(episode_names)
        if n == 0:
            return 0
        
        base_conf = self.SOURCE_CONFIDENCE.get(source, 0.5)
        count = 0
        skipped = {'no_gloss': 0, 'bad_time': 0, 'duration': 0}
        
        for i in range(n):
            try:
                # Video ID
                video_id = str(episode_names[i])
                if video_id.endswith('.mp4'):
                    video_id = video_id[:-4]
                
                # Gloss
                gloss = str(annot_words[i]).upper().strip()
                if not gloss or gloss in ('NONE', 'NAN', '', 'NULL'):
                    skipped['no_gloss'] += 1
                    continue
                
                # Time - handle multiple formats
                time_info = annot_times[i] if i < len(annot_times) else None
                start, end = self._parse_time(time_info)
                
                if start is None or end is None:
                    skipped['bad_time'] += 1
                    continue
                
                # Validate duration
                duration = end - start
                if duration < min_dur or duration > max_dur:
                    skipped['duration'] += 1
                    continue
                
                # Confidence
                conf = base_conf
                if i < len(annot_probs) and annot_probs[i] is not None:
                    try:
                        conf = float(annot_probs[i])
                    except:
                        pass
                
                # Create annotation
                ann = SignAnnotation(
                    video_id=video_id,
                    gloss=gloss,
                    start_time=start,
                    end_time=end,
                    source=source,
                    confidence=conf
                )
                
                self.annotations[video_id].append(ann)
                self.gloss_counts[gloss] += 1
                count += 1
                
            except Exception as e:
                continue
        
        if count == 0:
            print(f"  [DEBUG] Skipped: {skipped}")
        
        return count
    
    def _parse_time(self, time_info) -> Tuple[Optional[float], Optional[float]]:
        """
        Parse time information into (start, end).
        
        Handles:
        - (start, end) tuple/list
        - [start, end] list
        - Single float (timestamp) -> use default duration
        - np.ndarray
        """
        if time_info is None:
            return None, None
        
        try:
            # Handle numpy array
            if isinstance(time_info, np.ndarray):
                time_info = time_info.tolist()
            
            # Tuple or list with 2 elements
            if isinstance(time_info, (list, tuple)):
                if len(time_info) >= 2:
                    start = float(time_info[0])
                    end = float(time_info[1])
                    
                    # Handle frame indices (convert to seconds)
                    if start > 1000:
                        start = start / 25.0
                        end = end / 25.0
                    
                    return start, end
                elif len(time_info) == 1:
                    start = float(time_info[0])
                    if start > 1000:
                        start = start / 25.0
                    return start, start + self.DEFAULT_SIGN_DURATION
            
            # Single value (timestamp)
            if isinstance(time_info, (int, float)):
                start = float(time_info)
                if start > 1000:
                    start = start / 25.0
                return start, start + self.DEFAULT_SIGN_DURATION
            
        except:
            pass
        
        return None, None
    
    def _build_vocabulary(self, min_count: int = 2):
        """Build gloss vocabulary."""
        glosses = [g for g, c in self.gloss_counts.items() if c >= min_count]
        glosses = sorted(glosses)
        
        self.gloss_to_idx = {g: i for i, g in enumerate(glosses)}
        self.idx_to_gloss = {i: g for g, i in self.gloss_to_idx.items()}
    
    def get_training_samples(
        self,
        min_confidence: float = 0.5,
        available_videos: Optional[Set[str]] = None
    ) -> List[Tuple[str, int, int, int]]:
        """Get training samples as (video_id, start_idx, end_idx, gloss_idx)."""
        samples = []
        
        for video_id, anns in self.annotations.items():
            if available_videos is not None and video_id not in available_videos:
                continue
            
            for ann in anns:
                if ann.confidence < min_confidence:
                    continue
                if ann.gloss not in self.gloss_to_idx:
                    continue
                
                start_idx, end_idx = ann.to_feature_indices()
                gloss_idx = self.gloss_to_idx[ann.gloss]
                samples.append((video_id, start_idx, end_idx, gloss_idx))
        
        return samples
    
    def get_video_ids(self) -> List[str]:
        return list(self.annotations.keys())
    
    def get_vocabulary_size(self) -> int:
        return len(self.gloss_to_idx)
    
    def save_vocabulary(self, filepath: str):
        """Save vocabulary to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'gloss_to_idx': self.gloss_to_idx,
                'idx_to_gloss': {str(k): v for k, v in self.idx_to_gloss.items()},
                'gloss_counts': dict(self.gloss_counts),
                'source_stats': dict(self.source_stats)
            }, f, indent=2)
        print(f"Saved vocabulary to: {filepath}")
    
    def load_vocabulary(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.gloss_to_idx = data['gloss_to_idx']
        self.idx_to_gloss = {int(k): v for k, v in data['idx_to_gloss'].items()}
    
    def split_by_video(self, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[List[str], List[str], List[str]]:
        """Split videos into train/val/test."""
        video_ids = list(self.annotations.keys())
        train, val, test = [], [], []
        
        for vid in video_ids:
            h = int(hashlib.md5(vid.encode()).hexdigest(), 16)
            ratio = (h % 1000) / 1000
            
            if ratio < train_ratio:
                train.append(vid)
            elif ratio < train_ratio + val_ratio:
                val.append(vid)
            else:
                test.append(vid)
        
        return train, val, test
    
    def print_stats(self):
        """Print statistics."""
        print("\n" + "="*60)
        print("BSL-1K STATISTICS")
        print("="*60)
        
        total = sum(len(v) for v in self.annotations.values())
        print(f"Total annotations: {total:,}")
        print(f"Unique videos: {len(self.annotations):,}")
        print(f"Unique glosses: {len(self.gloss_counts):,}")
        print(f"Vocabulary size: {len(self.gloss_to_idx):,}")
        
        print(f"\nBy source:")
        for source, count in self.source_stats.items():
            print(f"  {source}: {count:,}")
        
        print(f"\nTop 10 glosses:")
        for gloss, count in sorted(self.gloss_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {gloss}: {count:,}")


def get_swin_video_ids(swin_dir: str) -> Set[str]:
    """Get video IDs that have SWIN features."""
    swin_path = Path(swin_dir)
    if not swin_path.exists():
        return set()
    return {f.stem for f in swin_path.glob("*.npy")}


# Test
if __name__ == "__main__":
    print("Testing BSL-1K Parser")
    print("="*60)
    
    bsl1k_dir = "D:/Signlytic_AI/code/bsl_translation_project/data/BSL-1K"
    swin_dir = "D:/Signlytic_AI/code/bsl_translation_project/data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32"
    
    parser = BSL1KParser(bsl1k_dir)
    parser.load_annotations()
    
    # Get SWIN videos
    swin_videos = get_swin_video_ids(swin_dir)
    print(f"\nSWIN videos: {len(swin_videos)}")
    
    # Get samples
    samples = parser.get_training_samples(min_confidence=0.5, available_videos=swin_videos)
    print(f"Training samples: {len(samples):,}")
    
    if samples:
        print(f"Sample: {samples[0]}")
    
    parser.print_stats()
    
    # Save vocabulary
    parser.save_vocabulary("D:/Signlytic_AI/code/bsl_translation_project/models/swin_recognition/vocabulary.json")