"""
BOBSL Pose Extraction Pipeline

Extracts body, hand, and face poses from BOBSL signing videos
using MediaPipe for AI sign language generation training.

Pipeline:
1. Parse CSLR annotations to get gloss timestamps
2. Extract video segments for each gloss
3. Run MediaPipe Holistic to get poses
4. Save pose sequences as training data

Usage:
    python scripts/extract_poses.py --limit 100  # Test with 100 glosses
    python scripts/extract_poses.py              # Process all
"""

import os
import re
import csv
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import cv2
from tqdm import tqdm

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_NEW_API = True
except ImportError:
    try:
        import mediapipe as mp
        MEDIAPIPE_AVAILABLE = hasattr(mp, 'solutions')
        MEDIAPIPE_NEW_API = False
    except ImportError:
        MEDIAPIPE_AVAILABLE = False
        MEDIAPIPE_NEW_API = False
        print("MediaPipe not installed. Run: pip install mediapipe")


@dataclass
class GlossAnnotation:
    """Single gloss annotation with timestamps."""
    gloss: str
    start_time: float
    end_time: float
    video_id: str
    english_sentence: str


@dataclass 
class PoseFrame:
    """Pose data for a single frame."""
    frame_idx: int
    timestamp: float
    # Pose landmarks (33 points)
    pose: Optional[List[Tuple[float, float, float]]]
    # Left hand landmarks (21 points)
    left_hand: Optional[List[Tuple[float, float, float]]]
    # Right hand landmarks (21 points)
    right_hand: Optional[List[Tuple[float, float, float]]]
    # Face landmarks (468 points - we'll store subset)
    face: Optional[List[Tuple[float, float, float]]]


class CSLRAnnotationParser:
    """Parse CSLR annotation CSV files."""
    
    # Regex to extract gloss and timestamps: gloss_name[start-end]
    GLOSS_PATTERN = re.compile(r'([^[\]]+)\[(\d+\.?\d*)-(\d+\.?\d*)\]')
    
    def __init__(self, annotations_dir: str):
        """
        Initialize parser.
        
        Args:
            annotations_dir: Path to CSLR annotations directory
        """
        self.annotations_dir = Path(annotations_dir)
        self.cslr_dir = self.annotations_dir / "continuous_sign_sequences" / "cslr-raw"
    
    def parse_gloss_sequence(self, gloss_str: str, video_id: str, english: str) -> List[GlossAnnotation]:
        """Parse gloss sequence string into list of annotations."""
        annotations = []
        
        if not gloss_str or not gloss_str.strip():
            return annotations
        
        for match in self.GLOSS_PATTERN.finditer(gloss_str):
            gloss_name = match.group(1).strip()
            start_time = float(match.group(2))
            end_time = float(match.group(3))
            
            # Normalize gloss name
            gloss_name = gloss_name.upper().replace(' ', '_')
            
            annotations.append(GlossAnnotation(
                gloss=gloss_name,
                start_time=start_time,
                end_time=end_time,
                video_id=video_id,
                english_sentence=english
            ))
        
        return annotations
    
    def parse_all(self, splits: List[str] = None) -> Dict[str, List[GlossAnnotation]]:
        """
        Parse all annotations.
        
        Args:
            splits: List of splits to parse ['train', 'val', 'test']
            
        Returns:
            Dict mapping split name to list of annotations
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        all_annotations = {}
        
        for split in splits:
            split_dir = self.cslr_dir / split
            if not split_dir.exists():
                print(f"Warning: {split} directory not found")
                continue
            
            annotations = []
            csv_files = list(split_dir.glob("*.csv"))
            
            for csv_file in tqdm(csv_files, desc=f"Parsing {split}"):
                video_id = csv_file.stem
                
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    
                    for row in reader:
                        gloss_str = row.get('approx gloss sequence', '')
                        english = row.get('english sentence', '')
                        
                        parsed = self.parse_gloss_sequence(gloss_str, video_id, english)
                        annotations.extend(parsed)
            
            all_annotations[split] = annotations
            print(f"{split}: {len(annotations)} gloss annotations from {len(csv_files)} files")
        
        return all_annotations
    
    def get_gloss_statistics(self, annotations: List[GlossAnnotation]) -> Dict:
        """Get statistics about glosses."""
        gloss_counts = defaultdict(int)
        gloss_durations = defaultdict(list)
        
        for ann in annotations:
            gloss_counts[ann.gloss] += 1
            duration = ann.end_time - ann.start_time
            gloss_durations[ann.gloss].append(duration)
        
        stats = {
            'total_annotations': len(annotations),
            'unique_glosses': len(gloss_counts),
            'top_glosses': sorted(gloss_counts.items(), key=lambda x: -x[1])[:50],
            'avg_duration': np.mean([ann.end_time - ann.start_time for ann in annotations]),
        }
        
        return stats


class PoseExtractor:
    """Extract poses from video using MediaPipe."""
    
    def __init__(self, model_complexity: int = 1):
        """
        Initialize pose extractor.
        
        Args:
            model_complexity: 0, 1, or 2 (higher = more accurate but slower)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not installed")
        
        self.model_complexity = model_complexity
        self.use_new_api = MEDIAPIPE_NEW_API
        
        if self.use_new_api:
            # New MediaPipe Tasks API (0.10+)
            # Download models if needed
            self._init_new_api()
        else:
            # Legacy API
            self._init_legacy_api()
    
    def _init_new_api(self):
        """Initialize with new MediaPipe Tasks API."""
        import urllib.request
        import os
        
        # Model paths
        model_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/models/mediapipe")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        pose_model = model_dir / "pose_landmarker_heavy.task"
        hand_model = model_dir / "hand_landmarker.task"
        
        # Download models if needed
        if not pose_model.exists():
            print("Downloading pose model...")
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, pose_model)
        
        if not hand_model.exists():
            print("Downloading hand model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, hand_model)
        
        # Create landmarkers
        base_options_pose = python.BaseOptions(model_asset_path=str(pose_model))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            output_segmentation_masks=False,
            num_poses=1
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        
        base_options_hand = python.BaseOptions(model_asset_path=str(hand_model))
        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            num_hands=2
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
    
    def _init_legacy_api(self):
        """Initialize with legacy MediaPipe API."""
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def extract_from_video(
        self,
        video_path: str,
        start_time: float,
        end_time: float,
        fps: int = 25
    ) -> List[PoseFrame]:
        """
        Extract poses from video segment.
        
        Args:
            video_path: Path to video file
            start_time: Start time in seconds
            end_time: End time in seconds
            fps: Video FPS (default 25 for BOBSL)
            
        Returns:
            List of PoseFrame objects
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return []
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        
        # Calculate frame range
        start_frame = int(start_time * video_fps)
        end_frame = int(end_time * video_fps)
        
        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        poses = []
        frame_idx = start_frame
        
        while frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame
            if self.use_new_api:
                pose_data = self._process_frame_new_api(rgb_frame)
            else:
                pose_data = self._process_frame_legacy(rgb_frame)
            
            pose_data.frame_idx = frame_idx
            pose_data.timestamp = frame_idx / video_fps
            
            poses.append(pose_data)
            frame_idx += 1
        
        cap.release()
        return poses
    
    def _process_frame_new_api(self, rgb_frame: np.ndarray) -> PoseFrame:
        """Process frame with new MediaPipe Tasks API."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect pose
        pose = None
        pose_result = self.pose_landmarker.detect(mp_image)
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            pose = [(lm.x, lm.y, lm.z) for lm in pose_result.pose_landmarks[0]]
        
        # Detect hands
        left_hand = None
        right_hand = None
        hand_result = self.hand_landmarker.detect(mp_image)
        if hand_result.hand_landmarks:
            for i, handedness in enumerate(hand_result.handedness):
                if i < len(hand_result.hand_landmarks):
                    hand_lms = [(lm.x, lm.y, lm.z) for lm in hand_result.hand_landmarks[i]]
                    # Note: MediaPipe reports handedness from camera's perspective
                    if handedness[0].category_name == "Left":
                        right_hand = hand_lms  # Flip for user perspective
                    else:
                        left_hand = hand_lms
        
        return PoseFrame(
            frame_idx=0,
            timestamp=0.0,
            pose=pose,
            left_hand=left_hand,
            right_hand=right_hand,
            face=None  # Face not available in new API without separate model
        )
    
    def _process_frame_legacy(self, rgb_frame: np.ndarray) -> PoseFrame:
        """Process frame with legacy MediaPipe API."""
        results = self.holistic.process(rgb_frame)
        return self._extract_landmarks_legacy(results)
    
    def _extract_landmarks_legacy(self, results) -> PoseFrame:
        """Extract landmarks from legacy MediaPipe results."""
        
        # Pose landmarks
        pose = None
        if results.pose_landmarks:
            pose = [(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark]
        
        # Left hand
        left_hand = None
        if results.left_hand_landmarks:
            left_hand = [(lm.x, lm.y, lm.z) for lm in results.left_hand_landmarks.landmark]
        
        # Right hand
        right_hand = None
        if results.right_hand_landmarks:
            right_hand = [(lm.x, lm.y, lm.z) for lm in results.right_hand_landmarks.landmark]
        
        # Face (subset - key points only for efficiency)
        face = None
        if results.face_landmarks:
            key_indices = [0, 13, 14, 61, 291, 33, 263, 159, 386, 70, 300]
            face = [(results.face_landmarks.landmark[i].x,
                    results.face_landmarks.landmark[i].y,
                    results.face_landmarks.landmark[i].z) 
                   for i in key_indices if i < len(results.face_landmarks.landmark)]
        
        return PoseFrame(
            frame_idx=0,
            timestamp=0.0,
            pose=pose,
            left_hand=left_hand,
            right_hand=right_hand,
            face=face
        )
    
    def close(self):
        """Release resources."""
        if self.use_new_api:
            self.pose_landmarker.close()
            self.hand_landmarker.close()
        else:
            self.holistic.close()


class BOBSLPoseDataset:
    """
    Create pose dataset from BOBSL videos.
    
    Output format:
    {
        "gloss": "HELLO",
        "video_id": "1234567890",
        "start_time": 10.5,
        "end_time": 11.2,
        "num_frames": 18,
        "poses": [
            {
                "frame_idx": 0,
                "timestamp": 10.5,
                "pose": [[x,y,z], ...],  # 33 points
                "left_hand": [[x,y,z], ...],  # 21 points
                "right_hand": [[x,y,z], ...],  # 21 points
                "face": [[x,y,z], ...]  # key points
            },
            ...
        ]
    }
    """
    
    def __init__(
        self,
        videos_dir: str,
        annotations_dir: str,
        output_dir: str
    ):
        """
        Initialize dataset builder.
        
        Args:
            videos_dir: Path to BOBSL MP4 videos
            annotations_dir: Path to annotations
            output_dir: Where to save pose data
        """
        self.videos_dir = Path(videos_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.parser = CSLRAnnotationParser(annotations_dir)
        self.extractor = None  # Lazy init
        
        # Build video index
        self.video_index = self._build_video_index()
        print(f"Found {len(self.video_index)} videos")
    
    def _build_video_index(self) -> Dict[str, str]:
        """Build index mapping video_id to file path."""
        index = {}
        
        for video_file in self.videos_dir.glob("**/*.mp4"):
            # Video ID is typically the filename without extension
            video_id = video_file.stem
            index[video_id] = str(video_file)
        
        return index
    
    def process(
        self,
        splits: List[str] = None,
        limit: int = None,
        skip_existing: bool = True
    ):
        """
        Process videos and extract poses.
        
        Args:
            splits: Which splits to process
            limit: Maximum glosses to process (for testing)
            skip_existing: Skip already processed glosses
        """
        if splits is None:
            splits = ['train', 'val', 'test']
        
        # Parse annotations
        print("Parsing annotations...")
        all_annotations = self.parser.parse_all(splits)
        
        # Initialize extractor
        if self.extractor is None:
            print("Initializing pose extractor...")
            self.extractor = PoseExtractor(model_complexity=1)
        
        # Process each split
        for split, annotations in all_annotations.items():
            print(f"\nProcessing {split} split ({len(annotations)} glosses)...")
            
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Apply limit
            if limit:
                annotations = annotations[:limit]
            
            processed = 0
            skipped = 0
            errors = 0
            
            for ann in tqdm(annotations, desc=split):
                # Sanitize gloss name for filename (replace problematic characters)
                safe_gloss = ann.gloss.replace('/', '-').replace('\\', '-').replace(':', '-').replace('*', '-').replace('?', '-').replace('"', '-').replace('<', '-').replace('>', '-').replace('|', '-')
                
                # Output path
                output_file = split_dir / f"{safe_gloss}_{ann.video_id}_{ann.start_time:.2f}.json"
                
                if skip_existing and output_file.exists():
                    skipped += 1
                    continue
                
                # Find video file
                video_path = self.video_index.get(ann.video_id)
                if not video_path:
                    errors += 1
                    continue
                
                # Extract poses
                try:
                    poses = self.extractor.extract_from_video(
                        video_path,
                        ann.start_time,
                        ann.end_time
                    )
                    
                    if not poses:
                        errors += 1
                        continue
                    
                    # Save
                    data = {
                        'gloss': ann.gloss,
                        'video_id': ann.video_id,
                        'start_time': ann.start_time,
                        'end_time': ann.end_time,
                        'english': ann.english_sentence,
                        'num_frames': len(poses),
                        'poses': [asdict(p) for p in poses]
                    }
                    
                    with open(output_file, 'w') as f:
                        json.dump(data, f)
                    
                    processed += 1
                    
                except Exception as e:
                    print(f"Error processing {ann.gloss}: {e}")
                    errors += 1
            
            print(f"{split}: processed={processed}, skipped={skipped}, errors={errors}")
        
        # Cleanup
        if self.extractor:
            self.extractor.close()
    
    def create_summary(self):
        """Create summary of extracted poses."""
        summary = {
            'splits': {},
            'glosses': defaultdict(int),
            'total_frames': 0
        }
        
        for split_dir in self.output_dir.iterdir():
            if not split_dir.is_dir():
                continue
            
            json_files = list(split_dir.glob("*.json"))
            summary['splits'][split_dir.name] = len(json_files)
            
            for json_file in json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    summary['glosses'][data['gloss']] += 1
                    summary['total_frames'] += data['num_frames']
        
        summary['unique_glosses'] = len(summary['glosses'])
        summary['glosses'] = dict(sorted(summary['glosses'].items(), key=lambda x: -x[1])[:100])
        
        # Save summary
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDataset Summary:")
        print(f"  Splits: {summary['splits']}")
        print(f"  Unique glosses: {summary['unique_glosses']}")
        print(f"  Total frames: {summary['total_frames']}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Extract poses from BOBSL videos")
    parser.add_argument("--videos-dir", type=str, default=None,
                       help="Path to BOBSL videos")
    parser.add_argument("--annotations-dir", type=str, default=None,
                       help="Path to annotations")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for poses")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of glosses to process")
    parser.add_argument("--splits", type=str, default="train,val,test",
                       help="Comma-separated splits to process")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only create summary of existing data")
    
    args = parser.parse_args()
    
    # Default paths
    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    
    if args.videos_dir is None:
        args.videos_dir = project_root / "data/processed/bobsl_v1_4_videos_mp4/bobsl/v1.4/original_data/videos/mp4"
    
    if args.annotations_dir is None:
        args.annotations_dir = project_root / "data/processed/annotations/bobsl/v1.4/manual_annotations"
    
    if args.output_dir is None:
        args.output_dir = project_root / "data/poses"
    
    # Create dataset
    dataset = BOBSLPoseDataset(
        videos_dir=str(args.videos_dir),
        annotations_dir=str(args.annotations_dir),
        output_dir=str(args.output_dir)
    )
    
    if args.summary_only:
        dataset.create_summary()
        return
    
    # Process
    splits = args.splits.split(',')
    dataset.process(splits=splits, limit=args.limit)
    
    # Create summary
    dataset.create_summary()


if __name__ == "__main__":
    main()