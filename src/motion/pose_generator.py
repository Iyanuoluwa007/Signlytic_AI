"""
Motion Generator Module for BSL

Generates natural signing motion from glosses.
Compatible with 2D renderer and future Blender avatar.
"""

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.signal import savgol_filter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json
import random


@dataclass
class PoseFrame:
    """Single frame of pose data."""
    body: np.ndarray  # (33, 3) body keypoints
    left_hand: np.ndarray  # (21, 3) left hand
    right_hand: np.ndarray  # (21, 3) right hand
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'pose': self.body.tolist(),
            'left_hand': self.left_hand.tolist(),
            'right_hand': self.right_hand.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PoseFrame':
        return cls(
            body=np.array(data.get('pose', np.zeros((33, 3)))),
            left_hand=np.array(data.get('left_hand', np.zeros((21, 3)))),
            right_hand=np.array(data.get('right_hand', np.zeros((21, 3))))
        )


@dataclass
class MotionConfig:
    """Motion generation configuration."""
    fps: int = 25
    min_sign_frames: int = 12
    max_sign_frames: int = 60
    transition_frames: int = 6
    hold_frames: int = 3  # Frames to hold at sign peak
    
    # Smoothing
    savgol_window: int = 7
    savgol_order: int = 3
    
    # Variation
    timing_jitter: float = 0.15
    position_noise: float = 0.005
    
    # Transition blending
    blend_velocity: bool = True  # Use Hermite (velocity-aware) vs linear


def get_rest_pose() -> PoseFrame:
    """Generate neutral rest pose."""
    body = np.zeros((33, 3))
    
    # Head
    body[0] = [0.5, 0.12, 0]  # Nose
    
    # Shoulders
    body[11] = [0.35, 0.25, 0]  # Left shoulder
    body[12] = [0.65, 0.25, 0]  # Right shoulder
    
    # Elbows (slightly bent, relaxed)
    body[13] = [0.30, 0.42, 0]  # Left elbow
    body[14] = [0.70, 0.42, 0]  # Right elbow
    
    # Wrists (at sides)
    body[15] = [0.28, 0.55, 0]  # Left wrist
    body[16] = [0.72, 0.55, 0]  # Right wrist
    
    # Hips
    body[23] = [0.42, 0.55, 0]
    body[24] = [0.58, 0.55, 0]
    
    # Create relaxed hands
    left_hand = _create_relaxed_hand(body[15])
    right_hand = _create_relaxed_hand(body[16])
    
    return PoseFrame(body=body, left_hand=left_hand, right_hand=right_hand)


def _create_relaxed_hand(wrist: np.ndarray) -> np.ndarray:
    """Create relaxed hand keypoints."""
    hand = np.zeros((21, 3))
    hand[0] = wrist
    
    # Simple relaxed finger positions
    for finger in range(5):
        base = 1 + finger * 4
        for joint in range(4):
            offset = [
                (finger - 2) * 0.008,
                0.015 + joint * 0.012,
                0
            ]
            hand[base + joint] = wrist + offset
    
    return hand


class MotionGenerator:
    """
    Generates natural BSL signing motion.
    
    Features:
    - Cubic Hermite velocity-aware blending
    - Savitzky-Golay temporal smoothing
    - Peak hold for sign clarity
    - Rest pose transitions
    """
    
    def __init__(
        self,
        poses_dir: str,
        config: Optional[MotionConfig] = None
    ):
        self.poses_dir = Path(poses_dir)
        self.config = config or MotionConfig()
        
        # Build pose index
        self.gloss_to_files: Dict[str, List[Path]] = {}
        self._build_index()
        
        # Cache for loaded poses
        self._pose_cache: Dict[str, List[PoseFrame]] = {}
    
    def _build_index(self):
        """Index available pose files."""
        if not self.poses_dir.exists():
            print(f"[WARN] Poses directory not found: {self.poses_dir}")
            return
        
        count = 0
        for split in ['train', 'val', 'test']:
            split_dir = self.poses_dir / split
            if not split_dir.exists():
                continue
            
            for json_file in split_dir.glob("*.json"):
                try:
                    gloss = json_file.stem.split('_')[0].upper()
                    if gloss not in self.gloss_to_files:
                        self.gloss_to_files[gloss] = []
                    self.gloss_to_files[gloss].append(json_file)
                    count += 1
                except:
                    continue
        
        print(f"[MotionGenerator] Indexed {len(self.gloss_to_files)} glosses, {count} files")
    
    def load_pose_sequence(self, gloss: str) -> Optional[List[PoseFrame]]:
        """Load pose frames for a gloss."""
        gloss = gloss.upper().strip()
        
        if gloss in self._pose_cache:
            return self._pose_cache[gloss]
        
        if gloss not in self.gloss_to_files:
            return None
        
        # Select random variant
        pose_file = random.choice(self.gloss_to_files[gloss])
        
        try:
            with open(pose_file, 'r') as f:
                data = json.load(f)
            
            frames = []
            for i, frame_data in enumerate(data.get('poses', [])):
                frame = PoseFrame.from_dict(frame_data)
                frame.timestamp = i / self.config.fps
                frames.append(frame)
            
            self._pose_cache[gloss] = frames
            return frames
        except Exception as e:
            print(f"[WARN] Error loading {gloss}: {e}")
            return None
    
    def generate_sign(
        self,
        gloss: str,
        target_frames: Optional[int] = None
    ) -> List[PoseFrame]:
        """Generate motion for a single sign."""
        frames = self.load_pose_sequence(gloss)
        
        if not frames:
            return []
        
        # Determine target length with jitter
        if target_frames is None:
            base_len = len(frames)
            jitter = random.uniform(1 - self.config.timing_jitter, 1 + self.config.timing_jitter)
            target_frames = int(base_len * jitter)
        
        target_frames = max(self.config.min_sign_frames, 
                           min(target_frames, self.config.max_sign_frames))
        
        # Interpolate to target length
        if len(frames) != target_frames:
            frames = self._interpolate_frames(frames, target_frames)
        
        # Add position variation
        frames = self._add_variation(frames)
        
        # Apply temporal smoothing
        frames = self._smooth_frames(frames)
        
        return frames
    
    def generate_sequence(
        self,
        glosses: List[str],
        add_rest_between: bool = False
    ) -> List[PoseFrame]:
        """Generate motion for a sequence of glosses."""
        if not glosses:
            return []
        
        all_frames = []
        
        for i, gloss in enumerate(glosses):
            sign_frames = self.generate_sign(gloss)
            
            if not sign_frames:
                print(f"[WARN] No motion for: {gloss}")
                continue
            
            if all_frames:
                # Blend transition
                all_frames = self._blend_transition(all_frames, sign_frames)
            else:
                all_frames = sign_frames
            
            # Add hold at peak
            if self.config.hold_frames > 0 and sign_frames:
                peak_idx = len(sign_frames) // 2
                peak_frame = sign_frames[min(peak_idx, len(sign_frames) - 1)]
                for _ in range(self.config.hold_frames):
                    all_frames.append(peak_frame)
            
            # Optional rest between signs
            if add_rest_between and i < len(glosses) - 1:
                rest_frames = self._generate_rest_transition(all_frames[-1])
                all_frames.extend(rest_frames)
        
        return all_frames
    
    def _interpolate_frames(
        self,
        frames: List[PoseFrame],
        target_len: int
    ) -> List[PoseFrame]:
        """Interpolate frames to target length using Hermite splines."""
        if len(frames) < 2:
            return frames * target_len if frames else []
        
        n = len(frames)
        t_orig = np.linspace(0, 1, n)
        t_new = np.linspace(0, 1, target_len)
        
        # Stack all keypoints
        body_stack = np.stack([f.body for f in frames])  # (n, 33, 3)
        left_stack = np.stack([f.left_hand for f in frames])
        right_stack = np.stack([f.right_hand for f in frames])
        
        # Interpolate
        body_interp = self._interpolate_array(body_stack, t_orig, t_new)
        left_interp = self._interpolate_array(left_stack, t_orig, t_new)
        right_interp = self._interpolate_array(right_stack, t_orig, t_new)
        
        # Reconstruct frames
        result = []
        for i in range(target_len):
            result.append(PoseFrame(
                body=body_interp[i],
                left_hand=left_interp[i],
                right_hand=right_interp[i],
                timestamp=t_new[i]
            ))
        
        return result
    
    def _interpolate_array(
        self,
        arr: np.ndarray,
        t_orig: np.ndarray,
        t_new: np.ndarray
    ) -> np.ndarray:
        """Interpolate 3D array along time axis."""
        n_orig, n_points, n_dims = arr.shape
        result = np.zeros((len(t_new), n_points, n_dims))
        
        for p in range(n_points):
            for d in range(n_dims):
                values = arr[:, p, d]
                
                if self.config.blend_velocity and n_orig >= 4:
                    # Compute velocities for Hermite spline
                    velocities = np.gradient(values, t_orig)
                    try:
                        spline = CubicHermiteSpline(t_orig, values, velocities)
                        result[:, p, d] = spline(t_new)
                    except:
                        # Fallback to linear
                        result[:, p, d] = np.interp(t_new, t_orig, values)
                else:
                    result[:, p, d] = np.interp(t_new, t_orig, values)
        
        return result
    
    def _blend_transition(
        self,
        frames1: List[PoseFrame],
        frames2: List[PoseFrame]
    ) -> List[PoseFrame]:
        """Blend end of sequence 1 with start of sequence 2."""
        n_blend = min(self.config.transition_frames, len(frames1) // 2, len(frames2) // 2)
        
        if n_blend < 2:
            return frames1 + frames2
        
        # Keep non-overlapping parts
        result = frames1[:-n_blend]
        
        # Blend overlapping region
        for i in range(n_blend):
            t = (i + 1) / (n_blend + 1)
            # Smooth step function
            weight = t * t * (3 - 2 * t)
            
            f1 = frames1[-(n_blend - i)]
            f2 = frames2[i]
            
            blended = PoseFrame(
                body=(1 - weight) * f1.body + weight * f2.body,
                left_hand=(1 - weight) * f1.left_hand + weight * f2.left_hand,
                right_hand=(1 - weight) * f1.right_hand + weight * f2.right_hand
            )
            result.append(blended)
        
        # Add rest of sequence 2
        result.extend(frames2[n_blend:])
        
        return result
    
    def _generate_rest_transition(self, last_frame: PoseFrame) -> List[PoseFrame]:
        """Generate transition to rest pose."""
        rest = get_rest_pose()
        n_frames = self.config.transition_frames
        
        frames = []
        for i in range(n_frames):
            t = (i + 1) / (n_frames + 1)
            weight = t * t * (3 - 2 * t)
            
            blended = PoseFrame(
                body=(1 - weight) * last_frame.body + weight * rest.body,
                left_hand=(1 - weight) * last_frame.left_hand + weight * rest.left_hand,
                right_hand=(1 - weight) * last_frame.right_hand + weight * rest.right_hand
            )
            frames.append(blended)
        
        return frames
    
    def _add_variation(self, frames: List[PoseFrame]) -> List[PoseFrame]:
        """Add subtle position variation."""
        noise = self.config.position_noise
        
        for frame in frames:
            frame.body += np.random.randn(*frame.body.shape) * noise
            frame.left_hand += np.random.randn(*frame.left_hand.shape) * noise
            frame.right_hand += np.random.randn(*frame.right_hand.shape) * noise
        
        return frames
    
    def _smooth_frames(self, frames: List[PoseFrame]) -> List[PoseFrame]:
        """Apply Savitzky-Golay smoothing."""
        if len(frames) < self.config.savgol_window:
            return frames
        
        # Stack arrays
        body_stack = np.stack([f.body for f in frames])
        left_stack = np.stack([f.left_hand for f in frames])
        right_stack = np.stack([f.right_hand for f in frames])
        
        # Smooth along time axis
        window = min(self.config.savgol_window, len(frames))
        if window % 2 == 0:
            window -= 1
        
        if window >= 3:
            body_smooth = savgol_filter(body_stack, window, self.config.savgol_order, axis=0)
            left_smooth = savgol_filter(left_stack, window, self.config.savgol_order, axis=0)
            right_smooth = savgol_filter(right_stack, window, self.config.savgol_order, axis=0)
        else:
            body_smooth, left_smooth, right_smooth = body_stack, left_stack, right_stack
        
        # Reconstruct
        for i, frame in enumerate(frames):
            frame.body = body_smooth[i]
            frame.left_hand = left_smooth[i]
            frame.right_hand = right_smooth[i]
        
        return frames
    
    def generate_for_blender(
        self,
        glosses: List[str],
        output_path: Optional[str] = None
    ) -> List[Dict]:
        """Generate motion in Blender-compatible format."""
        frames = self.generate_sequence(glosses)
        
        result = [f.to_dict() for f in frames]
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump({
                    'glosses': glosses,
                    'fps': self.config.fps,
                    'num_frames': len(result),
                    'poses': result
                }, f)
            print(f"Saved Blender motion to: {output_path}")
        
        return result
    
    def get_available_glosses(self) -> List[str]:
        return sorted(self.gloss_to_files.keys())


class SignAvatarsAdapter:
    """
    Placeholder for SignAvatars dataset integration.
    
    Will provide higher quality motion data when dataset is approved.
    """
    
    def __init__(self, signavatars_dir: str):
        self.signavatars_dir = Path(signavatars_dir)
        self.available = self._check_availability()
    
    def _check_availability(self) -> bool:
        """Check if SignAvatars data is available."""
        return self.signavatars_dir.exists() and any(self.signavatars_dir.iterdir())
    
    def load_motion(self, gloss: str) -> Optional[List[PoseFrame]]:
        """Load motion from SignAvatars."""
        if not self.available:
            return None
        
        # TODO: Implement when dataset is approved
        return None


# Convenience function
def generate_natural_motion(
    glosses: List[str],
    poses_dir: str = "D:/Signlytic_AI/code/bsl_translation_project/data/poses"
) -> List[Dict]:
    """Generate natural BSL motion sequence."""
    generator = MotionGenerator(poses_dir)
    frames = generator.generate_sequence(glosses)
    return [f.to_dict() for f in frames]


# Testing
if __name__ == "__main__":
    poses_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/poses")
    
    if poses_dir.exists():
        gen = MotionGenerator(str(poses_dir))
        
        glosses = ["HELLO", "GOOD", "YOU"]
        frames = gen.generate_sequence(glosses)
        
        print(f"Generated {len(frames)} frames for {glosses}")
        
        if frames:
            print(f"First frame body shape: {frames[0].body.shape}")
            print(f"First frame left hand shape: {frames[0].left_hand.shape}")
    else:
        print(f"Poses directory not found: {poses_dir}")