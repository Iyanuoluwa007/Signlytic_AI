"""
Natural Motion Generator for BSL Signing

Addresses the "robotic signing" issue by:
1. Smooth interpolation between poses
2. Motion dynamics (velocity, acceleration)
3. Easing functions for natural movement
4. Co-articulation between consecutive signs
5. Rest pose blending

Compatible with:
- Current 2D pose renderer
- Future Blender avatar integration
- SignAvatars dataset (when available)

Output format matches existing pose structure:
{
    'pose': [[x,y,z], ...],  # 33 body keypoints
    'left_hand': [[x,y,z], ...],  # 21 hand keypoints
    'right_hand': [[x,y,z], ...],  # 21 hand keypoints
}
"""

import numpy as np
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import random


@dataclass
class MotionConfig:
    """Configuration for motion generation."""
    # Timing
    fps: int = 25
    min_sign_frames: int = 15
    max_sign_frames: int = 60
    transition_frames: int = 8
    
    # Smoothing
    temporal_smoothing: float = 1.5
    spatial_smoothing: float = 0.5
    
    # Easing
    ease_in_duration: float = 0.15
    ease_out_duration: float = 0.15
    
    # Co-articulation
    coarticulation_overlap: float = 0.3
    
    # Variation
    position_noise: float = 0.01
    timing_variation: float = 0.1
    
    # Rest pose
    use_rest_pose: bool = True
    rest_pose_duration: float = 0.2


class EasingFunctions:
    """Easing functions for natural motion."""
    
    @staticmethod
    def ease_in_out_cubic(t: float) -> float:
        """Smooth acceleration and deceleration."""
        if t < 0.5:
            return 4 * t * t * t
        else:
            return 1 - pow(-2 * t + 2, 3) / 2
    
    @staticmethod
    def ease_in_out_quad(t: float) -> float:
        """Gentler easing."""
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - pow(-2 * t + 2, 2) / 2
    
    @staticmethod
    def ease_in_out_sine(t: float) -> float:
        """Sinusoidal easing."""
        return -(np.cos(np.pi * t) - 1) / 2
    
    @staticmethod
    def apply_ease(values: np.ndarray, ease_func, direction: str = 'both') -> np.ndarray:
        """Apply easing to a sequence of values."""
        n = len(values)
        t = np.linspace(0, 1, n)
        
        if direction == 'in':
            weights = np.array([ease_func(ti) for ti in t])
        elif direction == 'out':
            weights = np.array([ease_func(1 - ti) for ti in t])
        else:  # both
            weights = np.array([ease_func(ti) for ti in t])
        
        return values * weights[:, np.newaxis] if len(values.shape) > 1 else values * weights


class RestPoseGenerator:
    """Generates natural rest poses between signs."""
    
    # Default rest pose (neutral standing with arms at sides)
    DEFAULT_REST = {
        # Key body landmarks (simplified)
        'shoulders': np.array([[0.35, 0.25, 0], [0.65, 0.25, 0]]),  # L, R
        'elbows': np.array([[0.30, 0.45, 0], [0.70, 0.45, 0]]),
        'wrists': np.array([[0.32, 0.60, 0], [0.68, 0.60, 0]]),
        'hips': np.array([[0.42, 0.55, 0], [0.58, 0.55, 0]]),
    }
    
    def __init__(self, variation: float = 0.02):
        self.variation = variation
    
    def generate_rest_pose(self, num_frames: int) -> List[Dict]:
        """Generate rest pose frames with subtle natural movement."""
        poses = []
        
        for i in range(num_frames):
            # Add subtle breathing-like movement
            breath_phase = np.sin(2 * np.pi * i / num_frames) * 0.005
            
            pose = self._create_full_pose(breath_phase)
            poses.append(pose)
        
        return poses
    
    def _create_full_pose(self, offset: float = 0) -> Dict:
        """Create a complete pose dictionary."""
        # Body pose (33 keypoints)
        body = np.zeros((33, 3))
        
        # Head/face (0-10)
        body[0] = [0.5, 0.1 + offset, 0]  # Nose
        
        # Shoulders (11-12)
        body[11] = [0.35, 0.25 + offset, 0]
        body[12] = [0.65, 0.25 + offset, 0]
        
        # Elbows (13-14)
        body[13] = [0.30, 0.45, 0]
        body[14] = [0.70, 0.45, 0]
        
        # Wrists (15-16)
        body[15] = [0.32, 0.60, 0]
        body[16] = [0.68, 0.60, 0]
        
        # Hips (23-24)
        body[23] = [0.42, 0.55, 0]
        body[24] = [0.58, 0.55, 0]
        
        # Add variation
        body += np.random.randn(*body.shape) * self.variation
        
        # Hands (21 keypoints each) - relaxed position
        left_hand = self._create_relaxed_hand(body[15])
        right_hand = self._create_relaxed_hand(body[16])
        
        return {
            'pose': body.tolist(),
            'left_hand': left_hand.tolist(),
            'right_hand': right_hand.tolist()
        }
    
    def _create_relaxed_hand(self, wrist_pos: np.ndarray) -> np.ndarray:
        """Create relaxed hand keypoints relative to wrist."""
        hand = np.zeros((21, 3))
        
        # Wrist
        hand[0] = wrist_pos
        
        # Generate finger positions (simplified relaxed pose)
        for finger_idx in range(5):
            base_idx = 1 + finger_idx * 4
            
            for joint in range(4):
                offset_y = 0.02 + joint * 0.015
                offset_x = (finger_idx - 2) * 0.01
                
                hand[base_idx + joint] = wrist_pos + [offset_x, offset_y, 0]
        
        return hand


class MotionInterpolator:
    """Interpolates between poses for smooth motion."""
    
    def __init__(self, config: MotionConfig):
        self.config = config
    
    def interpolate_sequence(
        self,
        poses: List[Dict],
        target_frames: int,
        method: str = 'cubic'
    ) -> List[Dict]:
        """
        Interpolate pose sequence to target number of frames.
        
        Args:
            poses: List of pose dictionaries
            target_frames: Desired number of output frames
            method: Interpolation method ('linear', 'cubic', 'akima')
        """
        if len(poses) < 2:
            return poses * target_frames if poses else []
        
        # Extract arrays
        body_array = np.array([p['pose'] for p in poses])
        left_array = np.array([p['left_hand'] for p in poses])
        right_array = np.array([p['right_hand'] for p in poses])
        
        # Interpolate each component
        body_interp = self._interpolate_array(body_array, target_frames, method)
        left_interp = self._interpolate_array(left_array, target_frames, method)
        right_interp = self._interpolate_array(right_array, target_frames, method)
        
        # Apply temporal smoothing
        if self.config.temporal_smoothing > 0:
            body_interp = gaussian_filter1d(body_interp, self.config.temporal_smoothing, axis=0)
            left_interp = gaussian_filter1d(left_interp, self.config.temporal_smoothing, axis=0)
            right_interp = gaussian_filter1d(right_interp, self.config.temporal_smoothing, axis=0)
        
        # Reconstruct poses
        result = []
        for i in range(target_frames):
            result.append({
                'pose': body_interp[i].tolist(),
                'left_hand': left_interp[i].tolist(),
                'right_hand': right_interp[i].tolist()
            })
        
        return result
    
    def _interpolate_array(
        self,
        array: np.ndarray,
        target_frames: int,
        method: str
    ) -> np.ndarray:
        """Interpolate a pose array."""
        n_frames, n_points, n_dims = array.shape
        
        # Original time points
        t_orig = np.linspace(0, 1, n_frames)
        t_new = np.linspace(0, 1, target_frames)
        
        # Interpolate each keypoint
        result = np.zeros((target_frames, n_points, n_dims))
        
        for p in range(n_points):
            for d in range(n_dims):
                values = array[:, p, d]
                
                if method == 'cubic' and n_frames >= 4:
                    f = interpolate.interp1d(t_orig, values, kind='cubic', fill_value='extrapolate')
                elif method == 'akima' and n_frames >= 4:
                    f = interpolate.Akima1DInterpolator(t_orig, values)
                else:
                    f = interpolate.interp1d(t_orig, values, kind='linear', fill_value='extrapolate')
                
                result[:, p, d] = f(t_new)
        
        return result


class CoarticulationBlender:
    """Blends consecutive signs for natural co-articulation."""
    
    def __init__(self, config: MotionConfig):
        self.config = config
    
    def blend_signs(
        self,
        sign1_poses: List[Dict],
        sign2_poses: List[Dict],
        overlap_frames: Optional[int] = None
    ) -> List[Dict]:
        """
        Blend end of sign1 with beginning of sign2.
        
        Args:
            sign1_poses: Poses for first sign
            sign2_poses: Poses for second sign
            overlap_frames: Number of frames to blend
        """
        if not sign1_poses or not sign2_poses:
            return sign1_poses + sign2_poses
        
        if overlap_frames is None:
            overlap_frames = int(
                min(len(sign1_poses), len(sign2_poses)) * 
                self.config.coarticulation_overlap
            )
        
        overlap_frames = max(1, min(overlap_frames, len(sign1_poses) // 2, len(sign2_poses) // 2))
        
        # Split sequences
        sign1_main = sign1_poses[:-overlap_frames]
        sign1_blend = sign1_poses[-overlap_frames:]
        sign2_blend = sign2_poses[:overlap_frames]
        sign2_main = sign2_poses[overlap_frames:]
        
        # Blend overlapping frames
        blended = []
        for i in range(overlap_frames):
            # Smooth blending weight
            t = (i + 1) / (overlap_frames + 1)
            weight = EasingFunctions.ease_in_out_sine(t)
            
            blended_pose = self._blend_poses(sign1_blend[i], sign2_blend[i], weight)
            blended.append(blended_pose)
        
        return sign1_main + blended + sign2_main
    
    def _blend_poses(
        self,
        pose1: Dict,
        pose2: Dict,
        weight: float
    ) -> Dict:
        """Linearly blend two poses."""
        result = {}
        
        for key in ['pose', 'left_hand', 'right_hand']:
            arr1 = np.array(pose1.get(key, []))
            arr2 = np.array(pose2.get(key, []))
            
            if arr1.shape == arr2.shape:
                blended = (1 - weight) * arr1 + weight * arr2
                result[key] = blended.tolist()
            else:
                result[key] = pose1.get(key, [])
        
        return result


class NaturalMotionGenerator:
    """
    Main class for generating natural BSL motion sequences.
    
    Features:
    - Pose lookup from database
    - Smooth interpolation
    - Natural easing
    - Co-articulation between signs
    - Rest pose blending
    - Motion variation for realism
    """
    
    def __init__(
        self,
        poses_dir: str,
        config: Optional[MotionConfig] = None
    ):
        self.poses_dir = Path(poses_dir)
        self.config = config or MotionConfig()
        
        # Components
        self.interpolator = MotionInterpolator(self.config)
        self.coarticulator = CoarticulationBlender(self.config)
        self.rest_generator = RestPoseGenerator()
        
        # Build pose index
        self.gloss_to_files: Dict[str, List[Path]] = {}
        self._build_index()
    
    def _build_index(self):
        """Index available pose files by gloss."""
        if not self.poses_dir.exists():
            print(f"Warning: Poses directory not found: {self.poses_dir}")
            return
        
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
                except:
                    continue
        
        print(f"Indexed {len(self.gloss_to_files)} glosses for motion generation")
    
    def load_pose_sequence(self, gloss: str) -> Optional[List[Dict]]:
        """Load pose sequence for a gloss."""
        gloss = gloss.upper().strip()
        
        if gloss not in self.gloss_to_files:
            return None
        
        # Select random variant
        pose_file = random.choice(self.gloss_to_files[gloss])
        
        try:
            with open(pose_file, 'r') as f:
                data = json.load(f)
            
            poses = data.get('poses', [])
            
            # Convert frame format if needed
            result = []
            for frame in poses:
                result.append({
                    'pose': frame.get('pose', []),
                    'left_hand': frame.get('left_hand', []),
                    'right_hand': frame.get('right_hand', [])
                })
            
            return result
        except:
            return None
    
    def generate_sign_motion(
        self,
        gloss: str,
        target_duration: Optional[float] = None,
        add_easing: bool = True
    ) -> List[Dict]:
        """
        Generate natural motion for a single sign.
        
        Args:
            gloss: The gloss to generate
            target_duration: Optional duration in seconds
            add_easing: Whether to add ease-in/out
        """
        # Load base poses
        poses = self.load_pose_sequence(gloss)
        
        if not poses:
            print(f"Warning: No poses found for {gloss}")
            return []
        
        # Calculate target frames
        if target_duration:
            target_frames = int(target_duration * self.config.fps)
        else:
            # Add timing variation
            base_frames = len(poses)
            variation = random.uniform(1 - self.config.timing_variation, 
                                       1 + self.config.timing_variation)
            target_frames = int(base_frames * variation)
        
        target_frames = max(self.config.min_sign_frames, 
                           min(target_frames, self.config.max_sign_frames))
        
        # Interpolate to target length
        poses = self.interpolator.interpolate_sequence(poses, target_frames)
        
        # Apply easing
        if add_easing:
            poses = self._apply_motion_easing(poses)
        
        # Add position variation
        poses = self._add_position_variation(poses)
        
        return poses
    
    def generate_sequence(
        self,
        glosses: List[str],
        add_rest_poses: bool = True
    ) -> List[Dict]:
        """
        Generate complete motion sequence for multiple glosses.
        
        Args:
            glosses: List of glosses to generate
            add_rest_poses: Whether to add rest poses between signs
        """
        if not glosses:
            return []
        
        all_poses = []
        
        for i, gloss in enumerate(glosses):
            # Generate sign motion
            sign_poses = self.generate_sign_motion(gloss)
            
            if not sign_poses:
                continue
            
            if all_poses:
                # Blend with previous sign
                all_poses = self.coarticulator.blend_signs(all_poses, sign_poses)
            else:
                all_poses = sign_poses
            
            # Add rest pose between signs (except after last)
            if add_rest_poses and i < len(glosses) - 1 and self.config.use_rest_pose:
                rest_frames = int(self.config.rest_pose_duration * self.config.fps)
                rest_poses = self._generate_transition_to_rest(all_poses[-1], rest_frames)
                all_poses.extend(rest_poses)
        
        return all_poses
    
    def _apply_motion_easing(self, poses: List[Dict]) -> List[Dict]:
        """Apply ease-in and ease-out to motion."""
        n = len(poses)
        ease_in_frames = int(n * self.config.ease_in_duration)
        ease_out_frames = int(n * self.config.ease_out_duration)
        
        # Convert to array
        body_array = np.array([p['pose'] for p in poses])
        left_array = np.array([p['left_hand'] for p in poses])
        right_array = np.array([p['right_hand'] for p in poses])
        
        # Apply easing weights
        weights = np.ones(n)
        
        # Ease in
        for i in range(ease_in_frames):
            weights[i] = EasingFunctions.ease_in_out_cubic(i / ease_in_frames)
        
        # Ease out
        for i in range(ease_out_frames):
            idx = n - ease_out_frames + i
            weights[idx] *= EasingFunctions.ease_in_out_cubic(1 - i / ease_out_frames)
        
        # Blend with first/last frame based on weights
        first_body = body_array[0]
        last_body = body_array[-1]
        
        for i in range(n):
            if i < ease_in_frames:
                body_array[i] = (1 - weights[i]) * first_body + weights[i] * body_array[i]
            elif i >= n - ease_out_frames:
                body_array[i] = weights[i] * body_array[i] + (1 - weights[i]) * last_body
        
        # Similarly for hands
        first_left = left_array[0]
        first_right = right_array[0]
        
        for i in range(ease_in_frames):
            left_array[i] = (1 - weights[i]) * first_left + weights[i] * left_array[i]
            right_array[i] = (1 - weights[i]) * first_right + weights[i] * right_array[i]
        
        # Reconstruct poses
        result = []
        for i in range(n):
            result.append({
                'pose': body_array[i].tolist(),
                'left_hand': left_array[i].tolist(),
                'right_hand': right_array[i].tolist()
            })
        
        return result
    
    def _add_position_variation(self, poses: List[Dict]) -> List[Dict]:
        """Add subtle random variation to poses."""
        noise_level = self.config.position_noise
        
        for pose in poses:
            for key in ['pose', 'left_hand', 'right_hand']:
                arr = np.array(pose.get(key, []))
                if arr.size > 0:
                    noise = np.random.randn(*arr.shape) * noise_level
                    pose[key] = (arr + noise).tolist()
        
        return poses
    
    def _generate_transition_to_rest(
        self,
        last_pose: Dict,
        num_frames: int
    ) -> List[Dict]:
        """Generate smooth transition from last pose to rest."""
        rest_poses = self.rest_generator.generate_rest_pose(1)
        
        if not rest_poses:
            return []
        
        rest_pose = rest_poses[0]
        
        # Interpolate
        transition = []
        for i in range(num_frames):
            t = (i + 1) / (num_frames + 1)
            weight = EasingFunctions.ease_in_out_sine(t)
            
            blended = self.coarticulator._blend_poses(last_pose, rest_pose, weight)
            transition.append(blended)
        
        return transition
    
    def get_available_glosses(self) -> List[str]:
        """Get list of available glosses."""
        return sorted(self.gloss_to_files.keys())


# API compatible with existing system
def generate_natural_motion(
    glosses: List[str],
    poses_dir: str = "D:/Signlytic_AI/code/bsl_translation_project/data/poses",
    fps: int = 25
) -> List[Dict]:
    """
    Generate natural BSL motion sequence.
    
    Drop-in replacement for existing motion generation.
    
    Args:
        glosses: List of BSL glosses
        poses_dir: Path to pose data
        fps: Output frame rate
        
    Returns:
        List of pose dictionaries compatible with 2D renderer and Blender
    """
    config = MotionConfig(fps=fps)
    generator = NaturalMotionGenerator(poses_dir, config)
    
    return generator.generate_sequence(glosses)


# Testing
if __name__ == "__main__":
    print("Testing Natural Motion Generator...")
    
    poses_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/poses")
    
    if poses_dir.exists():
        generator = NaturalMotionGenerator(str(poses_dir))
        
        # Test single sign
        glosses = ["HELLO", "GOOD", "YOU"]
        
        for gloss in glosses:
            poses = generator.generate_sign_motion(gloss)
            if poses:
                print(f"{gloss}: {len(poses)} frames")
        
        # Test sequence
        sequence = generator.generate_sequence(glosses)
        print(f"\nFull sequence: {len(sequence)} frames")
        
        # Verify output format
        if sequence:
            sample = sequence[0]
            print(f"Pose keys: {list(sample.keys())}")
            print(f"Body shape: {len(sample['pose'])} keypoints")
            print(f"Left hand shape: {len(sample['left_hand'])} keypoints")
            print(f"Right hand shape: {len(sample['right_hand'])} keypoints")
    else:
        print(f"Poses directory not found: {poses_dir}")
    
    print("\n[OK] Motion generator test complete!")
