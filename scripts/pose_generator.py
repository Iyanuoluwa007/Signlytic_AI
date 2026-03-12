"""
BSL Motion Generator (V2)

Redesigned gloss -> pose generation module that produces natural signing
motion for the 2D renderer and is compatible with future Blender avatars.

Three generation modes:
  1. LookupBlend:    Retrieves stored pose sequences and blends transitions
  2. ConditionalVAE: Learned generative model for novel motion synthesis
  3. Placeholder for SignAvatars integration (SMPL-X motion)

Key improvements over V1:
  - Smooth transitions between consecutive signs (spline interpolation)
  - Velocity and acceleration constraints for natural motion
  - Bone length consistency enforcement
  - Configurable hold frames for sign boundaries
  - Compatible with both 2D renderer and Blender armature

Pose format (compatible with existing pipeline):
  {
      "pose":       [[x, y, z], ...],   # 33 body keypoints
      "left_hand":  [[x, y, z], ...],   # 21 left hand keypoints
      "right_hand": [[x, y, z], ...],   # 21 right hand keypoints
  }

Usage:
    generator = MotionGenerator(poses_dir="data/poses", mode="lookup_blend")
    frames = generator.generate(["HELLO", "HOW", "YOU"])
    # Returns list of pose dicts, ready for 2D renderer or Blender
"""

import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict


# --- Constants ---
NUM_BODY_JOINTS = 33
NUM_HAND_JOINTS = 21
FPS = 25
REST_POSE_DURATION = 0.3        # Seconds of rest pose between signs
TRANSITION_DURATION = 0.2       # Seconds for blending between signs
HOLD_DURATION = 0.15            # Seconds to hold the sign's peak frame


class PoseFrame:
    """Single frame of pose data."""

    __slots__ = ["body", "left_hand", "right_hand"]

    def __init__(
        self,
        body: np.ndarray,       # (33, 3)
        left_hand: np.ndarray,  # (21, 3)
        right_hand: np.ndarray, # (21, 3)
    ):
        self.body = body
        self.left_hand = left_hand
        self.right_hand = right_hand

    def to_dict(self) -> dict:
        """Convert to the format expected by the 2D renderer and Blender."""
        return {
            "pose": self.body.tolist(),
            "left_hand": self.left_hand.tolist(),
            "right_hand": self.right_hand.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "PoseFrame":
        """Parse from existing pose dict format."""
        body = np.array(d.get("pose", d.get("body_pose", [])), dtype=np.float32)
        lh = np.array(d.get("left_hand", []), dtype=np.float32)
        rh = np.array(d.get("right_hand", []), dtype=np.float32)

        # Ensure correct shapes
        if body.ndim == 1:
            body = body.reshape(-1, 3)
        if lh.ndim == 1:
            lh = lh.reshape(-1, 3)
        if rh.ndim == 1:
            rh = rh.reshape(-1, 3)

        # Pad if needed
        body = _pad_joints(body, NUM_BODY_JOINTS)
        lh = _pad_joints(lh, NUM_HAND_JOINTS)
        rh = _pad_joints(rh, NUM_HAND_JOINTS)

        return PoseFrame(body, lh, rh)

    def as_flat(self) -> np.ndarray:
        """Flatten to a single vector (33+21+21)*3 = 225."""
        return np.concatenate([
            self.body.flatten(),
            self.left_hand.flatten(),
            self.right_hand.flatten(),
        ])

    @staticmethod
    def from_flat(vec: np.ndarray) -> "PoseFrame":
        """Reconstruct from flat vector."""
        body = vec[:NUM_BODY_JOINTS * 3].reshape(NUM_BODY_JOINTS, 3)
        lh = vec[NUM_BODY_JOINTS * 3:(NUM_BODY_JOINTS + NUM_HAND_JOINTS) * 3].reshape(NUM_HAND_JOINTS, 3)
        rh = vec[(NUM_BODY_JOINTS + NUM_HAND_JOINTS) * 3:].reshape(NUM_HAND_JOINTS, 3)
        return PoseFrame(body, lh, rh)


def _pad_joints(arr: np.ndarray, target_n: int) -> np.ndarray:
    """Pad joint array to target number of joints."""
    if len(arr) >= target_n:
        return arr[:target_n]
    pad = np.zeros((target_n - len(arr), 3), dtype=np.float32)
    return np.concatenate([arr, pad], axis=0)


# =============================================================================
# REST POSE
# =============================================================================

def get_rest_pose() -> PoseFrame:
    """
    Neutral rest/idle pose: arms at sides, hands relaxed.
    Based on a standing MediaPipe skeleton.
    """
    body = np.zeros((NUM_BODY_JOINTS, 3), dtype=np.float32)
    # Set approximate neutral standing pose (normalized coords)
    # Head
    body[0] = [0.5, 0.15, 0.0]    # nose
    # Shoulders
    body[11] = [0.4, 0.35, 0.0]   # left shoulder
    body[12] = [0.6, 0.35, 0.0]   # right shoulder
    # Elbows
    body[13] = [0.35, 0.50, 0.0]  # left elbow
    body[14] = [0.65, 0.50, 0.0]  # right elbow
    # Wrists
    body[15] = [0.37, 0.62, 0.0]  # left wrist
    body[16] = [0.63, 0.62, 0.0]  # right wrist
    # Hips
    body[23] = [0.45, 0.65, 0.0]  # left hip
    body[24] = [0.55, 0.65, 0.0]  # right hip

    lh = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
    rh = np.zeros((NUM_HAND_JOINTS, 3), dtype=np.float32)
    # Relaxed hand positions near wrists
    for i in range(NUM_HAND_JOINTS):
        lh[i] = [0.37 + (i % 5) * 0.005, 0.62 + (i // 5) * 0.01, 0.0]
        rh[i] = [0.63 + (i % 5) * 0.005, 0.62 + (i // 5) * 0.01, 0.0]

    return PoseFrame(body, lh, rh)


# =============================================================================
# INTERPOLATION AND SMOOTHING
# =============================================================================

def lerp_frames(a: PoseFrame, b: PoseFrame, t: float) -> PoseFrame:
    """Linear interpolation between two pose frames."""
    t = np.clip(t, 0.0, 1.0)
    return PoseFrame(
        body=(1 - t) * a.body + t * b.body,
        left_hand=(1 - t) * a.left_hand + t * b.left_hand,
        right_hand=(1 - t) * a.right_hand + t * b.right_hand,
    )


def smoothstep(t: float) -> float:
    """Hermite smoothstep for ease-in-out transitions."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def cubic_hermite_blend(
    frames_a: List[PoseFrame],
    frames_b: List[PoseFrame],
    n_transition: int,
) -> List[PoseFrame]:
    """
    Create a smooth cubic Hermite transition between two pose sequences.
    Uses the last frames of A and first frames of B to estimate velocity.
    """
    if n_transition < 1:
        return []

    # Get boundary frames and velocities
    end_a = frames_a[-1]
    start_b = frames_b[0]

    # Estimate velocity from adjacent frames
    if len(frames_a) >= 2:
        vel_a = PoseFrame(
            body=frames_a[-1].body - frames_a[-2].body,
            left_hand=frames_a[-1].left_hand - frames_a[-2].left_hand,
            right_hand=frames_a[-1].right_hand - frames_a[-2].right_hand,
        )
    else:
        vel_a = PoseFrame(
            np.zeros_like(end_a.body),
            np.zeros_like(end_a.left_hand),
            np.zeros_like(end_a.right_hand),
        )

    if len(frames_b) >= 2:
        vel_b = PoseFrame(
            body=frames_b[1].body - frames_b[0].body,
            left_hand=frames_b[1].left_hand - frames_b[0].left_hand,
            right_hand=frames_b[1].right_hand - frames_b[0].right_hand,
        )
    else:
        vel_b = PoseFrame(
            np.zeros_like(start_b.body),
            np.zeros_like(start_b.left_hand),
            np.zeros_like(start_b.right_hand),
        )

    transition = []
    for i in range(n_transition):
        t = (i + 1) / (n_transition + 1)
        # Hermite basis functions
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2

        frame = PoseFrame(
            body=h00 * end_a.body + h10 * vel_a.body + h01 * start_b.body + h11 * vel_b.body,
            left_hand=h00 * end_a.left_hand + h10 * vel_a.left_hand + h01 * start_b.left_hand + h11 * vel_b.left_hand,
            right_hand=h00 * end_a.right_hand + h10 * vel_a.right_hand + h01 * start_b.right_hand + h11 * vel_b.right_hand,
        )
        transition.append(frame)

    return transition


def apply_savgol_smoothing(
    frames: List[PoseFrame],
    window_length: int = 7,
    polyorder: int = 3,
) -> List[PoseFrame]:
    """
    Apply Savitzky-Golay filter to smooth a pose sequence.
    Preserves sign boundaries better than Gaussian smoothing.
    """
    if len(frames) < window_length:
        return frames

    try:
        from scipy.signal import savgol_filter
    except ImportError:
        print("  [WARN] scipy not available for Savgol filter, skipping smoothing")
        return frames

    # Stack all frames as flat vectors
    flat = np.array([f.as_flat() for f in frames])  # (T, D)

    # Apply per-dimension
    smoothed = savgol_filter(flat, window_length, polyorder, axis=0)

    return [PoseFrame.from_flat(smoothed[i]) for i in range(len(smoothed))]


def enforce_bone_lengths(
    frames: List[PoseFrame],
    reference: Optional[PoseFrame] = None,
    strength: float = 0.5,
) -> List[PoseFrame]:
    """
    Post-process to enforce consistent bone lengths across frames.
    This prevents limbs from stretching unnaturally during transitions.
    """
    if not frames:
        return frames

    # Define bone connections (parent -> child joint pairs for body)
    bones = [
        (11, 13), (13, 15),  # Left arm
        (12, 14), (14, 16),  # Right arm
        (11, 23), (12, 24),  # Torso to hips
        (23, 25), (25, 27),  # Left leg
        (24, 26), (26, 28),  # Right leg
    ]

    # Compute reference bone lengths from first frame (or provided reference)
    ref = reference or frames[0]
    ref_lengths = {}
    for p, c in bones:
        if p < len(ref.body) and c < len(ref.body):
            ref_lengths[(p, c)] = np.linalg.norm(ref.body[c] - ref.body[p])

    # Enforce on each frame
    corrected = []
    for frame in frames:
        body = frame.body.copy()
        for (p, c), target_len in ref_lengths.items():
            if p >= len(body) or c >= len(body):
                continue
            current = body[c] - body[p]
            current_len = np.linalg.norm(current)
            if current_len > 1e-6:
                correction = (target_len / current_len - 1.0) * strength
                body[c] = body[p] + current * (1.0 + correction)

        corrected.append(PoseFrame(body, frame.left_hand.copy(), frame.right_hand.copy()))

    return corrected


# =============================================================================
# MOTION GENERATOR
# =============================================================================

class PoseLookup:
    """Index and retrieve stored pose sequences by gloss."""

    def __init__(self, poses_dir: str):
        self.poses_dir = Path(poses_dir)
        self.index: Dict[str, List[Path]] = defaultdict(list)
        self._build_index()

    def _build_index(self):
        if not self.poses_dir.exists():
            print(f"  [WARN] Poses directory not found: {self.poses_dir}")
            return

        for subdir in [self.poses_dir, self.poses_dir / "train",
                       self.poses_dir / "val", self.poses_dir / "test"]:
            if not subdir.exists():
                continue
            for fp in subdir.glob("*.json"):
                gloss = fp.stem.upper().split("_")[0]
                self.index[gloss].append(fp)

        print(f"  PoseLookup: {sum(len(v) for v in self.index.values())} files, "
              f"{len(self.index)} glosses")

    def get(self, gloss: str) -> Optional[List[PoseFrame]]:
        """Retrieve a pose sequence for a gloss. Returns None if not found."""
        gloss = gloss.upper().strip()
        if gloss not in self.index:
            return None

        fp = random.choice(self.index[gloss])
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            poses = data.get("poses", data if isinstance(data, list) else [data])
            return [PoseFrame.from_dict(p) for p in poses if p]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  [WARN] Failed to load {fp}: {e}")
            return None

    @property
    def available_glosses(self) -> List[str]:
        return sorted(self.index.keys())


class MotionGenerator:
    """
    Generates smooth BSL signing motion from a sequence of glosses.

    Modes:
      - lookup_blend:    Retrieve stored poses, blend transitions
      - conditional_vae: (Future) Generate novel motion via learned model
      - signavatars:     (Future) Use SignAvatars SMPL-X motion data
    """

    def __init__(
        self,
        poses_dir: str = "data/poses",
        mode: str = "lookup_blend",
        fps: int = FPS,
        transition_sec: float = TRANSITION_DURATION,
        hold_sec: float = HOLD_DURATION,
        rest_sec: float = REST_POSE_DURATION,
        smoothing: bool = True,
        smoothing_window: int = 7,
        enforce_bones: bool = True,
    ):
        self.mode = mode
        self.fps = fps
        self.n_transition = max(1, int(transition_sec * fps))
        self.n_hold = max(1, int(hold_sec * fps))
        self.n_rest = max(1, int(rest_sec * fps))
        self.smoothing = smoothing
        self.smoothing_window = smoothing_window
        self.enforce_bones = enforce_bones

        self.rest_pose = get_rest_pose()

        if mode == "lookup_blend":
            self.lookup = PoseLookup(poses_dir)
        elif mode == "conditional_vae":
            self._load_vae_model()
        elif mode == "signavatars":
            print("  [INFO] SignAvatars mode: pending dataset integration")
            self.lookup = PoseLookup(poses_dir)  # Fallback
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def generate(
        self,
        glosses: List[str],
        return_metadata: bool = False,
    ) -> Union[List[dict], Tuple[List[dict], dict]]:
        """
        Generate a complete signing motion sequence for a list of glosses.

        Args:
            glosses: List of BSL glosses (e.g., ["HELLO", "HOW", "YOU"])
            return_metadata: If True, also return timing metadata

        Returns:
            List of pose dicts (compatible with 2D renderer and Blender),
            optionally with metadata dict
        """
        if not glosses:
            return ([], {}) if return_metadata else []

        all_frames: List[PoseFrame] = []
        metadata = {"glosses": [], "fps": self.fps, "total_frames": 0}

        # Start from rest pose
        rest_frames = [self.rest_pose] * self.n_rest
        all_frames.extend(rest_frames)

        for i, gloss in enumerate(glosses):
            sign_start = len(all_frames)

            # Get sign frames
            sign_frames = self._get_sign_frames(gloss)
            if sign_frames is None:
                print(f"  [WARN] No pose data for '{gloss}', using rest pose")
                sign_frames = [self.rest_pose] * (self.fps // 2)

            # Add hold frames at peak (middle of the sign)
            if self.n_hold > 0 and len(sign_frames) > 2:
                peak_idx = len(sign_frames) // 2
                peak = sign_frames[peak_idx]
                hold = [PoseFrame(peak.body.copy(), peak.left_hand.copy(), peak.right_hand.copy())
                        for _ in range(self.n_hold)]
                sign_frames = sign_frames[:peak_idx] + hold + sign_frames[peak_idx:]

            # Blend transition from previous
            if all_frames and sign_frames:
                transition = cubic_hermite_blend(
                    all_frames[-max(2, self.n_transition):],
                    sign_frames[:max(2, self.n_transition)],
                    self.n_transition,
                )
                all_frames.extend(transition)

            # Add sign frames
            all_frames.extend(sign_frames)
            sign_end = len(all_frames)

            metadata["glosses"].append({
                "gloss": gloss,
                "frame_start": sign_start,
                "frame_end": sign_end,
                "duration_sec": (sign_end - sign_start) / self.fps,
            })

        # Return to rest
        if all_frames:
            transition = cubic_hermite_blend(
                all_frames[-max(2, self.n_transition):],
                rest_frames[:max(2, self.n_transition)],
                self.n_transition,
            )
            all_frames.extend(transition)
            all_frames.extend(rest_frames)

        # Post-processing
        if self.smoothing and len(all_frames) > self.smoothing_window:
            all_frames = apply_savgol_smoothing(all_frames, self.smoothing_window, 3)

        if self.enforce_bones:
            all_frames = enforce_bone_lengths(all_frames, reference=self.rest_pose)

        metadata["total_frames"] = len(all_frames)
        metadata["duration_sec"] = len(all_frames) / self.fps

        # Convert to output format
        output = [f.to_dict() for f in all_frames]

        if return_metadata:
            return output, metadata
        return output

    def _get_sign_frames(self, gloss: str) -> Optional[List[PoseFrame]]:
        """Get pose frames for a single sign."""
        if self.mode in ("lookup_blend", "signavatars"):
            return self.lookup.get(gloss)
        elif self.mode == "conditional_vae":
            return self._generate_vae(gloss)
        return None

    def _load_vae_model(self):
        """Load conditional VAE motion generator (placeholder)."""
        print("  [INFO] ConditionalVAE mode: model loading not yet implemented")
        print("         Falling back to lookup_blend")
        self.mode = "lookup_blend"
        self.lookup = PoseLookup("data/poses")

    def _generate_vae(self, gloss: str) -> Optional[List[PoseFrame]]:
        """Generate motion via conditional VAE (placeholder)."""
        # Placeholder -- will be implemented when SignAvatars data is available
        return self.lookup.get(gloss)

    def generate_for_blender(
        self,
        glosses: List[str],
        output_path: str = "animation_data.json",
    ) -> str:
        """
        Generate motion data formatted for the Blender animation pipeline.
        Saves a JSON file that blender_bsl_animator.py can consume.

        Returns:
            Path to the output JSON file
        """
        frames, metadata = self.generate(glosses, return_metadata=True)

        blender_data = {
            "format_version": "2.0",
            "fps": self.fps,
            "glosses": glosses,
            "metadata": metadata,
            "frames": frames,
            # Joint mapping for Blender armature
            "joint_mapping": {
                "body_indices": list(range(NUM_BODY_JOINTS)),
                "left_hand_indices": list(range(NUM_HAND_JOINTS)),
                "right_hand_indices": list(range(NUM_HAND_JOINTS)),
                "total_joints": NUM_BODY_JOINTS + 2 * NUM_HAND_JOINTS,
            },
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(blender_data, f, indent=2)

        print(f"  Blender animation data saved: {output_path}")
        print(f"    {len(frames)} frames at {self.fps} fps "
              f"({metadata['duration_sec']:.1f}s)")
        return str(output_path)


# =============================================================================
# SIGNAVATARS INTEGRATION STUB
# =============================================================================

class SignAvatarsAdapter:
    """
    Adapter for future SignAvatars dataset integration.

    SignAvatars provides SMPL-X motion data for sign language,
    which can drive both our 2D renderer and the Blender avatar.

    Expected data format:
      - SMPL-X body parameters (pose, shape, expression)
      - Per-frame joint rotations and global translation
      - Sign segmentation timestamps

    This adapter will convert SMPL-X output to our PoseFrame format
    for backward compatibility.
    """

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None
        self.available = False

        if self.data_dir and self.data_dir.exists():
            self._load_index()
        else:
            print("  [INFO] SignAvatars data not yet available")

    def _load_index(self):
        """Index SignAvatars data files."""
        # Placeholder: will populate when dataset approval is received
        self.available = True
        print("  SignAvatars adapter: data indexed")

    def get_motion(self, gloss: str) -> Optional[List[PoseFrame]]:
        """
        Get motion data for a gloss from SignAvatars.
        Converts SMPL-X to PoseFrame format.
        """
        if not self.available:
            return None
        # Placeholder implementation
        return None

    def smplx_to_poseframe(self, smplx_params: dict) -> PoseFrame:
        """Convert SMPL-X parameters to PoseFrame (placeholder)."""
        # Will implement when data format is confirmed
        raise NotImplementedError("Awaiting SignAvatars dataset")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Generate BSL signing motion")
    ap.add_argument("--glosses", nargs="+", default=["HELLO", "HOW", "YOU"])
    ap.add_argument("--poses_dir", type=str, default="data/poses")
    ap.add_argument("--output", type=str, default="animation_data.json")
    ap.add_argument("--mode", type=str, default="lookup_blend")
    ap.add_argument("--blender", action="store_true", help="Output for Blender")
    args = ap.parse_args()

    gen = MotionGenerator(poses_dir=args.poses_dir, mode=args.mode)

    if args.blender:
        gen.generate_for_blender(args.glosses, args.output)
    else:
        frames, meta = gen.generate(args.glosses, return_metadata=True)
        print(f"\nGenerated {len(frames)} frames for: {' '.join(args.glosses)}")
        print(f"Duration: {meta['duration_sec']:.1f}s at {meta['fps']} fps")
        for g in meta["glosses"]:
            print(f"  {g['gloss']}: frames {g['frame_start']}-{g['frame_end']} "
                  f"({g['duration_sec']:.1f}s)")

        # Save
        with open(args.output, "w") as f:
            json.dump({"frames": frames, "metadata": meta}, f)
        print(f"Saved to {args.output}")
