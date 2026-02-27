"""
Real-Time BSL Recognition from Camera or Video

Captures input frames, extracts poses with MediaPipe,
recognizes signs using trained model, and converts to speech.

Pipeline:
    Input Source -> MediaPipe -> Pose Sequence -> Recognition Model -> Gloss -> Text -> Speech

Usage:
    python scripts/realtime_recognition.py
    python scripts/realtime_recognition.py --video path/to/file.mp4
    python scripts/realtime_recognition.py --video path/to/file.mp4 --no-speech
"""

import os
import sys
import json
import time
import queue
import threading
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

import cv2
import torch

# Add project paths
project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "inference"))
sys.path.insert(0, str(project_root / "scripts"))

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("MediaPipe not available")


class RealtimePoseExtractor:
    """Real-time pose extraction from input frames."""
    
    def __init__(self, swap_hands: bool = False):
        """Initialize MediaPipe models."""
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError("MediaPipe not installed")
        self.swap_hands = swap_hands
        
        # Model paths
        model_dir = project_root / "models" / "mediapipe"
        pose_model = model_dir / "pose_landmarker_heavy.task"
        hand_model = model_dir / "hand_landmarker.task"
        
        if not pose_model.exists() or not hand_model.exists():
            raise FileNotFoundError("MediaPipe models not found. Run extract_poses.py first.")
        
        # Create landmarkers for VIDEO mode
        base_options_pose = python.BaseOptions(model_asset_path=str(pose_model))
        pose_options = vision.PoseLandmarkerOptions(
            base_options=base_options_pose,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=False,
            num_poses=1
        )
        self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        
        base_options_hand = python.BaseOptions(model_asset_path=str(hand_model))
        hand_options = vision.HandLandmarkerOptions(
            base_options=base_options_hand,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2
        )
        self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        
        self.frame_timestamp = 0
    
    def extract(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract pose from a single frame.
        
        Args:
            frame: BGR frame from camera or video
            
        Returns:
            Pose vector (75*3,) = 225 dimensions
        """
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Increment timestamp (milliseconds)
        self.frame_timestamp += 33  # ~30 fps
        
        # Extract pose
        pose_result = self.pose_landmarker.detect_for_video(mp_image, self.frame_timestamp)
        pose_data = [(0, 0, 0)] * 33
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
            pose_data = [(lm.x, lm.y, lm.z) for lm in pose_result.pose_landmarks[0]]
        
        # Extract hands
        hand_result = self.hand_landmarker.detect_for_video(mp_image, self.frame_timestamp)
        left_hand = [(0, 0, 0)] * 21
        right_hand = [(0, 0, 0)] * 21
        
        if hand_result.hand_landmarks:
            for i, handedness in enumerate(hand_result.handedness):
                if i < len(hand_result.hand_landmarks):
                    hand_lms = [(lm.x, lm.y, lm.z) for lm in hand_result.hand_landmarks[i]]
                    if self.swap_hands:
                        if handedness[0].category_name == "Left":
                            right_hand = hand_lms
                        else:
                            left_hand = hand_lms
                    else:
                        if handedness[0].category_name == "Left":
                            left_hand = hand_lms
                        else:
                            right_hand = hand_lms
        
        # Combine all keypoints
        all_keypoints = pose_data + left_hand + right_hand
        return np.array(all_keypoints, dtype=np.float32).flatten()
    
    def close(self):
        """Release resources."""
        self.pose_landmarker.close()
        self.hand_landmarker.close()


class SignBuffer:
    """Buffer to collect pose frames for sign recognition."""
    
    def __init__(self, window_size: int = 64, stride: int = 16, min_active_frames: int = 12):
        """
        Initialize buffer.
        
        Args:
            window_size: Number of frames to collect before recognition
            stride: How often to trigger recognition (frames)
            min_active_frames: Minimum active frames required to trigger recognition
        """
        self.window_size = window_size
        self.stride = stride
        self.min_active_frames = min_active_frames
        self.buffer = deque(maxlen=window_size)
        self.activity_buffer = deque(maxlen=window_size)
        self.frame_count = 0
    
    def add_frame(self, pose: np.ndarray, is_active: bool = True) -> Optional[np.ndarray]:
        """
        Add a frame to buffer.
        
        Returns:
            Pose sequence if ready for recognition, None otherwise
        """
        self.buffer.append(pose)
        self.activity_buffer.append(1 if is_active else 0)
        self.frame_count += 1
        
        # Return sequence every `stride` frames once buffer is full
        if len(self.buffer) >= self.window_size and self.frame_count % self.stride == 0:
            if sum(self.activity_buffer) < self.min_active_frames:
                return None
            return np.array(list(self.buffer), dtype=np.float32)
        
        return None
    
    def get_current(self) -> Optional[np.ndarray]:
        """Get current buffer contents."""
        if len(self.buffer) >= self.window_size // 2:
            # Pad if needed
            poses = list(self.buffer)
            while len(poses) < self.window_size:
                poses.append(poses[-1] if poses else np.zeros(225, dtype=np.float32))
            return np.array(poses, dtype=np.float32)
        return None

    def clear(self):
        """Clear buffered frames and activity history."""
        self.buffer.clear()
        self.activity_buffer.clear()
        self.frame_count = 0


class RealtimeRecognizer:
    """Real-time sign language recognition system."""
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        ema_alpha: float = 0.65,
        class_stats_path: Optional[str] = None,
        abstain_threshold: float = 0.12,
        margin_threshold: float = 0.02,
        logit_adjustment_tau: float = 0.7,
        disable_logit_adjustment: bool = False,
    ):
        """
        Initialize recognizer.
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary JSON
            device: 'cuda' or 'cpu'
            confidence_threshold: Minimum confidence for predictions
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.ema_alpha = float(np.clip(ema_alpha, 0.0, 1.0))
        self.smoothed_probs = None
        self.abstain_threshold = abstain_threshold
        self.margin_threshold = margin_threshold
        self.logit_adjustment_tau = logit_adjustment_tau
        self.disable_logit_adjustment = disable_logit_adjustment
        self.class_priors: Optional[torch.Tensor] = None
        
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.gloss_to_idx = json.load(f)
        self.idx_to_gloss = {v: k for k, v in self.gloss_to_idx.items()}
        
        # Load model
        from train_recognition import SignRecognitionModel
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        self.model = SignRecognitionModel(
            input_dim=checkpoint.get('input_dim', 225),
            d_model=checkpoint.get('d_model', 256),
            nhead=checkpoint.get('nhead', 8),
            num_layers=checkpoint.get('num_layers', 4),
            dim_feedforward=checkpoint.get('dim_feedforward', checkpoint.get('d_model', 256) * 4),
            num_classes=len(self.gloss_to_idx),
            dropout=checkpoint.get('dropout', 0.1),
            max_frames=checkpoint.get('max_frames', 64)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        self.max_frames = checkpoint.get('max_frames', 64)
        self.input_dim = checkpoint.get('input_dim', 225)
        self.checkpoint_val_acc = checkpoint.get('val_acc')
        self._load_class_priors(checkpoint, class_stats_path)
        
        print(f"Loaded model with {len(self.gloss_to_idx)} classes")
        if self.checkpoint_val_acc is not None:
            print(f"Checkpoint val_acc: {self.checkpoint_val_acc:.4f}")
            if self.checkpoint_val_acc < 0.25:
                print("Warning: low validation accuracy; realtime gloss predictions may be unstable.")
        if self.class_priors is not None and not self.disable_logit_adjustment:
            print(f"Logit adjustment enabled (tau={self.logit_adjustment_tau:.2f})")

    def _load_class_priors(self, checkpoint: Dict, class_stats_path: Optional[str]):
        """Load class priors from checkpoint or class_stats file."""
        priors = checkpoint.get("class_priors")

        if priors is None and class_stats_path:
            stats_path = Path(class_stats_path)
            if stats_path.exists():
                try:
                    with open(stats_path, "r") as f:
                        stats = json.load(f)
                    p = np.zeros(len(self.gloss_to_idx), dtype=np.float32)
                    for gloss, idx in self.gloss_to_idx.items():
                        p[idx] = float(stats.get("class_priors", {}).get(gloss, 0.0))
                    priors = p.tolist()
                except Exception as e:
                    print(f"Warning: failed to load class stats from {class_stats_path}: {e}")

        if priors is None:
            return

        p = np.asarray(priors, dtype=np.float32)
        if p.size != len(self.gloss_to_idx):
            print("Warning: class priors size mismatch, ignoring priors")
            return

        p = np.clip(p, 1e-8, None)
        p = p / np.clip(p.sum(), 1e-8, None)
        self.class_priors = torch.tensor(p, dtype=torch.float32, device=self.device)

    def _apply_logit_adjustment(self, logits: torch.Tensor) -> torch.Tensor:
        if self.disable_logit_adjustment or self.class_priors is None:
            return logits
        return logits - self.logit_adjustment_tau * torch.log(self.class_priors.unsqueeze(0))

    def reset(self):
        """Reset temporal smoothing state."""
        self.smoothed_probs = None

    def _prepare_inputs(self, poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pad/truncate sequence and build validity mask."""
        if poses.ndim == 3:
            poses = poses.reshape(poses.shape[0], -1)
        if poses.ndim != 2:
            raise ValueError(f"Expected (frames, features), got {poses.shape}")
        if poses.shape[0] == 0:
            poses = np.zeros((1, self.input_dim), dtype=np.float32)
        if poses.shape[1] != self.input_dim:
            raise ValueError(f"Expected feature dim {self.input_dim}, got {poses.shape[1]}")

        valid_mask = np.any(np.abs(poses) > 1e-6, axis=1)
        poses = poses.astype(np.float32, copy=False)

        if poses.shape[0] > self.max_frames:
            poses = poses[-self.max_frames:]
            valid_mask = valid_mask[-self.max_frames:]
        elif poses.shape[0] < self.max_frames:
            pad_size = self.max_frames - poses.shape[0]
            poses = np.pad(poses, ((0, pad_size), (0, 0)), mode='constant')
            valid_mask = np.pad(valid_mask, (0, pad_size), mode='constant', constant_values=False)

        return poses, valid_mask
    
    @torch.no_grad()
    def recognize(
        self,
        poses: np.ndarray,
        top_k: int = 5,
        return_details: bool = False
    ):
        """
        Recognize sign from pose sequence.
        
        Args:
            poses: (frames, 225) pose sequence
            top_k: Number of predictions to return
            return_details: Return abstain/margin info
            
        Returns:
            If return_details=False: List[(gloss, confidence)]
            If return_details=True: Dict with top_k, predicted_gloss, abstain, margin
        """
        poses, valid_mask = self._prepare_inputs(poses)

        if not bool(np.any(valid_mask)):
            details = {
                "predicted_gloss": "NO_SIGN",
                "confidence": 1.0,
                "abstain": True,
                "reason": "empty_mask",
                "margin": 0.0,
                "top_k": [("NO_SIGN", 1.0)],
            }
            return details if return_details else details["top_k"]
        
        # Convert to tensor
        poses_tensor = torch.tensor(poses, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(valid_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
        
        # Forward pass
        logits = self.model(poses_tensor, mask_tensor)
        logits = self._apply_logit_adjustment(logits)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        # Exponential moving average smooths frame-to-frame jitter
        if self.smoothed_probs is None:
            self.smoothed_probs = probs
        else:
            self.smoothed_probs = self.ema_alpha * self.smoothed_probs + (1.0 - self.ema_alpha) * probs
        smoothed = self.smoothed_probs / np.clip(self.smoothed_probs.sum(), 1e-8, None)
        
        k = min(top_k, len(smoothed))
        top_indices = np.argsort(smoothed)[::-1][:k]
        
        results = []
        for idx in top_indices:
            gloss = self.idx_to_gloss.get(int(idx), "UNKNOWN")
            results.append((gloss, float(smoothed[idx])))

        top1_conf = results[0][1] if results else 0.0
        top2_conf = results[1][1] if len(results) > 1 else 0.0
        margin = top1_conf - top2_conf
        abstain = (top1_conf < self.abstain_threshold) or (margin < self.margin_threshold)
        predicted_gloss = "NO_SIGN" if abstain else (results[0][0] if results else "NO_SIGN")

        details = {
            "predicted_gloss": predicted_gloss,
            "confidence": top1_conf,
            "abstain": abstain,
            "reason": "threshold" if abstain else "accepted",
            "margin": margin,
            "top_k": results,
        }
        if return_details:
            return details
        if abstain:
            return [("NO_SIGN", top1_conf)] + results
        return results


class TextToSpeechOutput:
    """Async text-to-speech output with Gloss-to-Text conversion."""
    
    def __init__(self, use_gloss_to_text: bool = True):
        """Initialize TTS with optional Gloss-to-Text."""
        self.speech_queue = queue.Queue()
        self.tts = None
        self.gloss_to_text = None
        self.use_gloss_to_text = use_gloss_to_text
        self.last_spoken = ""
        self.last_spoken_time = 0
        self.cooldown = 2.0  # Seconds between repeating same word
        
        # Initialize Gloss-to-Text converter
        if use_gloss_to_text:
            try:
                from gloss_to_text import GlossToText
                self.gloss_to_text = GlossToText(mode="simple")  # Use simple mode for speed
                print("Gloss-to-Text initialized")
            except Exception as e:
                print(f"Gloss-to-Text init failed: {e}")
                self.gloss_to_text = None
        
        # Start TTS thread
        self.running = True
        self.thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.thread.start()
    
    def _init_tts(self):
        """Lazy-initialize TTS."""
        if self.tts is None:
            try:
                from TTS.api import TTS
                self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
                print("TTS initialized")
            except Exception as e:
                print(f"TTS initialization failed: {e}")
                self.tts = False  # Mark as failed
    
    def _tts_worker(self):
        """Background thread for TTS."""
        import tempfile
        import subprocess
        
        while self.running:
            try:
                text = self.speech_queue.get(timeout=1.0)
                
                self._init_tts()
                if self.tts and self.tts is not False:
                    # Generate speech
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        output_path = f.name
                    
                    self.tts.tts_to_file(text=text, file_path=output_path)
                    
                    # Play audio (Windows)
                    subprocess.run(
                        ["powershell", "-c", f"(New-Object Media.SoundPlayer '{output_path}').PlaySync()"],
                        capture_output=True
                    )
                    
                    os.remove(output_path)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"TTS error: {e}")
    
    def speak(self, gloss: str):
        """Queue gloss for speaking (converts to natural text first)."""
        current_time = time.time()
        
        # Avoid repeating same word too quickly
        if gloss == self.last_spoken and current_time - self.last_spoken_time < self.cooldown:
            return
        
        self.last_spoken = gloss
        self.last_spoken_time = current_time
        
        # Convert gloss to natural text
        if self.gloss_to_text:
            try:
                text = self.gloss_to_text.convert([gloss])
                if text:
                    self.speech_queue.put(text)
                    return
            except:
                pass
        
        # Fallback: speak gloss directly
        self.speech_queue.put(gloss.replace("_", " ").lower())
    
    def stop(self):
        """Stop TTS thread."""
        self.running = False


def draw_landmarks(frame: np.ndarray, pose_extractor: RealtimePoseExtractor, pose_data: np.ndarray = None) -> np.ndarray:
    """Draw pose landmarks on frame for visualization."""
    h, w = frame.shape[:2]
    
    if pose_data is not None and len(pose_data) > 0:
        # Reshape: (225,) -> (75, 3) = 33 pose + 21 left + 21 right
        try:
            keypoints = pose_data.reshape(75, 3)
            
            # Body pose (indices 0-32)
            pose_kps = keypoints[:33]
            # Left hand (indices 33-53)
            left_hand_kps = keypoints[33:54]
            # Right hand (indices 54-74)
            right_hand_kps = keypoints[54:75]
            
            # Draw hands with connections (like the reference image)
            # Hand connections (MediaPipe hand landmark indices)
            hand_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),      # Index
                (0, 9), (9, 10), (10, 11), (11, 12), # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
                (5, 9), (9, 13), (13, 17)            # Palm
            ]
            
            # Draw left hand (green lines, red dots)
            _draw_hand(frame, left_hand_kps, hand_connections, w, h, (0, 255, 0), (0, 0, 255))
            
            # Draw right hand (green lines, red dots)
            _draw_hand(frame, right_hand_kps, hand_connections, w, h, (0, 255, 0), (0, 0, 255))
            
            # Draw body pose (optional - just shoulders and arms)
            body_connections = [
                (11, 12),  # Shoulders
                (11, 13), (13, 15),  # Left arm
                (12, 14), (14, 16),  # Right arm
            ]
            for i, j in body_connections:
                if i < len(pose_kps) and j < len(pose_kps):
                    x1, y1 = int(pose_kps[i][0] * w), int(pose_kps[i][1] * h)
                    x2, y2 = int(pose_kps[j][0] * w), int(pose_kps[j][1] * h)
                    if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
        except Exception as e:
            pass
    
    return frame


def _draw_hand(frame, hand_kps, connections, w, h, line_color, point_color):
    """Draw hand landmarks with connections."""
    # Check if hand is detected (not all zeros)
    if np.sum(np.abs(hand_kps)) < 0.01:
        return
    
    # Draw connections (green lines)
    for i, j in connections:
        if i < len(hand_kps) and j < len(hand_kps):
            x1, y1 = int(hand_kps[i][0] * w), int(hand_kps[i][1] * h)
            x2, y2 = int(hand_kps[j][0] * w), int(hand_kps[j][1] * h)
            if 0 < x1 < w and 0 < y1 < h and 0 < x2 < w and 0 < y2 < h:
                cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
    
    # Draw keypoints (red dots)
    for kp in hand_kps:
        x, y = int(kp[0] * w), int(kp[1] * h)
        if 0 < x < w and 0 < y < h:
            cv2.circle(frame, (x, y), 5, point_color, -1)


def _mirror_pose_for_display(pose_data: np.ndarray) -> np.ndarray:
    """Mirror x coordinates for drawing on mirrored preview."""
    if pose_data is None or len(pose_data) != 225:
        return pose_data
    keypoints = pose_data.reshape(75, 3).copy()
    keypoints[:, 0] = 1.0 - keypoints[:, 0]
    return keypoints.reshape(-1)


def _hand_activity(pose_data: np.ndarray, prev_pose_data: Optional[np.ndarray]) -> Tuple[bool, bool, float]:
    """Return left/right hand presence and mean hand motion."""
    if pose_data is None or len(pose_data) != 225:
        return False, False, 0.0

    keypoints = pose_data.reshape(75, 3)
    left_hand = keypoints[33:54]
    right_hand = keypoints[54:75]

    left_present = np.count_nonzero(np.linalg.norm(left_hand, axis=1) > 1e-4) >= 5
    right_present = np.count_nonzero(np.linalg.norm(right_hand, axis=1) > 1e-4) >= 5

    if prev_pose_data is None or len(prev_pose_data) != 225:
        return left_present, right_present, 0.0

    prev_keypoints = prev_pose_data.reshape(75, 3)
    delta = keypoints[33:75, :2] - prev_keypoints[33:75, :2]
    motion = float(np.mean(np.linalg.norm(delta, axis=1)))
    return left_present, right_present, motion


def main():
    parser = argparse.ArgumentParser(description="Real-time BSL recognition")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--vocab", type=str, default=None, help="Vocabulary JSON path")
    parser.add_argument("--class-stats", type=str, default=None, help="Class stats JSON path")
    parser.add_argument("--camera", type=int, default=None, help="Input camera index")
    parser.add_argument("--video", type=str, default=None, help="Input video file path")
    parser.add_argument("--no-speech", action="store_true", help="Disable TTS output")
    parser.add_argument("--recognition-only", action="store_true",
                       help="Input-source-to-gloss only (disable gloss-to-text and speech)")
    parser.add_argument("--window-size", type=int, default=48, help="Recognition window size")
    parser.add_argument("--stride", type=int, default=12, help="Recognition stride")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--top-k", type=int, default=5, help="Number of predictions to display")
    parser.add_argument("--ema-alpha", type=float, default=0.65,
                       help="Temporal smoothing factor for probabilities (0-1)")
    parser.add_argument("--abstain-threshold", type=float, default=0.12,
                       help="Abstain if top1 confidence is below this threshold")
    parser.add_argument("--margin-threshold", type=float, default=0.02,
                       help="Abstain if top1-top2 confidence margin is below this value")
    parser.add_argument("--logit-adjustment-tau", type=float, default=0.7,
                       help="Logit adjustment strength using class priors")
    parser.add_argument("--disable-logit-adjustment", action="store_true",
                       help="Disable prior-based logit adjustment")
    parser.add_argument("--min-active-frames", type=int, default=12,
                       help="Minimum active frames required in each recognition window")
    parser.add_argument("--motion-threshold", type=float, default=0.002,
                       help="Minimum mean hand motion for an active frame")
    parser.add_argument("--mirror-view", action="store_true",
                       help="Mirror preview window (recognition uses unmirrored frame)")
    parser.add_argument("--swap-hands", action="store_true",
                       help="Swap MediaPipe handedness labels (legacy mode)")
    parser.add_argument("--no-swap-hands", action="store_true",
                       help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.recognition_only:
        args.no_speech = True
    if args.camera is not None and args.video:
        parser.error("Use either --camera or --video, not both.")
    if args.camera is None and not args.video:
        args.camera = 0

    source_type = "camera"
    source_label = f"Camera #{args.camera}"
    video_path = None
    if args.video:
        video_path = Path(args.video).expanduser()
        if not video_path.exists() or not video_path.is_file():
            parser.error(f"Video file not found or is not a file: {video_path}")
        source_type = "video"
        source_label = f"Video: {video_path}"

    # Direct handedness mapping is now default.
    swap_hands = bool(args.swap_hands)
    # Backward-compatible alias: force direct mapping when specified.
    if args.no_swap_hands:
        swap_hands = False

    # Default paths
    if args.model is None:
        args.model = project_root / "models" / "sign_recognition" / "best_model.pt"
    if args.vocab is None:
        args.vocab = project_root / "models" / "sign_recognition" / "vocabulary.json"
    if args.class_stats is None:
        args.class_stats = project_root / "models" / "sign_recognition" / "class_stats.json"

    # Check model exists
    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        print("Train the model first: python scripts/train_recognition.py")
        return

    # Initialize components
    print("Initializing pose extractor...")
    pose_extractor = RealtimePoseExtractor(swap_hands=swap_hands)
    print(f"Handedness mapping: {'SWAPPED' if swap_hands else 'DIRECT'}")

    print("Loading recognition model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    recognizer = RealtimeRecognizer(
        str(args.model),
        str(args.vocab),
        device=device,
        confidence_threshold=args.threshold,
        ema_alpha=args.ema_alpha,
        class_stats_path=str(args.class_stats) if Path(args.class_stats).exists() else None,
        abstain_threshold=args.abstain_threshold,
        margin_threshold=args.margin_threshold,
        logit_adjustment_tau=args.logit_adjustment_tau,
        disable_logit_adjustment=args.disable_logit_adjustment,
    )

    # Initialize TTS
    tts_output = None
    if not args.no_speech:
        print("Initializing TTS with Gloss-to-Text...")
        tts_output = TextToSpeechOutput(use_gloss_to_text=not args.recognition_only)

    # Initialize sign buffer
    sign_buffer = SignBuffer(
        window_size=args.window_size,
        stride=args.stride,
        min_active_frames=args.min_active_frames
    )

    # Open input source
    video_delay_ms = 1
    if source_type == "video":
        print(f"Opening video file {video_path}...")
        cap = cv2.VideoCapture(str(video_path))
    else:
        print(f"Opening camera {args.camera}...")
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        if source_type == "video":
            print(f"Failed to open video file: {video_path}")
        else:
            print(f"Failed to open camera {args.camera}")
        return

    if source_type == "video":
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        video_delay_ms = max(1, int(round(1000.0 / source_fps)))
        print(f"Video FPS: {source_fps:.2f} (frame delay: {video_delay_ms} ms)")

    print("\n" + "=" * 50)
    print("BSL REAL-TIME RECOGNITION")
    print("=" * 50)
    print(f"Source: {source_label}")
    if args.recognition_only:
        print("Pipeline: Input Source -> Pose -> Gloss")
    else:
        print("Pipeline: Input Source -> Pose -> Gloss -> Text -> Speech")
    print(f"\nModel knows {len(recognizer.idx_to_gloss)} signs")
    print("\nControls:")
    print("  'q' - Quit")
    print("  's' - Toggle speech output")
    print("  'c' - Clear history")
    print("  '1-5' - Select from top predictions")
    print("=" * 50 + "\n")

    # Recognition state
    current_prediction = ""
    current_text = ""
    prediction_history = deque(maxlen=5)
    gloss_history = deque(maxlen=10)
    last_predictions = []
    speech_enabled = not args.no_speech
    prev_pose = None
    left_detected = False
    right_detected = False
    motion_score = 0.0

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                if source_type == "video":
                    print("Reached end of video file")
                else:
                    print("Failed to read frame from camera")
                break

            # Extract pose from unmirrored frame for consistency with training data.
            pose = pose_extractor.extract(raw_frame)
            left_detected, right_detected, motion_score = _hand_activity(pose, prev_pose)
            has_hands = left_detected or right_detected
            is_active_frame = has_hands and (prev_pose is None or motion_score >= args.motion_threshold)
            prev_pose = pose.copy()

            # Prepare display frame (optional mirrored preview).
            if args.mirror_view:
                frame = cv2.flip(raw_frame, 1)
                draw_pose = _mirror_pose_for_display(pose)
            else:
                frame = raw_frame.copy()
                draw_pose = pose

            frame = draw_landmarks(frame, pose_extractor, draw_pose)

            # Add to buffer and check for recognition
            pose_sequence = sign_buffer.add_frame(pose, is_active=is_active_frame)
            if pose_sequence is not None:
                rec = recognizer.recognize(pose_sequence, top_k=args.top_k, return_details=True)
                last_predictions = rec["top_k"]

                if rec["abstain"]:
                    current_prediction = "NO_SIGN"
                elif rec["top_k"]:
                    gloss, confidence = rec["top_k"][0]
                    prediction_history.append(gloss)

                    # Require consistency and confidence before accepting prediction.
                    if prediction_history.count(gloss) >= 3 and confidence >= args.threshold:
                        if gloss != current_prediction:
                            current_prediction = gloss
                            gloss_history.append(gloss)

                            if tts_output and tts_output.gloss_to_text:
                                try:
                                    current_text = tts_output.gloss_to_text.convert(list(gloss_history)[-5:])
                                except Exception:
                                    current_text = " ".join(list(gloss_history)[-5:]).replace("_", " ")
                            else:
                                current_text = " ".join(list(gloss_history)[-5:]).replace("_", " ")

                            print(f"Recognized: {gloss} ({confidence:.2f}) -> \"{current_text}\"")
                            if speech_enabled and tts_output:
                                tts_output.speak(gloss)

            if current_prediction:
                gloss_color = (0, 255, 0) if current_prediction != "NO_SIGN" else (0, 180, 255)
                cv2.putText(frame, f"Gloss: {current_prediction}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, gloss_color, 2)

            if current_text:
                cv2.putText(frame, f"Text: {current_text[:50]}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            history_str = " -> ".join(list(gloss_history)[-5:])
            cv2.putText(frame, f"History: {history_str[:60]}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            status = f"Speech: {'ON' if speech_enabled else 'OFF'} | Press 'c' to clear"
            cv2.putText(frame, status, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            hand_status = f"Hands: L={'Y' if left_detected else 'N'} R={'Y' if right_detected else 'N'} Motion:{motion_score:.4f}"
            color = (0, 255, 0) if has_hands else (0, 0, 255)
            cv2.putText(frame, hand_status, (10, frame.shape[0] - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if last_predictions:
                top_str = " | ".join([f"[{i + 1}]{g}:{c:.2f}" for i, (g, c) in enumerate(last_predictions[:5])])
                cv2.putText(frame, f"Top: {top_str[:90]}", (10, frame.shape[0] - 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            cv2.imshow("BSL Recognition", frame)

            key = cv2.waitKey(video_delay_ms if source_type == "video" else 1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if tts_output:
                    speech_enabled = not speech_enabled
                    print(f"Speech {'enabled' if speech_enabled else 'disabled'}")
                else:
                    print("Speech is disabled in current mode")
            elif key == ord('c'):
                gloss_history.clear()
                prediction_history.clear()
                sign_buffer.clear()
                recognizer.reset()
                current_prediction = ""
                current_text = ""
                prev_pose = None
                print("History cleared")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                idx = key - ord('1')
                if last_predictions and idx < len(last_predictions):
                    selected_gloss, conf = last_predictions[idx]
                    current_prediction = selected_gloss
                    gloss_history.append(selected_gloss)

                    if tts_output and tts_output.gloss_to_text:
                        try:
                            current_text = tts_output.gloss_to_text.convert(list(gloss_history)[-5:])
                        except Exception:
                            current_text = " ".join(list(gloss_history)[-5:]).replace("_", " ")
                    else:
                        current_text = " ".join(list(gloss_history)[-5:]).replace("_", " ")

                    print(f"Selected: {selected_gloss} -> \"{current_text}\"")
                    if speech_enabled and tts_output:
                        tts_output.speak(selected_gloss)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_extractor.close()
        if tts_output:
            tts_output.stop()
        print("\nShutdown complete")

if __name__ == "__main__":
    main()

