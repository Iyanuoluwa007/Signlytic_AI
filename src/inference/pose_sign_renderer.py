"""
2D pose-based signer renderer for Direction 2 Speech/Text -> BSL output.

This module builds a cached gloss-to-pose index from extracted pose JSON files
and renders skeleton animations as RGB frames and MP4 video files.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from statistics import median
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class PoseSignRenderer:
    """Render BSL gloss sequences as 2D skeleton animation."""

    POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (11, 23),
        (12, 24),
        (23, 24),
    )

    HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),
        (0, 17),
    )

    def __init__(
        self,
        project_root: Optional[Path] = None,
        pose_roots: Optional[Sequence[Path]] = None,
        cache_path: Optional[Path] = None,
        canvas_width: int = 640,
        canvas_height: int = 480,
        output_fps: int = 20,
        base_gloss_duration: float = 0.9,
        ema_alpha: float = 0.6,
    ) -> None:
        self.project_root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        self.pose_roots = [Path(p) for p in pose_roots] if pose_roots else [
            self.project_root / "data" / "poses",
            self.project_root / "data" / "poses_bsldict",
        ]
        self.cache_path = Path(cache_path) if cache_path else self.project_root / "data" / "processed" / "pose_sign_index.json"
        self.canvas_width = int(canvas_width)
        self.canvas_height = int(canvas_height)
        self.output_fps = int(output_fps)
        self.base_gloss_duration = float(base_gloss_duration)
        self.ema_alpha = float(ema_alpha)

        self.gloss_index: Dict[str, Dict[str, float]] = {}
        self._pose_seq_cache: Dict[str, List[Dict[str, np.ndarray]]] = {}

        self.build_index(force_rebuild=False)

    def build_index(self, force_rebuild: bool = False) -> Dict[str, Dict[str, float]]:
        """
        Build and cache canonical pose sample per gloss.

        Selection rule:
        1) Highest valid-hand-frame ratio
        2) Duration closest to median for that gloss
        3) Lexicographically smallest file path (deterministic tie-break)
        """
        if not force_rebuild and self.cache_path.exists():
            if self._load_index_from_cache():
                return self.gloss_index

        candidates_by_gloss: Dict[str, List[Dict[str, float]]] = {}
        scanned_files = 0
        split_names = ("train", "val", "test")

        for root in self.pose_roots:
            if not root.exists():
                continue

            for split in split_names:
                split_dir = root / split
                if not split_dir.exists():
                    continue

                for json_path in split_dir.glob("*.json"):
                    scanned_files += 1
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            sample = json.load(f)
                    except Exception:
                        continue

                    gloss_raw = str(sample.get("gloss", "")).strip()
                    if not gloss_raw:
                        continue
                    gloss = gloss_raw.upper()

                    poses = sample.get("poses", [])
                    if not isinstance(poses, list) or len(poses) == 0:
                        continue

                    num_frames = int(sample.get("num_frames") or len(poses))
                    duration = self._estimate_duration_seconds(sample, num_frames)
                    valid_ratio = self._compute_valid_hand_ratio(poses)

                    candidates_by_gloss.setdefault(gloss, []).append(
                        {
                            "path": str(json_path.resolve()),
                            "valid_ratio": float(valid_ratio),
                            "duration": float(duration),
                            "num_frames": float(num_frames),
                        }
                    )

        selected: Dict[str, Dict[str, float]] = {}
        for gloss, candidates in candidates_by_gloss.items():
            durations = [c["duration"] for c in candidates]
            duration_median = float(median(durations)) if durations else 0.0
            best = sorted(
                candidates,
                key=lambda c: (-c["valid_ratio"], abs(c["duration"] - duration_median), c["path"]),
            )[0]
            selected[gloss] = {
                "path": best["path"],
                "valid_ratio": float(best["valid_ratio"]),
                "duration": float(best["duration"]),
                "num_frames": int(best["num_frames"]),
                "candidate_count": len(candidates),
            }

        self.gloss_index = selected
        self._save_index_cache(scanned_files=scanned_files, gloss_count=len(selected))
        return self.gloss_index

    def get_coverage(self, glosses: Sequence[str]) -> Dict[str, object]:
        """Return pose coverage stats for a gloss sequence."""
        normalized = [str(g).strip().upper() for g in glosses if str(g).strip()]
        if not normalized:
            return {
                "coverage": 0.0,
                "available": [],
                "missing": [],
                "available_count": 0,
                "missing_count": 0,
            }

        available = [g for g in normalized if g in self.gloss_index]
        missing = [g for g in normalized if g not in self.gloss_index]
        coverage = 100.0 * len(available) / len(normalized)
        return {
            "coverage": coverage,
            "available": available,
            "missing": missing,
            "available_count": len(available),
            "missing_count": len(missing),
        }

    def render_sequence_frames(
        self,
        glosses: Sequence[str],
        speed: float = 1.0,
        max_total_seconds: float = 60.0,
    ) -> Iterator[Tuple[np.ndarray, str]]:
        """
        Yield rendered RGB frames and status messages for a gloss sequence.

        Speed model:
            per_gloss_duration = base_gloss_duration / speed
        """
        normalized_glosses = [str(g).strip().upper() for g in glosses if str(g).strip()]
        if not normalized_glosses:
            neutral = self._neutral_frame()
            status = "No glosses to render."
            yield self._draw_frame(
                landmarks=neutral,
                gloss="NO_GLOSS",
                missing=True,
                gloss_idx=1,
                gloss_total=1,
                frame_idx=1,
                frame_total=1,
                status=status,
            ), status
            return

        speed = float(np.clip(speed, 0.6, 1.6))
        per_gloss_seconds = self.base_gloss_duration / speed
        max_glosses = max(1, int(max_total_seconds / per_gloss_seconds))
        truncated = len(normalized_glosses) > max_glosses
        glosses_used = normalized_glosses[:max_glosses]

        frames_per_gloss = max(1, int(round(per_gloss_seconds * self.output_fps)))
        total_frames = frames_per_gloss * len(glosses_used)

        prev_smoothed: Optional[Dict[str, np.ndarray]] = None
        frame_counter = 0

        for gloss_i, gloss in enumerate(glosses_used, start=1):
            sampled_frames, missing = self._get_gloss_frames(gloss, frames_per_gloss)
            for local_i, landmarks in enumerate(sampled_frames, start=1):
                smoothed = self._smooth_landmarks(prev_smoothed, landmarks)
                prev_smoothed = smoothed
                frame_counter += 1

                status = f"Rendering {gloss_i}/{len(glosses_used)}: {gloss}"
                if missing:
                    status += " (MISSING)"
                if truncated and gloss_i == len(glosses_used) and local_i == len(sampled_frames):
                    status += " | Sequence truncated for stability."

                frame_rgb = self._draw_frame(
                    landmarks=smoothed,
                    gloss=gloss,
                    missing=missing,
                    gloss_idx=gloss_i,
                    gloss_total=len(glosses_used),
                    frame_idx=frame_counter,
                    frame_total=total_frames,
                    status=status,
                )
                yield frame_rgb, status

    def render_sequence_video(
        self,
        glosses: Sequence[str],
        output_path: Optional[str] = None,
        speed: float = 1.0,
        max_total_seconds: float = 60.0,
    ) -> str:
        """Render a gloss sequence directly to MP4."""
        target = output_path or tempfile.mktemp(suffix=".mp4")
        writer = cv2.VideoWriter(
            str(target),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.output_fps,
            (self.canvas_width, self.canvas_height),
        )

        if not writer.isOpened():
            raise RuntimeError("Failed to initialize MP4 writer for pose animation.")

        frames_written = 0
        try:
            for frame_rgb, _ in self.render_sequence_frames(
                glosses=glosses,
                speed=speed,
                max_total_seconds=max_total_seconds,
            ):
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                frames_written += 1
        finally:
            writer.release()

        if frames_written == 0:
            raise RuntimeError("No frames rendered for output video.")
        return str(target)

    def _load_index_from_cache(self) -> bool:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            cached = payload.get("glosses", {})
            if not isinstance(cached, dict) or not cached:
                return False
            # Validate that selected files still exist; otherwise rebuild.
            for meta in cached.values():
                path = Path(str(meta.get("path", "")))
                if not path.exists():
                    return False
            self.gloss_index = cached
            return True
        except Exception:
            return False

    def _save_index_cache(self, scanned_files: int, gloss_count: int) -> None:
        payload = {
            "version": 1,
            "scanned_files": int(scanned_files),
            "gloss_count": int(gloss_count),
            "glosses": self.gloss_index,
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _estimate_duration_seconds(self, sample: Dict, num_frames: int) -> float:
        start_time = sample.get("start_time")
        end_time = sample.get("end_time")
        try:
            start_val = float(start_time)
            end_val = float(end_time)
            if end_val > start_val:
                return float(end_val - start_val)
        except Exception:
            pass
        return float(max(1, num_frames) / max(1, self.output_fps))

    def _compute_valid_hand_ratio(self, poses: List[Dict]) -> float:
        valid = 0
        total = 0
        for frame in poses:
            if not isinstance(frame, dict):
                continue
            total += 1
            left = frame.get("left_hand")
            right = frame.get("right_hand")
            if self._has_nonzero_xy(left) or self._has_nonzero_xy(right):
                valid += 1
        if total == 0:
            return 0.0
        return float(valid / total)

    def _get_gloss_frames(self, gloss: str, target_frames: int) -> Tuple[List[Dict[str, np.ndarray]], bool]:
        meta = self.gloss_index.get(gloss)
        if not meta:
            neutral = self._neutral_frame()
            return [neutral for _ in range(target_frames)], True

        pose_path = str(meta.get("path", "")).strip()
        if not pose_path:
            neutral = self._neutral_frame()
            return [neutral for _ in range(target_frames)], True

        source = self._load_pose_sequence(pose_path)
        if not source:
            neutral = self._neutral_frame()
            return [neutral for _ in range(target_frames)], True

        if len(source) == target_frames:
            return source, False

        indices = np.linspace(0, len(source) - 1, target_frames).astype(int)
        sampled = [source[i] for i in indices]
        return sampled, False

    def _load_pose_sequence(self, pose_path: str) -> List[Dict[str, np.ndarray]]:
        if pose_path in self._pose_seq_cache:
            return self._pose_seq_cache[pose_path]

        try:
            with open(pose_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            pose_frames = payload.get("poses", [])
            sequence = [self._to_landmark_frame(frame) for frame in pose_frames if isinstance(frame, dict)]
            if not sequence:
                sequence = [self._neutral_frame()]
            self._pose_seq_cache[pose_path] = sequence
            # Cap cache size to avoid unbounded memory growth.
            if len(self._pose_seq_cache) > 256:
                oldest = next(iter(self._pose_seq_cache.keys()))
                del self._pose_seq_cache[oldest]
            return sequence
        except Exception:
            return [self._neutral_frame()]

    def _to_landmark_frame(self, frame: Dict) -> Dict[str, np.ndarray]:
        return {
            "pose": self._to_point_array(frame.get("pose"), expected=33),
            "left_hand": self._to_point_array(frame.get("left_hand"), expected=21),
            "right_hand": self._to_point_array(frame.get("right_hand"), expected=21),
        }

    def _to_point_array(self, points: object, expected: int) -> np.ndarray:
        if not points:
            return np.zeros((expected, 3), dtype=np.float32)

        try:
            arr = np.asarray(points, dtype=np.float32)
        except Exception:
            return np.zeros((expected, 3), dtype=np.float32)

        if arr.ndim != 2:
            if arr.size % 3 != 0:
                return np.zeros((expected, 3), dtype=np.float32)
            arr = arr.reshape(-1, 3)

        if arr.shape[1] < 3:
            arr = np.pad(arr, ((0, 0), (0, 3 - arr.shape[1])), mode="constant")
        arr = arr[:, :3]

        if arr.shape[0] < expected:
            arr = np.pad(arr, ((0, expected - arr.shape[0]), (0, 0)), mode="constant")
        elif arr.shape[0] > expected:
            arr = arr[:expected]

        return arr.astype(np.float32, copy=False)

    def _neutral_frame(self) -> Dict[str, np.ndarray]:
        pose = np.zeros((33, 3), dtype=np.float32)
        pose[0, :2] = (0.50, 0.18)   # nose
        pose[11, :2] = (0.42, 0.34)  # left shoulder
        pose[12, :2] = (0.58, 0.34)  # right shoulder
        pose[13, :2] = (0.37, 0.48)  # left elbow
        pose[14, :2] = (0.63, 0.48)  # right elbow
        pose[15, :2] = (0.34, 0.62)  # left wrist
        pose[16, :2] = (0.66, 0.62)  # right wrist
        pose[23, :2] = (0.45, 0.62)  # left hip
        pose[24, :2] = (0.55, 0.62)  # right hip

        left_hand = self._synthetic_hand(cx=pose[15, 0], cy=pose[15, 1], mirrored=False)
        right_hand = self._synthetic_hand(cx=pose[16, 0], cy=pose[16, 1], mirrored=True)

        return {
            "pose": pose,
            "left_hand": left_hand,
            "right_hand": right_hand,
        }

    def _synthetic_hand(self, cx: float, cy: float, mirrored: bool) -> np.ndarray:
        hand = np.zeros((21, 3), dtype=np.float32)
        x_dir = -1.0 if mirrored else 1.0
        hand[0, :2] = (cx, cy)

        # Simple neutral open hand geometry.
        finger_bases = [
            (0.018, -0.006),  # thumb
            (0.016, -0.020),  # index
            (0.006, -0.024),  # middle
            (-0.006, -0.020),  # ring
            (-0.016, -0.014),  # pinky
        ]
        lengths = [
            (0.010, 0.010, 0.008, 0.008),
            (0.012, 0.012, 0.010, 0.010),
            (0.014, 0.014, 0.012, 0.010),
            (0.012, 0.012, 0.010, 0.010),
            (0.010, 0.010, 0.008, 0.008),
        ]

        idx = 1
        for (bx, by), (l1, l2, l3, l4) in zip(finger_bases, lengths):
            x0 = cx + x_dir * bx
            y0 = cy + by
            hand[idx, :2] = (x0, y0)
            hand[idx + 1, :2] = (x0 + x_dir * 0.004, y0 - l1)
            hand[idx + 2, :2] = (x0 + x_dir * 0.006, y0 - l1 - l2)
            hand[idx + 3, :2] = (x0 + x_dir * 0.008, y0 - l1 - l2 - l3 - l4)
            idx += 4

        return hand

    def _smooth_landmarks(
        self,
        prev: Optional[Dict[str, np.ndarray]],
        current: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        if prev is None:
            return {
                "pose": current["pose"].copy(),
                "left_hand": current["left_hand"].copy(),
                "right_hand": current["right_hand"].copy(),
            }

        a = self.ema_alpha
        b = 1.0 - a
        return {
            "pose": a * current["pose"] + b * prev["pose"],
            "left_hand": a * current["left_hand"] + b * prev["left_hand"],
            "right_hand": a * current["right_hand"] + b * prev["right_hand"],
        }

    def _draw_frame(
        self,
        landmarks: Dict[str, np.ndarray],
        gloss: str,
        missing: bool,
        gloss_idx: int,
        gloss_total: int,
        frame_idx: int,
        frame_total: int,
        status: str,
    ) -> np.ndarray:
        canvas = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.uint8)
        canvas[:] = (18, 18, 24)

        pose_color = (250, 200, 60)
        left_color = (80, 220, 255)
        right_color = (120, 255, 120)

        self._draw_connections(canvas, landmarks["pose"], self.POSE_CONNECTIONS, pose_color, thickness=2)
        self._draw_points(canvas, landmarks["pose"], pose_color, radius=3)

        self._draw_connections(canvas, landmarks["left_hand"], self.HAND_CONNECTIONS, left_color, thickness=2)
        self._draw_points(canvas, landmarks["left_hand"], left_color, radius=2)

        self._draw_connections(canvas, landmarks["right_hand"], self.HAND_CONNECTIONS, right_color, thickness=2)
        self._draw_points(canvas, landmarks["right_hand"], right_color, radius=2)

        gloss_label = f"{gloss_idx}/{gloss_total}  {gloss}"
        cv2.putText(canvas, gloss_label[:80], (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2)

        if missing:
            cv2.putText(
                canvas,
                f"MISSING: {gloss}"[:80],
                (12, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (60, 160, 255),
                2,
            )

        cv2.putText(
            canvas,
            f"Frame {frame_idx}/{frame_total}",
            (12, self.canvas_height - 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (210, 210, 210),
            1,
        )
        cv2.putText(
            canvas,
            status[:100],
            (12, self.canvas_height - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )

        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def _draw_connections(
        self,
        image: np.ndarray,
        points: np.ndarray,
        connections: Sequence[Tuple[int, int]],
        color: Tuple[int, int, int],
        thickness: int = 1,
    ) -> None:
        for start_idx, end_idx in connections:
            if start_idx >= len(points) or end_idx >= len(points):
                continue
            p1 = points[start_idx]
            p2 = points[end_idx]
            if not (self._is_valid_point(p1) and self._is_valid_point(p2)):
                continue
            x1, y1 = self._to_pixel(p1[0], p1[1])
            x2, y2 = self._to_pixel(p2[0], p2[1])
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)

    def _draw_points(
        self,
        image: np.ndarray,
        points: np.ndarray,
        color: Tuple[int, int, int],
        radius: int = 2,
    ) -> None:
        for point in points:
            if not self._is_valid_point(point):
                continue
            x, y = self._to_pixel(point[0], point[1])
            cv2.circle(image, (x, y), radius, color, -1)

    def _to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        px = int(np.clip(float(x), 0.0, 1.0) * (self.canvas_width - 1))
        py = int(np.clip(float(y), 0.0, 1.0) * (self.canvas_height - 1))
        return px, py

    @staticmethod
    def _is_valid_point(point: np.ndarray) -> bool:
        if point is None or len(point) < 2:
            return False
        x = float(point[0])
        y = float(point[1])
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        return (abs(x) + abs(y)) > 1e-6

    @staticmethod
    def _has_nonzero_xy(hand_points: object) -> bool:
        if not hand_points:
            return False
        try:
            arr = np.asarray(hand_points, dtype=np.float32)
        except Exception:
            return False
        if arr.size == 0:
            return False
        if arr.ndim != 2:
            if arr.size % 3 != 0:
                return False
            arr = arr.reshape(-1, 3)
        if arr.shape[1] < 2:
            return False
        xy = arr[:, :2]
        return bool(np.any(np.abs(xy) > 1e-4))
