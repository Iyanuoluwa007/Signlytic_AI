"""
Leap Motion BSL Hand Data Loader

Loads the BSL Leap Motion CSV dataset for auxiliary hand supervision.
The Leap Motion data provides high-quality 3D hand joint positions and
orientations for BSL signs, which we use as a secondary supervision signal
to improve handshape accuracy in the recognition model.

The CSV contains per-frame hand joint data with columns for:
  - Hand type (left/right)
  - Palm position/orientation
  - 5 fingers x ~4 joints each = ~20 joint positions
  - Grab/pinch strength

We convert this into normalized hand feature vectors that can be
aligned with SWIN temporal features by gloss label for auxiliary training.

Usage:
    loader = LeapMotionLoader("data/Leap_Motion/BSL-leap-motion.csv")
    hand_data = loader.get_hand_features("HELLO")
    # Returns dict with left_hand, right_hand arrays
"""

import csv
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# Expected columns in the Leap Motion CSV (may vary by dataset version)
# We detect columns dynamically but expect these patterns
PALM_COLS = ["palm_position_x", "palm_position_y", "palm_position_z",
             "palm_normal_x", "palm_normal_y", "palm_normal_z"]

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
JOINT_NAMES = ["metacarpal", "proximal", "intermediate", "distal", "tip"]


class LeapMotionLoader:
    """
    Loads and processes BSL Leap Motion hand tracking data.

    The Leap Motion sensor captures precise 3D hand joint positions,
    which provide ground-truth hand shape information for BSL signs.
    """

    def __init__(self, csv_path: str, normalize: bool = True):
        """
        Args:
            csv_path: Path to BSL-leap-motion.csv
            normalize: If True, normalize joint positions to palm-relative coords
        """
        self.csv_path = Path(csv_path)
        self.normalize = normalize

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Leap Motion CSV not found: {self.csv_path}")

        self._raw_data: List[dict] = []
        self._gloss_hands: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._columns: List[str] = []
        self._joint_columns: List[str] = []

        self._load_csv()
        self._extract_features()

    # ---- Public API ----

    @property
    def num_samples(self) -> int:
        return len(self._raw_data)

    @property
    def num_glosses(self) -> int:
        return len(self._gloss_hands)

    @property
    def gloss_list(self) -> List[str]:
        return sorted(self._gloss_hands.keys())

    @property
    def hand_feature_dim(self) -> int:
        """Dimension of the hand feature vector."""
        # 21 joints x 3 coords = 63 per hand, or detected dynamically
        if self._gloss_hands:
            first_key = next(iter(self._gloss_hands))
            if self._gloss_hands[first_key]:
                return self._gloss_hands[first_key][0].shape[-1]
        return 63  # Default: 21 joints x 3

    def get_hand_features(self, gloss: str) -> Optional[np.ndarray]:
        """
        Get hand feature vectors for a gloss.

        Args:
            gloss: BSL gloss (uppercase)

        Returns:
            Array of shape (N, hand_feature_dim) where N = number of samples,
            or None if gloss not found
        """
        gloss = gloss.upper().strip()
        if gloss not in self._gloss_hands:
            return None
        return np.array(self._gloss_hands[gloss], dtype=np.float32)

    def get_mean_hand_template(self, gloss: str) -> Optional[np.ndarray]:
        """
        Get the mean hand shape template for a gloss.
        Useful as a soft target for auxiliary hand loss.

        Returns:
            Array of shape (hand_feature_dim,) or None
        """
        features = self.get_hand_features(gloss)
        if features is None or len(features) == 0:
            return None
        return features.mean(axis=0)

    def get_all_templates(self) -> Dict[str, np.ndarray]:
        """
        Get mean hand templates for all glosses.

        Returns:
            Dict mapping gloss -> mean hand feature vector
        """
        templates = {}
        for gloss in self._gloss_hands:
            tpl = self.get_mean_hand_template(gloss)
            if tpl is not None:
                templates[gloss] = tpl
        return templates

    def build_hand_target_matrix(
        self, gloss_vocab: Dict[str, int]
    ) -> np.ndarray:
        """
        Build a matrix of hand templates indexed by gloss label.

        Args:
            gloss_vocab: Mapping of gloss -> integer label

        Returns:
            Array of shape (num_classes, hand_feature_dim).
            Rows without Leap Motion data are zero-filled.
        """
        num_classes = len(gloss_vocab)
        feat_dim = self.hand_feature_dim
        matrix = np.zeros((num_classes, feat_dim), dtype=np.float32)
        mask = np.zeros(num_classes, dtype=np.float32)

        for gloss, idx in gloss_vocab.items():
            tpl = self.get_mean_hand_template(gloss)
            if tpl is not None:
                matrix[idx] = tpl
                mask[idx] = 1.0

        coverage = mask.sum() / num_classes * 100
        print(f"  Leap Motion hand coverage: {int(mask.sum())}/{num_classes} "
              f"glosses ({coverage:.1f}%)")
        return matrix, mask

    # ---- Internal ----

    def _load_csv(self):
        """Load raw CSV data."""
        print(f"Loading Leap Motion data from: {self.csv_path}")
        with open(self.csv_path, "r", encoding="utf-8") as f:
            # Detect delimiter
            sample = f.read(4096)
            f.seek(0)
            delimiter = "," if sample.count(",") > sample.count("\t") else "\t"

            reader = csv.DictReader(f, delimiter=delimiter)
            self._columns = reader.fieldnames or []
            self._raw_data = [row for row in reader]

        print(f"  Loaded {len(self._raw_data)} rows, {len(self._columns)} columns")

        # Identify joint position columns
        self._joint_columns = [
            c for c in self._columns
            if any(kw in c.lower() for kw in ["position", "tip", "joint", "bone"])
            and any(kw in c.lower() for kw in ["x", "y", "z"])
        ]

        if not self._joint_columns:
            # Fallback: find any numeric-looking columns that could be joints
            self._joint_columns = self._detect_joint_columns()

        print(f"  Detected {len(self._joint_columns)} joint coordinate columns")

    def _detect_joint_columns(self) -> List[str]:
        """Auto-detect which columns contain hand joint coordinates."""
        numeric_cols = []
        if not self._raw_data:
            return numeric_cols

        sample_row = self._raw_data[0]
        for col in self._columns:
            val = sample_row.get(col, "")
            try:
                float(val)
                # Skip non-joint columns
                col_lower = col.lower()
                skip_keywords = ["id", "frame", "timestamp", "label", "gloss",
                                 "class", "hand_type", "confidence", "grab", "pinch"]
                if not any(kw in col_lower for kw in skip_keywords):
                    numeric_cols.append(col)
            except (ValueError, TypeError):
                pass

        return numeric_cols

    def _detect_gloss_column(self) -> Optional[str]:
        """Find the column containing gloss/sign labels."""
        candidates = ["gloss", "label", "sign", "class", "word", "gesture"]
        for col in self._columns:
            if col.lower().strip() in candidates:
                return col
        # Try partial match
        for col in self._columns:
            for cand in candidates:
                if cand in col.lower():
                    return col
        return None

    def _detect_hand_type_column(self) -> Optional[str]:
        """Find the column indicating left/right hand."""
        candidates = ["hand", "hand_type", "handtype", "side"]
        for col in self._columns:
            if col.lower().strip() in candidates:
                return col
        for col in self._columns:
            for cand in candidates:
                if cand in col.lower():
                    return col
        return None

    def _extract_features(self):
        """Extract hand features grouped by gloss."""
        gloss_col = self._detect_gloss_column()
        hand_col = self._detect_hand_type_column()

        if gloss_col is None:
            print("  [WARN] Could not detect gloss/label column. "
                  "Available columns:", self._columns[:10])
            return

        print(f"  Gloss column: '{gloss_col}', Hand column: '{hand_col}'")

        for row in self._raw_data:
            gloss = row.get(gloss_col, "").upper().strip()
            if not gloss:
                continue

            # Extract joint coordinates
            coords = []
            for col in self._joint_columns:
                try:
                    coords.append(float(row.get(col, 0)))
                except (ValueError, TypeError):
                    coords.append(0.0)

            if not coords:
                continue

            feature = np.array(coords, dtype=np.float32)

            # Normalize to palm-relative coordinates if requested
            if self.normalize and len(feature) >= 3:
                feature = self._normalize_hand(feature)

            self._gloss_hands[gloss].append(feature)

        print(f"  Extracted features for {len(self._gloss_hands)} glosses")

    def _normalize_hand(self, feature: np.ndarray) -> np.ndarray:
        """
        Normalize hand coordinates to be palm-relative and scale-invariant.
        Assumes feature is a flat array of [x, y, z, x, y, z, ...] joint positions.
        """
        if len(feature) < 6:
            return feature

        # Reshape to (N, 3)
        n_joints = len(feature) // 3
        joints = feature[:n_joints * 3].reshape(n_joints, 3)

        # Center on first joint (palm or wrist)
        palm = joints[0].copy()
        joints = joints - palm

        # Scale by maximum distance from palm
        dists = np.linalg.norm(joints, axis=1)
        max_dist = dists.max()
        if max_dist > 1e-6:
            joints = joints / max_dist

        return joints.flatten()


class LeapMotionHandLoss:
    """
    Auxiliary loss module for hand shape supervision.

    Given a batch of predicted hand features and gloss labels,
    computes MSE against Leap Motion hand templates where available.
    """

    def __init__(
        self,
        hand_templates: np.ndarray,
        hand_mask: np.ndarray,
        loss_weight: float = 0.15,
    ):
        """
        Args:
            hand_templates: (num_classes, hand_dim) from build_hand_target_matrix
            hand_mask: (num_classes,) binary mask for which classes have hand data
            loss_weight: Scaling factor for this auxiliary loss
        """
        self.templates = hand_templates  # Will be converted to tensor in train script
        self.mask = hand_mask
        self.loss_weight = loss_weight

    def compute(self, hand_preds, labels):
        """
        Compute auxiliary hand loss.

        Args:
            hand_preds: (batch, hand_dim) predicted hand features
            labels: (batch,) integer class labels

        Returns:
            Weighted MSE loss (scalar), only for samples with hand data
        """
        # This is a skeleton -- actual PyTorch implementation in train_v2.py
        raise NotImplementedError("Use the PyTorch version in temporal_recognizer.py")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect Leap Motion BSL data")
    ap.add_argument("--csv", type=str, default="data/Leap_Motion/BSL-leap-motion.csv")
    args = ap.parse_args()

    loader = LeapMotionLoader(args.csv)
    print(f"\nTotal samples: {loader.num_samples}")
    print(f"Total glosses: {loader.num_glosses}")
    print(f"Hand feature dim: {loader.hand_feature_dim}")
    print(f"\nSample glosses: {loader.gloss_list[:20]}")

    # Show a template
    if loader.gloss_list:
        g = loader.gloss_list[0]
        tpl = loader.get_mean_hand_template(g)
        print(f"\nTemplate for '{g}': shape={tpl.shape}, "
              f"range=[{tpl.min():.3f}, {tpl.max():.3f}]")
