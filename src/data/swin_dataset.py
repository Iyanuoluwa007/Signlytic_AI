"""
SWIN Temporal Feature Dataset

Loads pre-extracted SWIN video features as temporal sequences and pairs
them with BSL-1K annotations for training the recognition model.

Key design:
  - Each .npy file has shape (T, 768) -- a temporal sequence, NOT a static vector
  - Annotation timestamps are converted to feature indices to extract segments
  - Augmentation operates on the temporal dimension (jitter, crop, warp)
  - Compatible with the existing BOBSL annotation pipeline

Usage:
    dataset = SwinTemporalDataset(
        features_dir="data/processed/features/bobsl/v1.4/video_features/swin_v1/...",
        annotations=annotations_list,  # From BSL1KParser or existing parser
        gloss_vocab=gloss_vocab,
        max_seq_len=256,
    )
    features, label, hand_target, confidence = dataset[0]
"""

import os
import json
import tarfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter


class SwinTemporalDataset(Dataset):
    """
    Dataset that loads SWIN temporal features and pairs with annotations.

    Each sample is a temporal segment of SWIN features corresponding to
    one annotated sign instance.
    """

    def __init__(
        self,
        features_dir: str,
        annotations: list,
        gloss_vocab: Dict[str, int],
        max_seq_len: int = 256,
        feature_dim: int = 768,
        feature_fps: float = 1.0,
        # Hand targets (optional, from LeapMotionLoader)
        hand_templates: Optional[np.ndarray] = None,
        hand_mask: Optional[np.ndarray] = None,
        # Augmentation
        augment: bool = False,
        temporal_jitter: float = 0.1,
        temporal_crop_ratio: Tuple[float, float] = (0.8, 1.0),
        feature_noise_std: float = 0.02,
        feature_dropout: float = 0.05,
        time_warp: bool = False,
        time_warp_sigma: float = 0.2,
        # Filtering
        min_confidence: float = 0.0,
        min_seq_len: int = 2,
    ):
        super().__init__()
        self.features_dir = Path(features_dir)
        self.gloss_vocab = gloss_vocab
        self.num_classes = len(gloss_vocab)
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.feature_fps = feature_fps
        self.augment = augment

        # Augmentation params
        self.temporal_jitter = temporal_jitter
        self.temporal_crop_ratio = temporal_crop_ratio
        self.feature_noise_std = feature_noise_std
        self.feature_dropout = feature_dropout
        self.time_warp = time_warp
        self.time_warp_sigma = time_warp_sigma

        # Hand supervision targets
        self.hand_templates = hand_templates  # (num_classes, hand_dim) or None
        self.hand_mask = hand_mask            # (num_classes,) or None

        # Build feature file index: video_id -> .npy path
        self._feature_index = self._build_feature_index()

        # Filter annotations to those with available features and valid glosses
        self.samples = self._prepare_samples(annotations, min_confidence, min_seq_len)

        # Class distribution for weighted sampling
        self._label_counts = Counter(s["label"] for s in self.samples)

        print(f"  SwinTemporalDataset: {len(self.samples)} samples, "
              f"{self.num_classes} classes, "
              f"features from {len(self._feature_index)} videos")

    # ---- Public ----

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, float]:
        """
        Returns:
            features:    (max_seq_len, feature_dim) padded temporal features
            label:       int class label
            hand_target: (hand_dim,) target hand features (zeros if unavailable)
            confidence:  float annotation confidence score
        """
        sample = self.samples[idx]
        video_id = sample["video_id"]
        start_idx = sample["start_idx"]
        end_idx = sample["end_idx"]
        label = sample["label"]
        confidence = sample["confidence"]

        # Load full video features
        features = self._load_features(video_id)
        if features is None:
            # Fallback: return zeros (should be rare after filtering)
            features = np.zeros((self.max_seq_len, self.feature_dim), dtype=np.float32)
            return (
                torch.from_numpy(features),
                label,
                self._get_hand_target(label),
                0.0,
            )

        # Extract temporal segment
        T = features.shape[0]
        s = min(start_idx, T - 1)
        e = min(end_idx, T)
        if e <= s:
            e = s + 1
        segment = features[s:e]  # (seg_len, feature_dim)

        # Augmentation (training only)
        if self.augment:
            segment = self._augment(segment)

        # Pad or truncate to max_seq_len
        segment = self._pad_or_truncate(segment)

        # Convert
        feat_tensor = torch.from_numpy(segment.astype(np.float32))
        hand_target = self._get_hand_target(label)

        return feat_tensor, label, hand_target, confidence

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced training."""
        counts = np.zeros(self.num_classes, dtype=np.float32)
        for label, count in self._label_counts.items():
            counts[label] = count
        # Avoid division by zero
        counts = np.maximum(counts, 1.0)
        weights = 1.0 / counts
        weights = weights / weights.sum() * self.num_classes
        return torch.from_numpy(weights)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """Create a weighted random sampler for balanced training."""
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[s["label"]].item() for s in self.samples]
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    # ---- Internal ----

    def _build_feature_index(self) -> Dict[str, Path]:
        """Map video_id -> feature .npy file path."""
        index = {}
        if not self.features_dir.exists():
            print(f"  [WARN] Features directory not found: {self.features_dir}")
            return index

        # Recursively find all .npy files
        for npy_path in sorted(self.features_dir.rglob("*.npy")):
            # Use stem as video_id, stripping any suffix like "_features"
            vid_id = npy_path.stem
            for suffix in ["_features", "_swin", "_feat"]:
                if vid_id.endswith(suffix):
                    vid_id = vid_id[: -len(suffix)]
            index[vid_id] = npy_path

        # Also check for .npz files
        for npz_path in sorted(self.features_dir.rglob("*.npz")):
            vid_id = npz_path.stem
            for suffix in ["_features", "_swin", "_feat"]:
                if vid_id.endswith(suffix):
                    vid_id = vid_id[: -len(suffix)]
            if vid_id not in index:
                index[vid_id] = npz_path

        return index

    def _load_features(self, video_id: str) -> Optional[np.ndarray]:
        """Load SWIN features for a video. Returns shape (T, feature_dim)."""
        path = self._feature_index.get(video_id)
        if path is None:
            return None

        try:
            if path.suffix == ".npz":
                data = np.load(path, allow_pickle=True)
                # Try common array names
                for key in ["features", "feat", "data", "arr_0"]:
                    if key in data:
                        arr = data[key]
                        break
                else:
                    arr = data[list(data.keys())[0]]
            else:
                arr = np.load(path, allow_pickle=True)

            # Ensure 2D: (T, feature_dim)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim == 3:
                # Could be (1, T, D) or (T, 1, D)
                arr = arr.squeeze()
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)

            # Convert float16 to float32 for training
            return arr.astype(np.float32)

        except Exception as e:
            print(f"  [WARN] Failed to load features for {video_id}: {e}")
            return None

    def _prepare_samples(
        self, annotations: list, min_confidence: float, min_seq_len: int
    ) -> List[dict]:
        """Filter and prepare training samples from annotations."""
        from .bsl1k_parser import BSL1KAnnotation, BSL1KParser

        samples = []
        skipped_no_features = 0
        skipped_no_gloss = 0
        skipped_short = 0
        skipped_low_conf = 0

        for ann in annotations:
            # Handle both BSL1KAnnotation objects and dicts
            if isinstance(ann, BSL1KAnnotation):
                video_id = ann.video_id
                gloss = ann.gloss
                start = ann.start
                end = ann.end
                confidence = ann.confidence
            elif isinstance(ann, dict):
                video_id = ann.get("video_id", ann.get("video", ""))
                gloss = ann.get("gloss", ann.get("label", "")).upper().strip()
                start = float(ann.get("start", ann.get("start_time", 0)))
                end = float(ann.get("end", ann.get("end_time", 0)))
                confidence = float(ann.get("confidence", 1.0))
            else:
                continue

            # Check gloss in vocabulary
            if gloss not in self.gloss_vocab:
                skipped_no_gloss += 1
                continue

            # Check confidence
            if confidence < min_confidence:
                skipped_low_conf += 1
                continue

            # Check features exist
            if video_id not in self._feature_index:
                skipped_no_features += 1
                continue

            # Convert timestamps to feature indices
            start_idx, end_idx = BSL1KParser.time_to_feature_index(
                start, end, self.feature_fps
            )

            # Check minimum length
            if end_idx - start_idx < min_seq_len:
                skipped_short += 1
                continue

            samples.append({
                "video_id": video_id,
                "gloss": gloss,
                "label": self.gloss_vocab[gloss],
                "start_idx": start_idx,
                "end_idx": end_idx,
                "confidence": confidence,
            })

        if skipped_no_features > 0:
            print(f"  Skipped {skipped_no_features} samples (no features)")
        if skipped_no_gloss > 0:
            print(f"  Skipped {skipped_no_gloss} samples (gloss not in vocab)")
        if skipped_short > 0:
            print(f"  Skipped {skipped_short} samples (too short)")
        if skipped_low_conf > 0:
            print(f"  Skipped {skipped_low_conf} samples (low confidence)")

        return samples

    def _get_hand_target(self, label: int) -> torch.Tensor:
        """Get hand supervision target for a class label."""
        if self.hand_templates is not None:
            target = torch.from_numpy(self.hand_templates[label].copy())
            return target
        # Return a zero vector if no hand data
        hand_dim = 63  # default 21 joints x 3
        return torch.zeros(hand_dim, dtype=torch.float32)

    def _pad_or_truncate(self, segment: np.ndarray) -> np.ndarray:
        """Pad or truncate temporal segment to max_seq_len."""
        T, D = segment.shape
        if T >= self.max_seq_len:
            # Uniform downsampling to preserve temporal structure
            indices = np.linspace(0, T - 1, self.max_seq_len, dtype=int)
            return segment[indices]
        else:
            # Pad with zeros at the end
            padded = np.zeros((self.max_seq_len, D), dtype=segment.dtype)
            padded[:T] = segment
            return padded

    # ---- Augmentation ----

    def _augment(self, segment: np.ndarray) -> np.ndarray:
        """Apply temporal and feature augmentations."""
        T, D = segment.shape

        # Temporal jitter: shift start/end slightly
        if self.temporal_jitter > 0 and T > 4:
            jitter = int(T * self.temporal_jitter)
            if jitter > 0:
                shift = np.random.randint(-jitter, jitter + 1)
                if shift > 0:
                    segment = np.concatenate(
                        [np.zeros((shift, D), dtype=segment.dtype), segment[:-shift]], axis=0
                    )
                elif shift < 0:
                    segment = np.concatenate(
                        [segment[-shift:], np.zeros((-shift, D), dtype=segment.dtype)], axis=0
                    )

        # Temporal crop
        lo, hi = self.temporal_crop_ratio
        if lo < 1.0 and T > 4:
            crop_ratio = np.random.uniform(lo, hi)
            crop_len = max(2, int(T * crop_ratio))
            start = np.random.randint(0, max(1, T - crop_len + 1))
            segment = segment[start : start + crop_len]

        # Time warping (elastic temporal distortion)
        if self.time_warp and len(segment) > 4:
            segment = self._time_warp(segment)

        # Feature noise
        if self.feature_noise_std > 0:
            noise = np.random.normal(0, self.feature_noise_std, segment.shape)
            segment = segment + noise.astype(segment.dtype)

        # Feature dropout (random zeroing)
        if self.feature_dropout > 0:
            mask = np.random.random(segment.shape) > self.feature_dropout
            segment = segment * mask.astype(segment.dtype)

        return segment

    def _time_warp(self, segment: np.ndarray) -> np.ndarray:
        """Apply elastic time warping to a sequence."""
        T, D = segment.shape
        # Generate smooth random warp path
        warp_points = np.random.normal(0, self.time_warp_sigma, size=T)
        # Smooth the warp
        from scipy.ndimage import gaussian_filter1d
        try:
            warp_points = gaussian_filter1d(warp_points, sigma=T * 0.1)
        except ImportError:
            # Fallback: simple moving average
            kernel = np.ones(max(3, T // 5)) / max(3, T // 5)
            warp_points = np.convolve(warp_points, kernel, mode="same")

        # Create warped indices
        original_indices = np.arange(T, dtype=np.float64)
        warped_indices = original_indices + warp_points
        warped_indices = np.clip(warped_indices, 0, T - 1)

        # Interpolate
        warped = np.zeros_like(segment)
        for d in range(D):
            warped[:, d] = np.interp(
                original_indices, warped_indices, segment[:, d]
            )
        return warped


class SwinFeatureExtractor:
    """
    Utility to extract features from the raw tar archive if the
    features_dir is not populated yet.
    """

    @staticmethod
    def extract_from_tar(
        tar_path: str, output_dir: str, max_files: Optional[int] = None
    ):
        """Extract SWIN features from tar archive."""
        tar_path = Path(tar_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not tar_path.exists():
            raise FileNotFoundError(f"Tar archive not found: {tar_path}")

        print(f"Extracting SWIN features from {tar_path}")
        count = 0
        with tarfile.open(tar_path, "r") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".npy")]
            for member in members:
                tar.extract(member, output_dir)
                count += 1
                if max_files and count >= max_files:
                    break
                if count % 500 == 0:
                    print(f"  Extracted {count}/{len(members)} files")

        print(f"  Extracted {count} feature files to {output_dir}")


def create_dataloaders(
    features_dir: str,
    annotations: list,
    gloss_vocab: Dict[str, int],
    max_seq_len: int = 256,
    feature_fps: float = 1.0,
    hand_templates: Optional[np.ndarray] = None,
    hand_mask: Optional[np.ndarray] = None,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.10,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders with stratified splitting.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split

    np.random.seed(seed)

    # Get labels for stratification
    labels = []
    for ann in annotations:
        if hasattr(ann, "gloss"):
            g = ann.gloss
        elif isinstance(ann, dict):
            g = ann.get("gloss", ann.get("label", "")).upper().strip()
        else:
            g = ""
        labels.append(gloss_vocab.get(g, -1))

    # Filter out annotations without valid labels
    valid_mask = [l >= 0 for l in labels]
    valid_anns = [a for a, v in zip(annotations, valid_mask) if v]
    valid_labels = [l for l, v in zip(labels, valid_mask) if v]

    # Stratified split
    try:
        train_anns, test_anns, train_labels, _ = train_test_split(
            valid_anns, valid_labels,
            test_size=test_split, stratify=valid_labels, random_state=seed,
        )
        val_ratio = val_split / (1.0 - test_split)
        train_anns, val_anns, _, _ = train_test_split(
            train_anns, train_labels,
            test_size=val_ratio, stratify=train_labels, random_state=seed,
        )
    except ValueError:
        # Fallback: random split if stratification fails
        print("  [WARN] Stratified split failed, using random split")
        n = len(valid_anns)
        indices = np.random.permutation(n)
        n_test = int(n * test_split)
        n_val = int(n * val_split)
        test_anns = [valid_anns[i] for i in indices[:n_test]]
        val_anns = [valid_anns[i] for i in indices[n_test : n_test + n_val]]
        train_anns = [valid_anns[i] for i in indices[n_test + n_val :]]

    print(f"  Split: train={len(train_anns)}, val={len(val_anns)}, test={len(test_anns)}")

    common_kwargs = dict(
        features_dir=features_dir,
        gloss_vocab=gloss_vocab,
        max_seq_len=max_seq_len,
        feature_fps=feature_fps,
        hand_templates=hand_templates,
        hand_mask=hand_mask,
    )

    train_ds = SwinTemporalDataset(annotations=train_anns, augment=True, **common_kwargs)
    val_ds = SwinTemporalDataset(annotations=val_anns, augment=False, **common_kwargs)
    test_ds = SwinTemporalDataset(annotations=test_anns, augment=False, **common_kwargs)

    # Use weighted sampler for training
    train_sampler = train_ds.get_weighted_sampler()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader
