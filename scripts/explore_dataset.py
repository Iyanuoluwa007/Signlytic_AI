#!/usr/bin/env python3
"""
Explore the BOBSL dataset structure and contents.
"""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ENABLE_VIDEO_ANALYSIS = False  # Set to True when videos are available.


def explore_directory(path: Path, max_depth: int = 3, current_depth: int = 0) -> None:
    """Recursively explore directory structure."""
    path = Path(path)
    indent = "  " * current_depth

    if current_depth >= max_depth or not path.exists():
        return

    items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    dirs = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]

    for d in dirs[:10]:
        print(f"{indent}{d.name}/")
        explore_directory(d, max_depth, current_depth + 1)

    if len(dirs) > 10:
        print(f"{indent}... and {len(dirs) - 10} more directories")

    for f in files[:5]:
        size_str = format_size(f.stat().st_size)
        print(f"{indent}{f.name} ({size_str})")

    if len(files) > 5:
        print(f"{indent}... and {len(files) - 5} more files")


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0

    return f"{size:.1f} PB"


def count_files_by_extension(path: Path) -> Counter:
    """Count files by extension."""
    path = Path(path)
    extensions: Counter = Counter()

    if not path.exists():
        return extensions

    for f in path.rglob("*"):
        if f.is_file():
            ext = f.suffix.lower() or "no_extension"
            extensions[ext] += 1

    return extensions


def analyze_annotation_files(annotations_dir: Path) -> None:
    """Analyze annotation file formats and show basic samples."""
    annotations_dir = Path(annotations_dir)

    print("\n" + "=" * 60)
    print("ANNOTATION FILES ANALYSIS")
    print("=" * 60)

    if not annotations_dir.exists():
        print("Annotations directory not found.")
        return

    for ann_file in sorted(annotations_dir.rglob("*")):
        if not ann_file.is_file():
            continue

        print(f"\n{ann_file.relative_to(annotations_dir)}")

        try:
            if ann_file.suffix.lower() == ".json":
                with ann_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                print("Type: JSON")
                if isinstance(data, dict):
                    keys = list(data.keys())
                    print(f"Keys (first 5): {keys[:5]}")
                    print(f"Total keys: {len(keys)}")
                elif isinstance(data, list):
                    print(f"List length: {len(data)}")
                    if data:
                        first = data[0]
                        if isinstance(first, dict):
                            print(f"First item keys: {list(first.keys())[:10]}")
                        else:
                            print(f"First item type: {type(first).__name__}")

            elif ann_file.suffix.lower() == ".csv":
                df = pd.read_csv(ann_file, nrows=5)
                print("Type: CSV")
                print(f"Columns: {list(df.columns)}")
                print(f"Preview shape: {df.shape}")

            elif ann_file.suffix.lower() == ".txt":
                with ann_file.open("r", encoding="utf-8", errors="replace") as f:
                    lines = [next(f, "").rstrip("\n") for _ in range(5)]
                print("Type: Text")
                for line in lines:
                    if line:
                        print(line[:120])

        except Exception as exc:
            print(f"Error reading: {exc}")


def analyze_features(features_dir: Path) -> None:
    """Analyze pre-extracted feature files and print a sample shape."""
    features_dir = Path(features_dir)

    print("\n" + "=" * 60)
    print("FEATURES ANALYSIS")
    print("=" * 60)

    if not features_dir.exists():
        print("Features directory not found.")
        return

    for subdir_name in ["i3d", "swin_v1"]:
        subdir = features_dir / subdir_name
        if not subdir.exists():
            continue

        feature_files = list(subdir.rglob("*.npy")) + list(subdir.rglob("*.npz"))
        print(f"\n{subdir_name} features")
        print(f"Total files: {len(feature_files)}")

        if not feature_files:
            continue

        sample_path = feature_files[0]
        try:
            sample = np.load(sample_path, allow_pickle=False)
            print(f"Sample file: {sample_path.name}")

            if isinstance(sample, np.lib.npyio.NpzFile):
                keys = list(sample.keys())
                print(f"Format: NPZ, arrays: {keys}")
                for key in keys[:5]:
                    print(f"{key}: shape={sample[key].shape}, dtype={sample[key].dtype}")
            else:
                print(f"Format: NPY, shape={sample.shape}, dtype={sample.dtype}")

        except Exception as exc:
            print(f"Error reading sample: {exc}")


def analyze_videos(videos_dir: Path) -> None:
    """Analyze video files and print basic stats for one sample."""
    videos_dir = Path(videos_dir)

    print("\n" + "=" * 60)
    print("VIDEOS ANALYSIS")
    print("=" * 60)

    if not videos_dir.exists():
        print("Videos directory not found.")
        return

    video_files = list(videos_dir.rglob("*.mp4"))
    print(f"Total video files: {len(video_files)}")

    total_size = sum(f.stat().st_size for f in video_files)
    print(f"Total video storage: {format_size(total_size)}")

    if not video_files:
        return

    try:
        import cv2

        sample_video = video_files[0]
        cap = cv2.VideoCapture(str(sample_video))

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        duration = (frame_count / fps) if fps and fps > 0 else 0.0

        print(f"\nSample video: {sample_video.name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Duration (s): {duration:.2f}")
        print(f"Frames: {frame_count}")

    except Exception as exc:
        print(f"Error analyzing sample video: {exc}")


def main() -> None:
    processed_dir = Path("data/processed")

    print("=" * 60)
    print("BOBSL DATASET EXPLORATION")
    print("=" * 60)

    print("\nDIRECTORY STRUCTURE")
    explore_directory(processed_dir)

    print("\nFILE TYPES")
    extensions = count_files_by_extension(processed_dir)
    for ext, count in extensions.most_common(10):
        print(f"{ext}: {count} files")

    annotations_dir = processed_dir / "annotations"
    if annotations_dir.exists():
        analyze_annotation_files(annotations_dir)

    features_dir = processed_dir / "features"
    if features_dir.exists():
        analyze_features(features_dir)

    if ENABLE_VIDEO_ANALYSIS:
        videos_dir = processed_dir / "videos"
        if videos_dir.exists():
            analyze_videos(videos_dir)
        else:
            print("\nVIDEOS ANALYSIS")
            print("Videos directory not found.")


if __name__ == "__main__":
    main()
