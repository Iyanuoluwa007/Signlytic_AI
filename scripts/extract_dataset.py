#!/usr/bin/env python3
"""
Extract BOBSL dataset archives and delete each archive after successful extraction.
"""

import tarfile
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def extract_tar_gz(archive_path: Path, destination: Path) -> None:
    """Extract a .tar.gz archive."""
    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=archive_path.name, unit="file"):
            tar.extract(member, destination, filter="data")


def extract_tar(archive_path: Path, destination: Path) -> None:
    """Extract a .tar archive."""
    with tarfile.open(archive_path, "r:") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc=archive_path.name, unit="file"):
            tar.extract(member, destination, filter="data")


def delete_archive(archive_path: Path) -> None:
    """Delete an archive file from disk."""
    archive_path.unlink()


def prepare_directories(processed_dir: Path) -> dict:
    """Create output directories for extracted data."""
    destinations = {
        "annotations": processed_dir / "annotations",
        # "videos": processed_dir / "videos",  # Disabled: insufficient disk space (~400+ GB required)
        "features": processed_dir / "features",
        "metadata": processed_dir / "metadata",
        "subtitles": processed_dir / "subtitles",
    }

    for path in destinations.values():
        path.mkdir(parents=True, exist_ok=True)

    return destinations


def get_extraction_map(destinations: dict) -> List[Tuple[str, Path]]:
    """Map archive filenames to output directories."""
    return [
        ("bobsl_v1_4_continuous_sign_sequences.tar.gz", destinations["annotations"]),
        ("bobsl_v1_4_fingerspelled_signs.tar.gz", destinations["annotations"]),
        ("bobsl_v1_4_isolated_signs.tar.gz", destinations["annotations"]),
        ("bobsl_v1_4_signing_aligned_subtitles.tar.gz", destinations["annotations"]),
        ("bobsl_v1_4_metadata.tar.gz", destinations["metadata"]),
        ("bobsl_v1_4_subtitles.tar.gz", destinations["subtitles"]),

        # ("bobsl_v1_4_videos_mp4.tar", destinations["videos"]),  # Disabled: insufficient disk space

        ("bobsl_v1_4_features_i3d.tar", destinations["features"]),
        ("bobsl_v1_4_features_swin_v1.tar", destinations["features"]),
    ]


def main() -> None:
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")

    destinations = prepare_directories(processed_dir)
    extraction_map = get_extraction_map(destinations)

    for filename, destination in extraction_map:
        archive_path = raw_dir / filename

        if not archive_path.exists():
            print(f"Archive not found, skipping: {archive_path}")
            continue

        try:
            if filename.endswith(".tar.gz"):
                extract_tar_gz(archive_path, destination)
            elif filename.endswith(".tar"):
                extract_tar(archive_path, destination)
            else:
                print(f"Unsupported archive format: {archive_path}")
                continue

            delete_archive(archive_path)
            print(f"Extracted and deleted: {archive_path.name}")

        except Exception as exc:
            print(f"Extraction failed for: {archive_path.name}")
            print(f"Error: {exc}")

    print("Dataset extraction completed.")
    print(f"Processed data directory: {processed_dir}")


if __name__ == "__main__":
    main()
