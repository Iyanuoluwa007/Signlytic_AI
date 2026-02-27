"""
Extract pose sequences from local BSLDict videos for recognition training.

Output schema matches scripts/extract_poses.py so it can be merged with data/poses.
"""

import re
import cv2
import json
import argparse
import hashlib
from pathlib import Path
from dataclasses import asdict
from typing import Dict, Optional, List

from tqdm import tqdm

from extract_poses import PoseExtractor


def stable_split(key: str) -> str:
    """Deterministic 80/10/10 split."""
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    if h < 80:
        return "train"
    if h < 90:
        return "val"
    return "test"


def normalize_text(text: str) -> str:
    """Canonical normalization for loose glossary matching."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def build_gloss_aliases(glosses: List[str]) -> Dict[str, str]:
    """Build loose alias map from vocabulary glosses."""
    alias_to_gloss: Dict[str, str] = {}
    for gloss in sorted(glosses):
        base = gloss.lower()
        variants = {
            base,
            base.replace("_", " "),
            base.replace("_", "-"),
            base.replace("_", ""),
            normalize_text(base),
            normalize_text(base).replace(" ", ""),
        }
        for v in variants:
            if v:
                alias_to_gloss.setdefault(v, gloss)
    return alias_to_gloss


def resolve_vocab_gloss(stem: str, alias_map: Dict[str, str]) -> Optional[str]:
    """Resolve a video stem to a vocab gloss using tolerant matching."""
    stem_lower = stem.lower()
    candidates = {
        stem_lower,
        stem_lower.replace("-", " "),
        stem_lower.replace("-", "_"),
        stem_lower.replace("_", " "),
        stem_lower.replace("_", "-"),
        stem_lower.replace("_", ""),
        stem_lower.replace("-", ""),
        normalize_text(stem_lower),
        normalize_text(stem_lower).replace(" ", ""),
    }
    for c in candidates:
        if c in alias_map:
            return alias_map[c]
    return None


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(frames / max(fps, 1e-6))


def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "-", name)


def build_sample_index(output_dir: Path):
    """Create a merged sample index for faster training startup."""
    index = []
    for split in ["train", "val", "test"]:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for path in split_dir.glob("*.json"):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                index.append(
                    {
                        "gloss": data["gloss"],
                        "file": str(path.resolve()),
                        "num_frames": int(data.get("num_frames", 0)),
                        "source": str(output_dir.resolve()),
                    }
                )
            except Exception:
                continue
    with open(output_dir / "sample_index.json", "w") as f:
        json.dump(index, f)
    print(f"Saved sample index: {output_dir / 'sample_index.json'} ({len(index)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Extract BSLDict pose sequences")
    parser.add_argument("--video-dir", type=str, default=None, help="Directory with BSLDict videos")
    parser.add_argument("--vocab", type=str, default=None, help="Recognition vocabulary JSON")
    parser.add_argument("--output-dir", type=str, default=None, help="Output pose directory")
    parser.add_argument("--max-videos", type=int, default=None, help="Limit number of videos")
    parser.add_argument("--max-seconds", type=float, default=None, help="Optional max seconds per video")
    parser.add_argument("--skip-existing", action="store_true", dest="skip_existing")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    parser.set_defaults(skip_existing=True)
    args = parser.parse_args()

    project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
    video_dir = Path(args.video_dir) if args.video_dir else project_root / "data" / "videos" / "bsl_signs"
    vocab_path = Path(args.vocab) if args.vocab else project_root / "models" / "sign_recognition" / "vocabulary.json"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "data" / "poses_bsldict"

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_glosses = list(vocab.keys())
    alias_map = build_gloss_aliases(vocab_glosses)
    print(f"Loaded vocabulary: {len(vocab_glosses)} glosses")

    videos = sorted(video_dir.glob("*.mp4"))
    if args.max_videos:
        videos = videos[:args.max_videos]
    print(f"Videos to process: {len(videos)}")

    extractor = PoseExtractor(model_complexity=1)
    processed = 0
    skipped = 0
    unresolved = 0
    errors = 0

    try:
        for video_path in tqdm(videos, desc="Extracting BSLDict poses"):
            gloss = resolve_vocab_gloss(video_path.stem, alias_map)
            if gloss is None:
                unresolved += 1
                continue

            split = stable_split(video_path.name)
            safe_gloss = sanitize_filename(gloss)
            safe_stem = sanitize_filename(video_path.stem)
            output_file = output_dir / split / f"{safe_gloss}_{safe_stem}_0.00.json"

            if args.skip_existing and output_file.exists():
                skipped += 1
                continue

            duration = get_video_duration(video_path)
            if duration <= 0:
                errors += 1
                continue
            if args.max_seconds is not None:
                duration = min(duration, args.max_seconds)

            try:
                poses = extractor.extract_from_video(str(video_path), start_time=0.0, end_time=duration)
                if not poses:
                    errors += 1
                    continue

                payload = {
                    "gloss": gloss,
                    "video_id": video_path.stem,
                    "start_time": 0.0,
                    "end_time": float(duration),
                    "english": "",
                    "num_frames": len(poses),
                    "poses": [asdict(p) for p in poses],
                }

                with open(output_file, "w") as f:
                    json.dump(payload, f)
                processed += 1
            except Exception:
                errors += 1
                continue
    finally:
        extractor.close()

    build_sample_index(output_dir)
    summary = {
        "processed": processed,
        "skipped": skipped,
        "unresolved_gloss": unresolved,
        "errors": errors,
        "video_dir": str(video_dir),
        "output_dir": str(output_dir),
        "vocab_size": len(vocab_glosses),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

