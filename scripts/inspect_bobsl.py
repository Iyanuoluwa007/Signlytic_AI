"""
Inspect BOBSL dataset structure and verify file integrity.

Checks:
- All project files intact after move
- Video files and naming convention
- Annotation format (gloss sequences, timing)
- Subtitle alignment
- Metadata structure

Usage:
    python scripts/inspect_bobsl.py
"""

import os
import sys
import json
from pathlib import Path
from collections import Counter

# Project root (updated for D: drive)
PROJECT_ROOT = Path("D:/Signlytic_AI/code/bsl_translation_project")


def check_file_integrity():
    """Verify all essential project files exist."""
    
    print("=" * 60)
    print("FILE INTEGRITY CHECK")
    print("=" * 60)
    
    essential_files = [
        # Core code
        "app.py",
        "src/inference/speech_to_bsl.py",
        "src/inference/gloss_to_text.py",
        "src/inference/avatar_renderer.py",
        
        # Scripts
        "scripts/download_bsl_videos.py",
        "scripts/test_audio_pipeline.py",
        "scripts/test_speech_to_bsl.py",
        
        # Data - vocabulary
        "data/processed/vocabulary.json",
        "data/processed/vocabulary_extended.json",
        "data/processed/voice_training.wav",
        
        # Data - BslDict
        "data/bsldict/bsldict/bsldict_v1.pkl",
        "data/bsldict/bsldict/bsldict_video_map.json",
        
        # Data - downloaded videos
        "data/videos/bsl_signs",
    ]
    
    essential_dirs = [
        "src",
        "scripts",
        "data",
        "data/processed",
        "data/videos/bsl_signs",
    ]
    
    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Exists: {PROJECT_ROOT.exists()}")
    
    if not PROJECT_ROOT.exists():
        print("\nERROR: Project root not found!")
        print("Expected: D:\\Signlytic_AI\\code\\bsl_translation_project")
        return False
    
    # Check directories
    print(f"\n--- Directories ---")
    missing_dirs = []
    for dir_path in essential_dirs:
        full_path = PROJECT_ROOT / dir_path
        status = "OK" if full_path.exists() else "MISSING"
        if not full_path.exists():
            missing_dirs.append(dir_path)
        print(f"  [{status}] {dir_path}")
    
    # Check files
    print(f"\n--- Essential Files ---")
    missing_files = []
    for file_path in essential_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.is_dir():
            if full_path.exists():
                count = len(list(full_path.glob("*")))
                print(f"  [OK] {file_path}/ ({count} items)")
            else:
                print(f"  [MISSING] {file_path}/")
                missing_files.append(file_path)
        else:
            if full_path.exists():
                size = full_path.stat().st_size
                size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
                print(f"  [OK] {file_path} ({size_str})")
            else:
                print(f"  [MISSING] {file_path}")
                missing_files.append(file_path)
    
    print(f"\n--- Summary ---")
    if missing_dirs or missing_files:
        print(f"Missing directories: {len(missing_dirs)}")
        print(f"Missing files: {len(missing_files)}")
        return False
    else:
        print("All essential files present!")
        return True


def inspect_directory(path, max_files=10, indent=0):
    """Inspect directory structure."""
    path = Path(path)
    prefix = "  " * indent
    
    if not path.exists():
        print(f"{prefix}NOT FOUND: {path}")
        return None, None
    
    if path.is_file():
        size = path.stat().st_size
        size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
        print(f"{prefix}File: {path.name} ({size_str})")
        
        if path.suffix == '.json':
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    print(f"{prefix}  Type: dict with {len(data)} keys")
                    keys = list(data.keys())[:10]
                    print(f"{prefix}  Keys: {keys}")
                elif isinstance(data, list):
                    print(f"{prefix}  Type: list with {len(data)} items")
            except Exception as e:
                print(f"{prefix}  Error: {e}")
        return None, None
    
    items = sorted(path.iterdir())
    dirs = [i for i in items if i.is_dir()]
    files = [i for i in items if i.is_file()]
    
    print(f"{prefix}Directory: {path.name}/")
    print(f"{prefix}  Subdirs: {len(dirs)}, Files: {len(files)}")
    
    if files:
        extensions = Counter(f.suffix for f in files)
        print(f"{prefix}  File types: {dict(extensions)}")
        for f in files[:max_files]:
            size = f.stat().st_size
            size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
            print(f"{prefix}    {f.name} ({size_str})")
    
    return dirs, files


def inspect_bobsl_data():
    """Inspect BOBSL dataset structure."""
    
    print("\n" + "=" * 60)
    print("BOBSL DATASET INSPECTION")
    print("=" * 60)
    
    data_dir = PROJECT_ROOT / "data" / "processed"
    
    for folder in ['annotations', 'metadata', 'subtitles', 'features']:
        print(f"\n--- {folder.upper()} ---")
        folder_path = data_dir / folder
        if folder_path.exists():
            inspect_directory(folder_path, max_files=5)
            
            # Check first subdir
            subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
            if subdirs:
                first_sub = sorted(subdirs)[0]
                print(f"\n  First subdir: {first_sub.name}/")
                sub_files = list(first_sub.glob("*"))[:3]
                for f in sub_files:
                    inspect_directory(f, indent=2)
        else:
            print(f"  NOT FOUND")


def inspect_bobsl_videos():
    """Inspect BOBSL video files."""
    
    print("\n" + "=" * 60)
    print("BOBSL VIDEOS INSPECTION")
    print("=" * 60)
    
    video_locations = [
        PROJECT_ROOT / "data" / "processed" / "bobsl_v1_4_videos_mp4" / "bobsl" / "v1.4" / "original_data" / "videos" / "mp4",
        PROJECT_ROOT / "data" / "bobsl" / "videos" / "mp4",
        PROJECT_ROOT / "data" / "processed" / "videos",
    ]
    
    video_path = None
    for loc in video_locations:
        if loc.exists():
            video_path = loc
            break
    
    if video_path is None:
        print("Video directory NOT FOUND")
        print("Checked:")
        for loc in video_locations:
            print(f"  {loc}")
        return
    
    print(f"\nVideo directory: {video_path}")
    
    print("\nCounting videos...")
    mp4_files = list(video_path.glob("**/*.mp4"))
    print(f"Total MP4 files: {len(mp4_files)}")
    
    if mp4_files:
        total_size = sum(f.stat().st_size for f in mp4_files)
        print(f"Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    
    subdirs = [d for d in video_path.iterdir() if d.is_dir()]
    print(f"Subdirectories: {len(subdirs)}")
    
    if subdirs:
        print(f"Sample: {[d.name for d in sorted(subdirs)[:10]]}")
        
        first_subdir = sorted(subdirs)[0]
        subdir_videos = list(first_subdir.glob("*.mp4"))
        print(f"\nFirst subdir '{first_subdir.name}': {len(subdir_videos)} videos")
        for v in subdir_videos[:5]:
            size = v.stat().st_size / 1024 / 1024
            print(f"    {v.name} ({size:.1f} MB)")


def inspect_bsldict_videos():
    """Inspect downloaded BslDict videos."""
    
    print("\n" + "=" * 60)
    print("BSLDICT VIDEOS (Downloaded Signs)")
    print("=" * 60)
    
    video_dir = PROJECT_ROOT / "data" / "videos" / "bsl_signs"
    
    if not video_dir.exists():
        print(f"NOT FOUND: {video_dir}")
        return
    
    mp4_files = list(video_dir.glob("*.mp4"))
    print(f"Downloaded sign videos: {len(mp4_files)}")
    
    if mp4_files:
        total_size = sum(f.stat().st_size for f in mp4_files)
        print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
        print(f"\nSample:")
        for v in sorted(mp4_files)[:10]:
            print(f"  {v.name}")


def test_imports():
    """Test Python imports."""
    
    print("\n" + "=" * 60)
    print("PYTHON IMPORT TEST")
    print("=" * 60)
    
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "inference"))
    
    modules = [
        ("speech_to_bsl", ["SpeechToBSL", "TextToGloss", "CoquiTTS"]),
        ("gloss_to_text", ["GlossToText"]),
        ("avatar_renderer", ["BSLAvatarRenderer"]),
    ]
    
    for module, classes in modules:
        try:
            mod = __import__(module)
            for cls in classes:
                if hasattr(mod, cls):
                    print(f"  [OK] {module}.{cls}")
                else:
                    print(f"  [WARN] {module}.{cls} not found")
        except ImportError as e:
            print(f"  [FAIL] {module}: {e}")


def main():
    print("=" * 60)
    print("BOBSL & PROJECT INSPECTION")
    print(f"Project: {PROJECT_ROOT}")
    print("=" * 60)
    
    integrity_ok = check_file_integrity()
    inspect_bobsl_data()
    inspect_bobsl_videos()
    inspect_bsldict_videos()
    test_imports()
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("  cd D:\\Signlytic_AI\\code\\bsl_translation_project")
    print("  conda activate BSL")
    print("  python app.py")


if __name__ == "__main__":
    main()