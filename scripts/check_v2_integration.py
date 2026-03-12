"""
BSL Recognition V2 - Integration Check (Fixed)

Uses correct absolute paths and handles columnar pickle format.
"""

import sys
import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict

# Hardcoded correct paths
PATHS = {
    'project_root': Path("D:/Signlytic_AI/code/bsl_translation_project"),
    'swin_features': Path("D:/Signlytic_AI/code/bsl_translation_project/data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32"),
    'bsl1k_dir': Path("D:/Signlytic_AI/code/bsl_translation_project/data/BSL-1K"),
    'leap_motion': Path("D:/Signlytic_AI/code/bsl_translation_project/data/Leap_Motion/BSL-leap-motion.csv"),
    'poses_dir': Path("D:/Signlytic_AI/code/bsl_translation_project/data/poses"),
    'output_dir': Path("D:/Signlytic_AI/code/bsl_translation_project/models/swin_recognition"),
}


def check_paths():
    """Check all paths exist."""
    print("\n--- PATH CHECKS ---")
    results = {}
    
    for name, path in PATHS.items():
        if path.exists():
            if path.is_dir():
                if name == 'swin_features':
                    npy_files = list(path.glob("*.npy"))
                    print(f"  [OK] {name}: {len(npy_files)} .npy files")
                    
                    if npy_files:
                        sample = np.load(npy_files[0])
                        print(f"       Sample shape: {sample.shape}")
                        if len(sample.shape) == 2:
                            print(f"       [OK] Temporal features: T={sample.shape[0]}, D={sample.shape[1]}")
                        else:
                            print(f"       [WARN] Unexpected shape")
                    
                    results[name] = len(npy_files)
                else:
                    contents = list(path.iterdir())
                    print(f"  [OK] {name}: {len(contents)} items")
                    results[name] = len(contents)
            else:
                size = path.stat().st_size / 1024
                print(f"  [OK] {name}: {size:.1f} KB")
                results[name] = size
        else:
            print(f"  [FAIL] {name}: NOT FOUND")
            print(f"         Expected: {path}")
            results[name] = None
    
    return results


def check_bsl1k():
    """Check BSL-1K annotations."""
    print("\n--- BSL-1K ANNOTATION CHECKS ---")
    
    bsl1k_dir = PATHS['bsl1k_dir']
    
    if not bsl1k_dir.exists():
        print("  [FAIL] BSL-1K directory not found")
        return {}
    
    results = {}
    
    # Find pickle files
    pkl_files = list(bsl1k_dir.rglob("*.pkl"))
    print(f"  Found {len(pkl_files)} pickle files:")
    
    total_annotations = 0
    all_glosses = set()
    all_videos = set()
    
    for pkl_file in pkl_files:
        print(f"\n  --- {pkl_file.parent.name}/{pkl_file.name} ---")
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Parse columnar format
            episode_names = data.get('episode_name', [])
            annot_words = data.get('annot_word', [])
            annot_times = data.get('annot_time', [])
            
            n = len(episode_names)
            print(f"    Entries: {n:,}")
            
            # Count unique
            unique_videos = set(str(v).replace('.mp4', '') for v in episode_names)
            unique_glosses = set(str(g).upper() for g in annot_words if g and str(g).upper() != 'NONE')
            
            print(f"    Unique videos: {len(unique_videos):,}")
            print(f"    Unique glosses: {len(unique_glosses):,}")
            
            total_annotations += n
            all_videos.update(unique_videos)
            all_glosses.update(unique_glosses)
            
            # Sample
            if n > 0:
                print(f"    Sample: video={episode_names[0]}, gloss={annot_words[0]}, time={annot_times[0]}")
            
            results[pkl_file.stem] = {'entries': n, 'videos': len(unique_videos), 'glosses': len(unique_glosses)}
            
        except Exception as e:
            print(f"    [ERROR] {e}")
            results[pkl_file.stem] = None
    
    print(f"\n  TOTALS:")
    print(f"    Total annotations: {total_annotations:,}")
    print(f"    Total unique videos: {len(all_videos):,}")
    print(f"    Total unique glosses: {len(all_glosses):,}")
    
    results['_totals'] = {
        'annotations': total_annotations,
        'videos': len(all_videos),
        'glosses': len(all_glosses)
    }
    
    return results


def check_swin_bsl1k_overlap():
    """Check overlap between SWIN features and BSL-1K annotations."""
    print("\n--- FEATURE-ANNOTATION ALIGNMENT ---")
    
    swin_dir = PATHS['swin_features']
    bsl1k_dir = PATHS['bsl1k_dir']
    
    if not swin_dir.exists() or not bsl1k_dir.exists():
        print("  [FAIL] Missing directories")
        return {}
    
    # Get SWIN video IDs
    swin_videos = set()
    for npy_file in swin_dir.glob("*.npy"):
        swin_videos.add(npy_file.stem)
    
    print(f"  SWIN videos: {len(swin_videos)}")
    
    # Get annotated video IDs
    annotated_videos = set()
    for pkl_file in bsl1k_dir.rglob("*.pkl"):
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            for vid in data.get('episode_name', []):
                video_id = str(vid).replace('.mp4', '')
                annotated_videos.add(video_id)
        except:
            pass
    
    print(f"  Annotated videos: {len(annotated_videos)}")
    
    # Overlap
    overlap = swin_videos & annotated_videos
    print(f"  Overlap: {len(overlap)}")
    
    coverage = len(overlap) / len(annotated_videos) * 100 if annotated_videos else 0
    print(f"  Coverage: {coverage:.1f}% of annotations have SWIN features")
    
    if len(overlap) > 0:
        print(f"  [OK] {len(overlap)} videos can be used for training")
    else:
        print(f"  [FAIL] No overlap - check video ID format matching")
        
        # Debug: show samples
        print(f"\n  Sample SWIN IDs: {list(swin_videos)[:3]}")
        print(f"  Sample annotation IDs: {list(annotated_videos)[:3]}")
    
    return {'swin': len(swin_videos), 'annotated': len(annotated_videos), 'overlap': len(overlap)}


def check_leap_motion():
    """Check Leap Motion data."""
    print("\n--- LEAP MOTION CHECK ---")
    
    leap_path = PATHS['leap_motion']
    
    if not leap_path.exists():
        print(f"  [FAIL] Not found: {leap_path}")
        return {}
    
    import csv
    
    with open(leap_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sum(1 for _ in reader)
    
    print(f"  [OK] {rows:,} rows, {len(header)} columns")
    print(f"  Columns: {header[:5]}...")
    
    return {'rows': rows, 'columns': len(header)}


def check_motion_module():
    """Check motion module."""
    print("\n--- MOTION MODULE CHECK ---")
    
    motion_dir = PATHS['project_root'] / "src" / "motion"
    
    if motion_dir.exists():
        files = list(motion_dir.glob("*.py"))
        print(f"  [OK] src/motion/ exists with {len(files)} files")
        for f in files:
            print(f"       - {f.name}")
        return True
    else:
        print(f"  [FAIL] src/motion/ not found")
        print(f"         Create: {motion_dir}")
        return False


def check_model_module():
    """Check model module."""
    print("\n--- MODEL MODULE CHECK ---")
    
    model_file = PATHS['project_root'] / "src" / "models" / "temporal_recognizer.py"
    
    if model_file.exists():
        print(f"  [OK] temporal_recognizer.py exists")
        
        # Check for encoder_type issue
        with open(model_file, 'r') as f:
            content = f.read()
        
        if 'temporal_transformer' in content and '"temporal_transformer"' not in content:
            print(f"  [WARN] May have encoder_type issues")
        
        return True
    else:
        print(f"  [FAIL] temporal_recognizer.py not found")
        return False


def main():
    print("="*60)
    print("  BSL RECOGNITION V2 - INTEGRATION CHECK (FIXED)")
    print("="*60)
    
    results = {}
    
    # 1. Check paths
    results['paths'] = check_paths()
    
    # 2. Check BSL-1K
    results['bsl1k'] = check_bsl1k()
    
    # 3. Check overlap
    results['alignment'] = check_swin_bsl1k_overlap()
    
    # 4. Check Leap Motion
    results['leap'] = check_leap_motion()
    
    # 5. Check modules
    results['motion'] = check_motion_module()
    results['model'] = check_model_module()
    
    # Summary
    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)
    
    issues = []
    
    if results['paths'].get('swin_features') is None:
        issues.append("SWIN features not found")
    
    if results.get('bsl1k', {}).get('_totals', {}).get('annotations', 0) == 0:
        issues.append("No BSL-1K annotations loaded")
    
    if results.get('alignment', {}).get('overlap', 0) == 0:
        issues.append("No SWIN-annotation overlap")
    
    if not results.get('motion'):
        issues.append("Missing src/motion/ module")
    
    if not results.get('model'):
        issues.append("Missing temporal_recognizer.py")
    
    if issues:
        print("\n  ISSUES TO FIX:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  [OK] All checks passed!")
        print("\n  Ready to train:")
        print("    python scripts/train_v2.py --config configs/recognition_v2.yaml")
    
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())