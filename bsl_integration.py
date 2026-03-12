"""
BSL System Integration

Main entry point that combines:
- SWIN-based sign recognition
- Natural motion generation
- Improved 2D rendering
- Future Blender avatar integration

Usage:
    # Full pipeline demo
    python bsl_integration.py --demo
    
    # Train recognition
    python bsl_integration.py --train-recognition
    
    # Generate motion
    python bsl_integration.py --generate "HELLO HOW YOU"
    
    # Real-time recognition
    python bsl_integration.py --realtime
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

# Project root
PROJECT_ROOT = Path("D:/Signlytic_AI/code/bsl_translation_project")
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
CONFIG = {
    # Paths
    'poses_dir': PROJECT_ROOT / "data" / "poses",
    'swin_features_dir': PROJECT_ROOT / "data" / "swin_features",
    'bsl1k_dir': PROJECT_ROOT / "data" / "BSL-1K",
    'leap_motion_path': PROJECT_ROOT / "data" / "Leap_Motion" / "BSL-leap-motion.csv",
    'swin_tar_path': PROJECT_ROOT / "data" / "raw" / "bobsl_v1_4_features_swin_v1_3.tar",
    'models_dir': PROJECT_ROOT / "models",
    
    # Model settings
    'feature_dim': 768,  # SWIN feature dimension
    'max_seq_len': 64,
    
    # Output
    'output_dir': PROJECT_ROOT / "outputs",
}


def check_system_status():
    """Check status of all system components."""
    print("\n" + "="*60)
    print("BSL SYSTEM STATUS CHECK")
    print("="*60)
    
    status = {}
    
    # 1. Pose data
    poses_dir = CONFIG['poses_dir']
    if poses_dir.exists():
        train_count = len(list((poses_dir / "train").glob("*.json"))) if (poses_dir / "train").exists() else 0
        status['poses'] = f"OK ({train_count} training files)"
    else:
        status['poses'] = "NOT FOUND"
    
    # 2. SWIN features
    swin_dir = CONFIG['swin_features_dir']
    swin_tar = CONFIG['swin_tar_path']
    if swin_dir.exists():
        npy_count = len(list(swin_dir.rglob("*.npy")))
        status['swin_features'] = f"EXTRACTED ({npy_count} files)"
    elif swin_tar.exists():
        size_gb = swin_tar.stat().st_size / (1024**3)
        status['swin_features'] = f"ARCHIVE ({size_gb:.1f} GB) - needs extraction"
    else:
        status['swin_features'] = "NOT FOUND"
    
    # 3. BSL-1K annotations
    bsl1k_dir = CONFIG['bsl1k_dir']
    if bsl1k_dir.exists():
        json_count = len(list(bsl1k_dir.rglob("*.json")))
        status['bsl1k'] = f"OK ({json_count} annotation files)"
    else:
        status['bsl1k'] = "NOT FOUND"
    
    # 4. Leap Motion
    leap_path = CONFIG['leap_motion_path']
    if leap_path.exists():
        size_mb = leap_path.stat().st_size / (1024**2)
        status['leap_motion'] = f"OK ({size_mb:.1f} MB)"
    else:
        status['leap_motion'] = "NOT FOUND"
    
    # 5. Trained models
    models_dir = CONFIG['models_dir']
    if models_dir.exists():
        model_files = list(models_dir.rglob("*.pt"))
        status['models'] = f"OK ({len(model_files)} model files)"
    else:
        status['models'] = "NO MODELS"
    
    # Print status
    for component, stat in status.items():
        icon = "[OK]" if "OK" in stat or "EXTRACTED" in stat else "[!!]"
        print(f"  {icon} {component}: {stat}")
    
    return status


def extract_swin_features():
    """Extract SWIN features from tar archive."""
    import tarfile
    
    tar_path = CONFIG['swin_tar_path']
    extract_dir = CONFIG['swin_features_dir']
    
    if not tar_path.exists():
        print(f"SWIN archive not found: {tar_path}")
        return False
    
    print(f"Extracting SWIN features...")
    print(f"  Source: {tar_path}")
    print(f"  Target: {extract_dir}")
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(tar_path, 'r') as tar:
        members = tar.getmembers()
        print(f"  Total files: {len(members)}")
        
        # Extract with progress
        for i, member in enumerate(members):
            tar.extract(member, extract_dir)
            if (i + 1) % 1000 == 0:
                print(f"  Extracted {i + 1}/{len(members)}...")
    
    print(f"  [OK] Extraction complete!")
    return True


def run_recognition_training(resume: Optional[str] = None):
    """Train the SWIN-based recognition model."""
    try:
        from scripts.train_swin_recognition import SWINRecognitionTrainer, TrainingConfig
    except ImportError:
        print("Training modules not found. Make sure scripts are in place.")
        print("Required: scripts/train_swin_recognition.py")
        return
    
    # Check prerequisites
    status = check_system_status()
    
    if "NOT FOUND" in status.get('swin_features', ''):
        print("\n[!] SWIN features need to be extracted first.")
        response = input("Extract now? (y/n): ")
        if response.lower() == 'y':
            extract_swin_features()
        else:
            return
    
    if "NOT FOUND" in status.get('bsl1k', ''):
        print("\n[!] BSL-1K annotations not found. Cannot train.")
        return
    
    # Create config
    config = TrainingConfig(
        swin_features_dir=str(CONFIG['swin_features_dir']),
        bsl1k_dir=str(CONFIG['bsl1k_dir']),
        leap_motion_path=str(CONFIG['leap_motion_path']),
        output_dir=str(CONFIG['models_dir'] / "swin_recognition"),
        epochs=100,
        batch_size=32
    )
    
    # Train
    trainer = SWINRecognitionTrainer(config)
    trainer.setup()
    
    if resume:
        trainer.load_checkpoint(resume)
    
    trainer.train()


def generate_motion(text: str, output_video: Optional[str] = None):
    """Generate natural BSL motion from text/glosses."""
    try:
        from motion.natural_motion_generator import NaturalMotionGenerator
        from rendering.improved_pose_renderer import ImprovedPoseRenderer
    except ImportError:
        print("Motion/rendering modules not found.")
        print("Required: motion/natural_motion_generator.py, rendering/improved_pose_renderer.py")
        return
    
    # Parse input
    import re
    glosses = re.findall(r'\b\w+\b', text.upper())
    print(f"Generating motion for: {' -> '.join(glosses)}")
    
    # Generate motion
    generator = NaturalMotionGenerator(str(CONFIG['poses_dir']))
    poses = generator.generate_sequence(glosses)
    
    if not poses:
        print("No poses generated. Check if glosses exist in database.")
        return
    
    print(f"Generated {len(poses)} frames")
    
    # Render
    renderer = ImprovedPoseRenderer()
    
    if output_video:
        renderer.render_to_video(poses, output_video, glosses)
    else:
        renderer.display_sequence(poses, glosses)


def run_demo():
    """Run full system demo."""
    print("\n" + "="*60)
    print("BSL SYSTEM DEMO")
    print("="*60)
    
    # Check status
    check_system_status()
    
    # Demo motion generation
    print("\n--- Motion Generation Demo ---")
    demo_glosses = ["HELLO", "GOOD", "YOU"]
    
    try:
        from motion.natural_motion_generator import NaturalMotionGenerator
        
        generator = NaturalMotionGenerator(str(CONFIG['poses_dir']))
        
        for gloss in demo_glosses:
            poses = generator.generate_sign_motion(gloss)
            if poses:
                print(f"  {gloss}: {len(poses)} frames (natural motion)")
            else:
                print(f"  {gloss}: NOT FOUND")
        
        # Generate sequence
        full_sequence = generator.generate_sequence(demo_glosses)
        print(f"\n  Full sequence: {len(full_sequence)} frames")
        print(f"  With co-articulation and easing applied")
        
    except ImportError as e:
        print(f"  [!] Motion module not available: {e}")
    
    print("\n--- Recognition Model Status ---")
    
    # Check for trained model
    model_path = CONFIG['models_dir'] / "swin_recognition" / "best_model.pt"
    if model_path.exists():
        print(f"  Trained model found: {model_path}")
    else:
        print(f"  No trained model found. Run --train-recognition first.")
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("""
Next steps:
1. If SWIN features need extraction: python bsl_integration.py --extract-swin
2. Train recognition model: python bsl_integration.py --train-recognition
3. Generate motion: python bsl_integration.py --generate "HELLO HOW YOU"
4. Run real-time demo: python bsl_integration.py --realtime
""")


def run_realtime():
    """Run real-time recognition with webcam."""
    print("\n--- Real-time Recognition ---")
    print("This feature requires trained SWIN model.")
    print("For now, use existing pose-based recognition:")
    print("  python scripts/realtime_recognition.py")


def main():
    parser = argparse.ArgumentParser(description="BSL System Integration")
    parser.add_argument("--status", action="store_true", help="Check system status")
    parser.add_argument("--extract-swin", action="store_true", help="Extract SWIN features")
    parser.add_argument("--train-recognition", action="store_true", help="Train recognition model")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint")
    parser.add_argument("--generate", type=str, default=None, help="Generate motion for text/glosses")
    parser.add_argument("--output", type=str, default=None, help="Output video path")
    parser.add_argument("--realtime", action="store_true", help="Run real-time recognition")
    parser.add_argument("--demo", action="store_true", help="Run system demo")
    args = parser.parse_args()
    
    if args.status:
        check_system_status()
    elif args.extract_swin:
        extract_swin_features()
    elif args.train_recognition:
        run_recognition_training(args.resume)
    elif args.generate:
        generate_motion(args.generate, args.output)
    elif args.realtime:
        run_realtime()
    elif args.demo:
        run_demo()
    else:
        # Default: show status and demo
        run_demo()


if __name__ == "__main__":
    main()
