"""
Inspect SWIN feature files to understand their structure.
"""

import numpy as np
from pathlib import Path
from collections import Counter


def main():
    # The actual SWIN features path from check_v2 output
    swin_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32")
    
    print("="*60)
    print("SWIN FEATURES INSPECTION")
    print("="*60)
    
    if not swin_dir.exists():
        print(f"[FAIL] Directory not found: {swin_dir}")
        
        # Try to find the actual location
        base = Path("D:/Signlytic_AI/code/bsl_translation_project/data")
        print(f"\nSearching for .npy files in {base}...")
        
        npy_files = list(base.rglob("*.npy"))[:20]
        print(f"Found {len(npy_files)} .npy files")
        
        for f in npy_files[:10]:
            print(f"  {f.relative_to(base)}")
        
        if npy_files:
            print(f"\nInspecting first file: {npy_files[0].name}")
            data = np.load(npy_files[0])
            print(f"  Shape: {data.shape}")
            print(f"  Dtype: {data.dtype}")
        return
    
    print(f"Directory: {swin_dir}")
    
    # Count files
    npy_files = list(swin_dir.rglob("*.npy"))
    print(f"Total .npy files: {len(npy_files)}")
    
    if not npy_files:
        print("[FAIL] No .npy files found!")
        return
    
    # Check structure
    subdirs = set()
    for f in npy_files[:100]:
        rel = f.relative_to(swin_dir)
        if len(rel.parts) > 1:
            subdirs.add(rel.parts[0])
    
    if subdirs:
        print(f"Subdirectories: {sorted(subdirs)[:10]}")
    
    # Sample multiple files
    print("\nSampling features:")
    shapes = []
    
    for i, npy_file in enumerate(npy_files[:10]):
        try:
            data = np.load(npy_file)
            shapes.append(data.shape)
            print(f"  {npy_file.name}: shape={data.shape}, dtype={data.dtype}")
            
            if i == 0:
                print(f"    Min: {data.min():.4f}, Max: {data.max():.4f}")
                print(f"    Mean: {data.mean():.4f}, Std: {data.std():.4f}")
                
        except Exception as e:
            print(f"  {npy_file.name}: ERROR - {e}")
    
    # Analyze shapes
    print("\nShape analysis:")
    shape_counts = Counter(str(s) for s in shapes)
    for shape, count in shape_counts.items():
        print(f"  {shape}: {count} files")
    
    # Check if 2D temporal features
    if shapes:
        sample_shape = shapes[0]
        if len(sample_shape) == 2:
            T, D = sample_shape
            print(f"\n[OK] TEMPORAL FEATURES CONFIRMED")
            print(f"  T = {T} timesteps")
            print(f"  D = {D} feature dimensions")
        elif len(sample_shape) == 1:
            print(f"\n[WARN] 1D features detected (shape: {sample_shape})")
            print("  This might be pooled features, not temporal sequences")
        else:
            print(f"\n[WARN] Unexpected shape: {sample_shape}")


if __name__ == "__main__":
    main()
