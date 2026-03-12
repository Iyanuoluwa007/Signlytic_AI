"""
Check SWIN features for NaN/Inf values and statistics.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm


def main():
    swin_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32")
    
    print("Checking SWIN features for issues...")
    print("="*60)
    
    npy_files = list(swin_dir.glob("*.npy"))
    print(f"Total files: {len(npy_files)}")
    
    issues = []
    stats = {
        'min_val': float('inf'),
        'max_val': float('-inf'),
        'nan_files': 0,
        'inf_files': 0,
        'extreme_files': 0
    }
    
    for npy_file in tqdm(npy_files, desc="Checking", ncols=80):
        try:
            data = np.load(npy_file, mmap_mode='r')
            
            # Sample some values (don't load entire file)
            sample_size = min(1000, data.shape[0])
            indices = np.random.choice(data.shape[0], sample_size, replace=False)
            sample = np.array(data[indices])
            
            # Check for NaN
            if np.isnan(sample).any():
                stats['nan_files'] += 1
                issues.append(f"NaN: {npy_file.name}")
            
            # Check for Inf
            if np.isinf(sample).any():
                stats['inf_files'] += 1
                issues.append(f"Inf: {npy_file.name}")
            
            # Track min/max
            sample_min = np.nanmin(sample)
            sample_max = np.nanmax(sample)
            
            stats['min_val'] = min(stats['min_val'], sample_min)
            stats['max_val'] = max(stats['max_val'], sample_max)
            
            # Check for extreme values
            if sample_max > 1000 or sample_min < -1000:
                stats['extreme_files'] += 1
                issues.append(f"Extreme: {npy_file.name} (min={sample_min:.1f}, max={sample_max:.1f})")
                
        except Exception as e:
            issues.append(f"Error: {npy_file.name} - {e}")
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Files with NaN: {stats['nan_files']}")
    print(f"Files with Inf: {stats['inf_files']}")
    print(f"Files with extreme values: {stats['extreme_files']}")
    print(f"Global min value: {stats['min_val']:.2f}")
    print(f"Global max value: {stats['max_val']:.2f}")
    
    if issues:
        print(f"\nFirst 20 issues:")
        for issue in issues[:20]:
            print(f"  {issue}")
    else:
        print("\nNo issues found!")
    
    # Also check dtype
    if npy_files:
        sample = np.load(npy_files[0], mmap_mode='r')
        print(f"\nDtype: {sample.dtype}")
        print(f"Shape: {sample.shape}")


if __name__ == "__main__":
    main()
