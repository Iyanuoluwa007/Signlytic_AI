"""
Check the actual format of extracted pose files.
"""

import json
from pathlib import Path

poses_dir = Path("D:/Signlytic_AI/code/bsl_translation_project/data/poses")

# Find a sample file
sample_file = None
for split in ['train', 'val', 'test']:
    split_dir = poses_dir / split
    if split_dir.exists():
        files = list(split_dir.glob("*.json"))
        if files:
            sample_file = files[0]
            break

if sample_file:
    print(f"Sample file: {sample_file}")
    print(f"Size: {sample_file.stat().st_size / 1024:.1f} KB")
    
    with open(sample_file, 'r') as f:
        data = json.load(f)
    
    print(f"\nTop-level keys: {list(data.keys())}")
    print(f"Gloss: {data.get('gloss')}")
    print(f"Num frames: {data.get('num_frames')}")
    
    poses = data.get('poses', [])
    print(f"\nPoses type: {type(poses)}")
    print(f"Poses length: {len(poses)}")
    
    if poses:
        print(f"\nFirst pose type: {type(poses[0])}")
        
        if isinstance(poses[0], dict):
            print(f"First pose keys: {list(poses[0].keys())}")
            
            # Check 'pose' key (body keypoints)
            body = poses[0].get('pose', [])
            print(f"\nBody ('pose' key) type: {type(body)}")
            print(f"Body length: {len(body)}")
            
            if body:
                print(f"First body keypoint type: {type(body[0])}")
                print(f"First body keypoint: {body[0]}")
            
            # Check left_hand
            left = poses[0].get('left_hand', [])
            print(f"\nLeft hand type: {type(left)}")
            print(f"Left hand length: {len(left)}")
            if left:
                print(f"First left keypoint: {left[0]}")
            
            # Check right_hand
            right = poses[0].get('right_hand', [])
            print(f"\nRight hand type: {type(right)}")
            print(f"Right hand length: {len(right)}")
            if right:
                print(f"First right keypoint: {right[0]}")
        
        elif isinstance(poses[0], list):
            print(f"First pose length: {len(poses[0])}")
            print(f"First pose sample: {poses[0][:10]}...")
else:
    print("No pose files found!")