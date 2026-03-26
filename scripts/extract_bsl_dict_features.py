"""
Extract SWIN features from BSL Dictionary videos.
Then train a BSL recognizer that works end-to-end.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_all_features():
    """Extract SWIN features from all BSL dictionary videos."""
    
    print("="*70)
    print("EXTRACTING SWIN FEATURES FROM BSL DICTIONARY")
    print("="*70)
    
    # Setup
    device = "cuda"
    video_dir = Path("data/videos/bsl_signs")
    output_dir = Path("data/bsl_dict_features")
    output_dir.mkdir(exist_ok=True)
    
    videos = list(video_dir.glob("*.mp4"))
    print(f"Videos to process: {len(videos)}")
    
    # Load SWIN model
    from torchvision.models.video import swin3d_t, Swin3D_T_Weights
    
    print("Loading Video-SWIN-T...")
    weights = Swin3D_T_Weights.DEFAULT
    model = swin3d_t(weights=weights)
    model.head = nn.Identity()
    model = model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Process videos
    gloss_to_features = {}
    
    for video_path in tqdm(videos, desc="Extracting features"):
        gloss = video_path.stem.lower()
        
        try:
            # Read frames
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            
            while len(frames) < 32:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frames.append(frame_resized)
            
            cap.release()
            
            if len(frames) == 0:
                continue
            
            # Pad if needed
            while len(frames) < 32:
                frames.append(frames[-1])
            
            frames = frames[:32]
            
            # To tensor: (B, C, T, H, W)
            frames_np = np.array(frames)
            frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 255.0
            frames_tensor = frames_tensor.unsqueeze(0).to(device)
            
            # Extract features
            with torch.no_grad():
                features = model(frames_tensor)
            
            gloss_to_features[gloss] = features.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"Error processing {video_path.name}: {e}")
    
    # Save features
    print(f"\nExtracted features for {len(gloss_to_features)} glosses")
    
    # Save as numpy arrays
    for gloss, feat in gloss_to_features.items():
        np.save(output_dir / f"{gloss}.npy", feat)
    
    # Save index
    with open(output_dir / "index.json", 'w') as f:
        json.dump(list(gloss_to_features.keys()), f)
    
    print(f"Saved to {output_dir}")
    return gloss_to_features


if __name__ == "__main__":
    extract_all_features()
