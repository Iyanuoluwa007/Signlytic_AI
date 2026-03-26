"""
BSL Dictionary Recognizer - Retrieval-Based
Uses nearest neighbor matching instead of classification.
Works perfectly for dictionary data with 1 sample per class.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
import json
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BSLDictRecognizer:
    """
    Retrieval-based BSL recognizer.
    Finds closest match using cosine similarity.
    """
    
    def __init__(
        self,
        features_dir: str = "data/bsl_dict_features",
        device: str = "cuda"
    ):
        self.device = device
        self.features_dir = Path(features_dir)
        
        # Load all pre-extracted features
        self._load_features()
        
        # Load SWIN model for new video extraction
        self._load_swin()
        
        logger.info(f"BSL Dict Recognizer ready: {len(self.glosses)} signs")
    
    def _load_features(self):
        """Load all pre-extracted features."""
        with open(self.features_dir / "index.json", 'r') as f:
            self.glosses = json.load(f)
        
        features_list = []
        valid_glosses = []
        
        for gloss in self.glosses:
            feat_path = self.features_dir / f"{gloss}.npy"
            if feat_path.exists():
                feat = np.load(feat_path).squeeze()
                features_list.append(feat)
                valid_glosses.append(gloss)
        
        self.glosses = valid_glosses
        self.features = np.stack(features_list)  # (N, 768)
        self.features_tensor = torch.from_numpy(self.features).float().to(self.device)
        
        # Normalize for cosine similarity
        self.features_norm = self.features_tensor / self.features_tensor.norm(dim=1, keepdim=True)
        
        logger.info(f"Loaded {len(self.glosses)} sign features")
    
    def _load_swin(self):
        """Load SWIN model for feature extraction."""
        from torchvision.models.video import swin3d_t, Swin3D_T_Weights
        
        logger.info("Loading Video-SWIN-T...")
        weights = Swin3D_T_Weights.DEFAULT
        self.swin = swin3d_t(weights=weights)
        self.swin.head = nn.Identity()
        self.swin = self.swin.to(self.device)
        self.swin.eval()
        logger.info("SWIN loaded!")
    
    def _extract_features(self, video_path: str) -> torch.Tensor:
        """Extract features from a video."""
        cap = cv2.VideoCapture(video_path)
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
            raise ValueError(f"No frames from {video_path}")
        
        while len(frames) < 32:
            frames.append(frames[-1])
        
        frames_np = np.array(frames[:32])
        frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 255.0
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.swin(frames_tensor)
        
        return features.squeeze()
    
    def recognize(self, video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Recognize BSL sign from video using cosine similarity.
        
        Returns:
            List of (gloss, similarity_score) tuples
        """
        # Extract features from input video
        query_feat = self._extract_features(video_path)
        query_norm = query_feat / query_feat.norm()
        
        # Compute cosine similarity with all stored features
        similarities = torch.matmul(self.features_norm, query_norm)
        
        # Get top-k matches
        top_scores, top_indices = similarities.topk(top_k)
        
        results = []
        for score, idx in zip(top_scores.cpu().numpy(), top_indices.cpu().numpy()):
            gloss = self.glosses[idx]
            results.append((gloss, float(score)))
        
        return results


def test_bsl_dict_recognizer():
    """Test the retrieval-based recognizer."""
    print("="*70)
    print("TESTING BSL DICTIONARY RECOGNIZER (RETRIEVAL)")
    print("="*70)
    
    recognizer = BSLDictRecognizer()
    
    # Test on dictionary videos
    video_dir = Path("data/videos/bsl_signs")
    test_videos = list(video_dir.glob("*.mp4"))[:50]
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for video_path in test_videos:
        expected = video_path.stem.lower()
        
        try:
            results = recognizer.recognize(str(video_path), top_k=5)
            predictions = [r[0] for r in results]
            
            is_top1 = predictions[0] == expected
            is_top5 = expected in predictions
            
            correct_top1 += int(is_top1)
            correct_top5 += int(is_top5)
            total += 1
            
            status = "[OK]" if is_top1 else ("[TOP5]" if is_top5 else "[MISS]")
            print(f"{status} {expected} -> {predictions[0]} ({results[0][1]*100:.1f}%)")
            
        except Exception as e:
            print(f"[ERR] {expected}: {e}")
    
    print(f"\n" + "="*70)
    print(f"RESULTS: {total} videos")
    print(f"  Top-1 Accuracy: {correct_top1}/{total} = {correct_top1/total*100:.1f}%")
    print(f"  Top-5 Accuracy: {correct_top5}/{total} = {correct_top5/total*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    test_bsl_dict_recognizer()
