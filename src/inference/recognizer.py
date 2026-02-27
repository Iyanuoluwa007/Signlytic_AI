#!/usr/bin/env python3
"""
Inference pipeline for BSL sign recognition.

Loads trained model and predicts glosses from video features.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.transformer import TransformerClassifier
from data.datasets import Vocabulary


class BSLRecognizer:
    """BSL sign recognition inference pipeline."""
    
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        model_type: str = 'transformer',
        device: str = None,
    ):
        """
        Initialize recognizer.
        
        Args:
            model_path: Path to trained model checkpoint
            vocab_path: Path to vocabulary JSON file
            model_type: 'transformer', 'transformer_cls', or 'temporal_mlp'
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Load vocabulary
        self.vocab = Vocabulary.load(vocab_path)
        self.num_classes = len(self.vocab)
        
        # Load model
        self.model = self._load_model(model_path, model_type)
        self.model.eval()
        
        print(f"BSLRecognizer initialized")
        print(f"  Device: {self.device}")
        print(f"  Model: {model_type}")
        print(f"  Vocabulary: {self.num_classes} glosses")
    
    def _load_model(self, model_path: str, model_type: str) -> torch.nn.Module:
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Try to get model type from checkpoint config
        config = checkpoint.get('config', {})
        saved_model_type = config.get('model', model_type)
        
        if saved_model_type != model_type:
            print(f"  Note: Checkpoint is '{saved_model_type}', using that instead of '{model_type}'")
            model_type = saved_model_type
        
        # Determine model architecture
        if model_type == 'transformer':
            from models.transformer import TransformerClassifier
            model = TransformerClassifier(
                input_dim=768,
                num_classes=self.num_classes,
                d_model=config.get('d_model', 256),
                nhead=config.get('nhead', 8),
                num_layers=config.get('num_layers', 4),
                dim_feedforward=config.get('d_model', 256) * 4,
                dropout=0.0,
            )
        elif model_type == 'transformer_cls':
            from models.transformer import TransformerClassifierWithCLS
            model = TransformerClassifierWithCLS(
                input_dim=768,
                num_classes=self.num_classes,
                d_model=config.get('d_model', 256),
                nhead=config.get('nhead', 8),
                num_layers=config.get('num_layers', 4),
                dim_feedforward=config.get('d_model', 256) * 4,
                dropout=0.0,
            )
        elif model_type == 'temporal_mlp':
            from models.classifier import TemporalMLPClassifier
            model = TemporalMLPClassifier(
                input_dim=768,
                hidden_dim=config.get('hidden_dim', 512),
                num_classes=self.num_classes,
                dropout=0.0,
                pooling='attention',
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        return model
    
    def extract_window(
        self,
        features: np.ndarray,
        timestamp: float,
        window_seconds: float = 2.0,
        fps: float = 25.0,
        max_frames: int = 64,
    ) -> np.ndarray:
        """
        Extract temporal window around a timestamp.
        
        Args:
            features: Full video features (T, 768)
            timestamp: Center timestamp in seconds
            window_seconds: Window duration
            fps: Frame rate
            max_frames: Maximum frames to return
            
        Returns:
            Window features (max_frames, 768)
        """
        total_frames = features.shape[0]
        center_frame = int(timestamp * fps)
        half_window = int(window_seconds * fps / 2)
        
        start_frame = max(0, center_frame - half_window)
        end_frame = min(total_frames, center_frame + half_window)
        
        window = features[start_frame:end_frame]
        
        # Convert float16 to float32
        if window.dtype == np.float16:
            window = window.astype(np.float32)
        
        # Pad or sample to max_frames
        if len(window) < max_frames:
            padding = np.zeros((max_frames - len(window), 768), dtype=np.float32)
            window = np.concatenate([window, padding], axis=0)
        elif len(window) > max_frames:
            indices = np.linspace(0, len(window) - 1, max_frames, dtype=int)
            window = window[indices]
        
        return window
    
    @torch.no_grad()
    def predict(
        self,
        features: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Predict gloss from features.
        
        Args:
            features: Window features (max_frames, 768) or (768,)
            top_k: Number of top predictions to return
            
        Returns:
            List of (gloss, probability) tuples
        """
        # Ensure correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.ndim == 2:
            features = features[np.newaxis, ...]  # Add batch dim
        
        # Convert to tensor
        x = torch.from_numpy(features).float().to(self.device)
        
        # Forward pass
        logits = self.model(x)
        probs = F.softmax(logits, dim=-1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs[0], k=top_k)
        
        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            gloss = self.vocab.decode(int(idx))
            results.append((gloss, float(prob)))
        
        return results
    
    @torch.no_grad()
    def predict_video(
        self,
        feature_path: str,
        timestamps: List[float],
        top_k: int = 1,
    ) -> List[Dict]:
        """
        Predict glosses at multiple timestamps in a video.
        
        Args:
            feature_path: Path to video features (.npy file)
            timestamps: List of timestamps to predict at
            top_k: Number of top predictions per timestamp
            
        Returns:
            List of prediction dictionaries
        """
        # Load features
        features = np.load(feature_path, mmap_mode='r')
        
        results = []
        for ts in timestamps:
            window = self.extract_window(features, ts)
            predictions = self.predict(window, top_k=top_k)
            
            results.append({
                'timestamp': ts,
                'predictions': predictions,
                'top_gloss': predictions[0][0],
                'confidence': predictions[0][1],
            })
        
        return results
    
    @torch.no_grad()
    def recognize_sequence(
        self,
        feature_path: str,
        window_stride: float = 1.0,
        confidence_threshold: float = 0.5,
    ) -> List[str]:
        """
        Recognize sequence of glosses from video features.
        
        Args:
            feature_path: Path to video features
            window_stride: Stride between windows in seconds
            confidence_threshold: Minimum confidence to include prediction
            
        Returns:
            List of recognized glosses
        """
        features = np.load(feature_path, mmap_mode='r')
        
        if features.dtype == np.float16:
            total_frames = features.shape[0]
        else:
            total_frames = features.shape[0]
        
        fps = 25.0
        duration = total_frames / fps
        
        # Generate timestamps
        timestamps = np.arange(1.0, duration - 1.0, window_stride)
        
        # Predict at each timestamp
        glosses = []
        prev_gloss = None
        
        for ts in timestamps:
            window = self.extract_window(features, ts)
            predictions = self.predict(window, top_k=1)
            gloss, confidence = predictions[0]
            
            # Filter by confidence and avoid repeats
            if confidence >= confidence_threshold and gloss != prev_gloss:
                glosses.append(gloss)
                prev_gloss = gloss
        
        return glosses


def load_recognizer(
    model_path: str = 'outputs/best_model_transformer.pt',
    vocab_path: str = 'data/processed/vocabulary.json',
    model_type: str = 'transformer',
) -> BSLRecognizer:
    """Convenience function to load recognizer with default paths."""
    return BSLRecognizer(
        model_path=model_path,
        vocab_path=vocab_path,
        model_type=model_type,
    )