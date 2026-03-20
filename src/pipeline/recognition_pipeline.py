"""
BSL Recognition + Translation Integration

Connects:
- SWIN Recognition Model (video features → glosses)
- Gloss → Text translation
- Text → Speech (Coqui TTS)

Usage:
    from src.pipeline.recognition_pipeline import BSLRecognitionPipeline
    
    pipeline = BSLRecognitionPipeline()
    result = pipeline.process_video_segment(features)
    # result = {'glosses': ['HELLO', 'HOW', 'YOU'], 'text': 'Hello, how are you?', 'audio_path': '...'}
"""

import sys
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# SWIN Recognition Model
# ============================================================

class TemporalTransformerEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.5,
        num_classes: int = 500
    ):
        super().__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 256, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        
        x = self.input_norm(x)
        x = self.input_proj(x)
        
        T_pos = min(T, self.pos_encoding.size(1))
        x[:, :T_pos] = x[:, :T_pos] + self.pos_encoding[:, :T_pos]
        
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
        
        x = self.norm(x)
        x = self.dropout(x)
        x = torch.clamp(x, -10, 10)
        
        return self.classifier(x)


# ============================================================
# Recognition Pipeline
# ============================================================

class SWINRecognizer:
    """SWIN-based sign language recognizer."""
    
    def __init__(
        self,
        model_path: str = None,
        vocab_path: str = None,
        device: str = "cuda"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Default paths
        project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
        if model_path is None:
            model_path = project_root / "models/swin_recognition_top500/best_model.pt"
        if vocab_path is None:
            vocab_path = project_root / "models/swin_recognition_top500/vocabulary.json"
        
        # Load vocabulary
        with open(vocab_path) as f:
            vocab = json.load(f)
        
        self.gloss_to_idx = vocab['gloss_to_idx']
        self.idx_to_gloss = {int(k): v for k, v in vocab['idx_to_gloss'].items()}
        self.num_classes = vocab['num_classes']
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = TemporalTransformerEncoder(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.max_seq_len = 64
        
        print(f"[SWINRecognizer] Loaded: {self.num_classes} classes")
        print(f"[SWINRecognizer] Device: {self.device}")
    
    def _preprocess(self, features: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess features for model input."""
        features = np.array(features, dtype=np.float32)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -50, 50)
        
        T, D = features.shape
        
        if T < self.max_seq_len:
            padding = np.zeros((self.max_seq_len - T, D), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
            mask = np.concatenate([
                np.zeros(T, dtype=bool),
                np.ones(self.max_seq_len - T, dtype=bool)
            ])
        else:
            features = features[:self.max_seq_len]
            mask = np.zeros(self.max_seq_len, dtype=bool)
        
        features = torch.from_numpy(features).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        
        return features, mask
    
    @torch.no_grad()
    def recognize(
        self,
        features: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Dict]:
        """
        Recognize sign from SWIN features.
        
        Args:
            features: (T, 768) SWIN feature array
            top_k: Number of top predictions to return
            threshold: Minimum probability threshold
            
        Returns:
            List of {'gloss': str, 'probability': float}
        """
        features, mask = self._preprocess(features)
        
        logits = self.model(features, mask)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            prob = float(probs[idx])
            if prob >= threshold:
                results.append({
                    'gloss': self.idx_to_gloss[idx],
                    'probability': prob
                })
        
        return results
    
    def recognize_sequence(
        self,
        features: np.ndarray,
        window_size: int = 16,
        stride: int = 8,
        top_k: int = 1,
        merge_threshold: float = 0.1
    ) -> List[Dict]:
        """
        Recognize sequence of signs using sliding window.
        
        Args:
            features: (T, 768) full video features
            window_size: Frames per window
            stride: Frames to advance
            top_k: Predictions per window
            merge_threshold: Merge consecutive same glosses
            
        Returns:
            List of {'gloss': str, 'start_frame': int, 'end_frame': int, 'probability': float}
        """
        T = len(features)
        predictions = []
        
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            window_features = features[start:end]
            
            results = self.recognize(window_features, top_k=top_k)
            if results:
                predictions.append({
                    'gloss': results[0]['gloss'],
                    'start_frame': start,
                    'end_frame': end,
                    'probability': results[0]['probability']
                })
        
        # Merge consecutive same glosses
        if not predictions:
            return []
        
        merged = [predictions[0]]
        for pred in predictions[1:]:
            if pred['gloss'] == merged[-1]['gloss']:
                merged[-1]['end_frame'] = pred['end_frame']
                merged[-1]['probability'] = max(merged[-1]['probability'], pred['probability'])
            else:
                merged.append(pred)
        
        return merged


# ============================================================
# Full Pipeline
# ============================================================

class BSLRecognitionPipeline:
    """
    End-to-end BSL Recognition Pipeline.
    
    Video Features → Glosses → Text → Speech
    """
    
    def __init__(
        self,
        recognizer_model: str = None,
        recognizer_vocab: str = None,
        gloss_to_text_model: str = None,
        tts_model: str = None,
        device: str = "cuda"
    ):
        self.device = device
        
        # Initialize recognizer
        print("[Pipeline] Loading SWIN recognizer...")
        self.recognizer = SWINRecognizer(
            model_path=recognizer_model,
            vocab_path=recognizer_vocab,
            device=device
        )
        
        # Gloss-to-text translator (placeholder - connect to existing model)
        self.gloss_to_text = None
        self._load_gloss_to_text(gloss_to_text_model)
        
        # TTS (placeholder - connect to Coqui)
        self.tts = None
        self._load_tts(tts_model)
        
        print("[Pipeline] Ready!")
    
    def _load_gloss_to_text(self, model_path: str = None):
        """Load gloss-to-text translation model."""
        try:
            from src.translation.gloss_to_text import GlossToTextTranslator
            project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
            if model_path is None:
                model_path = project_root / "models/gloss_to_text"
            self.gloss_to_text = GlossToTextTranslator(str(model_path))
            print("[Pipeline] Gloss-to-text loaded")
        except Exception as e:
            print(f"[Pipeline] Gloss-to-text not available: {e}")
            self.gloss_to_text = None
    
    def _load_tts(self, model_path: str = None):
        """Load TTS model."""
        try:
            from TTS.api import TTS
            self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            print("[Pipeline] TTS loaded")
        except Exception as e:
            print(f"[Pipeline] TTS not available: {e}")
            self.tts = None
    
    def process_segment(
        self,
        features: np.ndarray,
        top_k: int = 5
    ) -> Dict:
        """
        Process a single video segment.
        
        Args:
            features: (T, 768) SWIN features
            top_k: Number of recognition predictions
            
        Returns:
            {
                'predictions': [{'gloss': str, 'probability': float}, ...],
                'top_gloss': str,
                'text': str (if translator available),
                'audio_path': str (if TTS available)
            }
        """
        # Recognition
        predictions = self.recognizer.recognize(features, top_k=top_k)
        top_gloss = predictions[0]['gloss'] if predictions else "UNKNOWN"
        
        result = {
            'predictions': predictions,
            'top_gloss': top_gloss,
            'text': None,
            'audio_path': None
        }
        
        # Translation
        if self.gloss_to_text and predictions:
            try:
                text = self.gloss_to_text.translate([top_gloss])
                result['text'] = text
            except:
                pass
        
        # TTS
        if self.tts and result['text']:
            try:
                audio_path = f"/tmp/bsl_output_{hash(result['text'])}.wav"
                self.tts.tts_to_file(text=result['text'], file_path=audio_path)
                result['audio_path'] = audio_path
            except:
                pass
        
        return result
    
    def process_video(
        self,
        features: np.ndarray,
        window_size: int = 16,
        stride: int = 8
    ) -> Dict:
        """
        Process full video with sliding window.
        
        Returns:
            {
                'segments': [{'gloss': str, 'start_frame': int, ...}, ...],
                'gloss_sequence': ['HELLO', 'HOW', 'YOU'],
                'text': str,
                'audio_path': str
            }
        """
        # Recognize sequence
        segments = self.recognizer.recognize_sequence(
            features,
            window_size=window_size,
            stride=stride
        )
        
        gloss_sequence = [s['gloss'] for s in segments]
        
        result = {
            'segments': segments,
            'gloss_sequence': gloss_sequence,
            'text': None,
            'audio_path': None
        }
        
        # Translation
        if self.gloss_to_text and gloss_sequence:
            try:
                text = self.gloss_to_text.translate(gloss_sequence)
                result['text'] = text
            except:
                pass
        
        # TTS
        if self.tts and result['text']:
            try:
                audio_path = f"/tmp/bsl_video_output.wav"
                self.tts.tts_to_file(text=result['text'], file_path=audio_path)
                result['audio_path'] = audio_path
            except:
                pass
        
        return result


# ============================================================
# CLI Testing
# ============================================================

def test_pipeline():
    """Quick test of the pipeline."""
    import time
    
    print("="*70)
    print("BSL Recognition Pipeline Test")
    print("="*70)
    
    # Create dummy features
    print("\nCreating dummy features...")
    dummy_features = np.random.randn(32, 768).astype(np.float32)
    
    # Test recognizer only
    print("\nTesting recognizer...")
    t0 = time.time()
    recognizer = SWINRecognizer()
    load_time = time.time() - t0
    print(f"Load time: {load_time:.2f}s")
    
    t0 = time.time()
    results = recognizer.recognize(dummy_features, top_k=5)
    infer_time = time.time() - t0
    
    print(f"Inference time: {infer_time*1000:.1f}ms")
    print(f"Top predictions:")
    for r in results:
        print(f"  {r['gloss']}: {r['probability']*100:.2f}%")
    
    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)


if __name__ == "__main__":
    test_pipeline()
