"""
BSL AI End-to-End Demo

Demonstrates the full pipeline:
1. Load pre-extracted SWIN features
2. Recognize signs using Top500 model
3. Translate glosses to English text
4. (Optional) Generate speech

Usage:
    python scripts/demo_pipeline.py
    python scripts/demo_pipeline.py --video_id 1234567890
    python scripts/demo_pipeline.py --interactive
"""

import sys
from pathlib import Path
import argparse
import json
import time
import random

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


# ============================================================
# Model Definition (same as training)
# ============================================================

class TemporalTransformerEncoder(torch.nn.Module):
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
        
        self.input_norm = torch.nn.LayerNorm(input_dim)
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.pos_encoding = torch.nn.Parameter(torch.zeros(1, 256, d_model))
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = torch.nn.LayerNorm(d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(d_model, num_classes)
    
    def forward(self, x, mask=None):
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
# Demo Pipeline
# ============================================================

class BSLDemo:
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.project_root = Path("D:/Signlytic_AI/code/bsl_translation_project")
        self.model_dir = self.project_root / "models/swin_recognition_top500"
        self.swin_dir = self.project_root / "data/processed/features/bobsl/v1.4/video_features/swin_v1/video-swin-s_c8697_16f_bs32"
        
        # Load components
        self._load_vocabulary()
        self._load_recognition_model()
        self._load_gloss_to_text()
        
        print(f"\n[Demo Ready] Device: {self.device}")
    
    def _load_vocabulary(self):
        """Load Top500 vocabulary."""
        print("[Loading] Vocabulary...")
        vocab_path = self.model_dir / "vocabulary.json"
        with open(vocab_path) as f:
            vocab = json.load(f)
        
        self.gloss_to_idx = vocab['gloss_to_idx']
        self.idx_to_gloss = {int(k): v for k, v in vocab['idx_to_gloss'].items()}
        self.num_classes = vocab['num_classes']
        print(f"  -> {self.num_classes} glosses loaded")
    
    def _load_recognition_model(self):
        """Load SWIN recognition model."""
        print("[Loading] Recognition model...")
        model_path = self.model_dir / "best_model.pt"
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.model = TemporalTransformerEncoder(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  -> Top-1: {100*checkpoint['best_val_acc']:.2f}%, Top-5: {100*checkpoint['best_val_top5']:.2f}%")
    
    def _load_gloss_to_text(self):
        """Load gloss-to-text translation."""
        print("[Loading] Gloss-to-text translator...")
        try:
            from src.translation.gloss_to_text import GlossToTextTranslator
            model_path = self.project_root / "models/gloss_to_text"
            self.translator = GlossToTextTranslator(str(model_path))
            print("  -> Translator loaded")
        except Exception as e:
            print(f"  -> Translator not available: {e}")
            self.translator = None
    
    def get_available_videos(self, limit=10):
        """Get list of available video IDs."""
        videos = list(self.swin_dir.glob("*.npy"))[:limit]
        return [v.stem for v in videos]
    
    def load_features(self, video_id: str) -> np.ndarray:
        """Load SWIN features for a video."""
        feature_path = self.swin_dir / f"{video_id}.npy"
        if not feature_path.exists():
            raise FileNotFoundError(f"Features not found: {feature_path}")
        
        features = np.load(feature_path)
        return features.astype(np.float32)
    
    @torch.no_grad()
    def recognize_segment(self, features: np.ndarray, top_k: int = 5):
        """Recognize sign from features."""
        # Preprocess
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = np.clip(features, -50, 50)
        
        T, D = features.shape
        max_len = 64
        
        if T < max_len:
            padding = np.zeros((max_len - T, D), dtype=np.float32)
            features = np.concatenate([features, padding], axis=0)
            mask = np.concatenate([np.zeros(T, dtype=bool), np.ones(max_len - T, dtype=bool)])
        else:
            features = features[:max_len]
            mask = np.zeros(max_len, dtype=bool)
        
        # To tensor
        features_t = torch.from_numpy(features).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        
        # Inference
        logits = self.model(features_t, mask_t)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        # Top-k
        top_indices = np.argsort(probs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                'gloss': self.idx_to_gloss[idx],
                'probability': float(probs[idx])
            })
        
        return results
    
    def recognize_video(self, video_id: str, window_size: int = 16, stride: int = 8):
        """Recognize signs in a full video using sliding window."""
        features = self.load_features(video_id)
        T = len(features)
        
        print(f"\n[Video: {video_id}]")
        print(f"  Frames: {T}")
        print(f"  Window: {window_size}, Stride: {stride}")
        
        predictions = []
        
        for start in range(0, T - window_size + 1, stride):
            end = start + window_size
            segment = features[start:end]
            
            results = self.recognize_segment(segment, top_k=1)
            if results:
                predictions.append({
                    'gloss': results[0]['gloss'],
                    'probability': results[0]['probability'],
                    'start': start,
                    'end': end
                })
        
        # Merge consecutive same glosses
        if not predictions:
            return []
        
        merged = [predictions[0]]
        for pred in predictions[1:]:
            if pred['gloss'] == merged[-1]['gloss']:
                merged[-1]['end'] = pred['end']
                merged[-1]['probability'] = max(merged[-1]['probability'], pred['probability'])
            else:
                merged.append(pred)
        
        return merged
    
    def translate(self, glosses: list) -> str:
        """Translate glosses to English text."""
        if self.translator is None:
            # Simple fallback
            return " ".join(glosses).lower().replace("_", " ")
        
        try:
            return self.translator.translate(glosses)
        except:
            return " ".join(glosses).lower().replace("_", " ")
    
    def demo_single_video(self, video_id: str = None):
        """Run demo on a single video."""
        if video_id is None:
            videos = self.get_available_videos(limit=100)
            video_id = random.choice(videos)
        
        print("\n" + "="*70)
        print("BSL AI RECOGNITION DEMO")
        print("="*70)
        
        # Recognize
        t0 = time.time()
        predictions = self.recognize_video(video_id)
        recog_time = time.time() - t0
        
        print(f"\n[Recognition] ({recog_time*1000:.0f}ms)")
        print("-"*50)
        
        if not predictions:
            print("  No signs detected")
            return
        
        gloss_sequence = [p['gloss'] for p in predictions]
        
        for i, pred in enumerate(predictions):
            print(f"  {i+1}. {pred['gloss']:<15} ({pred['probability']*100:5.1f}%) "
                  f"[frames {pred['start']}-{pred['end']}]")
        
        print(f"\n[Gloss Sequence]")
        print(f"  {' -> '.join(gloss_sequence)}")
        
        # Translate
        t0 = time.time()
        text = self.translate(gloss_sequence)
        trans_time = time.time() - t0
        
        print(f"\n[Translation] ({trans_time*1000:.0f}ms)")
        print(f"  \"{text}\"")
        
        print("\n" + "="*70)
        
        return {
            'video_id': video_id,
            'predictions': predictions,
            'gloss_sequence': gloss_sequence,
            'text': text
        }
    
    def demo_random_segments(self, n_segments: int = 5):
        """Demo on random segments from different videos."""
        print("\n" + "="*70)
        print("BSL AI - RANDOM SEGMENT DEMO")
        print("="*70)
        
        videos = self.get_available_videos(limit=100)
        
        for i in range(n_segments):
            video_id = random.choice(videos)
            features = self.load_features(video_id)
            T = len(features)
            
            # Random segment
            start = random.randint(0, max(0, T - 32))
            end = min(start + 32, T)
            segment = features[start:end]
            
            t0 = time.time()
            results = self.recognize_segment(segment, top_k=5)
            infer_time = time.time() - t0
            
            print(f"\n[Segment {i+1}] Video: {video_id}, Frames: {start}-{end}")
            print(f"  Inference: {infer_time*1000:.1f}ms")
            print(f"  Predictions:")
            for r in results[:3]:
                print(f"    {r['gloss']:<15} {r['probability']*100:5.2f}%")
        
        print("\n" + "="*70)
    
    def interactive_mode(self):
        """Interactive demo mode."""
        print("\n" + "="*70)
        print("BSL AI - INTERACTIVE MODE")
        print("="*70)
        print("\nCommands:")
        print("  list     - Show available videos")
        print("  demo     - Demo random video")
        print("  <id>     - Demo specific video ID")
        print("  random   - Demo random segments")
        print("  quit     - Exit")
        print("")
        
        while True:
            try:
                cmd = input("BSL> ").strip().lower()
                
                if cmd == "quit" or cmd == "exit":
                    print("Goodbye!")
                    break
                elif cmd == "list":
                    videos = self.get_available_videos(limit=20)
                    print("Available videos:")
                    for v in videos:
                        print(f"  {v}")
                elif cmd == "demo":
                    self.demo_single_video()
                elif cmd == "random":
                    self.demo_random_segments()
                elif cmd:
                    try:
                        self.demo_single_video(cmd)
                    except FileNotFoundError:
                        print(f"Video not found: {cmd}")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


# ============================================================
# Metrics Summary
# ============================================================

def print_metrics_summary():
    """Print portfolio metrics summary."""
    print("\n" + "="*70)
    print("BSL AI SYSTEM - PERFORMANCE METRICS")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │  COMPONENT                    │  METRIC           │  VALUE     │
    ├─────────────────────────────────────────────────────────────────┤
    │  SWIN Recognition (Top500)    │  Top-1 Accuracy   │  2.36%     │
    │                               │  Top-5 Accuracy   │  9.16%     │
    │                               │  Classes          │  500       │
    │                               │  vs Random        │  ~460x     │
    ├─────────────────────────────────────────────────────────────────┤
    │  SWIN Recognition (Full)      │  Top-1 Accuracy   │  1.6%      │
    │                               │  Top-5 Accuracy   │  6.2%      │
    │                               │  Classes          │  22,113    │
    ├─────────────────────────────────────────────────────────────────┤
    │  Pose Recognition (Existing)  │  Top-5 Accuracy   │  74.5%     │
    │                               │  Classes          │  552       │
    │                               │  Inference Time   │  0.46ms    │
    ├─────────────────────────────────────────────────────────────────┤
    │  Gloss → Text Translation     │  ROUGE-L          │  92%       │
    │                               │  BLEU             │  0.60      │
    │                               │  WER              │  16%       │
    ├─────────────────────────────────────────────────────────────────┤
    │  Text → Gloss Translation     │  Token Accuracy   │  71.7%     │
    │                               │  Coverage         │  97.5%     │
    ├─────────────────────────────────────────────────────────────────┤
    │  Dataset                      │  Annotations      │  5.9M      │
    │                               │  Videos           │  1,940     │
    │                               │  Pose Sequences   │  34,437    │
    └─────────────────────────────────────────────────────────────────┘
    
    Author: Oke Iyanuoluwa Enoch
    Affiliation: Independent Robotics & AI Systems Engineer
    """)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BSL AI Demo")
    parser.add_argument("--video_id", type=str, default=None, help="Specific video ID")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--random", action="store_true", help="Demo random segments")
    parser.add_argument("--metrics", action="store_true", help="Show metrics summary")
    args = parser.parse_args()
    
    if args.metrics:
        print_metrics_summary()
        return
    
    # Initialize demo
    demo = BSLDemo()
    
    if args.interactive:
        demo.interactive_mode()
    elif args.random:
        demo.demo_random_segments(n_segments=5)
    else:
        demo.demo_single_video(video_id=args.video_id)


if __name__ == "__main__":
    main()
