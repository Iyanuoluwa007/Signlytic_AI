"""
BSL Video Recognition Pipeline
Extracts SWIN features from video and recognizes BSL signs.

Author: Oke Iyanuoluwa Enoch
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# BSLRecognizer model
class BSLRecognizer(nn.Module):
    def __init__(self, num_classes, input_dim=768, hidden_dim=512, num_layers=4, num_heads=8, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
    
    def forward(self, x, mask=None):
        B, T, _ = x.shape
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :T, :]
        
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)
        
        if mask is not None:
            x = (x * mask.unsqueeze(-1).float()).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        else:
            x = x.mean(dim=1)
        
        return self.classifier(self.norm(x))


class VideoSWINExtractor:
    """Extract SWIN features from video frames."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.transform = None
        self.is_image_model = False
        self._load_model()
    
    def _load_model(self):
        try:
            from torchvision.models.video import swin3d_t, Swin3D_T_Weights
            
            logger.info("Loading Video-SWIN-T model...")
            weights = Swin3D_T_Weights.DEFAULT
            self.model = swin3d_t(weights=weights)
            self.model.head = nn.Identity()
            self.model = self.model.to(self.device)
            self.model.eval()
            self.transform = weights.transforms()
            self.is_image_model = False
            logger.info("Video-SWIN-T loaded!")
        except Exception as e:
            logger.warning(f"Video model failed: {e}, using image fallback")
            self._load_fallback()
    
    def _load_fallback(self):
        from torchvision.models import swin_t, Swin_T_Weights
        from torchvision import transforms
        
        logger.info("Loading SWIN-T (image) fallback...")
        weights = Swin_T_Weights.DEFAULT
        self.model = swin_t(weights=weights)
        self.model.head = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.is_image_model = True
        logger.info("SWIN-T (image) loaded!")
    
    def extract_frames(self, video_path: str, max_frames: int = 32) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError(f"No frames from {video_path}")
        
        while len(frames) < max_frames:
            frames.append(frames[-1])
        
        return np.array(frames[:max_frames])
    
    @torch.no_grad()
    def extract_features(self, video_path: str) -> torch.Tensor:
        frames = self.extract_frames(video_path, max_frames=32)
        
        if self.is_image_model:
            features = []
            for frame in frames:
                x = self.transform(frame).unsqueeze(0).to(self.device)
                feat = self.model(x)
                features.append(feat.cpu())
            return torch.cat(features, dim=0)
        else:
            # Video model expects (B, C, T, H, W)
            frames_resized = []
            for frame in frames:
                frame_resized = cv2.resize(frame, (224, 224))
                frames_resized.append(frame_resized)
            
            frames_np = np.array(frames_resized)
            frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float() / 255.0
            frames_tensor = frames_tensor.unsqueeze(0).to(self.device)
            
            features = self.model(frames_tensor)
            return features.cpu()


class BSLVideoRecognizer:
    """End-to-end BSL video recognition."""
    
    def __init__(self, model_path: str = "models/bsl_recognition/best_model.pt", device: str = "cuda"):
        self.device = device
        self.model_path = Path(model_path)
        
        self._load_bsl_model()
        self.extractor = VideoSWINExtractor(device=device)
        logger.info("BSL Video Recognizer ready!")
    
    def _load_bsl_model(self):
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        self.num_classes = checkpoint['num_classes']
        self.label_map = checkpoint['label_map']
        self.idx_to_label = checkpoint['idx_to_label']
        
        self.model = BSLRecognizer(num_classes=self.num_classes)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"BSL model loaded: {self.num_classes} classes")
    
    @torch.no_grad()
    def recognize(self, video_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Recognize BSL sign from video."""
        features = self.extractor.extract_features(video_path)
        
        # Shape handling
        if features.dim() == 1:
            features = features.unsqueeze(0).unsqueeze(0)
        elif features.dim() == 2:
            features = features.unsqueeze(0)
        
        T = features.shape[1]
        if T < 32:
            pad = torch.zeros(1, 32 - T, features.shape[2])
            features = torch.cat([features, pad], dim=1)
        elif T > 32:
            features = features[:, :32, :]
        
        mask = torch.ones(1, 32, dtype=torch.bool)
        mask[0, min(T, 32):] = False
        
        features = features.to(self.device)
        mask = mask.to(self.device)
        
        logits = self.model(features, mask)
        probs = torch.softmax(logits, dim=-1)
        
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        
        results = []
        for prob, idx in zip(top_probs[0].cpu().numpy(), top_indices[0].cpu().numpy()):
            gloss = self.idx_to_label[int(idx)]
            results.append((gloss, float(prob)))
        
        return results


def test_bsl_recognizer():
    print("="*70)
    print("TESTING BSL VIDEO RECOGNIZER")
    print("="*70)
    
    test_videos = list(Path("data/videos/bsl_signs").glob("*.mp4"))
    
    if not test_videos:
        print("No test videos found")
        return
    
    recognizer = BSLVideoRecognizer()
    
    for video_path in test_videos[:5]:
        print(f"\nVideo: {video_path.name} (expected: {video_path.stem})")
        try:
            results = recognizer.recognize(str(video_path), top_k=3)
            print("  Predictions:")
            for gloss, conf in results:
                print(f"    {gloss}: {conf*100:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    test_bsl_recognizer()
