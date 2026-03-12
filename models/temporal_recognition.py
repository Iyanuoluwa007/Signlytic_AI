"""
Temporal Sign Recognition Model

Uses SWIN video features with Transformer encoder for BSL gloss recognition.
Designed to work with BSL-1K annotations and optional Leap Motion supervision.

Architecture:
    SWIN Features (T x D)
    → Positional Encoding
    → Transformer Encoder
    → Temporal Attention Pooling
    → Gloss Classification Head
    → (Optional) Hand Shape Auxiliary Head

Key Features:
    - Temporal modeling of sign sequences
    - Multi-resolution temporal attention
    - Confidence calibration for stable predictions
    - Compatible with weak supervision (BSL-1K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalAttentionPooling(nn.Module):
    """Attention-based pooling over temporal dimension."""
    
    def __init__(self, d_model: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
            mask: Optional padding mask
        Returns:
            Pooled tensor of shape (batch, d_model)
        """
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        
        attn_output, _ = self.attention(query, x, x, key_padding_mask=mask)
        return attn_output.squeeze(1)


class MultiScaleTemporalBlock(nn.Module):
    """Multi-scale temporal convolutions for capturing different temporal granularities."""
    
    def __init__(self, d_model: int, kernel_sizes: list = [3, 5, 7]):
        super().__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model // len(kernel_sizes), kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        # Conv1d expects (batch, channels, seq_len)
        x_t = x.transpose(1, 2)
        
        outputs = [conv(x_t) for conv in self.convs]
        combined = torch.cat(outputs, dim=1)
        
        # Back to (batch, seq_len, d_model)
        combined = combined.transpose(1, 2)
        
        return self.norm(combined + x)


class HandShapeHead(nn.Module):
    """Auxiliary head for hand shape classification (Leap Motion supervision)."""
    
    def __init__(self, d_model: int, num_handshapes: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, num_handshapes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TemporalSignRecognitionModel(nn.Module):
    """
    Main recognition model for BSL signs using temporal SWIN features.
    
    Args:
        feature_dim: Dimension of SWIN features (typically 768 or 1024)
        num_classes: Number of gloss classes
        num_layers: Number of transformer encoder layers
        num_heads: Number of attention heads
        d_model: Internal model dimension
        dropout: Dropout rate
        use_handshape_head: Whether to include auxiliary hand shape prediction
        num_handshapes: Number of hand shape classes (for Leap Motion)
    """
    
    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 1000,
        num_layers: int = 4,
        num_heads: int = 8,
        d_model: int = 512,
        dropout: float = 0.2,
        use_handshape_head: bool = False,
        num_handshapes: int = 64,
        use_multiscale: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Optional multi-scale temporal block
        self.use_multiscale = use_multiscale
        if use_multiscale:
            self.multiscale = MultiScaleTemporalBlock(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Temporal pooling
        self.temporal_pool = TemporalAttentionPooling(d_model, num_heads=4)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
        # Optional handshape auxiliary head
        self.use_handshape_head = use_handshape_head
        if use_handshape_head:
            self.handshape_head = HandShapeHead(d_model, num_handshapes)
        
        # Temperature for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: SWIN features of shape (batch, seq_len, feature_dim)
            mask: Padding mask of shape (batch, seq_len), True for padded positions
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing:
                - logits: Classification logits (batch, num_classes)
                - probs: Calibrated probabilities
                - handshape_logits: Optional handshape predictions
                - features: Optional intermediate features
        """
        # Input projection
        x = self.input_proj(features)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Multi-scale temporal processing
        if self.use_multiscale:
            x = self.multiscale(x)
        
        # Transformer encoding
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=mask)
        else:
            x = self.transformer(x)
        
        # Temporal pooling
        pooled = self.temporal_pool(x, mask)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Temperature-scaled probabilities
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        output = {
            'logits': logits,
            'probs': probs,
            'pooled_features': pooled
        }
        
        # Optional handshape prediction
        if self.use_handshape_head:
            output['handshape_logits'] = self.handshape_head(pooled)
        
        if return_features:
            output['encoded_features'] = x
        
        return output
    
    def predict(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Inference with top-k predictions.
        
        Returns:
            Dictionary with top_k predictions and confidences
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(features, mask)
            probs = output['probs']
            
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            return {
                'top_indices': top_indices,
                'top_probs': top_probs,
                'all_probs': probs
            }


class TemporalSmoothing(nn.Module):
    """
    Temporal smoothing for stable predictions across frames.
    Uses exponential moving average with learnable decay.
    """
    
    def __init__(self, num_classes: int, initial_alpha: float = 0.7):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(initial_alpha))
        self.register_buffer('prev_probs', torch.zeros(1, num_classes))
    
    def forward(self, probs: torch.Tensor, reset: bool = False) -> torch.Tensor:
        """
        Apply temporal smoothing.
        
        Args:
            probs: Current probabilities (batch, num_classes)
            reset: Reset smoothing state
        """
        if reset or self.prev_probs.size(0) != probs.size(0):
            self.prev_probs = probs.detach()
            return probs
        
        alpha = torch.sigmoid(self.alpha)
        smoothed = alpha * probs + (1 - alpha) * self.prev_probs
        self.prev_probs = smoothed.detach()
        
        return smoothed


class RecognitionModelWithSmoothing(nn.Module):
    """Wrapper that adds temporal smoothing to the base model."""
    
    def __init__(self, base_model: TemporalSignRecognitionModel):
        super().__init__()
        self.model = base_model
        self.smoother = TemporalSmoothing(base_model.num_classes)
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        apply_smoothing: bool = True,
        reset_smoothing: bool = False
    ) -> Dict[str, torch.Tensor]:
        output = self.model(features, mask)
        
        if apply_smoothing:
            output['smoothed_probs'] = self.smoother(output['probs'], reset=reset_smoothing)
        
        return output


def create_model(
    feature_dim: int = 768,
    num_classes: int = 1000,
    use_handshape: bool = False,
    pretrained_path: Optional[str] = None
) -> TemporalSignRecognitionModel:
    """
    Factory function to create the recognition model.
    
    Args:
        feature_dim: SWIN feature dimension
        num_classes: Number of gloss classes
        use_handshape: Whether to use handshape auxiliary head
        pretrained_path: Path to pretrained weights
    """
    model = TemporalSignRecognitionModel(
        feature_dim=feature_dim,
        num_classes=num_classes,
        num_layers=4,
        num_heads=8,
        d_model=512,
        dropout=0.2,
        use_handshape_head=use_handshape,
        use_multiscale=True
    )
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return model


# Testing
if __name__ == "__main__":
    print("Testing TemporalSignRecognitionModel...")
    
    # Create model
    model = create_model(
        feature_dim=768,
        num_classes=1000,
        use_handshape=True
    )
    
    # Test input (batch=4, seq_len=50, features=768)
    x = torch.randn(4, 50, 768)
    mask = torch.zeros(4, 50, dtype=torch.bool)
    mask[:, 45:] = True  # Pad last 5 frames
    
    # Forward pass
    output = model(x, mask, return_features=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Probs shape: {output['probs'].shape}")
    print(f"Pooled features shape: {output['pooled_features'].shape}")
    
    if 'handshape_logits' in output:
        print(f"Handshape logits shape: {output['handshape_logits'].shape}")
    
    # Test inference
    pred = model.predict(x, mask, top_k=5)
    print(f"\nTop-5 predictions shape: {pred['top_indices'].shape}")
    print(f"Top-5 probs shape: {pred['top_probs'].shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n[OK] Model test passed!")
