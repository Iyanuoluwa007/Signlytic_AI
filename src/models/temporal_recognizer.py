"""
Temporal Sign Recognition Model (V2)

Replaces the original Transformer classifier with a proper temporal model
that processes SWIN feature sequences (T, 768) through:

  1. Input projection: 768 -> d_model
  2. Positional encoding (sinusoidal)
  3. Temporal encoder (Transformer or BiLSTM)
  4. Temporal pooling (attention, CLS token, mean, or max)
  5. Classification head -> num_glosses
  6. Auxiliary hand branch (optional, for Leap Motion supervision)

The model treats each .npy feature file as a temporal sequence,
not a single-frame feature vector.

Compatible with:
  - Existing 2D pose renderer output
  - Future Blender avatar pipeline (same gloss predictions)
  - Future SignAvatars motion generation

Usage:
    model = TemporalRecognizer(
        feature_dim=768,
        num_classes=1843,
        d_model=512,
        nhead=8,
        num_layers=6,
        pooling="attention",
        hand_branch=True,
        hand_feature_dim=63,
    )
    logits, hand_pred = model(features)  # features: (B, T, 768)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TemporalAttentionPooling(nn.Module):
    """
    Attention-based pooling over the temporal dimension.
    Learns which frames are most informative for classification.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len) bool mask, True = valid, False = padding

        Returns:
            (batch, d_model) pooled representation
        """
        attn_weights = self.attention(x).squeeze(-1)  # (batch, seq_len)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)  # (batch, seq_len)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # (batch, d_model)
        return pooled


class HandBranch(nn.Module):
    """
    Auxiliary branch for hand shape prediction.
    Supervised by Leap Motion hand templates.
    """

    def __init__(self, d_model: int, hand_feature_dim: int = 63, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hand_feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, d_model) -> (batch, hand_feature_dim)"""
        return self.net(x)


class TemporalRecognizer(nn.Module):
    """
    Full temporal sign recognition model.

    Architecture:
        Input (B, T, 768)
        -> Linear projection (768 -> d_model)
        -> Positional encoding
        -> [CLS] token prepend (if using cls_token pooling)
        -> Temporal encoder (Transformer or BiLSTM)
        -> Temporal pooling
        -> Classification head -> (B, num_classes)
        -> [Optional] Hand branch -> (B, hand_dim)
    """

    def __init__(
        self,
        feature_dim: int = 768,
        num_classes: int = 1843,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.2,
        max_seq_len: int = 512,
        encoder_type: str = "transformer",  # "transformer" or "bilstm" or "hybrid"
        pooling: str = "attention",          # "attention", "cls_token", "mean", "max"
        # Hand branch
        hand_branch: bool = True,
        hand_feature_dim: int = 63,
        hand_hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.num_classes = num_classes
        self.pooling_type = pooling
        self.encoder_type = encoder_type
        self.use_hand_branch = hand_branch

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

        # CLS token (if using cls_token pooling)
        if pooling == "cls_token":
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Temporal encoder
        if encoder_type in ["transformer", "temporal_transformer"]:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-norm for more stable training
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
        elif encoder_type == "bilstm":
            self.encoder = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        elif encoder_type == "hybrid":
            # BiLSTM first for local patterns, then Transformer for global
            self.lstm = nn.LSTM(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=max(1, num_layers - 2)
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        # Temporal pooling
        if pooling == "attention":
            self.pool = TemporalAttentionPooling(d_model)
        elif pooling in ("mean", "max", "cls_token"):
            self.pool = None  # Handled in forward()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # Layer norm before classification
        self.pre_head_norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Auxiliary hand branch
        if hand_branch:
            self.hand_head = HandBranch(d_model, hand_feature_dim, hand_hidden_dim)
        else:
            self.hand_head = None

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "encoder" not in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: (batch, seq_len, feature_dim) SWIN temporal features
            mask: (batch, seq_len) bool mask, True=valid, False=padding

        Returns:
            logits: (batch, num_classes)
            hand_pred: (batch, hand_feature_dim) or None
        """
        B, T, _ = x.shape

        # Create padding mask from zero features if not provided
        if mask is None:
            mask = (x.abs().sum(dim=-1) > 0)  # (B, T)

        # Input projection
        x = self.input_proj(x)  # (B, T, d_model)

        # Prepend CLS token if needed
        if self.pooling_type == "cls_token":
            cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls, x], dim=1)  # (B, T+1, d_model)
            mask = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=mask.device), mask], dim=1
            )

        # Positional encoding
        x = self.pos_encoder(x)

        # Temporal encoding
        if self.encoder_type == "transformer":
            # Create attention mask (True = ignore for PyTorch Transformer)
            src_key_padding_mask = ~mask
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        elif self.encoder_type == "bilstm":
            x, _ = self.encoder(x)

        elif self.encoder_type == "hybrid":
            x, _ = self.lstm(x)
            src_key_padding_mask = ~mask
            x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Temporal pooling
        if self.pooling_type == "attention":
            pooled = self.pool(x, mask)
        elif self.pooling_type == "cls_token":
            pooled = x[:, 0, :]  # CLS token output
        elif self.pooling_type == "mean":
            # Masked mean
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        elif self.pooling_type == "max":
            x_masked = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
            pooled = x_masked.max(dim=1)[0]
        else:
            pooled = x.mean(dim=1)

        # Normalize
        pooled = self.pre_head_norm(pooled)

        # Classification
        logits = self.classifier(pooled)

        # Hand branch
        hand_pred = None
        if self.hand_head is not None:
            hand_pred = self.hand_head(pooled)

        return logits, hand_pred


class TemporalSmoother:
    """
    Post-prediction temporal smoothing for inference.
    Stabilizes gloss predictions over consecutive frames.
    """

    def __init__(
        self,
        method: str = "ema",
        ema_alpha: float = 0.7,
        median_window: int = 5,
        min_confidence: float = 0.3,
    ):
        self.method = method
        self.ema_alpha = ema_alpha
        self.median_window = median_window
        self.min_confidence = min_confidence
        self._history: list = []
        self._ema_state: Optional[torch.Tensor] = None

    def reset(self):
        self._history.clear()
        self._ema_state = None

    def smooth(self, logits: torch.Tensor) -> Tuple[int, float]:
        """
        Apply temporal smoothing to a single frame's logits.

        Args:
            logits: (num_classes,) raw logits for one frame

        Returns:
            (predicted_class, confidence)
        """
        probs = F.softmax(logits, dim=-1)

        if self.method == "ema":
            if self._ema_state is None:
                self._ema_state = probs.clone()
            else:
                self._ema_state = (
                    self.ema_alpha * probs + (1 - self.ema_alpha) * self._ema_state
                )
            smoothed = self._ema_state

        elif self.method == "median_filter":
            self._history.append(probs)
            if len(self._history) > self.median_window:
                self._history.pop(0)
            stacked = torch.stack(self._history, dim=0)
            smoothed = stacked.median(dim=0)[0]

        else:
            smoothed = probs

        pred = smoothed.argmax().item()
        conf = smoothed[pred].item()

        if conf < self.min_confidence:
            return -1, conf  # -1 = uncertain / no prediction

        return pred, conf

    def smooth_sequence(self, logits_seq: torch.Tensor) -> list:
        """
        Smooth a full sequence of predictions.

        Args:
            logits_seq: (T, num_classes)

        Returns:
            List of (predicted_class, confidence) tuples
        """
        self.reset()
        results = []
        for t in range(logits_seq.shape[0]):
            pred, conf = self.smooth(logits_seq[t])
            results.append((pred, conf))
        return results


def build_model(config: dict) -> TemporalRecognizer:
    """Build model from config dict (matches recognition_v2.yaml)."""
    model_cfg = config.get("model", {})
    hand_cfg = model_cfg.get("hand_branch", {})
    swin_cfg = config.get("swin_features", {})

    return TemporalRecognizer(
        feature_dim=swin_cfg.get("feature_dim", 768),
        num_classes=model_cfg.get("num_classes", 1843),
        d_model=model_cfg.get("d_model", 512),
        nhead=model_cfg.get("nhead", 8),
        num_layers=model_cfg.get("num_encoder_layers", 6),
        dim_feedforward=model_cfg.get("dim_feedforward", 2048),
        dropout=model_cfg.get("dropout", 0.2),
        max_seq_len=swin_cfg.get("max_seq_len", 512),
        encoder_type=model_cfg.get("type", "temporal_transformer"),
        pooling=model_cfg.get("pooling", "attention"),
        hand_branch=hand_cfg.get("enabled", True),
        hand_feature_dim=hand_cfg.get("hand_feature_dim", 63),
        hand_hidden_dim=hand_cfg.get("hand_hidden_dim", 128),
    )


if __name__ == "__main__":
    # Quick test
    print("Testing TemporalRecognizer...")
    model = TemporalRecognizer(
        feature_dim=768, num_classes=100, d_model=256,
        nhead=4, num_layers=2, hand_branch=True,
    )
    x = torch.randn(4, 64, 768)
    logits, hand = model(x)
    print(f"  Input:  {x.shape}")
    print(f"  Logits: {logits.shape}")
    print(f"  Hand:   {hand.shape if hand is not None else 'None'}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print("  [OK]")
