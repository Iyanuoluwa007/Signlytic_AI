#!/usr/bin/env python3
"""
Neural network classifiers for BSL sign recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron for sign classification using pooled features."""
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 1843,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class TemporalMLPClassifier(nn.Module):
    """MLP Classifier with temporal pooling for sequence input."""
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        num_classes: int = 1843,
        dropout: float = 0.3,
        pooling: str = 'mean',
    ):
        super().__init__()
        
        self.pooling = pooling
        
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        
        self.classifier = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
    
    def _pool_features(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool temporal features to fixed-size vector.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional mask of shape (batch_size, seq_len)
            
        Returns:
            Pooled tensor of shape (batch_size, input_dim)
        """
        if self.pooling == 'mean':
            if mask is not None:
                mask = mask.unsqueeze(-1).float()
                x = x * mask
                return x.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return x.mean(dim=1)
        
        elif self.pooling == 'max':
            if mask is not None:
                mask = mask.unsqueeze(-1)
                x = x.masked_fill(~mask, float('-inf'))
            return x.max(dim=1)[0]
        
        elif self.pooling == 'attention':
            attn_scores = self.attention(x).squeeze(-1)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1)
            return torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)
        
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with temporal pooling.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional padding mask of shape (batch_size, seq_len)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        pooled = self._pool_features(x, mask)
        return self.classifier(pooled)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
