"""
=============================================================================
SIGNLYTIC AI - COMPREHENSIVE IMPROVEMENT PLAN
=============================================================================

This script implements all improvements using SignAvatars data:

1. POSE-BASED RECOGNITION MODEL (New)
2. MULTI-MODAL FUSION (Video + Pose)  
3. SIGN MOTION GENERATION (Text → Pose)
4. CONTINUOUS SIGN RECOGNITION (CTC)
5. IMPROVED HAND POSE MODEL
6. CROSS-LINGUAL TRANSFER LEARNING

Author: Oke Iyanuoluwa Enoch
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# 1. POSE-BASED SIGN RECOGNITION MODEL
# =============================================================================

class PoseEncoder(nn.Module):
    """
    Transformer encoder for pose sequences.
    Input: (B, T, 169) pose features
    Output: (B, D) sign embeddings
    """
    def __init__(
        self,
        input_dim: int = 169,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.3,
        max_seq_len: int = 512,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, 169) pose features
            mask: (B, T) padding mask
        Returns:
            (B, hidden_dim) sequence embedding
        """
        B, T, _ = x.shape
        
        # Project and add positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :T, :]
        x = self.dropout(x)
        
        # Transformer encoding
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)
        
        # Mean pooling
        if mask is not None:
            x = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
        else:
            x = x.mean(dim=1)
        
        return self.norm(x)


class PoseSignRecognizer(nn.Module):
    """
    Complete pose-based sign recognition model.
    """
    def __init__(
        self,
        num_classes: int = 500,
        input_dim: int = 169,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.encoder = PoseEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
    def forward(self, poses: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            poses: (B, T, 169) pose sequences
            mask: (B, T) valid frame mask
        Returns:
            (B, num_classes) logits
        """
        embeddings = self.encoder(poses, mask)
        return self.classifier(embeddings)


# =============================================================================
# 2. MULTI-MODAL FUSION (Video + Pose)
# =============================================================================

class MultiModalFusion(nn.Module):
    """
    Fuses video features (SWIN) with pose features (SignAvatars).
    """
    def __init__(
        self,
        video_dim: int = 768,      # SWIN feature dim
        pose_dim: int = 169,       # SignAvatars pose dim
        hidden_dim: int = 512,
        num_classes: int = 500,
        fusion_type: str = 'concat',  # 'concat', 'attention', 'gated'
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Video encoder
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Pose encoder
        self.pose_encoder = PoseEncoder(
            input_dim=pose_dim,
            hidden_dim=hidden_dim,
            num_layers=3,
            dropout=dropout,
        )
        
        if fusion_type == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif fusion_type == 'attention':
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
            self.fusion = nn.LayerNorm(hidden_dim)
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.LayerNorm(hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
    
    def forward(
        self, 
        video_features: torch.Tensor,  # (B, T_v, 768) or (B, 768)
        pose_features: torch.Tensor,   # (B, T_p, 169)
        video_mask: torch.Tensor = None,
        pose_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        # Encode video
        if video_features.dim() == 3:
            video_emb = self.video_proj(video_features.mean(dim=1))
        else:
            video_emb = self.video_proj(video_features)
        
        # Encode pose
        pose_emb = self.pose_encoder(pose_features, pose_mask)
        
        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([video_emb, pose_emb], dim=-1)
            fused = self.fusion(fused)
        elif self.fusion_type == 'attention':
            # Cross-attention: video attends to pose
            fused, _ = self.cross_attn(
                video_emb.unsqueeze(1), 
                pose_emb.unsqueeze(1), 
                pose_emb.unsqueeze(1)
            )
            fused = self.fusion(fused.squeeze(1) + video_emb)
        elif self.fusion_type == 'gated':
            gate = self.gate(torch.cat([video_emb, pose_emb], dim=-1))
            fused = gate * video_emb + (1 - gate) * pose_emb
            fused = self.fusion(fused)
        
        return self.classifier(fused)


# =============================================================================
# 3. SIGN MOTION GENERATION (Text/Gloss → Pose)
# =============================================================================

class MotionDecoder(nn.Module):
    """
    Autoregressive decoder for generating sign motion sequences.
    """
    def __init__(
        self,
        vocab_size: int = 10000,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        output_dim: int = 169,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        
        # Text/gloss embedding
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.text_pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        
        # Text encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads // 2,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # Motion decoder
        self.motion_proj = nn.Linear(output_dim, hidden_dim)
        self.motion_pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.motion_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Cross-attention from text to motion
        self.cross_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
        # Length predictor
        self.length_predictor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode text/gloss sequence"""
        B, T = text_ids.shape
        x = self.text_embedding(text_ids)
        x = x + self.text_pos_encoding[:, :T, :]
        return self.text_encoder(x)
    
    def forward(
        self,
        text_ids: torch.Tensor,      # (B, T_text)
        target_motion: torch.Tensor = None,  # (B, T_motion, 169)
        max_len: int = 128,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_ids: Input text/gloss token IDs
            target_motion: Target motion sequence (for training)
            max_len: Maximum generation length (for inference)
        """
        B = text_ids.shape[0]
        device = text_ids.device
        
        # Encode text
        text_encoded = self.encode_text(text_ids)  # (B, T_text, embed_dim)
        text_for_cross = self.cross_proj(text_encoded)  # (B, T_text, hidden_dim)
        
        # Predict length
        text_pooled = text_encoded.mean(dim=1)
        pred_length = self.length_predictor(text_pooled).squeeze(-1)  # (B,)
        
        if target_motion is not None:
            # Teacher forcing during training
            T_motion = target_motion.shape[1]
            
            # Project motion and add positional encoding
            motion_input = self.motion_proj(target_motion)
            motion_input = motion_input + self.motion_pos_encoding[:, :T_motion, :]
            
            # Causal mask
            causal_mask = torch.triu(
                torch.ones(T_motion, T_motion, device=device), diagonal=1
            ).bool()
            
            # Decode
            decoded = self.motion_decoder(
                motion_input,
                text_for_cross,
                tgt_mask=causal_mask,
            )
            
            motion_output = self.output_proj(decoded)
            
            return {
                'motion': motion_output,
                'pred_length': pred_length,
            }
        else:
            # Autoregressive generation
            generated = []
            prev_motion = torch.zeros(B, 1, 169, device=device)
            
            for t in range(max_len):
                motion_input = self.motion_proj(prev_motion)
                motion_input = motion_input + self.motion_pos_encoding[:, :t+1, :]
                
                decoded = self.motion_decoder(motion_input, text_for_cross)
                next_pose = self.output_proj(decoded[:, -1:, :])
                
                generated.append(next_pose)
                prev_motion = torch.cat([prev_motion, next_pose], dim=1)
            
            return {
                'motion': torch.cat(generated, dim=1),
                'pred_length': pred_length,
            }


class SignMotionGenerator(nn.Module):
    """
    Complete text/gloss to sign motion generation system.
    """
    def __init__(self, vocab_size: int = 10000, **kwargs):
        super().__init__()
        self.decoder = MotionDecoder(vocab_size=vocab_size, **kwargs)
        
    def forward(self, text_ids, target_motion=None, max_len=128):
        return self.decoder(text_ids, target_motion, max_len)
    
    def generate(self, text_ids: torch.Tensor, max_len: int = 128) -> torch.Tensor:
        """Generate motion sequence from text"""
        self.eval()
        with torch.no_grad():
            output = self.decoder(text_ids, max_len=max_len)
        return output['motion']


# =============================================================================
# 4. CONTINUOUS SIGN RECOGNITION (CTC)
# =============================================================================

class ContinuousSignRecognizer(nn.Module):
    """
    CTC-based continuous sign language recognition.
    Recognizes sign sequences without explicit segmentation.
    """
    def __init__(
        self,
        input_dim: int = 169,
        hidden_dim: int = 512,
        num_classes: int = 500,  # + 1 for blank
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, num_classes + 1)  # +1 for CTC blank
        
        self.ctc_loss = nn.CTCLoss(blank=num_classes, zero_infinity=True)
        
    def forward(self, poses: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            poses: (B, T, 169) pose sequences
            mask: (B, T) valid frame mask
        Returns:
            (B, T, num_classes+1) log probabilities
        """
        B, T, _ = poses.shape
        
        x = self.input_proj(poses)
        x = x + self.pos_encoding[:, :T, :]
        
        if mask is not None:
            x = self.transformer(x, src_key_padding_mask=~mask)
        else:
            x = self.transformer(x)
        
        logits = self.output_proj(x)
        return F.log_softmax(logits, dim=-1)
    
    def compute_loss(
        self,
        poses: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss"""
        log_probs = self.forward(poses)  # (B, T, C)
        log_probs = log_probs.permute(1, 0, 2)  # (T, B, C) for CTC
        
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
    
    def decode_greedy(self, log_probs: torch.Tensor) -> List[List[int]]:
        """Greedy CTC decoding"""
        predictions = log_probs.argmax(dim=-1)  # (B, T)
        blank_id = log_probs.shape[-1] - 1
        
        results = []
        for pred in predictions:
            decoded = []
            prev = blank_id
            for p in pred:
                if p != blank_id and p != prev:
                    decoded.append(p.item())
                prev = p
            results.append(decoded)
        
        return results


# =============================================================================
# 5. HAND POSE REFINEMENT MODEL
# =============================================================================

class HandPoseRefiner(nn.Module):
    """
    Refines hand pose predictions using temporal context.
    Can improve noisy hand tracking.
    """
    def __init__(
        self,
        hand_dim: int = 45,  # 15 joints x 3
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=hand_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        
        self.refine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hand_dim),
        )
        
    def forward(self, hand_poses: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            hand_poses: (B, T, 45) hand pose sequence
            valid_mask: (B, T) validity mask
        Returns:
            (B, T, 45) refined hand poses
        """
        encoded, _ = self.lstm(hand_poses)
        refined = self.refine(encoded)
        
        # Residual connection
        output = hand_poses + 0.1 * refined
        
        # Only update invalid frames more aggressively
        if valid_mask is not None:
            invalid = ~valid_mask
            output = torch.where(
                invalid.unsqueeze(-1),
                hand_poses + refined,  # Full update for invalid
                output,  # Small update for valid
            )
        
        return output


# =============================================================================
# 6. TRAINING UTILITIES
# =============================================================================

class SignAvatarsTrainer:
    """
    Unified trainer for all SignAvatars-based models.
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        task: str = 'recognition',
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.task = task
        
        if task == 'recognition':
            self.criterion = nn.CrossEntropyLoss()
        elif task == 'motion_gen':
            self.criterion = nn.MSELoss()
        elif task == 'ctc':
            self.criterion = None  # Built into model
    
    def train_step(self, batch: Dict) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        if self.task == 'recognition':
            poses = batch['poses'].to(self.device)
            labels = batch['labels'].to(self.device)
            mask = batch.get('mask')
            if mask is not None:
                mask = mask.to(self.device)
            
            logits = self.model(poses, mask)
            loss = self.criterion(logits, labels)
            
        elif self.task == 'motion_gen':
            text_ids = batch['text_ids'].to(self.device)
            target_motion = batch['motion'].to(self.device)
            
            output = self.model(text_ids, target_motion)
            loss = self.criterion(output['motion'], target_motion)
            
        elif self.task == 'ctc':
            poses = batch['poses'].to(self.device)
            targets = batch['targets'].to(self.device)
            input_lengths = batch['input_lengths'].to(self.device)
            target_lengths = batch['target_lengths'].to(self.device)
            
            loss = self.model.compute_loss(poses, targets, input_lengths, target_lengths)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in dataloader:
            if self.task == 'recognition':
                poses = batch['poses'].to(self.device)
                labels = batch['labels'].to(self.device)
                mask = batch.get('mask')
                if mask is not None:
                    mask = mask.to(self.device)
                
                logits = self.model(poses, mask)
                loss = self.criterion(logits, labels)
                
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.shape[0]
                
            total_loss += loss.item()
        
        metrics = {'loss': total_loss / len(dataloader)}
        if self.task == 'recognition':
            metrics['accuracy'] = correct / total
        
        return metrics


# =============================================================================
# 7. INTEGRATION WITH EXISTING BSL SYSTEM
# =============================================================================

class EnhancedBSLRecognizer(nn.Module):
    """
    Enhanced BSL recognizer combining:
    - SWIN video features (existing)
    - SignAvatars pose features (new)
    - Multi-modal fusion
    """
    def __init__(
        self,
        num_classes: int = 500,
        swin_dim: int = 768,
        pose_dim: int = 169,
        hidden_dim: int = 512,
        use_pose: bool = True,
        use_video: bool = True,
    ):
        super().__init__()
        
        self.use_pose = use_pose
        self.use_video = use_video
        
        # Video branch (existing SWIN)
        if use_video:
            self.video_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=swin_dim,
                    nhead=8,
                    dim_feedforward=swin_dim * 4,
                    dropout=0.3,
                    batch_first=True,
                ),
                num_layers=4,
            )
            self.video_proj = nn.Linear(swin_dim, hidden_dim)
        
        # Pose branch (new SignAvatars)
        if use_pose:
            self.pose_encoder = PoseEncoder(
                input_dim=pose_dim,
                hidden_dim=hidden_dim,
                num_layers=4,
                dropout=0.3,
            )
        
        # Fusion
        if use_video and use_pose:
            self.fusion = MultiModalFusion(
                video_dim=hidden_dim,
                pose_dim=pose_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                fusion_type='gated',
            )
            # Override fusion's internal encoders since we have our own
            self.final_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.final_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes),
            )
    
    def forward(
        self,
        video_features: torch.Tensor = None,  # (B, T, 768)
        pose_features: torch.Tensor = None,   # (B, T, 169)
        video_mask: torch.Tensor = None,
        pose_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        
        embeddings = []
        
        if self.use_video and video_features is not None:
            video_enc = self.video_encoder(video_features)
            video_emb = self.video_proj(video_enc.mean(dim=1))
            embeddings.append(video_emb)
        
        if self.use_pose and pose_features is not None:
            pose_emb = self.pose_encoder(pose_features, pose_mask)
            embeddings.append(pose_emb)
        
        if len(embeddings) == 2:
            fused = torch.cat(embeddings, dim=-1)
        else:
            fused = embeddings[0]
        
        return self.final_classifier(fused)


# =============================================================================
# MAIN - Test All Components
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("SIGNLYTIC AI - ENHANCED MODELS TEST")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test data
    B, T = 4, 50  # Batch size, sequence length
    pose_features = torch.randn(B, T, 169)
    video_features = torch.randn(B, T, 768)
    labels = torch.randint(0, 500, (B,))
    
    # 1. Pose-based recognizer
    print("\n1. Testing PoseSignRecognizer...")
    model1 = PoseSignRecognizer(num_classes=500)
    out1 = model1(pose_features)
    print(f"   Input: {pose_features.shape} -> Output: {out1.shape}")
    assert out1.shape == (B, 500), "Shape mismatch!"
    print("   [PASS]")
    
    # 2. Multi-modal fusion
    print("\n2. Testing MultiModalFusion...")
    model2 = MultiModalFusion(num_classes=500, fusion_type='gated')
    out2 = model2(video_features, pose_features)
    print(f"   Video: {video_features.shape}, Pose: {pose_features.shape} -> Output: {out2.shape}")
    assert out2.shape == (B, 500), "Shape mismatch!"
    print("   [PASS]")
    
    # 3. Motion generator
    print("\n3. Testing SignMotionGenerator...")
    model3 = SignMotionGenerator(vocab_size=1000)
    text_ids = torch.randint(0, 1000, (B, 20))
    target_motion = torch.randn(B, 30, 169)
    out3 = model3(text_ids, target_motion)
    print(f"   Text: {text_ids.shape}, Target: {target_motion.shape} -> Motion: {out3['motion'].shape}")
    assert out3['motion'].shape == (B, 30, 169), "Shape mismatch!"
    print("   [PASS]")
    
    # 4. Continuous recognizer (CTC)
    print("\n4. Testing ContinuousSignRecognizer...")
    model4 = ContinuousSignRecognizer(num_classes=500)
    out4 = model4(pose_features)
    print(f"   Input: {pose_features.shape} -> Output: {out4.shape}")
    assert out4.shape == (B, T, 501), "Shape mismatch!"  # +1 for blank
    print("   [PASS]")
    
    # 5. Hand pose refiner
    print("\n5. Testing HandPoseRefiner...")
    model5 = HandPoseRefiner()
    hand_poses = torch.randn(B, T, 45)
    out5 = model5(hand_poses)
    print(f"   Input: {hand_poses.shape} -> Output: {out5.shape}")
    assert out5.shape == (B, T, 45), "Shape mismatch!"
    print("   [PASS]")
    
    # 6. Enhanced BSL recognizer
    print("\n6. Testing EnhancedBSLRecognizer...")
    model6 = EnhancedBSLRecognizer(num_classes=500, use_pose=True, use_video=True)
    out6 = model6(video_features=video_features, pose_features=pose_features)
    print(f"   Video+Pose -> Output: {out6.shape}")
    assert out6.shape == (B, 500), "Shape mismatch!"
    print("   [PASS]")
    
    # Count parameters
    print("\n" + "="*70)
    print("MODEL PARAMETER COUNTS")
    print("="*70)
    models = {
        'PoseSignRecognizer': model1,
        'MultiModalFusion': model2,
        'SignMotionGenerator': model3,
        'ContinuousSignRecognizer': model4,
        'HandPoseRefiner': model5,
        'EnhancedBSLRecognizer': model6,
    }
    
    for name, model in models.items():
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} parameters ({params/1e6:.2f}M)")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
