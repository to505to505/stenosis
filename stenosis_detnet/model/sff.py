"""Sequence Feature Fusion (SFF) module.

Custom multi-head self-attention mechanism that tracks lesions
across N=9 frames by fusing RoI features from all candidate boxes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config


class SequenceFeatureFusion(nn.Module):
    """Multi-head self-attention fusion across temporal frames.

    Takes 256×7×7 RoI features for each proposal across T frames,
    projects to Q/K/V, computes attention, and outputs enhanced features
    with a residual connection.

    Args:
        cfg: Config with sff_d_model, sff_num_heads, sff_epsilon, sff_dropout,
             C (=256), roi_output_size (=7).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T
        self.d_model = cfg.sff_d_model
        self.num_heads = cfg.sff_num_heads
        self.head_dim = cfg.sff_head_dim
        self.epsilon = cfg.sff_epsilon
        input_dim = cfg.sff_input_dim  # C * roi_size * roi_size = 256*7*7 = 12544

        # Three linear projections for Q, K, V
        self.linear_q = nn.Linear(input_dim, self.d_model)
        self.linear_k = nn.Linear(input_dim, self.d_model)
        self.linear_v = nn.Linear(input_dim, self.d_model)

        # Output projection after multi-head concat
        self.out_proj = nn.Linear(self.d_model, input_dim)

        # Layer norm for residual connection
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(cfg.sff_dropout)

    def forward(self, roi_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            roi_features: (S, T, C, roi_h, roi_w) RoI features for S proposals
                          across T frames. C=256, roi_h=roi_w=7.

        Returns:
            enhanced: (S, T, C, roi_h, roi_w) enhanced features with
                      attention-fused temporal context + residual.
        """
        S, T, C, rh, rw = roi_features.shape
        input_dim = C * rh * rw  # 12544

        # Flatten spatial: (S, T, input_dim)
        x = roi_features.reshape(S, T, input_dim)
        residual = x

        # Project to Q, K, V: each (S, T, d_model)
        Q = self.linear_q(x)
        K = self.linear_k(x)
        V = self.linear_v(x)

        # Reshape for multi-head: (S, num_heads, T, head_dim)
        H = self.num_heads
        d_k = self.head_dim
        Q = Q.reshape(S, T, H, d_k).permute(0, 2, 1, 3)  # (S, H, T, d_k)
        K = K.reshape(S, T, H, d_k).permute(0, 2, 1, 3)
        V = V.reshape(S, T, H, d_k).permute(0, 2, 1, 3)

        # Attention: Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (S, H, T, T)
        attn_scores = attn_scores / (d_k ** 0.5 + self.epsilon)

        # Softmax normalization with epsilon for numerical stability
        attn_weights = F.softmax(attn_scores, dim=-1)  # (S, H, T, T)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: attn_weights @ V
        attn_output = torch.matmul(attn_weights, V)  # (S, H, T, d_k)

        # Concatenate heads: (S, T, d_model)
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(S, T, self.d_model)

        # Output projection back to input_dim
        output = self.out_proj(attn_output)  # (S, T, input_dim)
        output = self.dropout(output)

        # Residual connection + layer norm
        enhanced = self.layer_norm(output + residual)

        # Reshape back to spatial: (S, T, C, rh, rw)
        enhanced = enhanced.reshape(S, T, C, rh, rw)
        return enhanced
