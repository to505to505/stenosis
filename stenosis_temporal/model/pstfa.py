"""Proposal-aware Spatio-Temporal Feature Aggregation (PSTFA) module.

Contains:
  - PSSTT: Proposal-Shifted Spatio-Temporal Tokenization
  - TFA:   Transformer-based Feature Aggregation
"""

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.ops import roi_align

from ..config import Config


class PSSTT(nn.Module):
    """Proposal-Shifted Spatio-Temporal Tokenization.

    For each proposal, generates T*(K+1) RoI tokens by applying spatial shifts
    (up/down/left/right) on all temporal frames, then projects to D dimensions.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.K = cfg.K
        self.T = cfg.T
        self.shift_fraction = cfg.shift_fraction
        self.roi_size = cfg.roi_output_size
        self.C = cfg.C

        # Spatial scale for FPN level "0" (stride 4)
        self.spatial_scale = 1.0 / 4.0

        # Linear projection: flatten RoI features → D
        roi_feat_dim = cfg.roi_output_size ** 2 * cfg.C  # 7*7*256 = 12544
        self.projection = nn.Linear(roi_feat_dim, cfg.D)

    def _generate_shifted_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Generate K+1 shifted versions of each box.

        Args:
            boxes: (num_boxes, 4) in x1y1x2y2 format

        Returns:
            shifted: (num_boxes, K+1, 4) — original + 4 directional shifts
        """
        x1, y1, x2, y2 = boxes.unbind(dim=-1)
        w = x2 - x1
        h = y2 - y1
        dx = w * self.shift_fraction
        dy = h * self.shift_fraction

        # Original (zero shift)
        orig = boxes.unsqueeze(1)  # (N, 1, 4)

        shifts = []
        # Up: shift y upward (decrease y)
        shifts.append(torch.stack([x1, y1 - dy, x2, y2 - dy], dim=-1).unsqueeze(1))
        # Down
        shifts.append(torch.stack([x1, y1 + dy, x2, y2 + dy], dim=-1).unsqueeze(1))
        # Left
        shifts.append(torch.stack([x1 - dx, y1, x2 - dx, y2], dim=-1).unsqueeze(1))
        # Right
        shifts.append(torch.stack([x1 + dx, y1, x2 + dx, y2], dim=-1).unsqueeze(1))

        # (N, K+1, 4) — original first, then shifts
        shifted = torch.cat([orig] + shifts, dim=1)

        # Clamp to image bounds
        shifted = shifted.clamp(min=0)
        shifted[..., 0::2] = shifted[..., 0::2].clamp(max=self.cfg.img_w)
        shifted[..., 1::2] = shifted[..., 1::2].clamp(max=self.cfg.img_h)

        return shifted

    def forward(
        self,
        features_per_frame: List[torch.Tensor],
        proposals_ref: torch.Tensor,
        ref_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            features_per_frame: list of T feature maps, each (C, H_feat, W_feat)
                                from a single FPN level (level 0 / stride-4).
            proposals_ref: (S, 4) proposals for the reference frame, x1y1x2y2
            ref_idx: index of the reference frame in the T-length sequence

        Returns:
            tokens: (S, T*(K+1), D) token sequence per proposal
        """
        S = proposals_ref.shape[0]
        K1 = self.K + 1  # 5
        device = proposals_ref.device

        # Generate spatially shifted boxes: (S, K+1, 4)
        shifted_boxes = self._generate_shifted_boxes(proposals_ref)

        # Collect RoI features across all frames
        # Order: reference frame first, then support frames in temporal order
        frame_order = [ref_idx] + [t for t in range(self.T) if t != ref_idx]

        all_roi_feats = []  # will be list of T tensors, each (S, K+1, C, roi, roi)

        for t in frame_order:
            feat = features_per_frame[t]  # (C, H_f, W_f)
            feat_4d = feat.unsqueeze(0)  # (1, C, H_f, W_f) — single "image"

            # Flatten shifted boxes for this frame: (S*K1, 4)
            boxes_flat = shifted_boxes.reshape(-1, 4)

            # roi_align expects list of boxes per image — all belong to image 0
            # Use batch index column
            batch_idx = torch.zeros(
                boxes_flat.shape[0], 1, device=device, dtype=boxes_flat.dtype
            )
            rois = torch.cat([batch_idx, boxes_flat], dim=1)  # (S*K1, 5)

            roi_feats = roi_align(
                feat_4d, rois,
                output_size=self.roi_size,
                spatial_scale=self.spatial_scale,  # 1/4 for FPN level "0" (stride 4)
                sampling_ratio=2,
            )  # (S*K1, C, roi, roi)

            roi_feats = roi_feats.reshape(S, K1, self.C, self.roi_size, self.roi_size)
            all_roi_feats.append(roi_feats)

        # Stack along temporal dim: (S, T, K+1, C, roi, roi)
        all_roi_feats = torch.stack(all_roi_feats, dim=1)

        # Reshape to (S, T*K1, C*roi*roi)
        all_roi_feats = all_roi_feats.reshape(S, self.T * K1, -1)

        # Linear projection → (S, T*K1, D)
        tokens = self.projection(all_roi_feats)
        return tokens


class TFA(nn.Module):
    """Transformer-based Feature Aggregation.

    Takes token sequences from PSSTT, adds positional embeddings,
    runs through a Transformer encoder, and fuses into RoI features.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = cfg.gradient_checkpointing
        num_tokens = cfg.num_tokens  # T*(K+1) = 25

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, cfg.D))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder (pre-norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.D,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.num_transformer_layers
        )

        # Feature fusion: 1×1 conv to aggregate the token sequence
        self.fusion_conv = nn.Conv1d(cfg.D, cfg.D, kernel_size=1)
        self.fusion_norm = nn.LayerNorm(cfg.D)

        # Reshape back to RoI feature map
        roi_feat_dim = cfg.roi_output_size ** 2 * cfg.C  # 7*7*256
        self.output_proj = nn.Linear(cfg.D, roi_feat_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (S, num_tokens, D)

        Returns:
            roi_features: (S, C, roi_size, roi_size) — aggregated RoI features
        """
        cfg = self.cfg
        S = tokens.shape[0]

        # Add positional embeddings
        x = tokens + self.pos_embed  # (S, num_tokens, D)

        # Transformer encoder (per-layer checkpointing to save VRAM)
        if self.training and self.gradient_checkpointing:
            for layer in self.transformer.layers:
                x = checkpoint(layer, x, use_reentrant=False)
        else:
            x = self.transformer(x)  # (S, num_tokens, D)

        # Feature fusion: 1×1 conv + mean pool
        # (S, num_tokens, D) → (S, D, num_tokens)
        x = x.permute(0, 2, 1)
        x = self.fusion_conv(x)  # (S, D, num_tokens)
        x = x.mean(dim=-1)  # (S, D) — global average pooling
        x = self.fusion_norm(x)

        # Project back to RoI feature size
        x = self.output_proj(x)  # (S, 7*7*256)
        x = x.reshape(S, cfg.C, cfg.roi_output_size, cfg.roi_output_size)
        return x


class PSTFA(nn.Module):
    """Combined PSSTT tokenization + TFA aggregation."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.psstt = PSSTT(cfg)
        self.tfa = TFA(cfg)
        self.cfg = cfg

    def forward(
        self,
        features_per_frame: List[torch.Tensor],
        proposals_ref: torch.Tensor,
        ref_idx: int,
    ) -> torch.Tensor:
        """
        Args:
            features_per_frame: list of T feature maps (C, H_f, W_f)
            proposals_ref: (S, 4) reference frame proposals
            ref_idx: index of reference frame

        Returns:
            aggregated_roi: (S, C, roi_size, roi_size)
        """
        S = proposals_ref.shape[0]
        chunk = self.cfg.proposal_chunk_size

        if S <= chunk:
            tokens = self.psstt(features_per_frame, proposals_ref, ref_idx)
            return self.tfa(tokens)

        # Process in chunks to manage memory
        parts = []
        for i in range(0, S, chunk):
            chunk_proposals = proposals_ref[i : i + chunk]
            tokens = self.psstt(features_per_frame, chunk_proposals, ref_idx)
            part = self.tfa(tokens)
            parts.append(part)
        return torch.cat(parts, dim=0)
