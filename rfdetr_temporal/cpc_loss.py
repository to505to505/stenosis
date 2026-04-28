"""Contrastive Predictive Coding (CPC) loss for Temporal RF-DETR.

Implements the log-bilinear scoring model

    f_k(x_{t+k}, c_t)  ∝  exp( z_{t+k}^T  W_k  c_t )

as a temporal regulariser on the fused centre-frame context ``c_t`` and the
raw backbone features ``z_{t±1}`` of the immediately neighbouring frames.
The trainable projection ``W`` (one per relative offset) lets the model
account for spatial displacement of the artery between consecutive frames
instead of forcing identity ``c_t == z_{t+1}``.

Loss
────
For each spatial position p in the centre frame, the projected context
``W c_t[p]`` must be more similar to the neighbour feature ``z_{t±1}[p]``
than to the neighbour features at any other spatial position.  This is
implemented as InfoNCE: cross-entropy over a (HW, HW) similarity matrix
where the positive target lies on the diagonal (same spatial location),
all off-diagonal entries are negatives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCPCLoss(nn.Module):
    """InfoNCE-style CPC loss between fused context and neighbour features.

    Args:
        hidden_dim: channel dimension of the features (must match the
            projector output, e.g. 256 for the P4 RF-DETR Small backbone).
        offsets: relative temporal offsets to predict.  Default ``(-1, 1)``
            uses one trainable projection for ``t-1`` and another for ``t+1``.
    """

    def __init__(self, hidden_dim: int = 256, offsets: tuple[int, ...] = (-1, 1)):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.offsets = tuple(int(o) for o in offsets)
        # One projection W_k per relative offset (separate parameters so each
        # k learns its own spatial-displacement transform).
        self.projections = nn.ModuleDict({
            self._key(o): nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
            for o in self.offsets
        })

    @staticmethod
    def _key(offset: int) -> str:
        return f"off_{offset:+d}"

    def _pair_loss(
        self, context: torch.Tensor, neighbour: torch.Tensor, offset: int,
    ) -> torch.Tensor:
        """InfoNCE for a single (context, neighbour) frame pair.

        Args:
            context:    (B, C, H, W) fused centre-frame features
            neighbour:  (B, C, H, W) raw backbone features of frame t+offset
        Returns:
            scalar loss (mean cross-entropy over batch and spatial positions)
        """
        B, C, H, W = context.shape
        HW = H * W

        # (B, HW, C)
        c = context.reshape(B, C, HW).transpose(1, 2)
        z = neighbour.reshape(B, C, HW).transpose(1, 2)

        # L2-normalise so the bilinear logits have stable scale.
        c = F.normalize(c, dim=-1)
        z = F.normalize(z, dim=-1)

        # Project context into the neighbour space:  W_k c_t
        c_proj = self.projections[self._key(offset)](c)              # (B, HW, C)

        # Similarity matrix: logits[b, p, q] = <W c[b,p], z[b,q]>
        logits = torch.bmm(c_proj, z.transpose(1, 2))                # (B, HW, HW)

        # Positive target = same spatial position (diagonal).
        target = torch.arange(HW, device=context.device).unsqueeze(0).expand(B, -1)
        # Cross-entropy expects (N, K) logits with (N,) targets.
        return F.cross_entropy(logits.reshape(B * HW, HW), target.reshape(B * HW))

    def forward(
        self, fused_context: torch.Tensor, raw_feats: torch.Tensor, centre: int,
    ) -> torch.Tensor:
        """
        Args:
            fused_context: (B, C, H, W)   – temporally-fused centre frame
            raw_feats:     (B, T, C, H, W) – per-frame raw backbone features
            centre:        index of the centre frame in ``raw_feats``
        Returns:
            scalar CPC loss averaged across the requested offsets
        """
        B, T, C, H, W = raw_feats.shape
        losses = []
        for off in self.offsets:
            t_idx = centre + off
            if t_idx < 0 or t_idx >= T:
                continue
            neighbour = raw_feats[:, t_idx]           # (B, C, H, W)
            losses.append(self._pair_loss(fused_context, neighbour, off))

        if not losses:
            return fused_context.new_zeros(())
        return torch.stack(losses).mean()
