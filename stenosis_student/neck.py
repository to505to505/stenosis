"""Detail-Aware Cross-Attention FPN.

Standard FPNs combine semantic (low-resolution, high-channel) and spatial
(high-resolution, low-channel) features by simple element-wise addition
after upsampling.  For tiny targets such as micro-stenoses (<50 % occlusion)
this can wash out boundary information.

We replace each top-down merge with a lightweight cross-attention module
where the high-resolution feature provides queries and the upsampled
semantic feature provides keys/values.  This lets every spatial token pull
context from the semantic map only where it is actually needed, while
preserving fine boundary detail.

Output: three feature maps ``P3', P4', P5'`` at strides 8 / 16 / 32, all
projected to ``cfg.fpn_dim`` channels.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class CrossAttnFusion(nn.Module):
    """Cross-attention fusion of a high-res spatial map with a low-res
    semantic map.

    Queries: tokens from the high-res map (``Hh*Wh, B, D``).
    Keys / values: tokens from the upsampled semantic map, optionally pooled
    to keep memory bounded at large resolutions.

    A learned per-channel gate blends the attention output with the
    high-res map (residual style).
    """

    def __init__(self, dim: int, num_heads: int = 4, kv_pool: int = 1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kv_pool = kv_pool
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True,
        )
        self.gate = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
        """Args:
            hi: ``(B, D, Hh, Wh)`` high-res spatial features (queries).
            lo: ``(B, D, Hl, Wl)`` low-res semantic features.
        Returns:
            ``(B, D, Hh, Wh)`` fused features.
        """
        B, D, Hh, Wh = hi.shape
        # Upsample lo to hi resolution
        lo_up = F.interpolate(lo, size=(Hh, Wh), mode="bilinear", align_corners=False)
        # Pool K/V to bound memory at high resolution
        if self.kv_pool > 1:
            lo_pooled = F.avg_pool2d(lo_up, kernel_size=self.kv_pool)
        else:
            lo_pooled = lo_up

        Hk, Wk = lo_pooled.shape[-2:]
        q = hi.flatten(2).transpose(1, 2)        # (B, Hh*Wh, D)
        kv = lo_pooled.flatten(2).transpose(1, 2)  # (B, Hk*Wk, D)
        q = self.norm_q(q)
        kv = self.norm_kv(kv)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, D, Hh, Wh)
        # Gated residual: hi + sigmoid(gate) * proj(attn)
        gate = torch.sigmoid(self.gate).view(1, D, 1, 1)
        return hi + gate * self.proj(attn_out)


class DetailAwareFPN(nn.Module):
    """Three-level FPN with cross-attention top-down fusion.

    Inputs (from the backbone, in ascending stride order):
        P3 @ stride 8,  P4 @ stride 16,  P5 @ stride 32.

    Outputs the same three levels, all with ``fpn_dim`` channels.
    """

    def __init__(self, cfg: Config, in_channels: List[int]):
        super().__init__()
        self.cfg = cfg
        D = cfg.fpn_dim
        assert len(in_channels) == 3, "DetailAwareFPN expects three input levels"

        self.lateral = nn.ModuleList([
            nn.Conv2d(c, D, kernel_size=1) for c in in_channels
        ])

        # Cross-attn fusion: P5→P4 and P4→P3 (semantic into spatial)
        self.fuse_p4 = CrossAttnFusion(D, num_heads=cfg.fpn_num_heads, kv_pool=1)
        self.fuse_p3 = CrossAttnFusion(D, num_heads=cfg.fpn_num_heads, kv_pool=cfg.fpn_kv_pool)

        # 3×3 smoothing convs
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(D, D, kernel_size=3, padding=1),
                nn.GroupNorm(32, D) if D % 32 == 0 else nn.GroupNorm(min(8, D), D),
                nn.ReLU(inplace=True),
            )
            for _ in range(3)
        ])

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """Args:
            feats: ``[P3, P4, P5]`` each ``(B, C_i, H_i, W_i)``.
        Returns:
            ``[P3', P4', P5']`` each ``(B, fpn_dim, H_i, W_i)``.
        """
        assert len(feats) == 3
        p3, p4, p5 = (lat(f) for lat, f in zip(self.lateral, feats))

        # Top-down fusion
        p4 = self.fuse_p4(p4, p5)
        p3 = self.fuse_p3(p3, p4)

        out = [self.smooth[0](p3), self.smooth[1](p4), self.smooth[2](p5)]
        return out
