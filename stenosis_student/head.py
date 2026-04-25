"""Anchor-free FCOS detection head.

Three parallel branches share a 4-conv tower per branch:
    - classification (sigmoid, ``num_classes`` channels)
    - bbox regression (4 channels: ``l, t, r, b`` distances in pixels)
    - centre-ness (1 channel, sigmoid)

Towers are shared across pyramid levels but the regression tower has a
per-level learnable scalar ``Scale`` to account for differing strides
(standard FCOS practice).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


def _make_tower(in_dim: int, num_convs: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1))
        gn_groups = 32 if in_dim % 32 == 0 else min(8, in_dim)
        layers.append(nn.GroupNorm(gn_groups, in_dim))
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class FCOSHead(nn.Module):
    """Detail-Aware FCOS head (anchor-free).

    Args:
        in_dim: channel count of every FPN level (must be uniform).
        num_classes: number of object classes (excludes background).
        num_levels: number of pyramid levels (3 for P3/P4/P5).
        strides: per-level strides in pixels (length ``num_levels``).
        num_convs: depth of each shared tower.
        prior_prob: focal-loss bias initialisation for the cls head.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_levels: int,
        strides: Tuple[int, ...],
        num_convs: int = 4,
        prior_prob: float = 0.01,
    ):
        super().__init__()
        assert len(strides) == num_levels
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.strides = tuple(strides)

        self.cls_tower = _make_tower(in_dim, num_convs)
        self.reg_tower = _make_tower(in_dim, num_convs)

        self.cls_logits = nn.Conv2d(in_dim, num_classes, kernel_size=3, padding=1)
        self.bbox_pred = nn.Conv2d(in_dim, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in range(num_levels)])

        # ── init ──
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(
        self, feats: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Args:
            feats: list of ``(B, in_dim, H_l, W_l)`` per level.
        Returns:
            cls_logits, bbox_reg, centerness — each a list of length
            ``num_levels``:
              cls   ``(B, num_classes, H, W)``
              bbox  ``(B, 4, H, W)`` — positive distances in pixels
              cent  ``(B, 1, H, W)`` — raw logits
        """
        assert len(feats) == self.num_levels
        cls_outs, reg_outs, ctr_outs = [], [], []
        for lvl, x in enumerate(feats):
            ct = self.cls_tower(x)
            rt = self.reg_tower(x)
            cls_outs.append(self.cls_logits(ct))
            ctr_outs.append(self.centerness(rt))
            # Scale + exp to keep positive distances; multiply by stride so
            # raw outputs live in a stride-normalised range (FCOS trick).
            reg = self.scales[lvl](self.bbox_pred(rt))
            reg = torch.exp(reg) * self.strides[lvl]
            reg_outs.append(reg)
        return cls_outs, reg_outs, ctr_outs


def make_locations(
    feature_sizes: List[Tuple[int, int]],
    strides: Tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> List[torch.Tensor]:
    """For every pyramid level, build ``(H*W, 2)`` tensor of pixel locations
    (centres of each grid cell, in input-image coordinates)."""
    assert len(feature_sizes) == len(strides)
    out = []
    for (H, W), s in zip(feature_sizes, strides):
        ys = (torch.arange(H, device=device, dtype=dtype) + 0.5) * s
        xs = (torch.arange(W, device=device, dtype=dtype) + 0.5) * s
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        out.append(torch.stack([xx.flatten(), yy.flatten()], dim=1))
    return out
