"""Temporal-consistency InfoNCE on RoI-pooled box embeddings.

Pools the student's FPN feature at the centre-frame GT boxes for each
clip on frame ``t`` and on neighbour frames ``t + Δ`` (Δ ∈ ``cfg.
temporal_consistency_neighbor_offsets``).  Uses the **same xyxy** on
neighbour frames — small inter-frame motion assumption.

Returns the InfoNCE loss; if no GT boxes are present the loss is a
zero connected to the input feature so backward still works.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from .config import Config
from .distill_losses import info_nce_loss


def pool_box_embeddings(
    feature_map: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    batch_idx: torch.Tensor,
    stride: float,
    pool_size: int = 7,
) -> torch.Tensor:
    """RoI-Align a feature map at given xyxy boxes (in input-image pixels).

    Args:
        feature_map: ``(B, C, H, W)``.
        boxes_xyxy:  ``(N, 4)`` boxes in input-image pixel coords.
        batch_idx:   ``(N,)`` int32/int64 batch index per box.
        stride:      input/feature spatial scale (e.g. 16 for P4).
        pool_size:   RoI-Align spatial output size (square).

    Returns:
        ``(N, C)`` average-pooled embeddings.
    """
    if boxes_xyxy.numel() == 0:
        C = feature_map.shape[1]
        return feature_map.new_zeros((0, C))

    rois = torch.cat([batch_idx.float().unsqueeze(1), boxes_xyxy.float()], dim=1)
    pooled = roi_align(
        feature_map.float(),
        rois.to(feature_map.device),
        output_size=(pool_size, pool_size),
        spatial_scale=1.0 / float(stride),
        sampling_ratio=2,
        aligned=True,
    )                                              # (N, C, P, P)
    # Collapse spatial dims to a single embedding per box.
    return pooled.mean(dim=(-1, -2))               # (N, C)


def _gather_boxes(
    targets: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
):
    """Stack per-image box lists into a flat ``(N, 4)`` + ``batch_idx``."""
    all_boxes, all_batch = [], []
    for b, t in enumerate(targets):
        boxes = t["boxes"].to(device)
        if boxes.numel() == 0:
            continue
        all_boxes.append(boxes)
        all_batch.append(torch.full((boxes.shape[0],), b,
                                     dtype=torch.long, device=device))
    if not all_boxes:
        return (torch.zeros((0, 4), device=device),
                torch.zeros((0,), dtype=torch.long, device=device))
    return torch.cat(all_boxes, dim=0), torch.cat(all_batch, dim=0)


def temporal_consistency_loss(
    multi_frame_feat: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    cfg: Config,
    stride: float,
) -> torch.Tensor:
    """Compute the InfoNCE temporal-consistency loss.

    Args:
        multi_frame_feat: ``(B, T, C, h, w)`` — the chosen FPN level for
            every frame in the clip.
        targets: list of per-clip target dicts (centre-frame xyxy abs px).
        cfg: training config.
        stride: input/feature scale of ``multi_frame_feat`` (e.g. 16).
    """
    B, T, C, h, w = multi_frame_feat.shape
    centre = T // 2
    device = multi_frame_feat.device

    boxes, batch_idx = _gather_boxes(targets, device)
    if boxes.shape[0] == 0:
        return multi_frame_feat.sum() * 0.0

    # Centre-frame anchors
    anchor_feat = multi_frame_feat[:, centre]      # (B, C, h, w)
    anchors = pool_box_embeddings(
        anchor_feat, boxes, batch_idx, stride, cfg.temporal_consistency_pool_size,
    )                                              # (N, C)

    losses = []
    for offset in cfg.temporal_consistency_neighbor_offsets:
        t_idx = centre + int(offset)
        if t_idx < 0 or t_idx >= T:
            continue
        neigh_feat = multi_frame_feat[:, t_idx]    # (B, C, h, w)
        positives = pool_box_embeddings(
            neigh_feat, boxes, batch_idx, stride,
            cfg.temporal_consistency_pool_size,
        )                                          # (N, C)
        # Negatives: every OTHER box in the batch (cross-clip and
        # within-clip non-self).  Build per-anchor by excluding the row.
        N = anchors.shape[0]
        if N <= 1:
            continue
        # Vectorised: each anchor sees the whole pool except itself.
        eye = torch.eye(N, dtype=torch.bool, device=device)
        neg_mask = ~eye                            # (N, N)
        # For each anchor i: negatives = positives[neg_mask[i]] (shape N-1, C)
        # We can compute the loss in a single pass without materialising
        # per-anchor negative tensors by replacing self-similarity with -inf
        # in the logits matrix.
        a = torch.nn.functional.normalize(anchors.float(), dim=1, eps=1e-6)
        p = torch.nn.functional.normalize(positives.float(), dim=1, eps=1e-6)
        T_ = cfg.temporal_consistency_temperature
        pos_logits = (a * p).sum(dim=1, keepdim=True) / T_              # (N, 1)
        # Negatives = all OTHER positives (i.e. neighbour embeddings) plus
        # all OTHER anchors. Concatenate both pools.
        neg_logits_p = a @ p.t() / T_                                   # (N, N)
        neg_logits_a = a @ a.t() / T_                                   # (N, N)
        neg_logits_p = neg_logits_p.masked_fill(eye, float("-inf"))
        neg_logits_a = neg_logits_a.masked_fill(eye, float("-inf"))
        logits = torch.cat([pos_logits, neg_logits_p, neg_logits_a], dim=1)
        target = torch.zeros(N, dtype=torch.long, device=device)
        losses.append(torch.nn.functional.cross_entropy(logits, target))

    if not losses:
        return multi_frame_feat.sum() * 0.0
    return torch.stack(losses).mean()
