"""Sliding-Window Temporal Consistency Loss.

Given two student forward passes that should yield (approximately) the same
detections on the same physical frame:

  - ``pred_a``: window A queried with ``predict_frame = centre + 1``  →
                 predictions for frame ``f_{t+1}``
  - ``pred_b``: window B (= window A shifted by +1) queried at its natural
                 centre → predictions also for frame ``f_{t+1}``

we Hungarian-match the top-K most confident predictions from each side
(treating side B as the matching reference, both sides receive gradients) and
penalise box-coordinate disagreement (L1 + 1−IoU) and class-distribution
disagreement (symmetric KL on sigmoid probabilities, since the codebase uses
single-class detection).

Outputs and signatures follow the standard RF-DETR head:
    pred["pred_logits"] : (B, Q, num_classes)
    pred["pred_boxes"]  : (B, Q, 4)   in cxcywh, normalised [0,1]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ─────────────────────────── geometry helpers ────────────────────────────

def _cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h,
                        cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def _pairwise_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """boxes in xyxy normalised coords. Returns (Na, Nb) IoU matrix."""
    Na, Nb = boxes_a.size(0), boxes_b.size(0)
    if Na == 0 or Nb == 0:
        return boxes_a.new_zeros(Na, Nb)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * \
             (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * \
             (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0)
    lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])    # (Na,Nb,2)
    rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp(min=1e-6)


# ─────────────────────────── main loss function ──────────────────────────

def temporal_consistency_loss(
    pred_a: dict,
    pred_b: dict,
    *,
    top_k: int = 20,
    l1_weight: float = 5.0,
    iou_cost_weight: float = 2.0,
    cls_cost_weight: float = 1.0,
    kl_weight: float = 1.0,
    box_l1_weight: float = 5.0,
) -> torch.Tensor:
    """Compute consistency loss between two query passes that target the same
    physical frame.

    Args:
        pred_a, pred_b: dicts with ``pred_logits`` (B,Q,C) and ``pred_boxes`` (B,Q,4).
        top_k: number of most-confident predictions kept per side per item.
        l1_weight, iou_cost_weight, cls_cost_weight: matching cost weights.
        kl_weight: weight on symmetric-KL classification consistency term.
        box_l1_weight: weight on L1 box term in the *loss* (separate from cost).

    Returns:
        Scalar loss averaged over batch items that have ≥1 valid match.
    """
    logits_a = pred_a["pred_logits"]   # (B, Q, C)
    boxes_a = pred_a["pred_boxes"]     # (B, Q, 4)  cxcywh, normalised
    logits_b = pred_b["pred_logits"]
    boxes_b = pred_b["pred_boxes"]

    assert logits_a.shape == logits_b.shape, (
        f"shape mismatch: {logits_a.shape} vs {logits_b.shape}"
    )
    B, Q, _C = logits_a.shape
    K = min(top_k, Q)

    probs_a = logits_a.sigmoid()       # (B,Q,C)
    probs_b = logits_b.sigmoid()
    score_a = probs_a.max(dim=-1).values   # (B,Q)
    score_b = probs_b.max(dim=-1).values

    # Per-item top-K selection
    _, topk_a = score_a.topk(K, dim=1)     # (B,K)
    _, topk_b = score_b.topk(K, dim=1)

    losses = []
    for i in range(B):
        ia, ib = topk_a[i], topk_b[i]
        ba = boxes_a[i, ia]                # (K,4)
        bb = boxes_b[i, ib]
        pa = probs_a[i, ia]                # (K,C)
        pb = probs_b[i, ib]

        # Build cost matrix on detached values (matching only).
        with torch.no_grad():
            ba_xyxy = _cxcywh_to_xyxy(ba)
            bb_xyxy = _cxcywh_to_xyxy(bb)
            iou = _pairwise_iou(ba_xyxy, bb_xyxy)        # (K,K)
            cost_iou = 1.0 - iou
            cost_l1 = torch.cdist(ba, bb, p=1)           # (K,K)
            # Class agreement cost: 1 − cosine on prob vectors (single-class
            # → reduces to |pa-pb| basically). Cheap & well-defined.
            cost_cls = 1.0 - F.cosine_similarity(
                pa.unsqueeze(1).expand(-1, K, -1),
                pb.unsqueeze(0).expand(K, -1, -1),
                dim=-1,
            )
            cost = (
                iou_cost_weight * cost_iou
                + l1_weight * cost_l1
                + cls_cost_weight * cost_cls
            )
            cost_np = cost.detach().cpu().numpy()
            row_idx, col_idx = linear_sum_assignment(cost_np)

        if len(row_idx) == 0:
            continue

        ri = torch.as_tensor(row_idx, device=ba.device, dtype=torch.long)
        ci = torch.as_tensor(col_idx, device=bb.device, dtype=torch.long)

        ba_m = ba[ri]      # (M,4)  with grad
        bb_m = bb[ci]
        pa_m = pa[ri]      # (M,C)
        pb_m = pb[ci]

        # Box L1 (gradients flow to both sides).
        loss_l1 = F.l1_loss(ba_m, bb_m, reduction="mean")

        # Symmetric KL on Bernoulli probs per class, averaged over (M,C).
        eps = 1e-6
        pa_c = pa_m.clamp(eps, 1.0 - eps)
        pb_c = pb_m.clamp(eps, 1.0 - eps)
        kl_ab = (
            pa_c * (pa_c.log() - pb_c.log())
            + (1.0 - pa_c) * ((1.0 - pa_c).log() - (1.0 - pb_c).log())
        )
        kl_ba = (
            pb_c * (pb_c.log() - pa_c.log())
            + (1.0 - pb_c) * ((1.0 - pb_c).log() - (1.0 - pa_c).log())
        )
        loss_kl = 0.5 * (kl_ab + kl_ba).mean()

        item_loss = box_l1_weight * loss_l1 + kl_weight * loss_kl
        losses.append(item_loss)

    if not losses:
        return logits_a.new_zeros(())

    return torch.stack(losses).mean()
