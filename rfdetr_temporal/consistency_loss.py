"""GT-anchored Sliding-Window Temporal Consistency Loss.

Both forward passes target the *same physical frame* f_{t+1}:

  - ``pred_a``: window A queried with ``predict_frame = centre + 1``
                 → predictions for frame f_{t+1} from the A-context.
  - ``pred_b``: window B (= window A shifted by +offset) queried at its natural
                 centre → predictions for the same frame f_{t+1} from the
                 B-context.

We Hungarian-match BOTH sides independently against the shared GT of f_{t+1}
(``targets``), using the same matcher the main detection criterion uses. Each
GT object thus produces a pair of slot indices (idx_A, idx_B) — those are the
slots that should answer for the same physical stenosis from both temporal
contexts, and we penalise their disagreement directly:

    L_cons = box_l1_weight * L1(boxes_a[idx_A], stop_grad(boxes_b[idx_B]))
           + kl_weight     * KL(B || A)   # one-directional: only A learns

Only A (the off-centre query) receives a gradient from this loss; B is detached
and is supervised exclusively by its own detection loss (loss_dict_b), so it
stays GT-anchored and acts as a clean teacher signal for A.

No top-K filtering: only GT-anchored slots contribute, so the loss focuses
exactly on the slots that should never flicker.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def temporal_consistency_loss(
    pred_a: dict,
    pred_b: dict,
    targets: list,
    matcher,
    *,
    kl_weight: float = 1.0,
    box_l1_weight: float = 5.0,
) -> torch.Tensor:
    """GT-anchored consistency between two query passes for the same frame.

    Args:
        pred_a, pred_b: dicts with
            ``pred_logits`` (B, Q, C) and ``pred_boxes`` (B, Q, 4) [cxcywh, normalised].
        targets: list of length B; each item ``{"boxes": (N,4), "labels": (N,)}``
            describing the GT of the *shared* physical frame.
        matcher: the criterion's HungarianMatcher (called with @torch.no_grad()).
        kl_weight: weight on symmetric-KL classification consistency term.
        box_l1_weight: weight on L1 box term.

    Returns:
        Scalar loss averaged over batch items that have ≥1 matched GT.
    """
    logits_a = pred_a["pred_logits"]   # (B, Q, C)
    boxes_a = pred_a["pred_boxes"]     # (B, Q, 4)  cxcywh, normalised
    logits_b = pred_b["pred_logits"]
    boxes_b = pred_b["pred_boxes"]

    assert logits_a.shape == logits_b.shape, (
        f"shape mismatch: {logits_a.shape} vs {logits_b.shape}"
    )
    B = logits_a.size(0)

    probs_a = logits_a.sigmoid()
    probs_b = logits_b.sigmoid()

    # Hungarian-match each side against the SAME GT.
    # matcher returns list[(pred_idx_tensor, tgt_idx_tensor)] of length B.
    indices_a = matcher({"pred_logits": logits_a.detach(),
                         "pred_boxes":  boxes_a.detach()}, targets)
    indices_b = matcher({"pred_logits": logits_b.detach(),
                         "pred_boxes":  boxes_b.detach()}, targets)

    losses = []
    eps = 1e-6
    for i in range(B):
        pa_idx, ta_idx = indices_a[i]   # slot idx in A, GT idx
        pb_idx, tb_idx = indices_b[i]   # slot idx in B, GT idx
        if pa_idx.numel() == 0 or pb_idx.numel() == 0:
            continue

        # Build (GT idx → A slot) and (GT idx → B slot) mappings, then keep
        # only GT objects matched on BOTH sides. This pairs A and B by the
        # physical stenosis they're responsible for.
        a_map = {int(t.item()): int(p.item()) for p, t in zip(pa_idx, ta_idx)}
        b_map = {int(t.item()): int(p.item()) for p, t in zip(pb_idx, tb_idx)}
        common_gt = sorted(set(a_map.keys()) & set(b_map.keys()))
        if not common_gt:
            continue

        a_slots = torch.tensor([a_map[g] for g in common_gt],
                               device=boxes_a.device, dtype=torch.long)
        b_slots = torch.tensor([b_map[g] for g in common_gt],
                               device=boxes_b.device, dtype=torch.long)

        ba_m = boxes_a[i, a_slots]    # (M, 4)  with grad
        bb_m = boxes_b[i, b_slots]
        pa_m = probs_a[i, a_slots]    # (M, C)  with grad
        pb_m = probs_b[i, b_slots]

        # Box L1: only A receives gradient (B is the GT-anchored teacher).
        loss_l1 = F.l1_loss(ba_m, bb_m.detach(), reduction="mean")

        # One-directional KL(B || A): pull A's distribution towards B.
        # B is detached — its gradient comes only from loss_dict_b.
        pa_c = pa_m.clamp(eps, 1.0 - eps)
        pb_c = pb_m.detach().clamp(eps, 1.0 - eps)
        loss_kl = (
            pb_c * (pb_c.log() - pa_c.log())
            + (1.0 - pb_c) * ((1.0 - pb_c).log() - (1.0 - pa_c).log())
        ).mean()

        item_loss = box_l1_weight * loss_l1 + kl_weight * loss_kl
        losses.append(item_loss)

    if not losses:
        return logits_a.new_zeros(())

    return torch.stack(losses).mean()
