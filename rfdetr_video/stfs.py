"""Query-Level Spatio-Temporal Feature Sharing (STFS).

Workflow on a window of T frames after the first decoder pass:

    1. ``track_queries`` chains confident query slots across frames using
       per-pair Hungarian assignment with a cost balancing IoU, centre
       distance and class probability. Tracks shorter than
       ``cfg.stfs_min_track_len`` are dropped as Hypothesis-False-Positive
       noise. For each surviving track, frames where the slot dropped
       (no match above ``cfg.stfs_match_iou_thresh``) are flagged
       Hypothesis-False-Negative (H-FN).

    2. ``inject_features`` replaces every H-FN slot's query embedding +
       reference point with the embedding / refpoint from the strongest
       frame in that track (max class probability). Implemented with
       ``torch.where`` so source-slot gradients flow into the decoder
       through the injected slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


@dataclass
class _Track:
    """A chain of (frame, slot) pairs across a window."""
    slots: List[int]                # length T; -1 ⇒ H-FN at that frame
    last_box: np.ndarray            # cxcywh, latest known box
    best_t: int                     # frame with the highest class prob seen
    best_p: float                   # the corresponding probability


# ─────────────────────────────────────────────────────────────────────
#  Geometry helpers (numpy, used inside torch.no_grad tracking)
# ─────────────────────────────────────────────────────────────────────
def _cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N,4) and (M,4) cxcywh box arrays."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_xy = _cxcywh_to_xyxy(a)
    b_xy = _cxcywh_to_xyxy(b)
    x1 = np.maximum(a_xy[:, None, 0], b_xy[None, :, 0])
    y1 = np.maximum(a_xy[:, None, 1], b_xy[None, :, 1])
    x2 = np.minimum(a_xy[:, None, 2], b_xy[None, :, 2])
    y2 = np.minimum(a_xy[:, None, 3], b_xy[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    a_area = (a_xy[:, 2] - a_xy[:, 0]) * (a_xy[:, 3] - a_xy[:, 1])
    b_area = (b_xy[:, 2] - b_xy[:, 0]) * (b_xy[:, 3] - b_xy[:, 1])
    union = a_area[:, None] + b_area[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ─────────────────────────────────────────────────────────────────────
#  Tracking
# ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def track_queries(
    pred_boxes: torch.Tensor,        # (B, T, Q, 4) cxcywh
    pred_logits: torch.Tensor,       # (B, T, Q, K)
    *,
    iou_weight: float,
    l1_weight: float,
    cls_weight: float,
    iou_gate: float,
    score_thresh: float,
    min_track_len: int,
) -> List[List[_Track]]:
    """Return per-batch lists of surviving tracks."""
    B, T, Q, _ = pred_boxes.shape
    probs = pred_logits.sigmoid().amax(dim=-1).cpu().numpy()    # (B, T, Q)
    boxes = pred_boxes.detach().cpu().numpy()                   # (B, T, Q, 4)

    out: List[List[_Track]] = []
    for b in range(B):
        # Seed tracks from confident slots on frame 0.
        tracks: List[_Track] = []
        for q in range(Q):
            if probs[b, 0, q] >= score_thresh:
                tracks.append(_Track(
                    slots=[q] + [-1] * (T - 1),
                    last_box=boxes[b, 0, q].copy(),
                    best_t=0,
                    best_p=float(probs[b, 0, q]),
                ))

        for t in range(1, T):
            # Candidate confident slots on frame t.
            cand_idx = np.where(probs[b, t] >= score_thresh)[0]
            if not tracks or cand_idx.size == 0:
                # No matches possible; queries that exist start new tracks.
                for q in cand_idx:
                    tracks.append(_Track(
                        slots=[-1] * t + [int(q)] + [-1] * (T - 1 - t),
                        last_box=boxes[b, t, q].copy(),
                        best_t=t,
                        best_p=float(probs[b, t, q]),
                    ))
                continue

            track_boxes = np.stack([tr.last_box for tr in tracks], axis=0)
            cand_boxes = boxes[b, t, cand_idx]
            cand_probs = probs[b, t, cand_idx]

            iou = _iou_matrix(track_boxes, cand_boxes)            # (N_tr, N_cand)
            l1 = np.abs(
                track_boxes[:, None, :2] - cand_boxes[None, :, :2]
            ).sum(-1)                                             # (N_tr, N_cand)
            cost = (
                iou_weight * (1.0 - iou)
                + l1_weight * l1
                + cls_weight * (1.0 - cand_probs)[None, :]
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            matched_cand: set = set()
            matched_track: set = set()
            for r, c in zip(row_ind, col_ind):
                if iou[r, c] < iou_gate:
                    continue                                      # H-FN at frame t
                tr = tracks[r]
                tr.slots[t] = int(cand_idx[c])
                tr.last_box = cand_boxes[c].copy()
                p = float(cand_probs[c])
                if p > tr.best_p:
                    tr.best_p = p
                    tr.best_t = t
                matched_track.add(r)
                matched_cand.add(c)

            # Unmatched candidates start new tracks.
            for c in range(len(cand_idx)):
                if c in matched_cand:
                    continue
                q = int(cand_idx[c])
                tracks.append(_Track(
                    slots=[-1] * t + [q] + [-1] * (T - 1 - t),
                    last_box=boxes[b, t, q].copy(),
                    best_t=t,
                    best_p=float(probs[b, t, q]),
                ))

        # H-FP filter: a slot count below ``min_track_len`` is suspect.
        kept = [
            tr for tr in tracks
            if sum(1 for s in tr.slots if s >= 0) >= min_track_len
        ]
        out.append(kept)
    return out


# ─────────────────────────────────────────────────────────────────────
#  Feature injection
# ─────────────────────────────────────────────────────────────────────
def inject_features(
    query_embed: torch.Tensor,       # (B, T, Q, D) — gradient required
    refpoint: torch.Tensor,          # (B, T, Q, 4) — gradient detached upstream OK
    tracks_per_batch: List[List[_Track]],
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace H-FN slot embeddings + refpoints with the source-frame
    counterpart from the same track. Returns autograd-safe new tensors.
    """
    B, T, Q, D = query_embed.shape
    device = query_embed.device

    # Build (B, T, Q) gather indices along the T axis. Default ``t`` ⇒
    # identity (no change). For each H-FN slot we set the source frame.
    src_t = torch.arange(T, device=device).view(1, T, 1).expand(B, T, Q).clone()
    src_q = torch.arange(Q, device=device).view(1, 1, Q).expand(B, T, Q).clone()
    inject_mask = torch.zeros(B, T, Q, dtype=torch.bool, device=device)

    for b, tracks in enumerate(tracks_per_batch):
        for tr in tracks:
            best_t = tr.best_t
            best_q = tr.slots[best_t]
            if best_q < 0:
                continue                              # safety
            for t in range(T):
                if tr.slots[t] < 0:
                    # Choose any anchor slot for the H-FN frame: prefer the
                    # *current* slot index from the strongest frame so the
                    # downstream class/box heads of slot ``best_q`` carry
                    # the rich representation across.
                    target_q = best_q
                    src_t[b, t, target_q] = best_t
                    src_q[b, t, target_q] = best_q
                    inject_mask[b, t, target_q] = True

    # Gather sources with autograd-safe indexing.
    src_idx_emb = (
        src_t.unsqueeze(-1) * Q + src_q.unsqueeze(-1)
    ).expand(-1, -1, -1, D)
    src_idx_box = (
        src_t.unsqueeze(-1) * Q + src_q.unsqueeze(-1)
    ).expand(-1, -1, -1, 4)
    flat_emb = query_embed.reshape(B, T * Q, D)
    flat_box = refpoint.reshape(B, T * Q, 4)
    src_emb = flat_emb.gather(1, src_idx_emb.reshape(B, T * Q, D)).reshape(B, T, Q, D)
    src_box = flat_box.gather(1, src_idx_box.reshape(B, T * Q, 4)).reshape(B, T, Q, 4)

    mix = float(alpha)
    mask_emb = inject_mask.unsqueeze(-1).to(query_embed.dtype)
    mask_box = inject_mask.unsqueeze(-1).to(refpoint.dtype)

    new_emb = torch.where(
        inject_mask.unsqueeze(-1),
        (1.0 - mix) * query_embed + mix * src_emb,
        query_embed,
    )
    new_box = torch.where(
        inject_mask.unsqueeze(-1),
        src_box,                      # geometry from the strong frame
        refpoint,
    )
    # Reference the alpha-blend mask values to silence linting on the
    # explicit dtype variables (kept available for future schedules).
    _ = mask_emb, mask_box
    return new_emb, new_box
