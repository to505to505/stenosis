"""Query-Level Spatio-Temporal Feature Sharing (STFS).

Workflow on a window of T frames after the first decoder pass:

    1. ``track_queries`` chains confident query slots across frames using
       per-pair Hungarian assignment with a cost balancing IoU, centre
       distance and class probability. Tracks shorter than
       ``cfg.stfs_min_track_len`` are dropped as Hypothesis-False-Positive
       noise. For each surviving track, frames where the slot dropped
       (no match above ``cfg.stfs_match_iou_thresh``) are flagged
       Hypothesis-False-Negative (H-FN).

    2. ``inject_features`` produces enriched ``(query_embed, refpoint)``
       tensors for the refinement decoder. For every H-FN slot:
         * embeddings are mixed with the strong source slot's embedding
           via a learnable cross-attention :class:`FeatureAggregator`
           (Query = current weak embedding, Key/Value = strong source
           embedding). This avoids the geometry-breaking hard copy of
           ``torch.where`` and is analogous to the RoI Aggregator in
           STQD-Det / SFF in Stenosis-DetNet.
                 * reference points are taken from the strong source frame and a
                     deterministic :class:`RefPointShift` builds a 5-point proposal
                     grid (centre/up/down/left/right). The refinement layer samples
                     visual features at these explicit hypotheses instead of asking
                     an MLP to blindly regress an exact geometric offset.
             The aggregator is zero-initialised on its attention output projection;
             RefPointShift itself has no trainable parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
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
#  Soft aggregation modules (RoI Aggregator / SFF + Proposal-Shift)
# ─────────────────────────────────────────────────────────────────────
class DynamicConv1D(nn.Module):
    """1-D Dynamic Convolution for query embedding vectors (N, D).

    Generates per-sample channel-wise kernel weights from the input via a
    2-layer MLP (analogous to the spatial ``DynamicConv`` in STQD-Det / GFE
    but for D-dimensional embedding vectors instead of (C, H, W) maps):

        kernels = ReLU(FC1(x))         # (N, D) — content-adaptive weights
        out     = proj(kernels * x)    # depthwise-equivalent channel scaling

    ``proj`` is zero-initialised so that at init ``forward(x) ≈ 0``,
    preserving the warm-start identity behaviour of the parent module.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.kernel_gen = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.proj = nn.Linear(d_model, d_model)
        # Zero-init → output is 0 at start → residual path is identity.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, D) → (N, D)."""
        return self.proj(self.kernel_gen(x) * x)


class FeatureAggregator(nn.Module):
    """Cross-attention + Dynamic Convolution RoI Aggregator (STQD-Det / SFF analogue).

    Given a weak query embedding from frame ``t`` and the strong source
    embedding from the best frame of the same track, returns a softly
    aggregated embedding:

        x   = LN(q  + MHA(query=q, key=src, value=src))   # cross-attn mix
        out = LN(x  + DynConv1D(x))                        # adaptive channel filter

    Both the MHA output projection and the ``DynConv1D`` projection are
    zero-initialised so the module starts as identity; during training it
    learns to import information from the source slot and to adaptively
    re-weight channels for morphologically deformed / blurred frames.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dyn_conv = DynamicConv1D(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Zero-init MHA output projection → identity behaviour at start.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, q: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """q, src: (N, D) → (N, D). N is the number of H-FN slots."""
        if q.numel() == 0:
            return q
        q_seq = q.unsqueeze(1)        # (N, 1, D)
        kv_seq = src.unsqueeze(1)
        out, _ = self.attn(query=q_seq, key=kv_seq, value=kv_seq, need_weights=False)
        x = self.norm1(q_seq + out).squeeze(1)      # (N, D) — after MHA + LN
        return self.norm2(x + self.dyn_conv(x))      # DynConv1D residual + LN


class RefPointShift(nn.Module):
    """Deterministic proposal-shift grid for STFS H-FN refpoints.

    The previous implementation used an MLP to regress ``Δcxcywh`` from
    query embeddings alone. That is geometrically blind: the module has no
    access to local pixels while deciding where the artery moved. This class
    instead returns five explicit hypotheses around the source refpoint:

        centre, up, down, left, right

    Each hypothesis uses the source wh expanded by ``padding_alpha``. The
    refinement layer then runs deformable cross-attention on these candidate
    boxes, grounding the choice in visual features. ``hidden_dim`` is kept in
    the constructor for checkpoint / CLI compatibility but is intentionally
    unused.
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: Optional[int] = None,
        padding_alpha: float = 1.5,
    ):
        super().__init__()
        self.padding_alpha = float(padding_alpha)
        _ = d_model, hidden_dim

    def forward(
        self,
        cur_emb: torch.Tensor,        # (N, D)  current weak embedding
        src_emb: torch.Tensor,        # (N, D)  strong source embedding
        src_ref: torch.Tensor,        # (N, 4)  cxcywh refpoint of source
    ) -> torch.Tensor:
        """Return candidate refpoints with shape ``(N, 5, 4)``."""
        if cur_emb.numel() == 0:
            return src_ref.new_zeros((0, 5, 4))
        _ = cur_emb, src_emb
        wh = (src_ref[..., 2:] * self.padding_alpha).clamp(min=1e-6)
        cxcy = src_ref[..., :2]
        half_step = 0.5 * wh
        offsets = torch.stack([
            torch.zeros_like(cxcy),
            torch.stack([torch.zeros_like(half_step[..., 0]), -half_step[..., 1]], dim=-1),
            torch.stack([torch.zeros_like(half_step[..., 0]), half_step[..., 1]], dim=-1),
            torch.stack([-half_step[..., 0], torch.zeros_like(half_step[..., 1])], dim=-1),
            torch.stack([half_step[..., 0], torch.zeros_like(half_step[..., 1])], dim=-1),
        ], dim=1)
        candidate_cxcy = (cxcy.unsqueeze(1) + offsets).clamp(0.0, 1.0)
        candidate_wh = wh.unsqueeze(1).expand(-1, 5, -1)
        return torch.cat([candidate_cxcy, candidate_wh], dim=-1)


# ─────────────────────────────────────────────────────────────────────
#  Feature injection
# ─────────────────────────────────────────────────────────────────────
def inject_features(
    query_embed: torch.Tensor,       # (B, T, Q, D) — gradient required
    refpoint: torch.Tensor,          # (B, T, Q, 4) — gradient detached upstream OK
    tracks_per_batch: List[List[_Track]],
    *,
    alpha: float,
    aggregator: Optional[FeatureAggregator] = None,
    shifter: Optional[RefPointShift] = None,
    return_shift_candidates: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[
    torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor,
]:
    """Build refined ``(query_embed, refpoint)`` for the refinement pass.

    For H-FN slots the embedding and refpoint are taken from the
    strongest in-track frame:

      * If ``aggregator`` is provided → the H-FN embedding is replaced by
        ``FeatureAggregator(weak, strong)`` (soft mix via cross-attention).
        Otherwise the legacy α-blend ``(1-α)·weak + α·strong`` is used.
            * If ``shifter`` is provided → a 5-point proposal grid is generated
                around the strong source refpoint. The centre candidate is used as
                the default refpoint and the full grid can be returned for sparse
                candidate refinement. Otherwise the strong source refpoint is copied
                as-is.

    Non-H-FN slots are returned unchanged.
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

    # ── Embedding path ──────────────────────────────────────────────
    if aggregator is not None:
        # Gather only the H-FN slots, run cross-attention once, scatter.
        flat_mask = inject_mask.reshape(-1)              # (B*T*Q,)
        if flat_mask.any():
            flat_q = query_embed.reshape(-1, D)
            flat_s = src_emb.reshape(-1, D)
            cur_sel = flat_q[flat_mask]                  # (N, D)
            src_sel = flat_s[flat_mask]                  # (N, D)
            mixed = aggregator(cur_sel, src_sel)         # (N, D)
            new_emb_flat = flat_q.clone()
            new_emb_flat[flat_mask] = mixed
            new_emb = new_emb_flat.reshape(B, T, Q, D)
        else:
            new_emb = query_embed
    else:
        mix = float(alpha)
        new_emb = torch.where(
            inject_mask.unsqueeze(-1),
            (1.0 - mix) * query_embed + mix * src_emb,
            query_embed,
        )

    # ── Reference-point path ────────────────────────────────────────
    shift_candidates: Optional[torch.Tensor] = None
    if shifter is not None:
        flat_mask = inject_mask.reshape(-1)
        if return_shift_candidates:
            shift_candidates = refpoint.unsqueeze(-2).expand(-1, -1, -1, 5, -1).clone()
        if flat_mask.any():
            flat_q = query_embed.reshape(-1, D)
            flat_s_emb = src_emb.reshape(-1, D)
            flat_s_box = src_box.reshape(-1, 4)
            flat_box_in = refpoint.reshape(-1, 4)
            cur_sel = flat_q[flat_mask]
            src_emb_sel = flat_s_emb[flat_mask]
            src_box_sel = flat_s_box[flat_mask]
            candidates = shifter(cur_sel, src_emb_sel, src_box_sel)  # (N, 5, 4)
            new_box_flat = flat_box_in.clone()
            new_box_flat[flat_mask] = candidates[:, 0]
            new_box = new_box_flat.reshape(B, T, Q, 4)
            if shift_candidates is not None:
                shift_candidates.reshape(-1, 5, 4)[flat_mask] = candidates
        else:
            new_box = refpoint
    else:
        new_box = torch.where(
            inject_mask.unsqueeze(-1),
            src_box,                      # geometry from the strong frame
            refpoint,
        )

    if return_shift_candidates:
        return new_emb, new_box, shift_candidates, inject_mask
    return new_emb, new_box
