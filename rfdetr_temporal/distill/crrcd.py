"""Cross-Resolution Relational Contrastive Distillation (CRRCD).

Operates on the **decoder hidden states** captured at the same query slots
on both the frozen HR teacher and the trainable LR student (the KD-DETR
specific-sampling branch already guarantees per-slot alignment via the
``forward_pre_hook`` injection — see ``DISTILLATION.md``).

Two small MLPs (Feature Relation Modules) model relations between
foreground / background slots:

    F^t  : (e^t_i, e^t_j)  →  v^t_{i,j}        (teacher–teacher)
    F^ts : (e^t_i, e^s_j)  →  v^{t,s}_{i,j}    (teacher–student)

A sigmoid-NCE critic ``h(u, v) = σ(cos(u, v) / τ)`` forces the cross-space
relation to mimic the teacher reference at matching slot pairs:

    L_rcd = - mean log h(v^t_{i,j}, v^{t,s}_{i,j})
            - mean Σ_{k≠j} log [1 − h(v^t_{i,j}, v^{t,s}_{i,k})]

Gradient flow:
    • e^t is detached at the FRM input → no gradient flows into the
      frozen teacher backbone / decoder (already frozen, this is belt &
      braces).
    • F^t and F^ts are trainable here.
    • e^s carries the graph back into the student decoder + temporal
      fusion, which is the desired learning signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _RelationMLP(nn.Module):
    """v = W2 · ReLU(W1 (e_i − e_j))."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        return self.net(diff)


class CRRCDLoss(nn.Module):
    """Cross-Resolution Relational Contrastive Distillation loss."""

    def __init__(
        self,
        hidden_dim: int,
        relation_dim: int,
        frm_hidden_dim: int,
        num_fg: int,
        num_bg: int,
        num_negatives: int,
        temperature: float,
    ):
        super().__init__()
        self.F_t = _RelationMLP(hidden_dim, frm_hidden_dim, relation_dim)
        self.F_ts = _RelationMLP(hidden_dim, frm_hidden_dim, relation_dim)
        self.K_fg = int(num_fg)
        self.K_bg = int(num_bg)
        self.n_neg = int(num_negatives)
        self.tau = float(temperature)

    def forward(
        self,
        teacher_hs: torch.Tensor,   # (B, Q, D)  detached
        student_hs: torch.Tensor,   # (B, Q, D)  with grad
        weights: torch.Tensor,      # (B, Q)     teacher max-fg confidence
    ) -> torch.Tensor:
        assert teacher_hs.shape == student_hs.shape, (
            f"shape mismatch: teacher_hs {tuple(teacher_hs.shape)} vs "
            f"student_hs {tuple(student_hs.shape)}"
        )
        assert weights.shape[:2] == teacher_hs.shape[:2], (
            f"weights {tuple(weights.shape)} must match (B, Q) of hs"
        )

        # Belt & braces — never let any gradient reach the teacher.
        e_t = teacher_hs.detach()
        e_s = student_hs

        B, Q, D = e_t.shape
        K_fg = min(self.K_fg, Q)
        K_bg = min(self.K_bg, Q)
        if K_fg == 0 or K_bg < 2:
            return e_s.new_zeros(())

        # Top-K_fg by foreground weight; bottom-K_bg by foreground weight.
        fg_idx = weights.topk(K_fg, dim=1).indices                      # (B, K_fg)
        bg_idx = weights.topk(K_bg, dim=1, largest=False).indices       # (B, K_bg)

        def gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
            return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))

        et_fg = gather(e_t, fg_idx)        # (B, K_fg, D)
        et_bg = gather(e_t, bg_idx)        # (B, K_bg, D)
        es_bg = gather(e_s, bg_idx)        # (B, K_bg, D)  — carries grad

        # Pairwise differences  (B, K_fg, K_bg, D).
        diff_t = et_fg.unsqueeze(2) - et_bg.unsqueeze(1)
        diff_ts = et_fg.unsqueeze(2) - es_bg.unsqueeze(1)

        v_t = self.F_t(diff_t)             # (B, K_fg, K_bg, R)
        v_ts = self.F_ts(diff_ts)          # (B, K_fg, K_bg, R)

        # Sigmoid critic on cosine similarity (numerically stable via
        # F.logsigmoid). For each anchor (b, i, j):
        #   pos: <v_t[b,i,j], v_ts[b,i,j]>
        #   neg: <v_t[b,i,j], v_ts[b,i,k]> for k ≠ j
        v_t_n = F.normalize(v_t, dim=-1)
        v_ts_n = F.normalize(v_ts, dim=-1)

        # Full similarity tensor: (B, K_fg, K_bg, K_bg) — last two dims are (j, k).
        sim = torch.einsum("bijd,bikd->bijk", v_t_n, v_ts_n) / max(self.tau, 1e-6)

        device = sim.device
        eye = torch.eye(K_bg, device=device, dtype=torch.bool)         # (K_bg, K_bg)
        eye = eye.view(1, 1, K_bg, K_bg)                               # broadcastable

        # Positive term: diagonal (j == k).
        pos_sim = torch.diagonal(sim, dim1=-2, dim2=-1)                # (B, K_fg, K_bg)
        log_h_pos = F.logsigmoid(pos_sim)

        # Negative term: log(1 - σ(s)) = logsigmoid(-s); zero out diagonal.
        log_one_minus_h_neg = F.logsigmoid(-sim).masked_fill(eye, 0.0)

        # Optional negative subsampling (per anchor) to honour the "n
        # negatives" parameter from the CRRCD spec; otherwise use all
        # K_bg-1 off-diagonal slots.
        if 0 < self.n_neg < (K_bg - 1):
            rand = torch.rand_like(sim).masked_fill(eye, -1.0)
            sel = rand.topk(self.n_neg, dim=-1).indices                # (B, K_fg, K_bg, n_neg)
            neg_term = log_one_minus_h_neg.gather(-1, sel).sum(dim=-1)  # (B, K_fg, K_bg)
        else:
            neg_term = log_one_minus_h_neg.sum(dim=-1)                  # (B, K_fg, K_bg)

        # Mean over anchors for scale stability across (B, K_fg, K_bg).
        loss = -(log_h_pos.mean() + neg_term.mean())
        return loss
