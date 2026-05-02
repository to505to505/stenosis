"""Multi-frame count-consistency loss (L_num).

For a window of T frames, count how many predictions exceed a fixed
score threshold on each frame. Penalise the absolute deviation of each
per-frame count ``n_t`` from the window's median ``n_r``:

    L_num = (1/T) Σ_t |n_t − n_r|.

A soft surrogate (sigmoid temperature ``soft_temp``) is used so the
gradient flows through the count, otherwise the hard ``(p > thr).sum()``
would be non-differentiable.
"""

from __future__ import annotations

import torch


def num_consistency_loss(
    pred_logits: torch.Tensor,
    threshold: float,
    soft_temp: float = 0.05,
) -> torch.Tensor:
    """Args:
        pred_logits: (B, T, Q, K) raw classification logits.
        threshold: confidence above which a query is counted as a box.
        soft_temp: temperature of the soft-count sigmoid surrogate;
            smaller values approach a hard step but reduce gradient.
    Returns:
        Scalar tensor.
    """
    if pred_logits.dim() != 4:
        raise ValueError(
            f"pred_logits must be (B, T, Q, K), got {tuple(pred_logits.shape)}"
        )
    B, T, Q, _K = pred_logits.shape
    if T < 2:
        return pred_logits.new_zeros(())

    # Per-query foreground prob (max over classes).
    p = pred_logits.sigmoid().amax(dim=-1)            # (B, T, Q)

    # Soft count: sum_q sigmoid((p - thr) / temp).
    soft_temp = max(float(soft_temp), 1e-6)
    soft_indicator = torch.sigmoid((p - float(threshold)) / soft_temp)  # (B, T, Q)
    n_t = soft_indicator.sum(dim=-1)                  # (B, T)

    # Median across T (consensus count). ``median`` is non-differentiable
    # at the median index, but its detached form is fine — we only need
    # a stable target; the gradient flows through ``n_t``.
    n_r = n_t.detach().median(dim=1, keepdim=True).values  # (B, 1)

    return (n_t - n_r).abs().mean()
