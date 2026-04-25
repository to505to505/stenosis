"""Temporal Shift Module (TSM).

Parameter-free temporal mixing for sequences folded into the batch dimension
as ``(B*T, C, H, W)``.  A small fraction of channels is shifted forward and
backward in time so neighbouring frames can exchange information without
adding any learnable parameters or noticeable FLOPs.

Reference: Lin et al., "TSM: Temporal Shift Module for Efficient Video
Understanding" (ICCV 2019).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def temporal_shift(x: torch.Tensor, T: int, fold_div: int = 8) -> torch.Tensor:
    """Bidirectional temporal channel shift.

    Args:
        x: ``(B*T, C, H, W)`` tensor with frames packed along the batch axis,
            ordered as ``[b0_t0, b0_t1, ..., b0_t(T-1), b1_t0, ...]``.
        T: number of frames per clip.
        fold_div: channels are split into ``fold_div`` folds; one fold is
            shifted forward, one backward, the rest is unchanged.

    Returns:
        Tensor of the same shape and dtype as ``x``.
    """
    if T <= 1:
        return x
    BT, C, H, W = x.shape
    assert BT % T == 0, f"batch dim {BT} not divisible by T={T}"
    B = BT // T
    fold = max(C // fold_div, 1)

    x = x.view(B, T, C, H, W)
    out = torch.zeros_like(x)
    # forward shift: channel fold [0:fold] from t-1 → t
    out[:, 1:, :fold] = x[:, :-1, :fold]
    # backward shift: channel fold [fold:2*fold] from t+1 → t
    out[:, :-1, fold:2 * fold] = x[:, 1:, fold:2 * fold]
    # unchanged
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
    return out.view(BT, C, H, W)


class TSMState:
    """Mutable container so a single ``T`` value can be shared by many hooks
    and updated dynamically (e.g. for inference with a different clip length).
    """

    def __init__(self, T: int, fold_div: int = 8, enabled: bool = True):
        self.T = T
        self.fold_div = fold_div
        self.enabled = enabled


def _make_pre_hook(state: TSMState):
    def _hook(_module, inputs):
        if not state.enabled:
            return None
        # Module called as forward(x); pre-hooks receive the args tuple.
        if not inputs:
            return None
        x = inputs[0]
        if not torch.is_tensor(x) or x.dim() != 4:
            return None
        shifted = temporal_shift(x, state.T, state.fold_div)
        return (shifted,) + inputs[1:]
    return _hook


def install_tsm_hooks(
    modules: List[nn.Module],
    T: int,
    fold_div: int = 8,
    enabled: bool = True,
) -> TSMState:
    """Attach a forward-pre-hook to each module in ``modules``.

    Returns the shared :class:`TSMState`.  Toggle ``state.enabled`` or change
    ``state.T`` at runtime.  The handles are stored as ``module._tsm_handle``
    so they can be removed later if needed.
    """
    state = TSMState(T=T, fold_div=fold_div, enabled=enabled)
    hook = _make_pre_hook(state)
    for m in modules:
        handle = m.register_forward_pre_hook(hook)
        m._tsm_handle = handle  # type: ignore[attr-defined]
    return state
