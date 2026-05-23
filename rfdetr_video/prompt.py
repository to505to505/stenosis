"""Temporally-propagated learnable prompts for parameter-efficient tuning.

The "Prompt Tuning" baseline: the 2-D RF-DETR is fully frozen. A small
bank of learnable "visual prompts" ``P ∈ (n_prompts, d_model)`` is
evolved across the T frames of a video window via a ``GRUCell``
conditioned on the pooled backbone feature of each frame. The evolved
per-frame prompt is then *added* to the first ``n_prompts`` rows of the
decoder's query content (``tgt``) before the frozen decoder runs.

Architecturally this matches the literature description: a small number
of learnable prompt tokens that propagate frame-to-frame to accumulate
motion information, and which can only influence the detector through
the existing query/decoder interface (no new spatial path).
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


def _largest_feature_channel(srcs: Iterable[torch.Tensor]) -> int:
    """Pick the channel count of the lowest-resolution feature map.

    This is the canonical "semantic" feature in a multi-scale FPN — the
    one used by most detection heads. Using a single fixed scale keeps
    the projector dimensions stable across forward calls.
    """
    last = list(srcs)[-1]
    return int(last.shape[1])


class TemporalPromptBank(nn.Module):
    """Learnable prompts propagated across frames via a GRUCell.

    Args:
        n_prompts: number of learnable prompt tokens.
        d_model: hidden dimension of the detector (== decoder D).
        feat_channels: channel count of the pooled backbone feature
            (set once at first forward via :meth:`lazy_init_projector`
            if not provided here).
        init_std: stddev for the prompt-bank random initialisation.
    """

    def __init__(
        self,
        n_prompts: int,
        d_model: int,
        feat_channels: int | None = None,
        init_std: float = 0.02,
    ):
        super().__init__()
        if n_prompts < 1:
            raise ValueError(f"n_prompts must be >= 1, got {n_prompts}")
        self.n_prompts = int(n_prompts)
        self.d_model = int(d_model)
        self.init_std = float(init_std)

        self.P0 = nn.Parameter(
            torch.randn(self.n_prompts, self.d_model) * self.init_std,
        )
        self.gru = nn.GRUCell(self.d_model, self.d_model)

        if feat_channels is not None:
            self.feat_proj: nn.Module = nn.Linear(int(feat_channels), self.d_model)
        else:
            self.feat_proj = nn.Identity()
        self._proj_initialised: bool = feat_channels is not None

    def lazy_init_projector(self, feat_channels: int, device, dtype) -> None:
        if self._proj_initialised:
            return
        self.feat_proj = nn.Linear(int(feat_channels), self.d_model).to(
            device=device, dtype=dtype,
        )
        self._proj_initialised = True

    def forward(self, srcs: list, B: int, T: int) -> torch.Tensor:
        """Args:
            srcs: list of backbone feature maps, each ``(B*T, C_i, h_i, w_i)``.
                The lowest-resolution map is pooled per frame.
            B, T: batch and time dims of the video window.
        Returns:
            ``(B, T, n_prompts, d_model)`` per-frame propagated prompts.
        """
        if not srcs:
            raise ValueError("srcs must be a non-empty list of feature maps")
        feat = srcs[-1]  # (B*T, C, h, w)
        if feat.dim() != 4:
            raise ValueError(
                f"expected 4-D feature map, got {tuple(feat.shape)}",
            )
        BT, C, _h, _w = feat.shape
        if BT != B * T:
            raise ValueError(
                f"feat.shape[0]={BT} does not match B*T={B * T}",
            )

        self.lazy_init_projector(C, feat.device, feat.dtype)

        c = feat.mean(dim=(-1, -2))  # (B*T, C)
        c = self.feat_proj(c)  # (B*T, D)
        c = c.reshape(B, T, self.d_model)

        # (B*n_prompts, D) hidden state for the GRU; broadcast P0 to batch.
        P = self.P0.unsqueeze(0).expand(B, self.n_prompts, self.d_model)
        P = P.reshape(B * self.n_prompts, self.d_model)

        outputs = []
        for t in range(T):
            c_t = c[:, t, :].unsqueeze(1).expand(-1, self.n_prompts, -1)
            c_flat = c_t.reshape(B * self.n_prompts, self.d_model)
            P = self.gru(c_flat, P)
            outputs.append(P.reshape(B, self.n_prompts, self.d_model))

        return torch.stack(outputs, dim=1)  # (B, T, N, D)
