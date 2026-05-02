"""Frozen RF-DETR-Large teacher — video flavour.

Identical to :class:`rfdetr_temporal.distill.teacher.FrozenRFDETRTeacher`
plus a ``forward_video`` method that flattens ``(B, T, 3, S, S) → (B*T,
3, S, S)`` before invoking the underlying teacher, so the per-frame
CRRCD distillation is spatially aligned to the per-frame student
forward.
"""

from __future__ import annotations

from typing import Dict

import torch

from rfdetr_temporal.distill.teacher import FrozenRFDETRTeacher

from ..config import Config


class VideoFrozenRFDETRTeacher(FrozenRFDETRTeacher):
    """Teacher exposing a batched ``forward_video`` over T frames."""

    def __init__(self, cfg: Config):
        # Reuse the base class wholesale; it accepts the same Config
        # field set we expose in :class:`rfdetr_video.config.Config`.
        super().__init__(cfg)  # type: ignore[arg-type]

    @torch.no_grad()
    def forward_video(self, frames_hr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
            frames_hr: (B, T, 3, S, S) ImageNet-normalised HR frames.
        Returns:
            Same keys as :meth:`forward`, with the leading dim flattened
            to ``B*T``.
        """
        assert frames_hr.dim() == 5, (
            f"forward_video expects (B, T, 3, S, S), got "
            f"{tuple(frames_hr.shape)}"
        )
        B, T, C, H, W = frames_hr.shape
        return self.forward(frames_hr.reshape(B * T, C, H, W))

    @torch.no_grad()
    def forward_video_general(
        self,
        frames_hr: torch.Tensor,
        refpoint_w: torch.Tensor,
        query_feat_w: torch.Tensor,
        min_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """General-sampling variant — flattens video frames and reuses
        the base class's :meth:`forward_general`.
        """
        assert frames_hr.dim() == 5
        B, T, C, H, W = frames_hr.shape
        return self.forward_general(
            frames_hr.reshape(B * T, C, H, W),
            refpoint_w, query_feat_w,
            min_weight=min_weight,
        )
