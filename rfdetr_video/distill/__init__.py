"""Distillation utilities for the Video RF-DETR.

Re-exports the existing CRRCD loss + distillation loss primitives from
:mod:`rfdetr_temporal.distill`; the only new piece is
:class:`VideoFrozenRFDETRTeacher` which adds a per-frame
``forward_video`` entry point so the teacher can score all T frames in
a batched call.
"""

from rfdetr_temporal.distill.crrcd import CRRCDLoss
from rfdetr_temporal.distill.losses import distillation_loss

from .teacher import VideoFrozenRFDETRTeacher

__all__ = ["CRRCDLoss", "distillation_loss", "VideoFrozenRFDETRTeacher"]
