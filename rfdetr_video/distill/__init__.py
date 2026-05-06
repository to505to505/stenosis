"""Distillation utilities for the Video RF-DETR.

Re-exports the existing CRRCD loss + distillation loss primitives from
:mod:`rfdetr_temporal.distill`; video-specific pieces add per-frame
teacher forwarding and STFS feature-space alignment.
"""

from rfdetr_temporal.distill.crrcd import CRRCDLoss
from rfdetr_temporal.distill.losses import distillation_loss

from .feature_alignment import stfs_feature_alignment_loss
from .teacher import VideoFrozenRFDETRTeacher

__all__ = [
    "CRRCDLoss",
    "distillation_loss",
    "stfs_feature_alignment_loss",
    "VideoFrozenRFDETRTeacher",
]
