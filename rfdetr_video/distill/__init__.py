"""Distillation utilities for Video RF-DETR."""

from rfdetr_temporal.distill.crrcd import CRRCDLoss
from rfdetr_temporal.distill.losses import distillation_loss

from .teacher import VideoFrozenRFDETRTeacher

__all__ = [
    "CRRCDLoss",
    "distillation_loss",
    "VideoFrozenRFDETRTeacher",
]
