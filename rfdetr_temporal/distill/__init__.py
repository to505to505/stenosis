"""HR→LR query-aligned distillation for Temporal RF-DETR."""

from .teacher import FrozenRFDETRTeacher
from .losses import distillation_loss

__all__ = ["FrozenRFDETRTeacher", "distillation_loss"]
