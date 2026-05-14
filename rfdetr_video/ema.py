"""Exponential moving average of a model's trainable parameters.

Frozen parameters (``requires_grad=False``) are never copied — the EMA is a
plain dict of the trainable parameter tensors. ``applied_to`` temporarily swaps
the EMA tensors into a live model for evaluation / checkpoint saving and then
restores the live parameters, so no second full model is held in memory.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay,
            )

    @contextmanager
    def applied_to(self, model: nn.Module):
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.copy_(backup[name])
