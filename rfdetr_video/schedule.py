"""LR scheduler factory for Video RF-DETR.

``cosine`` decays every param group from its base LR toward ``eta_min`` over the
epoch budget; ``multistep`` preserves the legacy step schedule. The per-iteration
linear warmup in ``train.py`` is unchanged and hands off to whichever scheduler
this returns.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def build_scheduler(optimizer, cfg):
    if cfg.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    if cfg.lr_schedule == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=list(cfg.lr_step_milestones),
            gamma=cfg.lr_gamma,
        )
    raise ValueError(
        f"unknown lr_schedule={cfg.lr_schedule!r}; expected 'cosine' or 'multistep'"
    )
