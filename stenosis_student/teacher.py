"""Frozen 2D teacher for feature distillation.

Wraps the RF-DETR Small backbone+projector from a fine-tuned temporal
checkpoint and exposes only the P4 feature map (stride 16, 256 channels).
The teacher consumes the *clean* centre frame of each clip — no temporal
fusion, no decoder, no detection heads.

All parameters are frozen and the module is kept in eval mode.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .config import Config


class FrozenRFDETRTeacher(nn.Module):
    """Frozen RF-DETR backbone exposing the P4 feature map.

    Usage::

        teacher = FrozenRFDETRTeacher(cfg).to(device).eval()
        with torch.no_grad():
            feat = teacher(centre_clean)        # (B, 256, H/16, W/16)
    """

    def __init__(self, cfg: Config):
        super().__init__()
        # Local imports so users without the rfdetr stack can still import
        # ``stenosis_student`` modules that don't touch the teacher.
        from rfdetr.config import RFDETRSmallConfig
        from rfdetr.models.lwdetr import build_model_from_config
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        self._nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
        self.cfg = cfg

        model_cfg = RFDETRSmallConfig(num_classes=cfg.num_classes)
        lwdetr = build_model_from_config(model_cfg)

        ckpt_path = cfg.distill_teacher_ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        msg = lwdetr.load_state_dict(state_dict, strict=False)
        print(f"[Teacher] Loaded {ckpt_path}  missing={len(msg.missing_keys)}  "
              f"unexpected={len(msg.unexpected_keys)}")

        # Keep only the Joiner(Backbone, PosEmbed); discard everything else
        # so the teacher's parameter footprint stays minimal.
        self.backbone = lwdetr.backbone

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @property
    def hidden_dim(self) -> int:
        return int(self.cfg.distill_teacher_hidden_dim)

    def train(self, mode: bool = True):  # type: ignore[override]
        # Always stay in eval mode regardless of the parent's .train() call.
        return super().train(False)

    @torch.no_grad()
    def forward(self, centre_clean: torch.Tensor) -> torch.Tensor:
        """Args:
            centre_clean: ``(B, 3, H, W)`` un-corrupted centre frames,
                ImageNet-normalised, on the same device as the teacher.

        Returns:
            ``(B, hidden_dim, H/16, W/16)`` projected P4 features.
        """
        assert centre_clean.dim() == 4, \
            f"teacher expects (B, 3, H, W), got {tuple(centre_clean.shape)}"
        nested = self._nested_tensor_from_tensor_list(centre_clean)
        features, _poss = self.backbone(nested)
        # RF-DETR Small with projector_scale=["P4"] yields one level.
        src, _mask = features[-1].decompose()
        return src
