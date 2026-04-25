"""End-to-end temporal student detector.

    9-frame clip ``(B, 9, 3, H, W)``
        → fold to ``(B*9, 3, H, W)``
        → ConvNeXt-V2-Tiny backbone with TSM hooks (multi-scale feats)
        → reshape to ``(B, 9, C, h, w)`` and slice centre frame (idx=4)
        → Detail-Aware Cross-Attention FPN
        → FCOS head (cls / reg / centre-ness)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import ConvNeXtV2TinyBackbone
from .config import Config
from .distill_losses import FeatureAdapter
from .head import FCOSHead
from .loss import FCOSLoss
from .neck import DetailAwareFPN
from .postprocess import postprocess as fcos_postprocess


class StenosisStudent(nn.Module):
    """ConvNeXt-V2-Tiny + TSM + Detail-Aware FCOS detector."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T
        self.centre = cfg.centre_index

        self.backbone = ConvNeXtV2TinyBackbone(cfg)
        in_channels = list(self.backbone.out_channels)
        self.neck = DetailAwareFPN(cfg, in_channels=in_channels)
        self.head = FCOSHead(
            in_dim=cfg.fpn_dim,
            num_classes=cfg.num_classes,
            num_levels=len(in_channels),
            strides=self.backbone.out_strides,
            num_convs=cfg.head_num_convs,
            prior_prob=cfg.head_prior_prob,
        )
        self.criterion = FCOSLoss(cfg)

        # ── Stage-4: optional feature adapter for distillation ──────
        self.feature_adapter: Optional[FeatureAdapter] = None
        if cfg.distill_enabled:
            self.feature_adapter = FeatureAdapter(
                in_dim=cfg.fpn_dim,
                out_dim=cfg.distill_teacher_hidden_dim,
            )

    # ─── helpers ──────────────────────────────────────────────────────
    def _set_tsm_T(self, T: int) -> None:
        """Update TSM T at runtime (e.g. for inference with a single frame)."""
        if self.backbone.tsm_state is not None:
            self.backbone.tsm_state.T = T
            self.backbone.tsm_state.enabled = T > 1

    def _backbone_multi(self, frames: torch.Tensor) -> Tuple[List[torch.Tensor], int, int]:
        """Run the backbone over a clip and return per-stage features in
        ``(B, T, C, h, w)`` layout (TSM-mixed across the T axis)."""
        if frames.dim() == 4:
            B, _, _, _ = frames.shape
            T = 1
            x = frames
        else:
            B, T, _, _, _ = frames.shape
            x = frames.reshape(B * T, *frames.shape[-3:])
        self._set_tsm_T(T)
        feats = self.backbone(x)  # list of (B*T, C, h, w)
        out = []
        for f in feats:
            _, Cl, h, w = f.shape
            out.append(f.reshape(B, T, Cl, h, w))
        return out, B, T

    def extract_centre_features(self, frames: torch.Tensor) -> List[torch.Tensor]:
        """Run the backbone over all T frames, return centre-frame features."""
        feats_bt, _B, T = self._backbone_multi(frames)
        out = []
        for f_bt in feats_bt:
            centre = self.centre if T > 1 else 0
            centre = min(centre, T - 1)
            out.append(f_bt[:, centre])
        return out

    # ─── forward ──────────────────────────────────────────────────────
    def forward(
        self,
        frames: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        *,
        return_extras: bool = False,
    ):
        """Args:
            frames: ``(B, T, 3, H, W)`` (or ``(B, 3, H, W)`` for T=1).
            targets: list of length B of dicts with
                ``"boxes": (n, 4)`` xyxy abs px and ``"labels": (n,)``.
            return_extras: when True, also return Stage-4 auxiliary tensors
                (multi-frame FPN level + adapted student feature map for the
                distillation level).  Returned even when ``targets is None``.

        Returns:
            * Train (``targets`` given, no extras): dict of losses.
            * Eval  (no targets, no extras): tuple ``(cls, reg, ctr)``.
            * With ``return_extras=True``: a dict
              ``{"head_outputs": (cls, reg, ctr) or losses,
                 "fused": list of fused FPN levels (centre frame),
                 "multi_frame_fpn_level": (B, T, fpn_dim, h, w)
                                          (lateral-only, level
                                          ``cfg.temporal_consistency_level_idx``),
                 "student_distill_feat": (B, teacher_hidden, h, w) or None}``.
        """
        feats_bt, B, T = self._backbone_multi(frames)

        # Centre-frame features fed to the neck.
        centre_idx = self.centre if T > 1 else 0
        centre_idx = min(centre_idx, T - 1)
        centre_feats = [f_bt[:, centre_idx] for f_bt in feats_bt]
        fused = self.neck(centre_feats)
        cls_logits, bbox_reg, centerness = self.head(fused)

        if not return_extras:
            if targets is not None:
                return self.criterion(cls_logits, bbox_reg, centerness, targets)
            return cls_logits, bbox_reg, centerness

        # ── Stage-4 extras ─────────────────────────────────────────
        cfg = self.cfg
        # Multi-frame FPN level for temporal consistency: pass each frame's
        # backbone feature through the SAME lateral 1×1 conv used by the
        # neck.  No cross-attention smoothing here — we just need a per-frame
        # embedding map at the correct stride/dim for RoI pooling.
        multi_frame_fpn_level = None
        if cfg.temporal_consistency_enabled:
            lvl = cfg.temporal_consistency_level_idx
            lateral = self.neck.lateral[lvl]
            f_bt = feats_bt[lvl]                       # (B, T, C, h, w)
            B_, T_, Cl, h, w = f_bt.shape
            flat = f_bt.reshape(B_ * T_, Cl, h, w)
            proj = lateral(flat)                       # (B*T, fpn_dim, h, w)
            multi_frame_fpn_level = proj.reshape(
                B_, T_, proj.shape[1], proj.shape[2], proj.shape[3],
            )

        # Adapted student feature for distillation.
        student_distill_feat = None
        if cfg.distill_enabled and self.feature_adapter is not None:
            lvl = cfg.distill_student_level_idx
            student_distill_feat = self.feature_adapter(fused[lvl])

        head_out = (
            self.criterion(cls_logits, bbox_reg, centerness, targets)
            if targets is not None
            else (cls_logits, bbox_reg, centerness)
        )
        return {
            "head_outputs": head_out,
            "fused": fused,
            "multi_frame_fpn_level": multi_frame_fpn_level,
            "student_distill_feat": student_distill_feat,
        }

    @torch.no_grad()
    def predict(
        self, frames: torch.Tensor, image_size: Optional[int | Tuple[int, int]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """Convenience: forward + postprocess."""
        if image_size is None:
            image_size = self.cfg.img_size
        cls_logits, bbox_reg, centerness = self.forward(frames)
        return fcos_postprocess(cls_logits, bbox_reg, centerness, self.cfg, image_size)

    # ─── param groups ─────────────────────────────────────────────────
    def get_param_groups(self) -> List[Dict]:
        backbone_params, other_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("backbone."):
                backbone_params.append(p)
            else:
                other_params.append(p)
        groups = [{"params": other_params, "lr": self.cfg.lr}]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.cfg.lr_backbone})
        return groups
