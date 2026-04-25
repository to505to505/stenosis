"""FCOS loss: focal classification + GIoU regression + BCE centre-ness."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .head import make_locations


def _giou_loss(pred_xyxy: torch.Tensor, target_xyxy: torch.Tensor) -> torch.Tensor:
    """Generalised IoU loss; both inputs ``(N, 4)`` xyxy.  Returns ``(N,)``."""
    x1 = torch.max(pred_xyxy[:, 0], target_xyxy[:, 0])
    y1 = torch.max(pred_xyxy[:, 1], target_xyxy[:, 1])
    x2 = torch.min(pred_xyxy[:, 2], target_xyxy[:, 2])
    y2 = torch.min(pred_xyxy[:, 3], target_xyxy[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = (pred_xyxy[:, 2] - pred_xyxy[:, 0]).clamp(min=0) * \
             (pred_xyxy[:, 3] - pred_xyxy[:, 1]).clamp(min=0)
    area_t = (target_xyxy[:, 2] - target_xyxy[:, 0]).clamp(min=0) * \
             (target_xyxy[:, 3] - target_xyxy[:, 1]).clamp(min=0)
    union = area_p + area_t - inter + 1e-7
    iou = inter / union

    cx1 = torch.min(pred_xyxy[:, 0], target_xyxy[:, 0])
    cy1 = torch.min(pred_xyxy[:, 1], target_xyxy[:, 1])
    cx2 = torch.max(pred_xyxy[:, 2], target_xyxy[:, 2])
    cy2 = torch.max(pred_xyxy[:, 3], target_xyxy[:, 3])
    enclose = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-7
    giou = iou - (enclose - union) / enclose
    return 1.0 - giou


def _centerness_target(reg_targets: torch.Tensor) -> torch.Tensor:
    """``reg_targets``: ``(N, 4)`` ``(l, t, r, b)``.  Returns ``(N,)``."""
    lr = reg_targets[:, [0, 2]]
    tb = reg_targets[:, [1, 3]]
    cn = (lr.min(dim=1).values / lr.max(dim=1).values.clamp(min=1e-6)) * \
         (tb.min(dim=1).values / tb.max(dim=1).values.clamp(min=1e-6))
    return cn.clamp(min=0.0).sqrt()


def _decode_distances(locations: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """``locations``: ``(N, 2)`` xy.  ``distances``: ``(N, 4)`` ltrb. → ``(N, 4)`` xyxy."""
    x1 = locations[:, 0] - distances[:, 0]
    y1 = locations[:, 1] - distances[:, 1]
    x2 = locations[:, 0] + distances[:, 2]
    y2 = locations[:, 1] + distances[:, 3]
    return torch.stack([x1, y1, x2, y2], dim=1)


class FCOSLoss(nn.Module):
    """FCOS-style multi-level loss.

    Args at call time:
        cls_logits, bbox_reg, centerness — each a list of length L of tensors
            with shapes ``(B, num_classes, H_l, W_l)``, ``(B, 4, H_l, W_l)``,
            ``(B, 1, H_l, W_l)``.  ``bbox_reg`` already in pixel units
            (positive distances).
        targets: list of length B, each a dict with
            ``"boxes": (n_i, 4) xyxy abs px`` and
            ``"labels": (n_i,) int64``.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        self.strides = tuple(cfg.stage_strides[i] for i in cfg.fpn_stage_indices)
        self.reg_ranges = cfg.reg_ranges
        self.radius = cfg.centre_sample_radius

    @torch.no_grad()
    def _assign_targets(
        self,
        locations_per_level: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build per-location classification + regression targets.

        Returns:
            labels: ``(B, total_locs)`` int64 — class id, or
                ``num_classes`` for background.
            reg_targets: ``(B, total_locs, 4)`` float — ``(l, t, r, b)`` in pixels
                (only meaningful for positive locations).
        """
        device = locations_per_level[0].device
        # Per-level info (L,)
        L = len(locations_per_level)
        sizes = [loc.shape[0] for loc in locations_per_level]
        all_locs = torch.cat(locations_per_level, dim=0)  # (N, 2)
        N = all_locs.shape[0]
        # Per-location stride and reg range
        loc_stride = torch.cat([
            torch.full((sz,), self.strides[l], device=device, dtype=torch.float32)
            for l, sz in enumerate(sizes)
        ])
        loc_lo = torch.cat([
            torch.full((sz,), self.reg_ranges[l][0], device=device, dtype=torch.float32)
            for l, sz in enumerate(sizes)
        ])
        loc_hi = torch.cat([
            torch.full((sz,), self.reg_ranges[l][1], device=device, dtype=torch.float32)
            for l, sz in enumerate(sizes)
        ])

        B = len(targets)
        labels_out = torch.full((B, N), self.num_classes, device=device, dtype=torch.long)
        reg_out = torch.zeros((B, N, 4), device=device, dtype=torch.float32)

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(device)        # (M, 4)
            gt_labels = targets[b]["labels"].to(device)      # (M,)
            M = gt_boxes.shape[0]
            if M == 0:
                continue

            # ltrb distances from each location to every GT
            xs = all_locs[:, 0].unsqueeze(1)  # (N, 1)
            ys = all_locs[:, 1].unsqueeze(1)
            l = xs - gt_boxes[:, 0].unsqueeze(0)  # (N, M)
            t = ys - gt_boxes[:, 1].unsqueeze(0)
            r = gt_boxes[:, 2].unsqueeze(0) - xs
            b_ = gt_boxes[:, 3].unsqueeze(0) - ys
            ltrb = torch.stack([l, t, r, b_], dim=2)         # (N, M, 4)

            # 1) Inside-box test
            inside_box = ltrb.min(dim=2).values > 0          # (N, M)

            # 2) Centre-sampling: location must be near GT centre
            cx = 0.5 * (gt_boxes[:, 0] + gt_boxes[:, 2])     # (M,)
            cy = 0.5 * (gt_boxes[:, 1] + gt_boxes[:, 3])
            radius = loc_stride.unsqueeze(1) * self.radius   # (N, 1)
            cl = xs - (cx.unsqueeze(0) - radius)
            ct = ys - (cy.unsqueeze(0) - radius)
            cr = (cx.unsqueeze(0) + radius) - xs
            cb = (cy.unsqueeze(0) + radius) - ys
            in_centre = torch.stack([cl, ct, cr, cb], dim=2).min(dim=2).values > 0

            # 3) Per-level regression range
            max_dist = ltrb.max(dim=2).values                # (N, M)
            in_range = (max_dist >= loc_lo.unsqueeze(1)) & (max_dist <= loc_hi.unsqueeze(1))

            valid = inside_box & in_centre & in_range        # (N, M)

            # 4) For ambiguous locations, pick GT with smallest area
            areas = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * \
                    (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)  # (M,)
            INF = 1e10
            area_mat = areas.unsqueeze(0).expand(N, M).clone()       # (N, M)
            area_mat[~valid] = INF
            min_area, gt_idx = area_mat.min(dim=1)                   # (N,)
            pos_mask = min_area < INF

            # Assign labels & regression targets to positives
            labels_out[b, pos_mask] = gt_labels[gt_idx[pos_mask]]
            reg_out[b, pos_mask] = ltrb[torch.arange(N, device=device)[pos_mask], gt_idx[pos_mask]]

        return labels_out, reg_out

    def forward(
        self,
        cls_logits: List[torch.Tensor],
        bbox_reg: List[torch.Tensor],
        centerness: List[torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        device = cls_logits[0].device
        feature_sizes = [tuple(c.shape[-2:]) for c in cls_logits]
        locations = make_locations(feature_sizes, self.strides, device)

        labels, reg_targets = self._assign_targets(locations, targets)
        all_locs = torch.cat(locations, dim=0)  # (N, 2)

        # Flatten predictions (B, N, *)
        B = cls_logits[0].shape[0]
        cls_flat = torch.cat([
            c.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            for c in cls_logits
        ], dim=1)                                      # (B, N, C)
        reg_flat = torch.cat([
            r.permute(0, 2, 3, 1).reshape(B, -1, 4) for r in bbox_reg
        ], dim=1)                                      # (B, N, 4)
        ctr_flat = torch.cat([
            c.permute(0, 2, 3, 1).reshape(B, -1) for c in centerness
        ], dim=1)                                      # (B, N)

        N = cls_flat.shape[1]
        # ── Classification (focal) ──
        pos_mask = labels < self.num_classes              # (B, N)
        num_pos = max(pos_mask.sum().item(), 1)

        cls_targets = torch.zeros_like(cls_flat)
        if pos_mask.any():
            b_idx, n_idx = pos_mask.nonzero(as_tuple=True)
            cls_targets[b_idx, n_idx, labels[b_idx, n_idx]] = 1.0

        cls_loss = _sigmoid_focal_loss(
            cls_flat, cls_targets,
            alpha=self.cfg.focal_alpha, gamma=self.cfg.focal_gamma,
        ).sum() / num_pos

        # ── Regression (GIoU) + Centre-ness (BCE) ──
        if pos_mask.any():
            b_idx, n_idx = pos_mask.nonzero(as_tuple=True)
            pred_dist = reg_flat[b_idx, n_idx]            # (P, 4)
            tgt_dist = reg_targets[b_idx, n_idx]          # (P, 4)
            locs = all_locs[n_idx]                        # (P, 2)
            pred_xyxy = _decode_distances(locs, pred_dist)
            tgt_xyxy = _decode_distances(locs, tgt_dist)

            ctr_target = _centerness_target(tgt_dist)     # (P,)
            # Use centre-ness target as a quality weight on the box loss
            denom = ctr_target.sum().clamp(min=1e-6)
            reg_loss = (_giou_loss(pred_xyxy, tgt_xyxy) * ctr_target).sum() / denom

            ctr_pred = ctr_flat[b_idx, n_idx]
            ctr_loss = F.binary_cross_entropy_with_logits(
                ctr_pred, ctr_target, reduction="mean",
            )
        else:
            reg_loss = reg_flat.sum() * 0.0
            ctr_loss = ctr_flat.sum() * 0.0

        out = {
            "loss_cls": cls_loss * self.cfg.cls_loss_weight,
            "loss_box": reg_loss * self.cfg.reg_loss_weight,
            "loss_ctr": ctr_loss * self.cfg.centerness_loss_weight,
        }
        out["loss"] = out["loss_cls"] + out["loss_box"] + out["loss_ctr"]
        return out


def _sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Element-wise sigmoid focal loss.  Both inputs same shape."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss
