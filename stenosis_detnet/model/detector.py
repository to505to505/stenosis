"""End-to-end Stenosis-DetNet detector.

Combines Backbone → GA-RPN → SFF → Detection Heads into a single model.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from ..config import Config
from .backbone import Backbone
from .ga_rpn import GuidedAnchoringRPN
from .sff import SequenceFeatureFusion
from .heads import DetectionHeads


def encode_boxes(reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """Encode GT boxes relative to proposals (Faster R-CNN style).

    Args:
        reference_boxes: (N, 4) GT boxes x1y1x2y2
        proposals: (N, 4) proposal boxes x1y1x2y2

    Returns:
        targets: (N, 4) encoded as (dx, dy, dw, dh)
    """
    px = (proposals[:, 0] + proposals[:, 2]) / 2
    py = (proposals[:, 1] + proposals[:, 3]) / 2
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1.0)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1.0)

    gx = (reference_boxes[:, 0] + reference_boxes[:, 2]) / 2
    gy = (reference_boxes[:, 1] + reference_boxes[:, 3]) / 2
    gw = (reference_boxes[:, 2] - reference_boxes[:, 0]).clamp(min=1.0)
    gh = (reference_boxes[:, 3] - reference_boxes[:, 1]).clamp(min=1.0)

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(deltas: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """Decode box deltas relative to proposals → absolute x1y1x2y2."""
    px = (proposals[:, 0] + proposals[:, 2]) / 2
    py = (proposals[:, 1] + proposals[:, 3]) / 2
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1.0)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1.0)

    dx, dy, dw, dh = deltas.unbind(dim=1)
    dw = dw.clamp(max=4.0)
    dh = dh.clamp(max=4.0)

    cx = dx * pw + px
    cy = dy * ph + py
    w = torch.exp(dw) * pw
    h = torch.exp(dh) * ph

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


class StenosisDetNet(nn.Module):
    """End-to-end two-stage temporal stenosis detector.

    Pipeline:
        1. Backbone (ResNet-50 + FPN) extracts multi-scale features
        2. GA-RPN generates proposals + RoI features (256×7×7)
        3. SFF fuses proposal features across T=9 frames
        4. Detection heads classify (stenosis/non-stenosis) + regress boxes

    Input:  (B, T, 1, H, W) sequence of grayscale frames
    Output: training  → dict of losses
            inference → list of detections per frame
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = Backbone(cfg)
        self.ga_rpn = GuidedAnchoringRPN(cfg)
        self.sff = SequenceFeatureFusion(cfg)
        self.heads = DetectionHeads(cfg)

    def _assign_targets_to_proposals(
        self,
        proposals: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign GT targets to proposals based on IoU.

        Returns:
            labels: (S,) — 0=background, 1=stenosis
            matched_gt_boxes: (S, 4)
        """
        device = proposals.device
        S = proposals.shape[0]

        if gt_boxes.numel() == 0:
            return (
                torch.zeros(S, dtype=torch.int64, device=device),
                torch.zeros(S, 4, dtype=torch.float32, device=device),
            )

        iou_matrix = box_ops.box_iou(proposals, gt_boxes)
        max_iou, matched_idx = iou_matrix.max(dim=1)

        labels = torch.zeros(S, dtype=torch.int64, device=device)
        labels[max_iou >= self.cfg.fg_iou_thresh] = 1

        matched_gt_boxes = gt_boxes[matched_idx]
        matched_gt_boxes[labels == 0] = 0

        return labels, matched_gt_boxes

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[List[Dict]]] = None,
    ):
        """
        Args:
            images: (B, T, 1, H, W) sequence of grayscale frames
            targets: list of B elements, each list of T dicts with
                     'boxes' (N_i, 4) and 'labels' (N_i,)

        Returns:
            training: dict with loss keys
            inference: list of dicts with 'boxes', 'scores', 'labels',
                       'batch_idx', 'frame_idx'
        """
        B, T = images.shape[:2]
        device = images.device
        cfg = self.cfg

        # ── 1. Flatten to (B*T, 1, H, W), run backbone ────────────────
        flat_images = images.reshape(B * T, 1, cfg.img_h, cfg.img_w)
        features = self.backbone(flat_images)

        # ── 2. Build flat targets for GA-RPN ───────────────────────────
        flat_targets = None
        if targets is not None:
            flat_targets = []
            for b in range(B):
                for t in range(T):
                    flat_targets.append({
                        "boxes": targets[b][t]["boxes"].to(device),
                        "labels": targets[b][t]["labels"].to(device),
                    })

        # ── 3. GA-RPN → proposals + roi features ──────────────────────
        proposals, rpn_losses, roi_features = self.ga_rpn(
            features, flat_targets, cfg.img_h, cfg.img_w
        )
        # proposals: list of B*T tensors, each (S, 4)
        # roi_features: (B*T*S, C, roi_size, roi_size)

        S = cfg.S
        C = cfg.C
        rs = cfg.roi_output_size

        # ── 4. Reshape RoI features for SFF ───────────────────────────
        # roi_features: (B*T*S, C, rs, rs) → group by batch and proposal
        # We need (S, T, C, rs, rs) per batch element
        # Current layout: proposals are per-frame, indexed [b*T+t][s]

        all_cls_logits = []
        all_box_deltas = []
        all_proposals_for_ref = []
        all_ref_indices = []

        for b in range(B):
            # Gather roi features for all T frames of this batch element
            # Shape: (T, S, C, rs, rs)
            frame_roi_list = []
            frame_proposals = []
            for t in range(T):
                idx = b * T + t
                start = idx * S
                end = start + S
                frame_roi_list.append(roi_features[start:end])  # (S, C, rs, rs)
                frame_proposals.append(proposals[idx])           # (S, 4)

            frame_rois = torch.stack(frame_roi_list, dim=1)  # (S, T, C, rs, rs)

            # ── 5. SFF: fuse across temporal dimension ────────────────
            enhanced_rois = self.sff(frame_rois)  # (S, T, C, rs, rs)

            # ── 6. For each frame, run detection heads ────────────────
            for t in range(T):
                ref_rois = enhanced_rois[:, t]          # (S, C, rs, rs)
                ref_proposals = frame_proposals[t]       # (S, 4)

                cls_logits, box_deltas = self.heads(ref_rois)

                all_cls_logits.append(cls_logits)
                all_box_deltas.append(box_deltas)
                all_proposals_for_ref.append(ref_proposals)
                all_ref_indices.append((b, t))

        # ── 7. Compute losses or decode detections ─────────────────────
        if self.training:
            det_cls_loss = torch.tensor(0.0, device=device)
            det_reg_loss = torch.tensor(0.0, device=device)
            num_frames = 0

            for i, (b, t) in enumerate(all_ref_indices):
                gt_boxes = targets[b][t]["boxes"].to(device)
                gt_labels = targets[b][t]["labels"].to(device)
                ref_proposals = all_proposals_for_ref[i]

                assigned_labels, matched_gt = self._assign_targets_to_proposals(
                    ref_proposals, gt_boxes, gt_labels
                )

                # Classification loss
                det_cls_loss += F.cross_entropy(
                    all_cls_logits[i], assigned_labels
                )

                # Regression loss (foreground only)
                fg_mask = assigned_labels > 0
                if fg_mask.any():
                    fg_proposals = ref_proposals[fg_mask]
                    fg_gt = matched_gt[fg_mask]
                    fg_deltas = all_box_deltas[i][fg_mask]
                    target_deltas = encode_boxes(fg_gt, fg_proposals)
                    det_reg_loss += F.smooth_l1_loss(fg_deltas, target_deltas)

                num_frames += 1

            num_frames = max(num_frames, 1)
            losses = {
                "loss_ga_loc": rpn_losses.get("loss_ga_loc", torch.tensor(0.0, device=device)),
                "loss_ga_shape": rpn_losses.get("loss_ga_shape", torch.tensor(0.0, device=device)),
                "loss_rpn_cls": rpn_losses.get("loss_rpn_cls", torch.tensor(0.0, device=device)),
                "loss_rpn_bbox": rpn_losses.get("loss_rpn_bbox", torch.tensor(0.0, device=device)),
                "det_cls_loss": det_cls_loss / num_frames,
                "det_reg_loss": det_reg_loss / num_frames,
            }
            losses["total_loss"] = sum(losses.values())
            return losses

        else:
            # Inference
            results = []
            for i, (b, t) in enumerate(all_ref_indices):
                scores = F.softmax(all_cls_logits[i], dim=-1)[:, 1]
                deltas = all_box_deltas[i]
                ref_proposals = all_proposals_for_ref[i]

                pred_boxes = decode_boxes(deltas, ref_proposals)
                pred_boxes = torch.cat([
                    pred_boxes[:, 0:1].clamp(0, cfg.img_w),
                    pred_boxes[:, 1:2].clamp(0, cfg.img_h),
                    pred_boxes[:, 2:3].clamp(0, cfg.img_w),
                    pred_boxes[:, 3:4].clamp(0, cfg.img_h),
                ], dim=1)

                keep = scores > cfg.score_thresh
                scores = scores[keep]
                pred_boxes = pred_boxes[keep]

                if scores.numel() > 0:
                    nms_keep = box_ops.nms(pred_boxes, scores, cfg.nms_thresh)
                    nms_keep = nms_keep[:cfg.detections_per_img]
                    scores = scores[nms_keep]
                    pred_boxes = pred_boxes[nms_keep]

                results.append({
                    "boxes": pred_boxes,
                    "scores": scores,
                    "labels": torch.ones(len(scores), dtype=torch.int64, device=device),
                    "batch_idx": b,
                    "frame_idx": t,
                })

            return results

    def init_weights(self):
        """Xavier initialization for non-backbone parameters."""
        for name, module in self.named_modules():
            if name.startswith("backbone.resnet_fpn"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d) and not name.startswith("backbone"):
                if module.weight.requires_grad:
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
