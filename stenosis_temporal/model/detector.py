"""Full Spatio-Temporal Stenosis Detector.

Combines FPE → PSTFA → MTO into an end-to-end detection model.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops

from ..config import Config
from .fpe import FPE
from .pstfa import PSTFA
from .mto import MTO


def encode_boxes(reference_boxes: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """Encode GT boxes relative to proposals (same as Faster R-CNN).

    Args:
        reference_boxes: (N, 4) GT boxes x1y1x2y2
        proposals: (N, 4) proposal boxes x1y1x2y2

    Returns:
        targets: (N, 4) encoded as (dx, dy, dw, dh)
    """
    px = (proposals[:, 0] + proposals[:, 2]) / 2
    py = (proposals[:, 1] + proposals[:, 3]) / 2
    pw = proposals[:, 2] - proposals[:, 0]
    ph = proposals[:, 3] - proposals[:, 1]

    gx = (reference_boxes[:, 0] + reference_boxes[:, 2]) / 2
    gy = (reference_boxes[:, 1] + reference_boxes[:, 3]) / 2
    gw = reference_boxes[:, 2] - reference_boxes[:, 0]
    gh = reference_boxes[:, 3] - reference_boxes[:, 1]

    pw = pw.clamp(min=1.0)
    ph = ph.clamp(min=1.0)

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw.clamp(min=1.0) / pw)
    dh = torch.log(gh.clamp(min=1.0) / ph)

    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(deltas: torch.Tensor, proposals: torch.Tensor) -> torch.Tensor:
    """Decode box deltas relative to proposals → absolute x1y1x2y2."""
    px = (proposals[:, 0] + proposals[:, 2]) / 2
    py = (proposals[:, 1] + proposals[:, 3]) / 2
    pw = (proposals[:, 2] - proposals[:, 0]).clamp(min=1.0)
    ph = (proposals[:, 3] - proposals[:, 1]).clamp(min=1.0)

    dx, dy, dw, dh = deltas.unbind(dim=1)
    # Clamp dw/dh to prevent exp overflow
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


class StenosisTemporalDetector(nn.Module):
    """End-to-end spatio-temporal stenosis detector.

    Input:  (B, T, 1, H, W)  sequence of grayscale frames
    Output: training  → dict of losses
            inference → list of detections per reference frame
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.fpe = FPE(cfg)
        self.pstfa = PSTFA(cfg)
        self.mto = MTO(cfg)

    def _assign_targets_to_proposals(
        self,
        proposals: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign GT targets to proposals based on IoU.

        Args:
            proposals: (S, 4)
            gt_boxes: (G, 4)
            gt_labels: (G,)

        Returns:
            labels: (S,) — 0=background, 1=stenosis
            matched_gt_boxes: (S, 4) — GT box for each proposal (zero for bg)
        """
        device = proposals.device
        S = proposals.shape[0]

        if gt_boxes.numel() == 0:
            return (
                torch.zeros(S, dtype=torch.int64, device=device),
                torch.zeros(S, 4, dtype=torch.float32, device=device),
            )

        iou_matrix = box_ops.box_iou(proposals, gt_boxes)  # (S, G)
        max_iou, matched_idx = iou_matrix.max(dim=1)  # (S,)

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
            targets: list of B elements, each a list of T dicts with
                     'boxes' (N_i, 4) and 'labels' (N_i,)

        Returns:
            training: dict with loss keys
            inference: list of B dicts, each with 'boxes', 'scores', 'labels'
        """
        B, T = images.shape[:2]
        device = images.device
        cfg = self.cfg

        # ── 1. Reshape to (B*T, 1, H, W) and run FPE ──────────────────
        flat_images = images.reshape(B * T, 1, cfg.img_h, cfg.img_w)

        # Build flat targets for RPN (one per frame)
        flat_targets = None
        if targets is not None:
            flat_targets = []
            for b in range(B):
                for t in range(T):
                    flat_targets.append({
                        "boxes": targets[b][t]["boxes"].to(device),
                        "labels": targets[b][t]["labels"].to(device),
                    })

        features, proposals, rpn_losses, _ = self.fpe(flat_images, flat_targets)
        del flat_images, flat_targets  # free input tensors early

        # ── 2. For each batch element, run PSTFA on each reference frame ──
        # We pick each frame as reference and aggregate across its temporal context.
        # features is an OrderedDict; we use level "0" (highest resolution) for PSTFA.
        # Reshape per-batch, per-frame.

        all_cls_logits = []
        all_box_deltas = []
        all_proposals_for_ref = []
        all_ref_indices = []  # (batch_idx, frame_idx)

        for b in range(B):
            # Extract per-frame features from FPN level "0" for this batch element
            frame_features = []
            for t in range(T):
                idx = b * T + t
                frame_features.append(features["0"][idx])  # (C, H_f, W_f)

            for t in range(T):
                idx = b * T + t
                ref_proposals = proposals[idx]  # (S, 4)

                # During training, append GT boxes to proposals so
                # high-IoU examples are guaranteed for the detection head
                if self.training and targets is not None:
                    gt_boxes_t = targets[b][t]["boxes"].to(device)
                    if gt_boxes_t.numel() > 0:
                        ref_proposals = torch.cat([ref_proposals, gt_boxes_t], dim=0)

                # PSTFA: aggregate spatio-temporal features
                aggregated_roi = self.pstfa(
                    frame_features, ref_proposals, ref_idx=t
                )  # (S', C, roi_size, roi_size) where S'=S+G during training

                # MTO: classify + regress
                cls_logits, box_deltas = self.mto(aggregated_roi)
                del aggregated_roi  # free PSTFA output early

                all_cls_logits.append(cls_logits)
                all_box_deltas.append(box_deltas)
                all_proposals_for_ref.append(ref_proposals)
                all_ref_indices.append((b, t))

        # ── 3. Compute losses or detections ────────────────────────────
        if self.training:
            det_cls_loss = torch.tensor(0.0, device=device)
            det_reg_loss = torch.tensor(0.0, device=device)
            num_frames = 0

            # Balanced sampling parameters (standard Faster R-CNN)
            roi_batch_size = 128
            roi_positive_fraction = 0.25
            num_pos_max = int(roi_batch_size * roi_positive_fraction)  # 32

            for i, (b, t) in enumerate(all_ref_indices):
                gt_boxes = targets[b][t]["boxes"].to(device)
                gt_labels = targets[b][t]["labels"].to(device)
                ref_proposals = all_proposals_for_ref[i]

                # Assign GT to proposals
                assigned_labels, matched_gt = self._assign_targets_to_proposals(
                    ref_proposals, gt_boxes, gt_labels
                )

                # ── Balanced sampling: sample roi_batch_size proposals ──
                pos_idx = torch.where(assigned_labels > 0)[0]
                neg_idx = torch.where(assigned_labels == 0)[0]
                num_pos = min(pos_idx.numel(), num_pos_max)
                num_neg = min(neg_idx.numel(), roi_batch_size - num_pos)

                if num_pos > 0:
                    perm_pos = torch.randperm(pos_idx.numel(), device=device)[:num_pos]
                    pos_idx = pos_idx[perm_pos]
                if num_neg > 0:
                    perm_neg = torch.randperm(neg_idx.numel(), device=device)[:num_neg]
                    neg_idx = neg_idx[perm_neg]

                sampled_idx = torch.cat([pos_idx, neg_idx])
                if sampled_idx.numel() == 0:
                    num_frames += 1
                    continue

                sampled_labels = assigned_labels[sampled_idx]
                sampled_gt = matched_gt[sampled_idx]
                sampled_proposals = ref_proposals[sampled_idx]
                sampled_cls_logits = all_cls_logits[i][sampled_idx]
                sampled_box_deltas = all_box_deltas[i][sampled_idx]

                # Classification loss on balanced sample
                det_cls_loss += F.cross_entropy(
                    sampled_cls_logits, sampled_labels
                )

                # Regression loss (only on foreground proposals)
                fg_mask = sampled_labels > 0
                if fg_mask.any():
                    fg_proposals = sampled_proposals[fg_mask]
                    fg_gt = sampled_gt[fg_mask]
                    fg_deltas = sampled_box_deltas[fg_mask]
                    target_deltas = encode_boxes(fg_gt, fg_proposals)
                    det_reg_loss += F.smooth_l1_loss(fg_deltas, target_deltas)

                num_frames += 1

            num_frames = max(num_frames, 1)
            losses = {
                "rpn_objectness_loss": rpn_losses.get("loss_objectness", torch.tensor(0.0, device=device)),
                "rpn_box_loss": rpn_losses.get("loss_rpn_box_reg", torch.tensor(0.0, device=device)),
                "det_cls_loss": det_cls_loss / num_frames,
                "det_reg_loss": det_reg_loss / num_frames,
            }
            losses["total_loss"] = sum(losses.values())
            return losses

        else:
            # Inference: decode boxes and apply NMS per reference frame
            results = []
            for i, (b, t) in enumerate(all_ref_indices):
                scores = F.softmax(all_cls_logits[i], dim=-1)[:, 1]  # stenosis score
                deltas = all_box_deltas[i]
                ref_proposals = all_proposals_for_ref[i]

                pred_boxes = decode_boxes(deltas, ref_proposals)
                pred_boxes[:, 0::2] = pred_boxes[:, 0::2].clamp(0, cfg.img_w)
                pred_boxes[:, 1::2] = pred_boxes[:, 1::2].clamp(0, cfg.img_h)

                # Filter by score
                keep = scores > cfg.score_thresh
                scores = scores[keep]
                pred_boxes = pred_boxes[keep]

                # NMS
                if scores.numel() > 0:
                    nms_keep = box_ops.nms(pred_boxes, scores, cfg.nms_thresh)
                    nms_keep = nms_keep[: cfg.detections_per_img]
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
        """Xavier initialization for all non-backbone parameters."""
        for name, module in self.named_modules():
            # Skip backbone (pretrained) and channel adapter (has special 1/3 init)
            if name.startswith("fpe.backbone") or name == "fpe.channel_adapter":
                continue
            # Skip RPN (initialized by torchvision)
            if name.startswith("fpe.rpn"):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
