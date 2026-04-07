"""Loss functions for STQD-Det.

Combines:
  - Focal Loss (classification)
  - L1 Loss (bounding box regression)
  - GIoU Loss (bounding box regression)
  - Consistency Loss L_num (sequence detection count consistency)

Uses Hungarian matching to assign predictions to ground truth.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import (
    generalized_box_iou_loss,
    sigmoid_focal_loss,
    box_iou,
)
from scipy.optimize import linear_sum_assignment

from ..config import Config


class HungarianMatcher(nn.Module):
    """Matches predictions to ground truth using Hungarian algorithm.

    Cost combines classification probability, L1 box distance, and GIoU.
    """

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        img_w: int = 512,
        img_h: int = 512,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.img_w = img_w
        self.img_h = img_h

    @torch.no_grad()
    def forward(
        self,
        cls_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cls_logits: (P, num_classes) predicted class logits.
            pred_boxes: (P, 4) predicted boxes in xyxy.
            gt_boxes: (M, 4) ground truth boxes in xyxy.
            gt_labels: (M,) ground truth labels.

        Returns:
            pred_indices: (K,) matched prediction indices.
            gt_indices: (K,) matched GT indices.
        """
        P = cls_logits.shape[0]
        M = gt_boxes.shape[0]

        if M == 0:
            return (
                torch.empty(0, dtype=torch.long, device=cls_logits.device),
                torch.empty(0, dtype=torch.long, device=cls_logits.device),
            )

        # Classification cost: use sigmoid focal-style
        cls_probs = cls_logits.sigmoid()  # (P, num_classes)
        # Cost for each (pred, gt) pair based on gt class
        cost_class = -cls_probs[:, gt_labels]  # (P, M) — negative prob

        # L1 box cost (normalized to [0,1])
        img_scale = torch.tensor(
            [self.img_w, self.img_h, self.img_w, self.img_h],
            device=pred_boxes.device, dtype=pred_boxes.dtype,
        )
        cost_bbox = torch.cdist(
            pred_boxes / img_scale, gt_boxes / img_scale, p=1
        )  # (P, M)

        # GIoU cost — sanitize predicted boxes to ensure non-zero area
        pred_safe = pred_boxes.clone()
        pred_safe[:, 2] = torch.max(pred_safe[:, 2], pred_safe[:, 0] + 1)
        pred_safe[:, 3] = torch.max(pred_safe[:, 3], pred_safe[:, 1] + 1)
        iou = box_iou(pred_safe, gt_boxes)  # (P, M)
        cost_giou = -iou  # (P, M) — negative IoU

        # Combined cost
        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        # Hungarian matching — sanitize to prevent invalid entries
        C_np = C.detach().cpu().numpy()
        C_np = np.nan_to_num(C_np, nan=1e6, posinf=1e6, neginf=-1e6)
        row_idx, col_idx = linear_sum_assignment(C_np)

        return (
            torch.as_tensor(row_idx, dtype=torch.long, device=cls_logits.device),
            torch.as_tensor(col_idx, dtype=torch.long, device=cls_logits.device),
        )


class STQDDetCriterion(nn.Module):
    """Combined loss for STQD-Det.

    Total = focal_cls + λ_l1 * L1_reg + λ_giou * GIoU_reg + λ_num * L_num

    Args:
        cfg: Config with loss weights and hyperparameters.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.matcher = HungarianMatcher(
            cost_class=2.0,
            cost_bbox=cfg.lambda_l1,
            cost_giou=cfg.lambda_giou,
            img_w=cfg.img_w,
            img_h=cfg.img_h,
        )
        self.num_classes = cfg.num_classes

    def forward(
        self,
        layer_outputs: list[dict],
        gt_boxes_per_frame: list[torch.Tensor],
        gt_labels_per_frame: list[torch.Tensor],
        voted_count: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for all decoder layer outputs.

        Args:
            layer_outputs: List of dicts from decoder, each with:
                "cls_logits": (sum_N, P, num_classes)
                "box_pred": (sum_N, P, 4) xyxy absolute
            gt_boxes_per_frame: List of (M_n, 4) GT boxes per frame.
            gt_labels_per_frame: List of (M_n,) GT labels per frame.
            voted_count: n_r from STFS voting (for consistency loss).
                If None, consistency loss is not computed.

        Returns:
            losses: Dict with "loss_cls", "loss_l1", "loss_giou",
                "loss_consistency", "total_loss".
        """
        device = layer_outputs[0]["cls_logits"].device
        N = len(gt_boxes_per_frame)
        num_layers = len(layer_outputs)

        # Force fp32 for all loss math to prevent AMP fp16 overflow.
        # Also sanitize: fp16 forward pass can produce inf/NaN which we must
        # clamp to finite values so backward doesn't corrupt model weights.
        img_bound = float(max(self.cfg.img_w, self.cfg.img_h))
        layer_outputs = [
            {
                "cls_logits": lo["cls_logits"].float().clamp(-50, 50),
                "box_pred": lo["box_pred"].float().clamp(0, img_bound),
            }
            for lo in layer_outputs
        ]

        total_cls = torch.tensor(0.0, device=device)
        total_l1 = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)

        # Pre-compute inverse image scale for L1 normalization (once)
        img_scale_inv = torch.tensor(
            [1.0 / self.cfg.img_w, 1.0 / self.cfg.img_h,
             1.0 / self.cfg.img_w, 1.0 / self.cfg.img_h],
            device=device,
        )

        # Pre-compute Hungarian matching on the LAST decoder layer only,
        # then reuse indices for all layers. This reduces N_layers × N_frames
        # scipy CPU round-trips to just N_frames calls.
        last_layer = layer_outputs[-1]
        cached_matches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for n in range(N):
            gt_boxes = gt_boxes_per_frame[n].to(device)
            gt_labels = gt_labels_per_frame[n].to(device)
            pred_idx, gt_idx = self.matcher(
                last_layer["cls_logits"][n],
                last_layer["box_pred"][n],
                gt_boxes, gt_labels,
            )
            cached_matches.append((pred_idx, gt_idx))

        for layer_out in layer_outputs:
            cls_logits = layer_out["cls_logits"]  # (sum_N, P, num_classes)
            box_pred = layer_out["box_pred"]       # (sum_N, P, 4)

            layer_cls = torch.tensor(0.0, device=device)
            layer_l1 = torch.tensor(0.0, device=device)
            layer_giou = torch.tensor(0.0, device=device)
            num_matched = 0

            for n in range(N):
                gt_boxes = gt_boxes_per_frame[n].to(device)
                gt_labels = gt_labels_per_frame[n].to(device)

                pred_cls = cls_logits[n]   # (P, num_classes)
                pred_box = box_pred[n]     # (P, 4)

                # Reuse cached matching indices
                pred_idx, gt_idx = cached_matches[n]

                # Classification loss (focal loss for all predictions)
                target_classes = torch.zeros(
                    (pred_cls.shape[0], self.num_classes),
                    device=device,
                )
                if len(pred_idx) > 0:
                    # Vectorized one-hot assignment
                    target_classes[pred_idx, gt_labels[gt_idx]] = 1.0

                cls_loss = sigmoid_focal_loss(
                    pred_cls, target_classes,
                    alpha=self.cfg.focal_alpha,
                    gamma=self.cfg.focal_gamma,
                    reduction="sum",
                )
                layer_cls = layer_cls + cls_loss

                # Box losses only for matched predictions
                if len(pred_idx) > 0:
                    matched_pred = pred_box[pred_idx]
                    matched_gt = gt_boxes[gt_idx]

                    l1 = F.l1_loss(
                        matched_pred * img_scale_inv,
                        matched_gt * img_scale_inv,
                        reduction="sum",
                    )
                    layer_l1 = layer_l1 + l1

                    # Sanitize predicted boxes: clamp to min 1px width/height
                    # to prevent NaN from zero-area boxes early in training
                    mp = matched_pred.float()
                    mp_w = (mp[:, 2] - mp[:, 0]).clamp(min=1.0)
                    mp_h = (mp[:, 3] - mp[:, 1]).clamp(min=1.0)
                    mp_safe = torch.stack([mp[:, 0], mp[:, 1], mp[:, 0] + mp_w, mp[:, 1] + mp_h], dim=-1)

                    giou = generalized_box_iou_loss(
                        mp_safe, matched_gt.float(), reduction="sum"
                    )
                    if not torch.isfinite(giou):
                        giou = torch.tensor(0.0, device=device)
                    layer_giou = layer_giou + giou

                    num_matched += len(pred_idx)

            # Normalize by number of matched pairs (avoid div by zero)
            denominator = max(num_matched, 1)
            total_cls = total_cls + layer_cls / denominator
            total_l1 = total_l1 + layer_l1 / denominator
            total_giou = total_giou + layer_giou / denominator

        # Average over decoder layers
        total_cls = total_cls / num_layers
        total_l1 = total_l1 / num_layers
        total_giou = total_giou / num_layers

        # Consistency loss L_num
        loss_consistency = torch.tensor(0.0, device=device)
        if voted_count is not None:
            # L_num = (1/N) * Σ |n_boxes,n - n_r| + β
            # Use soft (differentiable) count: sum of max-class sigmoid
            # probabilities instead of hard threshold > 0.5
            for n in range(N):
                n_boxes_n = layer_outputs[-1]["cls_logits"][n].sigmoid().max(dim=-1).values.sum()
                loss_consistency = loss_consistency + torch.abs(n_boxes_n - voted_count)
            loss_consistency = loss_consistency / N + self.cfg.beta_consistency

        # Total loss
        total_loss = (
            total_cls
            + self.cfg.lambda_l1 * total_l1
            + self.cfg.lambda_giou * total_giou
            + self.cfg.lambda_num * loss_consistency
        )

        return {
            "loss_cls": total_cls,
            "loss_l1": total_l1,
            "loss_giou": total_giou,
            "loss_consistency": loss_consistency,
            "total_loss": total_loss,
        }
