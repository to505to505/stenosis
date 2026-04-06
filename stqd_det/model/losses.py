"""Loss functions for STQD-Det.

Combines:
  - Focal Loss (classification)
  - L1 Loss (bounding box regression)
  - GIoU Loss (bounding box regression)
  - Consistency Loss L_num (sequence detection count consistency)

Uses Hungarian matching to assign predictions to ground truth.
"""

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
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

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

        # L1 box cost
        cost_bbox = torch.cdist(pred_boxes, gt_boxes, p=1)  # (P, M)

        # GIoU cost
        iou = box_iou(pred_boxes, gt_boxes)  # (P, M)
        cost_giou = -iou  # (P, M) — negative IoU

        # Combined cost
        C = (
            self.cost_class * cost_class
            + self.cost_bbox * cost_bbox
            + self.cost_giou * cost_giou
        )

        # Hungarian matching
        C_np = C.detach().cpu().numpy()
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

        total_cls = torch.tensor(0.0, device=device)
        total_l1 = torch.tensor(0.0, device=device)
        total_giou = torch.tensor(0.0, device=device)

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

                # Hungarian matching
                pred_idx, gt_idx = self.matcher(
                    pred_cls, pred_box, gt_boxes, gt_labels
                )

                # Classification loss (focal loss for all predictions)
                # Create target: -1 for unmatched (background), gt_label for matched
                target_classes = torch.full(
                    (pred_cls.shape[0], self.num_classes),
                    0.0, device=device,
                )
                if len(pred_idx) > 0:
                    # One-hot for matched predictions
                    for pi, gi in zip(pred_idx, gt_idx):
                        target_classes[pi, gt_labels[gi]] = 1.0

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

                    # L1 loss
                    l1 = F.l1_loss(matched_pred, matched_gt, reduction="sum")
                    layer_l1 = layer_l1 + l1

                    # GIoU loss
                    giou = generalized_box_iou_loss(
                        matched_pred, matched_gt, reduction="sum"
                    )
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
            for n in range(N):
                n_boxes_n = (layer_outputs[-1]["cls_logits"][n].sigmoid().max(dim=-1).values > 0.5).sum().float()
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
