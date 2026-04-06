"""Guided Anchoring Region Proposal Network (GA-RPN).

Ported from mmdetection's GARPNHead / GuidedAnchorHead.
Predicts anchor location, shape, and generates guided proposals.
Uses FPN feature maps as input and outputs proposals + losses.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign, batched_nms, box_iou

from ..config import Config
from .feature_adaption import FeatureAdaption


# ── Anchor utilities ────────────────────────────────────────────────────


def generate_square_anchors(
    feat_h: int, feat_w: int, stride: int, scale: int, device: torch.device
) -> torch.Tensor:
    """Generate square anchor centers on a feature map grid.

    Returns:
        anchors: (H*W, 4) in x1y1x2y2 format
    """
    half = (scale * stride) / 2.0
    shift_y = torch.arange(0, feat_h, device=device).float() * stride + stride / 2
    shift_x = torch.arange(0, feat_w, device=device).float() * stride + stride / 2
    yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    anchors = torch.stack([xx - half, yy - half, xx + half, yy + half], dim=1)
    return anchors


def generate_approx_anchors(
    feat_h: int,
    feat_w: int,
    stride: int,
    sizes: Tuple,
    ratios: Tuple,
    device: torch.device,
) -> torch.Tensor:
    """Generate multi-scale multi-ratio approximate anchors.

    Returns:
        anchors: (H*W*A, 4) in x1y1x2y2 format where A = len(sizes)*len(ratios)
    """
    shift_y = torch.arange(0, feat_h, device=device).float() * stride + stride / 2
    shift_x = torch.arange(0, feat_w, device=device).float() * stride + stride / 2
    yy, xx = torch.meshgrid(shift_y, shift_x, indexing="ij")
    shifts = torch.stack([xx.reshape(-1), yy.reshape(-1),
                          xx.reshape(-1), yy.reshape(-1)], dim=1)

    # Base anchors at origin
    base_anchors = []
    for s in sizes:
        for r in ratios:
            h = s * math.sqrt(r)
            w = s / math.sqrt(r)
            base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])
    base_anchors = torch.tensor(base_anchors, device=device, dtype=torch.float32)

    # Broadcast: (H*W, 1, 4) + (1, A, 4)
    anchors = shifts.unsqueeze(1) + base_anchors.unsqueeze(0)
    return anchors.reshape(-1, 4)


def guided_anchor_transform(
    square_anchors: torch.Tensor,
    shape_pred: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Transform square anchors using predicted shape.

    Args:
        square_anchors: (H*W, 4) base square anchors
        shape_pred: (H*W, 2) predicted log-space w/h deltas

    Returns:
        guided_anchors: (H*W, 4) deformed anchors
    """
    cx = (square_anchors[:, 0] + square_anchors[:, 2]) / 2
    cy = (square_anchors[:, 1] + square_anchors[:, 3]) / 2
    bw = square_anchors[:, 2] - square_anchors[:, 0]
    bh = square_anchors[:, 3] - square_anchors[:, 1]

    dw = shape_pred[:, 0].clamp(max=4.0)
    dh = shape_pred[:, 1].clamp(max=4.0)
    new_w = bw * torch.exp(dw)
    new_h = bh * torch.exp(dh)

    x1 = cx - new_w / 2
    y1 = cy - new_h / 2
    x2 = cx + new_w / 2
    y2 = cy + new_h / 2
    return torch.stack([x1, y1, x2, y2], dim=1)


# ── Losses ──────────────────────────────────────────────────────────────


def bounded_iou_loss(
    pred: torch.Tensor, target: torch.Tensor, beta: float = 0.2
) -> torch.Tensor:
    """Bounded IoU loss for anchor shape regression.

    Args:
        pred: (N, 4) predicted anchors x1y1x2y2
        target: (N, 4) target anchors x1y1x2y2

    Returns:
        loss: scalar
    """
    pred_cx = (pred[:, 0] + pred[:, 2]) / 2
    pred_cy = (pred[:, 1] + pred[:, 3]) / 2
    pred_w = (pred[:, 2] - pred[:, 0]).clamp(min=1.0)
    pred_h = (pred[:, 3] - pred[:, 1]).clamp(min=1.0)

    target_cx = (target[:, 0] + target[:, 2]) / 2
    target_cy = (target[:, 1] + target[:, 3]) / 2
    target_w = (target[:, 2] - target[:, 0]).clamp(min=1.0)
    target_h = (target[:, 3] - target[:, 1]).clamp(min=1.0)

    dx = (pred_cx - target_cx).abs() / target_w
    dy = (pred_cy - target_cy).abs() / target_h

    loss_w = 1 - torch.min(pred_w, target_w) / torch.max(pred_w, target_w)
    loss_h = 1 - torch.min(pred_h, target_h) / torch.max(pred_h, target_h)

    loss = torch.stack([dx, dy, loss_w, loss_h], dim=-1)
    loss = F.smooth_l1_loss(loss, torch.zeros_like(loss), beta=beta, reduction="none")
    return loss.mean()


def focal_loss(
    pred: torch.Tensor, target: torch.Tensor,
    alpha: float = 0.25, gamma: float = 2.0,
) -> torch.Tensor:
    """Focal loss for location prediction (computed in fp32 for AMP safety)."""
    pred = pred.float()
    target = target.float()
    pred_sigmoid = pred.sigmoid()
    pt = torch.where(target == 1, pred_sigmoid, 1 - pred_sigmoid)
    at = torch.where(target == 1, alpha, 1 - alpha)
    loss = -at * (1 - pt) ** gamma * torch.log(pt.clamp(min=1e-6))
    return loss.mean()


# ── Location target generation ──────────────────────────────────────────


def ga_loc_targets(
    gt_boxes_list: List[torch.Tensor],
    feat_sizes: List[Tuple[int, int]],
    strides: Tuple[int, ...],
    center_ratio: float = 0.2,
    img_h: int = 512,
    img_w: int = 512,
) -> List[torch.Tensor]:
    """Generate location prediction targets for each FPN level.

    For each GT box, the positive region is the center_ratio portion
    of the box projected onto the appropriate feature level.

    Returns:
        loc_targets: list of (N, 1, H_l, W_l) tensors per level
    """
    device = gt_boxes_list[0].device if len(gt_boxes_list) > 0 and gt_boxes_list[0].numel() > 0 else torch.device("cpu")
    batch_size = len(gt_boxes_list)
    num_levels = len(feat_sizes)

    targets = []
    for lvl in range(num_levels):
        fh, fw = feat_sizes[lvl]
        stride = strides[lvl]
        lvl_targets = torch.zeros(batch_size, 1, fh, fw, device=device)

        for b in range(batch_size):
            gt_boxes = gt_boxes_list[b]
            if gt_boxes.numel() == 0:
                continue
            for box in gt_boxes:
                x1, y1, x2, y2 = box
                bw = x2 - x1
                bh = y2 - y1
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # Assign box to level based on area
                area = bw * bh
                target_level = max(0, min(num_levels - 1,
                    int(math.log2(max(area.sqrt().item(), 1)) - 2)))

                if target_level != lvl:
                    continue

                # Center region
                r = center_ratio
                cx1 = cx - bw * r / 2
                cy1 = cy - bh * r / 2
                cx2 = cx + bw * r / 2
                cy2 = cy + bh * r / 2

                # Map to feature coords
                fx1 = max(0, int(cx1.item() / stride))
                fy1 = max(0, int(cy1.item() / stride))
                fx2 = min(fw, int(cx2.item() / stride) + 1)
                fy2 = min(fh, int(cy2.item() / stride) + 1)

                if fx1 < fx2 and fy1 < fy2:
                    lvl_targets[b, 0, fy1:fy2, fx1:fx2] = 1.0

        targets.append(lvl_targets)
    return targets


# ── Shape target generation ─────────────────────────────────────────────


def ga_shape_targets(
    approx_anchors_list: List[torch.Tensor],
    square_anchors_list: List[torch.Tensor],
    gt_boxes_list: List[torch.Tensor],
    feat_sizes: List[Tuple[int, int]],
    pos_iou_thr: float = 0.5,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Generate shape regression targets.

    For each positive anchor location, compute the shape delta to
    transform the square anchor to best match the assigned GT box.

    Returns:
        shape_targets: list of (N, 2, H, W) delta targets per level
        shape_weights: list of (N, 2, H, W) loss weights (0 or 1)
    """
    batch_size = len(gt_boxes_list)
    num_levels = len(feat_sizes)

    all_targets = []
    all_weights = []

    for lvl in range(num_levels):
        fh, fw = feat_sizes[lvl]
        approx_per_loc = approx_anchors_list[lvl].shape[0] // (fh * fw) if fh * fw > 0 else 1
        device = approx_anchors_list[lvl].device

        lvl_targets = torch.zeros(batch_size, 2, fh, fw, device=device)
        lvl_weights = torch.zeros(batch_size, 2, fh, fw, device=device)

        for b in range(batch_size):
            gt_boxes = gt_boxes_list[b]
            if gt_boxes.numel() == 0:
                continue

            sq_anchors = square_anchors_list[lvl]  # (H*W, 4)
            approx = approx_anchors_list[lvl]      # (H*W*A, 4)

            # Compute IoU between approx anchors and GT
            iou_matrix = box_iou(approx, gt_boxes)  # (H*W*A, G)

            # Reshape to (H*W, A, G), take max over A
            n_locs = fh * fw
            if iou_matrix.shape[0] > 0:
                iou_reshaped = iou_matrix.reshape(n_locs, approx_per_loc, -1)
                max_iou_per_loc, _ = iou_reshaped.max(dim=1)  # (H*W, G)
                best_gt_per_loc = max_iou_per_loc.argmax(dim=1)  # (H*W,)
                best_iou_per_loc = max_iou_per_loc.max(dim=1).values  # (H*W,)

                pos_mask = best_iou_per_loc >= pos_iou_thr
                if pos_mask.any():
                    pos_indices = pos_mask.nonzero(as_tuple=True)[0]
                    matched_gt = gt_boxes[best_gt_per_loc[pos_indices]]
                    pos_sq = sq_anchors[pos_indices]

                    # Compute shape deltas
                    sq_w = (pos_sq[:, 2] - pos_sq[:, 0]).clamp(min=1.0)
                    sq_h = (pos_sq[:, 3] - pos_sq[:, 1]).clamp(min=1.0)
                    gt_w = (matched_gt[:, 2] - matched_gt[:, 0]).clamp(min=1.0)
                    gt_h = (matched_gt[:, 3] - matched_gt[:, 1]).clamp(min=1.0)

                    dw = torch.log(gt_w / sq_w)
                    dh = torch.log(gt_h / sq_h)

                    # Map flat indices to (y, x) grid coords
                    gy = pos_indices // fw
                    gx = pos_indices % fw

                    lvl_targets[b, 0, gy, gx] = dw
                    lvl_targets[b, 1, gy, gx] = dh
                    lvl_weights[b, :, gy, gx] = 1.0

        all_targets.append(lvl_targets)
        all_weights.append(lvl_weights)

    return all_targets, all_weights


# ── GA-RPN Head ─────────────────────────────────────────────────────────


class GuidedAnchoringRPN(nn.Module):
    """Guided Anchoring RPN ported from mmdetection.

    For each FPN level:
    1. Predict anchor locations (where to place anchors)
    2. Predict anchor shapes (aspect ratio / size deformation)
    3. Adapt features via deformable convolution
    4. Classify & regress on adapted features
    5. Generate proposals via guided anchors + NMS
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        in_channels = cfg.C  # 256

        # Shared conv before branches
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)

        # Location prediction → (N, 1, H, W)
        self.conv_loc = nn.Conv2d(in_channels, 1, 1)

        # Shape prediction → (N, 2, H, W) — log(dw), log(dh)
        self.conv_shape = nn.Conv2d(in_channels, 2, 1)

        # Feature adaption using deformable convolution
        self.feature_adaption = FeatureAdaption(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            deform_groups=cfg.ga_deform_groups,
        )

        # Classification on adapted features: foreground vs background
        self.conv_cls = nn.Conv2d(in_channels, 1, 1)

        # Box regression on adapted features: 4 deltas
        self.conv_reg = nn.Conv2d(in_channels, 4, 1)

        # RoI Align for extracting proposal features
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=cfg.roi_output_size,
            sampling_ratio=2,
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.rpn_conv, self.conv_cls, self.conv_reg]:
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.conv_loc.weight, 0, 0.01)
        nn.init.zeros_(self.conv_loc.bias)
        nn.init.zeros_(self.conv_shape.weight)
        nn.init.zeros_(self.conv_shape.bias)

    def _forward_single_level(
        self, x: torch.Tensor, stride: int, scale: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for a single FPN level.

        Returns:
            cls_score: (N, 1, H, W)
            bbox_pred: (N, 4, H, W)
            shape_pred: (N, 2, H, W)
            loc_pred: (N, 1, H, W)
        """
        x = F.relu(self.rpn_conv(x), inplace=False)

        loc_pred = self.conv_loc(x)
        shape_pred = self.conv_shape(x)

        # Adapt features based on predicted shape
        x_adapted = self.feature_adaption(x, shape_pred)

        # Classification and regression on adapted features
        cls_score = self.conv_cls(x_adapted)
        bbox_pred = self.conv_reg(x_adapted)

        return cls_score, bbox_pred, shape_pred, loc_pred

    def _get_proposals_single_level(
        self,
        cls_score: torch.Tensor,
        bbox_pred: torch.Tensor,
        shape_pred: torch.Tensor,
        loc_pred: torch.Tensor,
        stride: int,
        scale: int,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """Generate proposals for one FPN level.

        Returns:
            proposals: (N, K, 4) top-K proposals per image
        """
        N, _, fh, fw = cls_score.shape
        device = cls_score.device

        # Generate square base anchors
        sq_anchors = generate_square_anchors(fh, fw, stride, scale, device)  # (H*W, 4)

        # Transform to guided anchors using shape prediction
        shape_flat = shape_pred.permute(0, 2, 3, 1).reshape(N, fh * fw, 2)

        # Location mask: only keep positions with high location score
        loc_mask = loc_pred.sigmoid() >= self.cfg.ga_loc_filter_thr  # (N, 1, H, W)
        loc_mask = loc_mask.reshape(N, fh * fw)

        # Scores from classification
        scores = cls_score.sigmoid().reshape(N, fh * fw)

        # Box deltas
        deltas = bbox_pred.permute(0, 2, 3, 1).reshape(N, fh * fw, 4)

        all_proposals = []
        for b in range(N):
            # Get guided anchors for this image
            guided = guided_anchor_transform(sq_anchors, shape_flat[b], stride)

            # Apply location mask
            mask = loc_mask[b]
            if mask.sum() == 0:
                # Fallback: keep all positions
                mask = torch.ones_like(mask, dtype=torch.bool)

            valid_guided = guided[mask]
            valid_scores = scores[b][mask]
            valid_deltas = deltas[b][mask]

            # Decode box deltas relative to guided anchors
            pred_boxes = self._decode_deltas(valid_deltas, valid_guided)

            # Clamp to image (out-of-place to avoid autograd inplace error)
            pred_boxes = torch.cat([
                pred_boxes[:, 0:1].clamp(0, img_w),
                pred_boxes[:, 1:2].clamp(0, img_h),
                pred_boxes[:, 2:3].clamp(0, img_w),
                pred_boxes[:, 3:4].clamp(0, img_h),
            ], dim=1)

            all_proposals.append((pred_boxes, valid_scores))

        return all_proposals

    @staticmethod
    def _decode_deltas(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode box deltas relative to anchors."""
        cx = (anchors[:, 0] + anchors[:, 2]) / 2
        cy = (anchors[:, 1] + anchors[:, 3]) / 2
        w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1.0)
        h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1.0)

        dx, dy, dw, dh = deltas.unbind(dim=1)
        dw = dw.clamp(max=4.0)
        dh = dh.clamp(max=4.0)

        pred_cx = dx * w + cx
        pred_cy = dy * h + cy
        pred_w = torch.exp(dw) * w
        pred_h = torch.exp(dh) * h

        x1 = pred_cx - pred_w / 2
        y1 = pred_cy - pred_h / 2
        x2 = pred_cx + pred_w / 2
        y2 = pred_cy + pred_h / 2
        return torch.stack([x1, y1, x2, y2], dim=1)

    @staticmethod
    def _encode_deltas(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Encode GT boxes relative to anchors."""
        cx = (anchors[:, 0] + anchors[:, 2]) / 2
        cy = (anchors[:, 1] + anchors[:, 3]) / 2
        w = (anchors[:, 2] - anchors[:, 0]).clamp(min=1.0)
        h = (anchors[:, 3] - anchors[:, 1]).clamp(min=1.0)

        gx = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gy = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        gw = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
        gh = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)

        dx = (gx - cx) / w
        dy = (gy - cy) / h
        dw = torch.log(gw / w)
        dh = torch.log(gh / h)
        return torch.stack([dx, dy, dw, dh], dim=1)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
        img_h: int = 512,
        img_w: int = 512,
    ) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """
        Args:
            features: FPN feature maps dict "0".."3" + "pool"
            targets: list[dict] with 'boxes' and 'labels'
            img_h, img_w: image dimensions

        Returns:
            proposals: list[Tensor] — (S, 4) per image
            losses: dict of RPN losses (empty during eval)
            roi_features: (N*S, C, roi_size, roi_size) pooled features
        """
        cfg = self.cfg
        strides = cfg.ga_strides
        scale = cfg.ga_square_anchor_scale

        # Process each FPN level
        level_keys = ["0", "1", "2", "3"]
        cls_scores = []
        bbox_preds = []
        shape_preds = []
        loc_preds = []
        feat_sizes = []

        for i, key in enumerate(level_keys):
            feat = features[key]
            fh, fw = feat.shape[2], feat.shape[3]
            feat_sizes.append((fh, fw))
            cs, bp, sp, lp = self._forward_single_level(feat, strides[i], scale)
            cls_scores.append(cs)
            bbox_preds.append(bp)
            shape_preds.append(sp)
            loc_preds.append(lp)

        N = cls_scores[0].shape[0]
        device = cls_scores[0].device

        # ── Generate proposals ──────────────────────────────────────────
        all_proposals_per_level = []
        for i in range(len(level_keys)):
            level_proposals = self._get_proposals_single_level(
                cls_scores[i], bbox_preds[i], shape_preds[i], loc_preds[i],
                strides[i], scale, img_h, img_w,
            )
            all_proposals_per_level.append(level_proposals)

        # Merge proposals across levels and apply NMS per image
        pre_nms_top = (cfg.rpn_pre_nms_top_n_train if self.training
                       else cfg.rpn_pre_nms_top_n_test)
        post_nms_top = (cfg.rpn_post_nms_top_n_train if self.training
                        else cfg.rpn_post_nms_top_n_test)

        proposals = []
        for b in range(N):
            boxes_list = []
            scores_list = []
            for lvl in range(len(level_keys)):
                boxes_b, scores_b = all_proposals_per_level[lvl][b]
                boxes_list.append(boxes_b)
                scores_list.append(scores_b)

            if boxes_list:
                all_boxes = torch.cat(boxes_list, dim=0)
                all_scores = torch.cat(scores_list, dim=0)
            else:
                all_boxes = torch.zeros(0, 4, device=device)
                all_scores = torch.zeros(0, device=device)

            # Pre-NMS top-K
            if all_scores.numel() > pre_nms_top:
                _, topk_idx = all_scores.topk(pre_nms_top)
                all_boxes = all_boxes[topk_idx]
                all_scores = all_scores[topk_idx]

            # NMS
            if all_boxes.numel() > 0:
                keep = batched_nms(
                    all_boxes,
                    all_scores,
                    torch.zeros(len(all_scores), device=device, dtype=torch.int64),
                    cfg.rpn_nms_thresh,
                )
                keep = keep[:post_nms_top]
                all_boxes = all_boxes[keep]

            # Pad or truncate to exactly S proposals
            if all_boxes.shape[0] > cfg.S:
                all_boxes = all_boxes[:cfg.S]
            elif all_boxes.shape[0] < cfg.S:
                if all_boxes.shape[0] == 0:
                    all_boxes = torch.zeros(cfg.S, 4, device=device)
                else:
                    pad = all_boxes[-1:].expand(cfg.S - all_boxes.shape[0], -1)
                    all_boxes = torch.cat([all_boxes, pad], dim=0)

            proposals.append(all_boxes)

        # ── RoI Align ──────────────────────────────────────────────────
        image_sizes = [(img_h, img_w)] * N
        roi_features = self.roi_align(features, proposals, image_sizes)

        # ── Compute losses (training only) ─────────────────────────────
        losses = {}
        if self.training and targets is not None:
            gt_boxes_list = [t["boxes"].to(device) for t in targets]

            # Location loss
            loc_targets = ga_loc_targets(
                gt_boxes_list, feat_sizes, strides[:len(level_keys)],
                center_ratio=cfg.ga_center_ratio,
                img_h=img_h, img_w=img_w,
            )
            loss_loc = torch.tensor(0.0, device=device)
            for lvl in range(len(level_keys)):
                loss_loc += focal_loss(
                    loc_preds[lvl].reshape(-1),
                    loc_targets[lvl].reshape(-1),
                )
            losses["loss_ga_loc"] = loss_loc / len(level_keys)

            # Shape loss using Bounded IoU
            sq_anchors_list = []
            approx_anchors_list = []
            for i, key in enumerate(level_keys):
                fh, fw = feat_sizes[i]
                sq = generate_square_anchors(fh, fw, strides[i], scale, device)
                sq_anchors_list.append(sq)
                sizes_i = cfg.ga_approx_anchor_sizes[i]
                ratios_i = cfg.ga_approx_anchor_ratios[i]
                approx = generate_approx_anchors(
                    fh, fw, strides[i], sizes_i, ratios_i, device
                )
                approx_anchors_list.append(approx)

            shape_tgts, shape_wgts = ga_shape_targets(
                approx_anchors_list, sq_anchors_list, gt_boxes_list,
                feat_sizes, pos_iou_thr=0.5,
            )

            loss_shape = torch.tensor(0.0, device=device)
            n_pos = 0
            for lvl in range(len(level_keys)):
                fh, fw = feat_sizes[lvl]
                wgt = shape_wgts[lvl]  # (N, 2, H, W)
                if wgt.sum() > 0:
                    sq = sq_anchors_list[lvl]  # (H*W, 4)
                    for b in range(N):
                        pos_mask = wgt[b, 0] > 0  # (H, W)
                        if not pos_mask.any():
                            continue
                        pos_idx = pos_mask.reshape(-1).nonzero(as_tuple=True)[0]
                        pos_sq = sq[pos_idx]
                        pred_shapes_b = shape_preds[lvl][b].permute(1, 2, 0).reshape(-1, 2)
                        pos_pred_shapes = pred_shapes_b[pos_idx]

                        # Transform square anchors with predicted shape
                        guided = guided_anchor_transform(pos_sq, pos_pred_shapes, strides[lvl])
                        # Target: transform with target shape
                        tgt_shapes_b = shape_tgts[lvl][b].permute(1, 2, 0).reshape(-1, 2)
                        pos_tgt_shapes = tgt_shapes_b[pos_idx]
                        target_anchors = guided_anchor_transform(pos_sq, pos_tgt_shapes, strides[lvl])

                        loss_shape += bounded_iou_loss(guided, target_anchors)
                        n_pos += 1

            losses["loss_ga_shape"] = loss_shape / max(n_pos, 1)

            # Classification loss (using guided anchors)
            loss_cls = torch.tensor(0.0, device=device)
            loss_bbox = torch.tensor(0.0, device=device)
            n_cls = 0

            for lvl in range(len(level_keys)):
                fh, fw = feat_sizes[lvl]
                sq = sq_anchors_list[lvl]

                for b in range(N):
                    gt_b = gt_boxes_list[b]
                    sp = shape_preds[lvl][b].permute(1, 2, 0).reshape(-1, 2)
                    guided = guided_anchor_transform(sq, sp.detach(), strides[lvl])

                    # Assign guided anchors to GT
                    if gt_b.numel() == 0:
                        cls_target = torch.zeros(fh * fw, device=device, dtype=torch.long)
                        # Still compute cls loss for negatives
                        pred_cls = cls_scores[lvl][b].reshape(-1)
                        loss_cls += F.binary_cross_entropy_with_logits(
                            pred_cls, cls_target.float()
                        )
                        n_cls += 1
                        continue

                    iou_mat = box_iou(guided, gt_b)
                    max_iou, matched = iou_mat.max(dim=1)

                    cls_target = (max_iou >= 0.7).long()
                    cls_target[max_iou < 0.3] = 0
                    # Ignore between 0.3 and 0.7
                    ignore = (max_iou >= 0.3) & (max_iou < 0.7)

                    pred_cls = cls_scores[lvl][b].reshape(-1)
                    valid_mask = ~ignore
                    if valid_mask.any():
                        loss_cls += F.binary_cross_entropy_with_logits(
                            pred_cls[valid_mask], cls_target[valid_mask].float()
                        )

                    # Regression loss on positives
                    pos_mask = cls_target == 1
                    if pos_mask.any():
                        pos_guided = guided[pos_mask]
                        pos_gt = gt_b[matched[pos_mask]]
                        pos_deltas = bbox_preds[lvl][b].permute(1, 2, 0).reshape(-1, 4)[pos_mask]
                        target_deltas = self._encode_deltas(pos_gt, pos_guided)
                        loss_bbox += F.smooth_l1_loss(pos_deltas, target_deltas)

                    n_cls += 1

            losses["loss_rpn_cls"] = loss_cls / max(n_cls, 1)
            losses["loss_rpn_bbox"] = loss_bbox / max(n_cls, 1)

        return proposals, losses, roi_features
