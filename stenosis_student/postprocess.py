"""Inference-time decoding for the FCOS head: scoring + NMS."""

from __future__ import annotations

from typing import Dict, List

import torch
from torchvision.ops import batched_nms

from .config import Config
from .head import make_locations


@torch.no_grad()
def postprocess(
    cls_logits: List[torch.Tensor],
    bbox_reg: List[torch.Tensor],
    centerness: List[torch.Tensor],
    cfg: Config,
    image_size: int | tuple,
) -> List[Dict[str, torch.Tensor]]:
    """Decode FCOS predictions into per-image detections.

    Args:
        cls_logits, bbox_reg, centerness: per-level predictions.
        cfg: model config (uses score_thresh, nms_thresh, *_topk, strides).
        image_size: square ``int`` or ``(H, W)``.

    Returns:
        List of length ``B`` of dicts with
            ``"boxes": (K, 4)`` xyxy abs px,
            ``"scores": (K,)``,
            ``"labels": (K,)``.
    """
    device = cls_logits[0].device
    if isinstance(image_size, int):
        H_img = W_img = image_size
    else:
        H_img, W_img = image_size

    strides = tuple(cfg.stage_strides[i] for i in cfg.fpn_stage_indices)
    feature_sizes = [tuple(c.shape[-2:]) for c in cls_logits]
    locations = make_locations(feature_sizes, strides, device)

    B = cls_logits[0].shape[0]
    C = cfg.num_classes

    results: List[Dict[str, torch.Tensor]] = []
    for b in range(B):
        all_boxes, all_scores, all_labels = [], [], []
        for lvl in range(len(cls_logits)):
            cls = cls_logits[lvl][b].permute(1, 2, 0).reshape(-1, C)  # (HW, C)
            reg = bbox_reg[lvl][b].permute(1, 2, 0).reshape(-1, 4)
            ctr = centerness[lvl][b].permute(1, 2, 0).reshape(-1)
            loc = locations[lvl]                                      # (HW, 2)

            scores = torch.sigmoid(cls) * torch.sigmoid(ctr).unsqueeze(1)  # (HW, C)
            max_scores, max_cls = scores.max(dim=1)                   # (HW,)
            keep = max_scores >= cfg.score_thresh
            if keep.sum() == 0:
                continue
            max_scores = max_scores[keep]
            max_cls = max_cls[keep]
            reg = reg[keep]
            loc = loc[keep]

            if max_scores.numel() > cfg.pre_nms_topk:
                top = max_scores.topk(cfg.pre_nms_topk).indices
                max_scores = max_scores[top]
                max_cls = max_cls[top]
                reg = reg[top]
                loc = loc[top]

            x1 = (loc[:, 0] - reg[:, 0]).clamp(0, W_img)
            y1 = (loc[:, 1] - reg[:, 1]).clamp(0, H_img)
            x2 = (loc[:, 0] + reg[:, 2]).clamp(0, W_img)
            y2 = (loc[:, 1] + reg[:, 3]).clamp(0, H_img)
            boxes = torch.stack([x1, y1, x2, y2], dim=1)
            all_boxes.append(boxes)
            all_scores.append(max_scores)
            all_labels.append(max_cls)

        if not all_boxes:
            results.append({
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), device=device, dtype=torch.long),
            })
            continue

        boxes = torch.cat(all_boxes, dim=0)
        scores = torch.cat(all_scores, dim=0)
        labels = torch.cat(all_labels, dim=0)

        keep = batched_nms(boxes, scores, labels, cfg.nms_thresh)
        keep = keep[: cfg.post_nms_topk]
        results.append({
            "boxes": boxes[keep],
            "scores": scores[keep],
            "labels": labels[keep],
        })
    return results
