"""Evaluation utilities for Temporal RF-DETR.

Computes COCO-style mAP metrics on centre-frame predictions.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast

from .config import Config

# ─────────────────────────────────────────────────────────────────────
#  Core metric helpers (same as stenosis_temporal/evaluate.py)
# ─────────────────────────────────────────────────────────────────────

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map(all_detections, all_ground_truths, iou_threshold=0.5):
    det_boxes, det_scores, det_img_ids = [], [], []
    for img_id, det in enumerate(all_detections):
        n = det["scores"].shape[0]
        det_boxes.append(det["boxes"])
        det_scores.append(det["scores"])
        det_img_ids.extend([img_id] * n)

    if not det_boxes:
        return 0.0

    det_boxes = np.concatenate(det_boxes, 0)
    det_scores = np.concatenate(det_scores, 0)
    det_img_ids = np.array(det_img_ids)

    order = np.argsort(-det_scores)
    det_boxes = det_boxes[order]
    det_img_ids = det_img_ids[order]

    total_gt = sum(gt.shape[0] for gt in all_ground_truths)
    if total_gt == 0:
        return 0.0

    gt_matched = {
        i: np.zeros(gt.shape[0], dtype=bool)
        for i, gt in enumerate(all_ground_truths) if gt.shape[0] > 0
    }
    tp = np.zeros(len(det_scores))
    fp = np.zeros(len(det_scores))

    for d in range(len(det_scores)):
        img_id = det_img_ids[d]
        d_box = det_boxes[d]
        gt = all_ground_truths[img_id]
        if gt.shape[0] == 0:
            fp[d] = 1
            continue
        ixmin = np.maximum(gt[:, 0], d_box[0])
        iymin = np.maximum(gt[:, 1], d_box[1])
        ixmax = np.minimum(gt[:, 2], d_box[2])
        iymax = np.minimum(gt[:, 3], d_box[3])
        iw = np.maximum(ixmax - ixmin, 0.0)
        ih = np.maximum(iymax - iymin, 0.0)
        inter = iw * ih
        det_area = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
        gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        union = det_area + gt_area - inter
        iou = inter / np.maximum(union, 1e-6)
        best = np.argmax(iou)
        if iou[best] >= iou_threshold and img_id in gt_matched and not gt_matched[img_id][best]:
            tp[d] = 1
            gt_matched[img_id][best] = True
        else:
            fp[d] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / total_gt
    precisions = cum_tp / (cum_tp + cum_fp)
    return compute_ap(recalls, precisions)


def compute_max_recall(all_detections, all_ground_truths, iou_threshold=0.5, max_dets=100):
    total_gt = sum(gt.shape[0] for gt in all_ground_truths)
    if total_gt == 0:
        return 0.0
    total_tp = 0
    for det, gt in zip(all_detections, all_ground_truths):
        if gt.shape[0] == 0:
            continue
        scores = det["scores"]
        boxes = det["boxes"]
        if len(scores) == 0:
            continue
        order = np.argsort(-scores)[:max_dets]
        boxes = boxes[order]
        matched = np.zeros(gt.shape[0], dtype=bool)
        for d_box in boxes:
            ixmin = np.maximum(gt[:, 0], d_box[0])
            iymin = np.maximum(gt[:, 1], d_box[1])
            ixmax = np.minimum(gt[:, 2], d_box[2])
            iymax = np.minimum(gt[:, 3], d_box[3])
            iw = np.maximum(ixmax - ixmin, 0.0)
            ih = np.maximum(iymax - iymin, 0.0)
            inter = iw * ih
            det_area = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
            gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
            union = det_area + gt_area - inter
            iou = inter / np.maximum(union, 1e-6)
            best = np.argmax(iou)
            if iou[best] >= iou_threshold and not matched[best]:
                matched[best] = True
                total_tp += 1
    return total_tp / total_gt


def f1_confidence_sweep(all_detections, all_ground_truths, iou_threshold=0.5, n=101):
    total_gt = sum(gt.shape[0] for gt in all_ground_truths)
    if total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0
    best_f1, best_p, best_r, best_thr = 0.0, 0.0, 0.0, 0.0
    for thr in np.linspace(0.0, 1.0, n):
        tp = fp = fn = 0
        for det, gt in zip(all_detections, all_ground_truths):
            mask = det["scores"] >= thr
            filt = det["boxes"][mask]
            n_gt = gt.shape[0]
            if n_gt == 0:
                fp += filt.shape[0]
                continue
            if filt.shape[0] == 0:
                fn += n_gt
                continue
            matched = np.zeros(n_gt, dtype=bool)
            order = np.argsort(-det["scores"][mask])
            for d_box in filt[order]:
                ixmin = np.maximum(gt[:, 0], d_box[0])
                iymin = np.maximum(gt[:, 1], d_box[1])
                ixmax = np.minimum(gt[:, 2], d_box[2])
                iymax = np.minimum(gt[:, 3], d_box[3])
                iw = np.maximum(ixmax - ixmin, 0.0)
                ih = np.maximum(iymax - iymin, 0.0)
                inter = iw * ih
                det_a = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
                gt_a = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
                union = det_a + gt_a - inter
                iou = inter / np.maximum(union, 1e-6)
                best = np.argmax(iou)
                if iou[best] >= iou_threshold and not matched[best]:
                    matched[best] = True
                    tp += 1
                else:
                    fp += 1
            fn += int(np.sum(~matched))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        if f1 > best_f1:
            best_f1, best_p, best_r, best_thr = f1, precision, recall, thr
    return best_f1, best_p, best_r, best_thr


# ─────────────────────────────────────────────────────────────────────
#  Main evaluate function (called from train.py)
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, postprocess, cfg: Config, device):
    """Run evaluation on a DataLoader, return metrics dict."""
    model.eval()
    criterion.eval()

    all_detections = []
    all_ground_truths = []
    total_loss = 0.0
    n_batches = 0

    for images, targets_list, _fnames in loader:
        images = images.to(device)
        B = images.shape[0]
        centre = cfg.T // 2

        # Centre-frame targets
        centre_targets = []
        for sample_targets in targets_list:
            t = sample_targets[centre]
            centre_targets.append({
                "boxes": t["boxes"].to(device),
                "labels": t["labels"].to(device),
                "orig_size": torch.tensor([cfg.img_size, cfg.img_size], device=device),
            })

        with autocast(enabled=cfg.amp):
            outputs = model(images)
            loss_dict = criterion(outputs, centre_targets)
            loss = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict if k in criterion.weight_dict
            )
        total_loss += loss.item()
        n_batches += 1

        # Post-process predictions to absolute xyxy
        orig_sizes = torch.stack([t["orig_size"] for t in centre_targets])
        results = postprocess(outputs, orig_sizes)

        for b in range(B):
            scores = results[b]["scores"].cpu().numpy()
            boxes  = results[b]["boxes"].cpu().numpy()

            all_detections.append({"boxes": boxes, "scores": scores})

            # Ground truth in absolute xyxy
            gt_cxcywh = centre_targets[b]["boxes"].cpu().numpy()
            if gt_cxcywh.shape[0] > 0:
                cx, cy, w, h = gt_cxcywh[:, 0], gt_cxcywh[:, 1], gt_cxcywh[:, 2], gt_cxcywh[:, 3]
                x1 = (cx - w / 2) * cfg.img_size
                y1 = (cy - h / 2) * cfg.img_size
                x2 = (cx + w / 2) * cfg.img_size
                y2 = (cy + h / 2) * cfg.img_size
                gt_xyxy = np.column_stack([x1, y1, x2, y2])
            else:
                gt_xyxy = np.zeros((0, 4), dtype=np.float32)
            all_ground_truths.append(gt_xyxy)

    # Compute metrics
    ap50 = evaluate_map(all_detections, all_ground_truths, 0.5)
    ap75 = evaluate_map(all_detections, all_ground_truths, 0.75)
    iou_thrs = np.arange(0.5, 1.0, 0.05)
    ap5095 = float(np.mean([
        evaluate_map(all_detections, all_ground_truths, t) for t in iou_thrs
    ]))
    mar = float(np.mean([
        compute_max_recall(all_detections, all_ground_truths, t) for t in iou_thrs
    ]))
    f1, prec, rec, conf = f1_confidence_sweep(all_detections, all_ground_truths)

    return {
        "AP@0.5": ap50,
        "AP@0.75": ap75,
        "AP@0.5:0.95": ap5095,
        "mAR": mar,
        "F1": f1,
        "precision": prec,
        "recall": rec,
        "best_conf": conf,
        "val_loss": total_loss / max(n_batches, 1),
    }
