"""Evaluation utilities for the stenosis student.

Computes COCO-style mAP / max-recall / best-F1 metrics on centre-frame
predictions.  Box-handling is identical to ``rfdetr_temporal/evaluate.py``
but adapted for FCOS targets which are already in absolute xyxy pixels.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.cuda.amp import autocast

from .config import Config
from .postprocess import postprocess as fcos_postprocess


# ─── core metric helpers (copied / adapted from rfdetr_temporal) ────────
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
            det_a = (d_box[2] - d_box[0]) * (d_box[3] - d_box[1])
            gt_a = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
            union = det_a + gt_a - inter
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


# ─── main entry point ───────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> Dict[str, float]:
    model.eval()

    all_detections: List[Dict[str, np.ndarray]] = []
    all_ground_truths: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    for images, _centre_clean, targets, _fnames in loader:
        images = images.to(device, non_blocking=True)
        gpu_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                       for t in targets]
        with autocast(enabled=cfg.amp):
            cls_logits, bbox_reg, centerness = model(images)
            losses = model.criterion(cls_logits, bbox_reg, centerness, gpu_targets)
        total_loss += float(losses["loss"].item())
        n_batches += 1

        results = fcos_postprocess(cls_logits, bbox_reg, centerness, cfg, cfg.img_size)
        for b in range(len(results)):
            all_detections.append({
                "boxes": results[b]["boxes"].cpu().numpy(),
                "scores": results[b]["scores"].cpu().numpy(),
            })
            gt = targets[b]["boxes"].cpu().numpy()
            all_ground_truths.append(gt if gt.shape[0] > 0
                                      else np.zeros((0, 4), dtype=np.float32))

    ap50 = evaluate_map(all_detections, all_ground_truths, 0.5)
    ap75 = evaluate_map(all_detections, all_ground_truths, 0.75)
    iou_thrs = np.arange(0.5, 1.0, 0.05)
    ap5095 = float(np.mean([evaluate_map(all_detections, all_ground_truths, t)
                            for t in iou_thrs]))
    mar = float(np.mean([compute_max_recall(all_detections, all_ground_truths, t)
                         for t in iou_thrs]))
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
