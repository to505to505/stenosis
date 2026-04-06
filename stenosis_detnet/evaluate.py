"""Evaluation script — compute COCO-style mAP metrics for Stenosis-DetNet."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stenosis_detnet.config import Config
from stenosis_detnet.dataset import get_dataloader
from stenosis_detnet.model.detector import StenosisDetNet


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using the all-point interpolation method (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def evaluate_map(
    all_detections: list,
    all_ground_truths: list,
    iou_threshold: float = 0.5,
) -> float:
    """Compute AP at a given IoU threshold."""
    det_boxes = []
    det_scores = []
    det_img_ids = []

    for img_id, det in enumerate(all_detections):
        n = det["scores"].shape[0]
        det_boxes.append(det["boxes"])
        det_scores.append(det["scores"])
        det_img_ids.extend([img_id] * n)

    if not det_boxes:
        return 0.0

    det_boxes = np.concatenate(det_boxes, axis=0)
    det_scores = np.concatenate(det_scores, axis=0)
    det_img_ids = np.array(det_img_ids)

    order = np.argsort(-det_scores)
    det_boxes = det_boxes[order]
    det_scores = det_scores[order]
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

    for d_idx in range(len(det_scores)):
        img_id = det_img_ids[d_idx]
        det_box = det_boxes[d_idx]
        gt = all_ground_truths[img_id]

        if gt.shape[0] == 0:
            fp[d_idx] = 1
            continue

        ixmin = np.maximum(gt[:, 0], det_box[0])
        iymin = np.maximum(gt[:, 1], det_box[1])
        ixmax = np.minimum(gt[:, 2], det_box[2])
        iymax = np.minimum(gt[:, 3], det_box[3])
        iw = np.maximum(ixmax - ixmin, 0.0)
        ih = np.maximum(iymax - iymin, 0.0)
        inter = iw * ih

        det_area = (det_box[2] - det_box[0]) * (det_box[3] - det_box[1])
        gt_area = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
        union = det_area + gt_area - inter
        iou = inter / np.maximum(union, 1e-6)

        best_gt = np.argmax(iou)
        best_iou = iou[best_gt]

        if best_iou >= iou_threshold and img_id in gt_matched and not gt_matched[img_id][best_gt]:
            tp[d_idx] = 1
            gt_matched[img_id][best_gt] = True
        else:
            fp[d_idx] = 1

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / total_gt
    precisions = cum_tp / (cum_tp + cum_fp)

    return compute_ap(recalls, precisions)


def compute_max_recall(
    all_detections: list,
    all_ground_truths: list,
    iou_threshold: float = 0.5,
    max_dets: int = 100,
) -> float:
    """Compute max recall at a given IoU threshold."""
    total_gt = sum(gt.shape[0] for gt in all_ground_truths)
    if total_gt == 0:
        return 0.0

    total_tp = 0
    for det, gt in zip(all_detections, all_ground_truths):
        if gt.shape[0] == 0:
            continue
        boxes = det["boxes"]
        scores = det["scores"]
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
            best_gt = np.argmax(iou)
            if iou[best_gt] >= iou_threshold and not matched[best_gt]:
                matched[best_gt] = True
                total_tp += 1
    return total_tp / total_gt


def f1_confidence_sweep(
    all_detections: list,
    all_ground_truths: list,
    iou_threshold: float = 0.5,
    num_thresholds: int = 101,
) -> tuple:
    """Sweep confidence thresholds and return (best_f1, precision, recall, threshold)."""
    total_gt = sum(gt.shape[0] for gt in all_ground_truths)
    if total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0

    best_f1, best_p, best_r, best_thr = 0.0, 0.0, 0.0, 0.0

    for thr in np.linspace(0.0, 1.0, num_thresholds):
        tp, fp, fn = 0, 0, 0
        for det, gt in zip(all_detections, all_ground_truths):
            scores = det["scores"]
            boxes = det["boxes"]
            mask = scores >= thr
            filt_boxes = boxes[mask]

            n_gt = gt.shape[0]
            if n_gt == 0:
                fp += filt_boxes.shape[0]
                continue
            if filt_boxes.shape[0] == 0:
                fn += n_gt
                continue

            matched = np.zeros(n_gt, dtype=bool)
            filt_scores = scores[mask]
            order = np.argsort(-filt_scores)
            for d_box in filt_boxes[order]:
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
                best_gt_idx = np.argmax(iou)
                if iou[best_gt_idx] >= iou_threshold and not matched[best_gt_idx]:
                    matched[best_gt_idx] = True
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


@torch.no_grad()
def run_evaluation(model, loader, device, cfg):
    """Run model on dataset and compute full metrics."""
    model.eval()
    all_detections = []
    all_ground_truths = []

    for images, targets in loader:
        images = images.to(device)
        B, T = images.shape[:2]

        with autocast('cuda', enabled=cfg.amp):
            results = model(images, None)

        for res in results:
            b, t = res["batch_idx"], res["frame_idx"]
            all_detections.append({
                "boxes": res["boxes"].cpu().numpy(),
                "scores": res["scores"].cpu().numpy(),
            })
            gt = targets[b][t]["boxes"].numpy()
            all_ground_truths.append(gt)

    # AP metrics
    ap50 = evaluate_map(all_detections, all_ground_truths, iou_threshold=0.5)
    ap75 = evaluate_map(all_detections, all_ground_truths, iou_threshold=0.75)

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    aps = [evaluate_map(all_detections, all_ground_truths, iou_threshold=t)
           for t in iou_thresholds]
    ap5095 = np.mean(aps)

    recalls_per_iou = [compute_max_recall(all_detections, all_ground_truths, iou_threshold=t)
                       for t in iou_thresholds]
    mar = float(np.mean(recalls_per_iou))

    best_f1, best_precision, best_recall, best_conf = f1_confidence_sweep(
        all_detections, all_ground_truths, iou_threshold=0.5,
    )

    return {
        "AP@0.5": ap50,
        "AP@0.75": ap75,
        "AP@0.5:0.95": ap5095,
        "mAR": mar,
        "F1": best_f1,
        "precision": best_precision,
        "recall": best_recall,
        "best_conf_threshold": best_conf,
        "num_detections": len(all_detections),
        "num_gt": sum(gt.shape[0] for gt in all_ground_truths),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating on {args.split} split...")
    loader = get_dataloader(args.split, cfg, shuffle=False)

    model = StenosisDetNet(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    metrics = run_evaluation(model, loader, device, cfg)
    print(f"\nResults on {args.split}:")
    print(f"  AP@0.5:      {metrics['AP@0.5']:.4f}")
    print(f"  AP@0.75:     {metrics['AP@0.75']:.4f}")
    print(f"  AP@0.5:0.95: {metrics['AP@0.5:0.95']:.4f}")
    print(f"  mAR:         {metrics['mAR']:.4f}")
    print(f"  F1:          {metrics['F1']:.4f} (P={metrics['precision']:.4f}, R={metrics['recall']:.4f})")
    print(f"  Best conf:   {metrics['best_conf_threshold']:.3f}")
    print(f"  Detections:  {metrics['num_detections']}, GT: {metrics['num_gt']}")


if __name__ == "__main__":
    main()
