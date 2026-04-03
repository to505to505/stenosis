"""Evaluation script — compute mAP on test/valid set."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast

from detect_model import build_faster_rcnn
from detect_dataset import get_dataloader


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """All-point interpolation AP (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_map(all_dets, all_gts, iou_threshold: float = 0.5) -> float:
    det_boxes, det_scores, det_img_ids = [], [], []
    for img_id, det in enumerate(all_dets):
        n = det["scores"].shape[0]
        if n == 0:
            continue
        det_boxes.append(det["boxes"])
        det_scores.append(det["scores"])
        det_img_ids.extend([img_id] * n)

    if not det_boxes:
        return 0.0

    det_boxes = np.concatenate(det_boxes)
    det_scores = np.concatenate(det_scores)
    det_img_ids = np.array(det_img_ids)

    order = np.argsort(-det_scores)
    det_boxes = det_boxes[order]
    det_img_ids = det_img_ids[order]

    total_gt = sum(gt.shape[0] for gt in all_gts)
    if total_gt == 0:
        return 0.0

    gt_matched = {i: np.zeros(gt.shape[0], dtype=bool)
                  for i, gt in enumerate(all_gts) if gt.shape[0] > 0}

    tp = np.zeros(len(det_scores))
    fp = np.zeros(len(det_scores))

    for d in range(len(det_scores)):
        img_id = det_img_ids[d]
        db = det_boxes[d]
        gt = all_gts[img_id]
        if gt.shape[0] == 0:
            fp[d] = 1
            continue
        ixmin = np.maximum(gt[:, 0], db[0])
        iymin = np.maximum(gt[:, 1], db[1])
        ixmax = np.minimum(gt[:, 2], db[2])
        iymax = np.minimum(gt[:, 3], db[3])
        inter = np.maximum(ixmax - ixmin, 0) * np.maximum(iymax - iymin, 0)
        union = ((db[2]-db[0])*(db[3]-db[1]) +
                 (gt[:,2]-gt[:,0])*(gt[:,3]-gt[:,1]) - inter)
        iou = inter / np.maximum(union, 1e-6)
        best = np.argmax(iou)
        if iou[best] >= iou_threshold and img_id in gt_matched and not gt_matched[img_id][best]:
            tp[d] = 1
            gt_matched[img_id][best] = True
        else:
            fp[d] = 1

    cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
    recalls = cum_tp / total_gt
    precisions = cum_tp / (cum_tp + cum_fp)
    return compute_ap(recalls, precisions)


@torch.no_grad()
def run_evaluation(model, loader, device, amp_enabled=True):
    model.eval()
    all_dets, all_gts = [], []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        with autocast("cuda", enabled=amp_enabled):
            outputs = model(images)

        for out, tgt in zip(outputs, targets):
            # Filter to class 1 (stenosis)
            mask = out["labels"] == 1
            all_dets.append({
                "boxes": out["boxes"][mask].cpu().numpy(),
                "scores": out["scores"][mask].cpu().numpy(),
            })
            gt_boxes = tgt["boxes"].numpy()
            all_gts.append(gt_boxes)

    ap50 = evaluate_map(all_dets, all_gts, 0.5)
    aps = [evaluate_map(all_dets, all_gts, t) for t in np.arange(0.5, 1.0, 0.05)]
    ap5095 = np.mean(aps)

    return {
        "AP@0.5": ap50,
        "AP@0.5:0.95": ap5095,
        "num_images": len(all_gts),
        "num_gt": sum(g.shape[0] for g in all_gts),
        "num_dets": sum(d["scores"].shape[0] for d in all_dets),
    }


def main():
    parser = argparse.ArgumentParser("Evaluate ViTDet Faster R-CNN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="/home/dsa/stenosis/data/dataset2_split")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    print(f"Evaluating on {args.split}...")
    loader = get_dataloader(args.data_root, args.split, args.img_size,
                            args.batch_size, args.num_workers, shuffle=False)

    model = build_faster_rcnn(num_classes=2, img_size=args.img_size)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    print(f"Loaded: {args.checkpoint}")

    metrics = run_evaluation(model, loader, device, args.amp)
    print(f"\n{args.split} results:")
    print(f"  AP@0.5:      {metrics['AP@0.5']:.4f}")
    print(f"  AP@0.5:0.95: {metrics['AP@0.5:0.95']:.4f}")
    print(f"  Images: {metrics['num_images']}  GT: {metrics['num_gt']}  Dets: {metrics['num_dets']}")


if __name__ == "__main__":
    main()
