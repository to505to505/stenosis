"""Run RF-DETR Large (arcade2x_704_reg) inference on dataset2_split/test."""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

from rfdetr import RFDETRLarge

ROOT = Path("/home/dsa/stenosis")
MODEL = {
    "name": "rfdetr_large_arcade2x_704_reg",
    "ckpt": ROOT / "rfdetr_runs" / "rfdetr_large_arcade2x_704_reg" / "checkpoint_best_total.pth",
    "out_dir": ROOT / "rfdetr_runs" / "rfdetr_large_arcade2x_704_reg",
    "resolution": 704,
}
DATASET = {
    "name": "dataset2_split_test",
    "img_dir": ROOT / "data" / "dataset2_split" / "test" / "images",
    "lbl_dir": ROOT / "data" / "dataset2_split" / "test" / "labels",
    "img_ext": "*.jpg",
}
CONF_EVAL = 0.01
CONF_VIS = 0.15


def read_yolo_boxes(lbl_path, w, h):
    boxes = []
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return boxes
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def main():
    img_dir = DATASET["img_dir"]
    lbl_dir = DATASET["lbl_dir"]
    all_images = sorted(img_dir.glob(DATASET["img_ext"]))
    print(f"Dataset: {DATASET['name']}  ({len(all_images)} images)")

    model = RFDETRLarge(
        pretrain_weights=str(MODEL["ckpt"]),
        resolution=MODEL["resolution"],
        device="cuda:0",
    )

    targets_list, preds_list, raw_preds = [], [], {}
    t0 = time.time()
    for i, p in enumerate(all_images):
        img_pil = Image.open(p).convert("RGB")
        w, h = img_pil.size
        gt = read_yolo_boxes(lbl_dir / (p.stem + ".txt"), w, h)
        targets_list.append({
            "boxes": torch.tensor(gt, dtype=torch.float32) if gt else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(len(gt), dtype=torch.long),
        })
        det = model.predict(img_pil, threshold=CONF_EVAL)
        if det is not None and len(det) > 0:
            preds_list.append({
                "boxes": torch.tensor(det.xyxy, dtype=torch.float32),
                "scores": torch.tensor(det.confidence, dtype=torch.float32),
                "labels": torch.zeros(len(det), dtype=torch.long),
            })
            raw_preds[p.name] = {"boxes": det.xyxy.tolist(), "scores": det.confidence.tolist()}
        else:
            preds_list.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long),
            })
            raw_preds[p.name] = {"boxes": [], "scores": []}
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(all_images)}")
    print(f"Inference done in {time.time()-t0:.1f}s")

    results = {}
    for iou_thresh in [0.25, 0.50]:
        metric = MeanAveragePrecision(iou_thresholds=[iou_thresh])
        metric.update(preds_list, targets_list)
        r = metric.compute()
        results[f"mAP@{iou_thresh}"] = round(r["map"].item(), 4)
        results[f"mAR@{iou_thresh}"] = round(r["mar_100"].item(), 4)
        print(f"  mAP@{iou_thresh:.2f} = {results[f'mAP@{iou_thresh}']:.4f}   mAR@{iou_thresh:.2f} = {results[f'mAR@{iou_thresh}']:.4f}")

    for iou_thresh in [0.25, 0.3, 0.5]:
        total_gt = matched_gt = total_fp = 0
        for target, pred in zip(targets_list, preds_list):
            gt_boxes = target["boxes"]
            mask = pred["scores"] >= CONF_VIS
            pred_boxes = pred["boxes"][mask]
            if len(gt_boxes) > 0:
                total_gt += len(gt_boxes)
                if len(pred_boxes) > 0:
                    iou_matrix = box_iou(gt_boxes, pred_boxes)
                    max_iou_per_gt, _ = iou_matrix.max(dim=1)
                    matched_gt += (max_iou_per_gt >= iou_thresh).sum().item()
                    max_iou_per_pred, _ = iou_matrix.max(dim=0)
                    total_fp += (max_iou_per_pred < iou_thresh).sum().item()
            else:
                total_fp += len(pred_boxes)
        recall = matched_gt / total_gt if total_gt else 0.0
        precision = matched_gt / (matched_gt + total_fp) if (matched_gt + total_fp) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[f"recall@IoU{iou_thresh}_conf{CONF_VIS}"] = round(recall, 4)
        results[f"precision@IoU{iou_thresh}_conf{CONF_VIS}"] = round(precision, 4)
        results[f"f1@IoU{iou_thresh}_conf{CONF_VIS}"] = round(f1, 4)
        results[f"matched_gt@IoU{iou_thresh}_conf{CONF_VIS}"] = matched_gt
        results["total_gt"] = total_gt
        print(f"  Recall @IoU≥{iou_thresh}, conf≥{CONF_VIS}: {recall:.4f} ({matched_gt}/{total_gt})  P={precision:.4f}  F1={f1:.4f}")

    results["name"] = MODEL["name"]
    results["num_images"] = len(all_images)
    out_path = MODEL["out_dir"] / f"eval_{DATASET['name']}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")
    preds_path = MODEL["out_dir"] / f"preds_{DATASET['name']}.json"
    with open(preds_path, "w") as f:
        json.dump(raw_preds, f)
    print(f"Saved: {preds_path}")


if __name__ == "__main__":
    main()
