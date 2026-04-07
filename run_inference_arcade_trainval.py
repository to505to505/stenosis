"""Run RF-DETR inference from arcade_dataset2_trainval model on 3 test sets.

Evaluates:
  1. dataset2_split/test
  2. cadica_split_50plus/train (treated as test)
  3. stenosis_arcade_singlelabel/test

Saves per-dataset JSON results to rfdetr_runs/arcade_dataset2_trainval/eval_*.json
"""

import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

from rfdetr import RFDETRSmall

ROOT = Path("/home/dsa/stenosis")
CKPT = ROOT / "rfdetr_runs" / "arcade_dataset2_trainval" / "checkpoint_best_total.pth"
OUT_DIR = ROOT / "rfdetr_runs" / "arcade_dataset2_trainval"

CONF_EVAL = 0.01  # low threshold for mAP computation
CONF_VIS = 0.15   # threshold for recall / visualization

EVAL_SETS = [
    {
        "name": "dataset2_split_test",
        "img_dir": ROOT / "data" / "dataset2_split" / "test" / "images",
        "lbl_dir": ROOT / "data" / "dataset2_split" / "test" / "labels",
        "img_ext": "*.jpg",
    },
    {
        "name": "cadica_50plus_train",
        "img_dir": ROOT / "data" / "cadica_split_50plus" / "train" / "images",
        "lbl_dir": ROOT / "data" / "cadica_split_50plus" / "train" / "labels",
        "img_ext": "*.png",
    },
    {
        "name": "stenosis_arcade_singlelabel_test",
        "img_dir": ROOT / "data" / "stenosis_arcade_singlelabel" / "test" / "images",
        "lbl_dir": ROOT / "data" / "stenosis_arcade_singlelabel" / "test" / "labels",
        "img_ext": "*.png",
    },
]


def read_yolo_boxes(lbl_path, w, h):
    """Read YOLO-format label → list of [x1, y1, x2, y2]."""
    boxes = []
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return boxes
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append([
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ])
    return boxes


def evaluate_dataset(model, cfg):
    """Run inference + compute metrics for one dataset."""
    img_dir = cfg["img_dir"]
    lbl_dir = cfg["lbl_dir"]
    name = cfg["name"]
    img_ext = cfg["img_ext"]

    all_images = sorted(img_dir.glob(img_ext))
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}  ({len(all_images)} images)")
    print(f"{'='*60}")

    targets_list = []
    preds_list = []
    # Store raw predictions for notebook visualization
    raw_preds = {}

    t0 = time.time()
    for i, p in enumerate(all_images):
        img_pil = Image.open(p).convert("RGB")
        w, h = img_pil.size

        # GT
        gt = read_yolo_boxes(lbl_dir / (p.stem + ".txt"), w, h)
        target = {
            "boxes": torch.tensor(gt, dtype=torch.float32) if gt else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(len(gt), dtype=torch.long),
        }
        targets_list.append(target)

        # Predict
        det = model.predict(img_pil, threshold=CONF_EVAL)
        if det is not None and len(det) > 0:
            pred = {
                "boxes": torch.tensor(det.xyxy, dtype=torch.float32),
                "scores": torch.tensor(det.confidence, dtype=torch.float32),
                "labels": torch.zeros(len(det), dtype=torch.long),
            }
            raw_preds[p.name] = {
                "boxes": det.xyxy.tolist(),
                "scores": det.confidence.tolist(),
            }
        else:
            pred = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long),
            }
            raw_preds[p.name] = {"boxes": [], "scores": []}

        preds_list.append(pred)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(all_images)}")

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s ({len(all_images)/elapsed:.1f} img/s)")

    # ── mAP at various IoU thresholds ──
    results = {"name": name, "num_images": len(all_images)}

    for iou_thresh in [0.25, 0.50]:
        metric = MeanAveragePrecision(iou_thresholds=[iou_thresh])
        metric.update(preds_list, targets_list)
        r = metric.compute()
        map_val = r["map"].item()
        mar_val = r["mar_100"].item()
        results[f"mAP@{iou_thresh}"] = round(map_val, 4)
        results[f"mAR@{iou_thresh}"] = round(mar_val, 4)
        print(f"  mAP@{iou_thresh:.2f} = {map_val:.4f}   mAR@{iou_thresh:.2f} = {mar_val:.4f}")

    # ── Recall at conf >= 0.15 ──
    for iou_thresh in [0.25, 0.3, 0.5]:
        total_gt = 0
        matched_gt = 0
        total_fp = 0
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
                    # FP: pred boxes that don't match any GT
                    max_iou_per_pred, _ = iou_matrix.max(dim=0)
                    total_fp += (max_iou_per_pred < iou_thresh).sum().item()
                else:
                    pass  # no preds = 0 matches
            else:
                # No GT, all preds are FP
                total_fp += len(pred_boxes)

        recall = matched_gt / total_gt if total_gt > 0 else 0.0
        precision = matched_gt / (matched_gt + total_fp) if (matched_gt + total_fp) > 0 else 0.0
        results[f"recall@IoU{iou_thresh}_conf{CONF_VIS}"] = round(recall, 4)
        results[f"precision@IoU{iou_thresh}_conf{CONF_VIS}"] = round(precision, 4)
        results[f"matched_gt@IoU{iou_thresh}_conf{CONF_VIS}"] = matched_gt
        results[f"total_gt"] = total_gt
        print(f"  Recall @IoU≥{iou_thresh}, conf≥{CONF_VIS}: {recall:.4f} ({matched_gt}/{total_gt})  Precision: {precision:.4f}")

    # ── Save results ──
    out_path = OUT_DIR / f"eval_{name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Metrics saved to {out_path}")

    # Save raw predictions
    preds_path = OUT_DIR / f"preds_{name}.json"
    with open(preds_path, "w") as f:
        json.dump(raw_preds, f)
    print(f"  Predictions saved to {preds_path}")

    return results


def main():
    print(f"Loading model from {CKPT}")
    model = RFDETRSmall(pretrain_weights=str(CKPT), device="cuda:0")
    print("Model loaded")

    all_results = {}
    for cfg in EVAL_SETS:
        results = evaluate_dataset(model, cfg)
        all_results[cfg["name"]] = results

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(f"\n{name} ({r['num_images']} images):")
        print(f"  mAP@0.25={r['mAP@0.25']:.4f}  mAP@0.50={r['mAP@0.5']:.4f}")
        print(f"  Recall@IoU0.25,conf0.15={r['recall@IoU0.25_conf0.15']:.4f}")
        print(f"  Recall@IoU0.3,conf0.15={r['recall@IoU0.3_conf0.15']:.4f}")

    # Save combined summary
    summary_path = OUT_DIR / "eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined summary saved to {summary_path}")


if __name__ == "__main__":
    main()
