"""Run RF-DETR inference on cadica_50plus_new with two models:
  1. arcade_dataset2_trainval
  2. dataset2_augs

Saves per-model JSON results + raw predictions for notebook visualization.
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

MODELS = [
    {
        "name": "arcade_dataset2_trainval",
        "ckpt": ROOT / "rfdetr_runs" / "arcade_dataset2_trainval" / "checkpoint_best_total.pth",
        "out_dir": ROOT / "rfdetr_runs" / "arcade_dataset2_trainval",
    },
    {
        "name": "dataset2_augs",
        "ckpt": ROOT / "rfdetr_runs" / "dataset2_augs" / "checkpoint_best_total.pth",
        "out_dir": ROOT / "rfdetr_runs" / "dataset2_augs",
    },
]

DATASET = {
    "name": "cadica_50plus_new",
    "img_dir": ROOT / "data" / "cadica_50plus_new" / "images",
    "lbl_dir": ROOT / "data" / "cadica_50plus_new" / "labels",
    "img_ext": "*.png",
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
        boxes.append([
            (cx - bw / 2) * w, (cy - bh / 2) * h,
            (cx + bw / 2) * w, (cy + bh / 2) * h,
        ])
    return boxes


def run_inference(model, img_paths, lbl_dir):
    """Run inference, return targets_list, preds_list, raw_preds."""
    targets_list = []
    preds_list = []
    raw_preds = {}

    t0 = time.time()
    for i, p in enumerate(img_paths):
        img_pil = Image.open(p).convert("RGB")
        w, h = img_pil.size

        gt = read_yolo_boxes(lbl_dir / (p.stem + ".txt"), w, h)
        target = {
            "boxes": torch.tensor(gt, dtype=torch.float32) if gt else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros(len(gt), dtype=torch.long),
        }
        targets_list.append(target)

        det = model.predict(img_pil, threshold=CONF_EVAL)
        if det is not None and len(det) > 0:
            pred = {
                "boxes": torch.tensor(det.xyxy, dtype=torch.float32),
                "scores": torch.tensor(det.confidence, dtype=torch.float32),
                "labels": torch.zeros(len(det), dtype=torch.long),
            }
            raw_preds[p.name] = {"boxes": det.xyxy.tolist(), "scores": det.confidence.tolist()}
        else:
            pred = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long),
            }
            raw_preds[p.name] = {"boxes": [], "scores": []}

        preds_list.append(pred)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(img_paths)}")

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s ({len(img_paths)/elapsed:.1f} img/s)")
    return targets_list, preds_list, raw_preds


def compute_metrics(targets_list, preds_list):
    results = {}

    for iou_thresh in [0.25, 0.50]:
        metric = MeanAveragePrecision(iou_thresholds=[iou_thresh])
        metric.update(preds_list, targets_list)
        r = metric.compute()
        map_val = r["map"].item()
        mar_val = r["mar_100"].item()
        results[f"mAP@{iou_thresh}"] = round(map_val, 4)
        results[f"mAR@{iou_thresh}"] = round(mar_val, 4)
        print(f"  mAP@{iou_thresh:.2f} = {map_val:.4f}   mAR@{iou_thresh:.2f} = {mar_val:.4f}")

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
                    max_iou_per_pred, _ = iou_matrix.max(dim=0)
                    total_fp += (max_iou_per_pred < iou_thresh).sum().item()
            else:
                total_fp += len(pred_boxes)

        recall = matched_gt / total_gt if total_gt > 0 else 0.0
        precision = matched_gt / (matched_gt + total_fp) if (matched_gt + total_fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[f"recall@IoU{iou_thresh}_conf{CONF_VIS}"] = round(recall, 4)
        results[f"precision@IoU{iou_thresh}_conf{CONF_VIS}"] = round(precision, 4)
        results[f"f1@IoU{iou_thresh}_conf{CONF_VIS}"] = round(f1, 4)
        results[f"matched_gt@IoU{iou_thresh}_conf{CONF_VIS}"] = matched_gt
        results["total_gt"] = total_gt
        print(f"  Recall @IoU≥{iou_thresh}, conf≥{CONF_VIS}: {recall:.4f} ({matched_gt}/{total_gt})  "
              f"Precision: {precision:.4f}  F1: {f1:.4f}")

    return results


def main():
    img_dir = DATASET["img_dir"]
    lbl_dir = DATASET["lbl_dir"]
    img_ext = DATASET["img_ext"]
    all_images = sorted(img_dir.glob(img_ext))
    print(f"Dataset: {DATASET['name']}  ({len(all_images)} images)\n")

    all_results = {}

    for mcfg in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {mcfg['name']}")
        print(f"Checkpoint: {mcfg['ckpt']}")
        print(f"{'='*60}")

        model = RFDETRSmall(pretrain_weights=str(mcfg["ckpt"]), device="cuda:0")
        targets_list, preds_list, raw_preds = run_inference(model, all_images, lbl_dir)
        results = compute_metrics(targets_list, preds_list)
        results["name"] = mcfg["name"]
        results["num_images"] = len(all_images)
        all_results[mcfg["name"]] = results

        # Save per-model results
        out_path = mcfg["out_dir"] / f"eval_{DATASET['name']}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Metrics saved to {out_path}")

        preds_path = mcfg["out_dir"] / f"preds_{DATASET['name']}.json"
        with open(preds_path, "w") as f:
            json.dump(raw_preds, f)
        print(f"  Predictions saved to {preds_path}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY on cadica_50plus_new")
    print(f"{'='*70}")
    for name, r in all_results.items():
        print(f"\n{name}:")
        print(f"  mAP@0.25={r['mAP@0.25']:.4f}  mAP@0.50={r['mAP@0.5']:.4f}")
        for iou in [0.25, 0.3, 0.5]:
            key_r = f"recall@IoU{iou}_conf{CONF_VIS}"
            key_p = f"precision@IoU{iou}_conf{CONF_VIS}"
            key_f = f"f1@IoU{iou}_conf{CONF_VIS}"
            print(f"  IoU≥{iou} conf≥{CONF_VIS}: R={r[key_r]:.4f} P={r[key_p]:.4f} F1={r[key_f]:.4f}")


if __name__ == "__main__":
    main()
