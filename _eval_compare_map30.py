"""Compare mAP@0.25/0.30/0.50 for reg, wd1e3, lsj on both test sets."""
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision

from rfdetr import RFDETRLarge

ROOT = Path("/home/dsa/stenosis")

MODELS = [
    {
        "name": "rfdetr_large_arcade2x_704_reg",
        "ckpt": ROOT / "rfdetr_runs" / "rfdetr_large_arcade2x_704_reg" / "checkpoint_best_ema.pth",
    },
    {
        "name": "rfdetr_large_arcade2x_704_wd1e3",
        "ckpt": ROOT / "rfdetr_runs" / "rfdetr_large_arcade2x_704_wd1e3" / "checkpoint_best_ema.pth",
    },
    {
        "name": "rfdetr_large_arcade2x_704_lsj",
        "ckpt": ROOT / "rfdetr_runs" / "rfdetr_large_arcade2x_704_lsj" / "checkpoint_best_ema.pth",
    },
]

DATASETS = [
    {
        "name": "dataset2_split_test",
        "img_dir": ROOT / "data" / "dataset2_split" / "test" / "images",
        "lbl_dir": ROOT / "data" / "dataset2_split" / "test" / "labels",
        "img_ext": "*.jpg",
    },
    {
        "name": "cadica_50plus_new",
        "img_dir": ROOT / "data" / "cadica_50plus_new" / "images",
        "lbl_dir": ROOT / "data" / "cadica_50plus_new" / "labels",
        "img_ext": "*.png",
    },
]

RESOLUTION = 704
CONF_EVAL = 0.01
IOU_THRESHOLDS = [0.25, 0.30, 0.50]


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


def collect_preds(model, dataset):
    img_dir = dataset["img_dir"]
    lbl_dir = dataset["lbl_dir"]
    all_images = sorted(img_dir.glob(dataset["img_ext"]))
    targets_list, preds_list = [], []
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
        else:
            preds_list.append({
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros(0, dtype=torch.float32),
                "labels": torch.zeros(0, dtype=torch.long),
            })
    print(f"    Inference: {time.time()-t0:.1f}s ({len(all_images)} imgs)")
    return targets_list, preds_list


def compute_maps(preds_list, targets_list):
    results = {}
    for iou in IOU_THRESHOLDS:
        metric = MeanAveragePrecision(iou_thresholds=[iou])
        metric.update(preds_list, targets_list)
        r = metric.compute()
        results[f"mAP@{iou}"] = round(r["map"].item(), 4)
    return results


def main():
    all_results = {}

    for mconf in MODELS:
        model_name = mconf["name"]
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        model = RFDETRLarge(
            pretrain_weights=str(mconf["ckpt"]),
            resolution=RESOLUTION,
            device="cuda:0",
        )
        all_results[model_name] = {}
        for ds in DATASETS:
            print(f"  Dataset: {ds['name']}")
            targets_list, preds_list = collect_preds(model, ds)
            maps = compute_maps(preds_list, targets_list)
            all_results[model_name][ds["name"]] = maps
            for k, v in maps.items():
                print(f"    {k} = {v:.4f}")
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for ds in DATASETS:
        dsname = ds["name"]
        print(f"\n{dsname}:")
        header = f"{'Model':<40} " + " ".join(f"mAP@{iou:<5}" for iou in IOU_THRESHOLDS)
        print(header)
        print("-" * len(header))
        for mconf in MODELS:
            name = mconf["name"]
            vals = " ".join(f"{all_results[name][dsname].get(f'mAP@{iou}', 0):.4f}  " for iou in IOU_THRESHOLDS)
            print(f"{name:<40} {vals}")

    out = ROOT / "_compare_map30_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
