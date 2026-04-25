"""Evaluate best Temporal RF-DETR checkpoint on CADICA val split, per stenosis category.

The model predicts single-class (class 0 = stenosis). To get per-category metrics
we look up the original CADICA labels (which have class names like p0_20, p50_70, etc.)
and match each GT box to its original category via IoU matching.

Usage:
    python eval_cadica_per_category.py
    python eval_cadica_per_category.py --checkpoint rfdetr_temporal/runs/cadica_temporal_v1/best.pth
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from rfdetr_temporal.config import Config
from rfdetr_temporal.dataset import TemporalStenosisDataset, collate_fn
from rfdetr_temporal.evaluate import (
    evaluate_map,
    f1_confidence_sweep,
    compute_max_recall,
)
from rfdetr_temporal.model import TemporalRFDETR, _build_criterion

CADICA_CATS = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]
CAT_SET = set(CADICA_CATS)
CADICA_BASE_LABELS = Path("data/cadica_base/labels")


def parse_original_label(label_path: Path):
    """Parse original CADICA label file -> list of (category, x1, y1, x2, y2) absolute."""
    entries = []
    if not label_path.exists():
        return entries
    text = label_path.read_text().strip()
    if not text:
        return entries
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5 or parts[0] not in CAT_SET:
            continue
        cat = parts[0]
        x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        entries.append((cat, x, y, x + w, y + h))
    return entries


def match_gt_to_categories(gt_xyxy: np.ndarray, fname: str):
    """Match each GT box to its original CADICA category via IoU with original labels."""
    stem = Path(fname).stem
    original = parse_original_label(CADICA_BASE_LABELS / f"{stem}.txt")
    categories = []
    for i in range(gt_xyxy.shape[0]):
        box = gt_xyxy[i]
        best_cat, best_iou = "unknown", 0.0
        for cat, ox1, oy1, ox2, oy2 in original:
            ix1 = max(box[0], ox1); iy1 = max(box[1], oy1)
            ix2 = min(box[2], ox2); iy2 = min(box[3], oy2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (box[2]-box[0])*(box[3]-box[1]) + (ox2-ox1)*(oy2-oy1) - inter
            iou = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou, best_cat = iou, cat
        categories.append(best_cat)
    return categories


@torch.no_grad()
def run_evaluation(checkpoint_path: str, dataset_path: str, T: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = Path(checkpoint_path).parent
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg = Config(**{k: v for k, v in cfg_dict.items() if hasattr(Config, k)})
    else:
        cfg = Config()
    cfg.data_root = Path(dataset_path)
    cfg.T = T

    model = TemporalRFDETR(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    criterion, postprocess = _build_criterion(cfg)
    criterion = criterion.to(device).eval()

    ds = TemporalStenosisDataset("valid", cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, collate_fn=collate_fn, pin_memory=True)

    all_detections, all_gt_xyxy, all_gt_categories = [], [], []
    centre = cfg.T // 2

    for images, targets_list, fnames_batch in loader:
        images = images.to(device)
        B = images.shape[0]

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

        orig_sizes = torch.stack([t["orig_size"] for t in centre_targets])
        results = postprocess(outputs, orig_sizes)

        for b in range(B):
            scores = results[b]["scores"].cpu().numpy()
            boxes = results[b]["boxes"].cpu().numpy()
            all_detections.append({"boxes": boxes, "scores": scores})

            gt_cxcywh = centre_targets[b]["boxes"].cpu().numpy()
            if gt_cxcywh.shape[0] > 0:
                cx, cy, w, h = gt_cxcywh[:, 0], gt_cxcywh[:, 1], gt_cxcywh[:, 2], gt_cxcywh[:, 3]
                gt_xyxy = np.column_stack([
                    (cx - w/2) * cfg.img_size, (cy - h/2) * cfg.img_size,
                    (cx + w/2) * cfg.img_size, (cy + h/2) * cfg.img_size,
                ])
            else:
                gt_xyxy = np.zeros((0, 4), dtype=np.float32)
            all_gt_xyxy.append(gt_xyxy)

            centre_fname = fnames_batch[b][centre]
            all_gt_categories.append(match_gt_to_categories(gt_xyxy, centre_fname))

    # ── Overall ──
    iou_thrs = np.arange(0.5, 1.0, 0.05)
    ap50 = evaluate_map(all_detections, all_gt_xyxy, 0.5)
    ap75 = evaluate_map(all_detections, all_gt_xyxy, 0.75)
    ap5095 = float(np.mean([evaluate_map(all_detections, all_gt_xyxy, t) for t in iou_thrs]))
    mar = float(np.mean([compute_max_recall(all_detections, all_gt_xyxy, t) for t in iou_thrs]))
    f1, prec, rec, conf = f1_confidence_sweep(all_detections, all_gt_xyxy)
    total_gt = sum(g.shape[0] for g in all_gt_xyxy)

    print(f"\n{'='*60}\nOVERALL ({total_gt} GT boxes)\n{'='*60}")
    print(f"  AP@0.5: {ap50:.4f}  AP@0.75: {ap75:.4f}  AP@0.5:0.95: {ap5095:.4f}")
    print(f"  mAR: {mar:.4f}  F1: {f1:.4f} (P={prec:.4f} R={rec:.4f} conf={conf:.2f})")

    # ── Per-category ──
    print(f"\n{'='*60}\nPER-CATEGORY METRICS\n{'='*60}")
    results_table = {}

    for cat in CADICA_CATS:
        cat_dets, cat_gts = [], []
        for i in range(len(all_detections)):
            mask = [c == cat for c in all_gt_categories[i]]
            if not any(mask):
                continue
            cat_gts.append(all_gt_xyxy[i][np.array(mask)])
            cat_dets.append(all_detections[i])

        n_gt = sum(g.shape[0] for g in cat_gts)
        if n_gt == 0:
            print(f"\n  {cat}: no GT boxes")
            results_table[cat] = {"n_gt": 0}
            continue

        a50 = evaluate_map(cat_dets, cat_gts, 0.5)
        a75 = evaluate_map(cat_dets, cat_gts, 0.75)
        a5095 = float(np.mean([evaluate_map(cat_dets, cat_gts, t) for t in iou_thrs]))
        f1_c, p_c, r_c, conf_c = f1_confidence_sweep(cat_dets, cat_gts)

        results_table[cat] = {
            "n_gt": n_gt, "AP@0.5": a50, "AP@0.75": a75, "AP@0.5:0.95": a5095,
            "F1": f1_c, "precision": p_c, "recall": r_c, "best_conf": conf_c,
        }
        print(f"\n  {cat} ({n_gt} GT):")
        print(f"    AP@0.5={a50:.4f}  AP@0.75={a75:.4f}  AP50-95={a5095:.4f}")
        print(f"    F1={f1_c:.4f} (P={p_c:.4f} R={r_c:.4f} conf={conf_c:.2f})")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print(f"{'Category':<12} {'N_GT':>5} {'AP@50':>7} {'AP@75':>7} {'AP50-95':>8} {'F1':>6} {'P':>6} {'R':>6}")
    print("-" * 60)
    for cat in CADICA_CATS:
        r = results_table.get(cat, {})
        if r.get("n_gt", 0) == 0:
            print(f"{cat:<12} {0:>5}    —       —        —      —      —      —")
        else:
            print(f"{cat:<12} {r['n_gt']:>5} {r['AP@0.5']:>7.4f} {r['AP@0.75']:>7.4f} "
                  f"{r['AP@0.5:0.95']:>8.4f} {r['F1']:>6.4f} {r['precision']:>6.4f} {r['recall']:>6.4f}")
    print("=" * 60)

    out_path = run_dir / "per_category_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="rfdetr_temporal/runs/cadica_temporal_v1/best.pth")
    p.add_argument("--dataset", default="data/cadica_split_90_10")
    p.add_argument("--T", type=int, default=5)
    args = p.parse_args()
    run_evaluation(args.checkpoint, args.dataset, args.T)
