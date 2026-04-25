"""Visualize Temporal RF-DETR predictions on CADICA val images.

Draws GT boxes (green) and predicted boxes (red, with score) on centre frames.
Saves a grid of sample images to the run directory.

Usage:
    python visualize_cadica_predictions.py
    python visualize_cadica_predictions.py --n 30 --conf 0.15
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from rfdetr_temporal.config import Config
from rfdetr_temporal.dataset import TemporalStenosisDataset, collate_fn
from rfdetr_temporal.model import TemporalRFDETR, _build_criterion

CADICA_CATS = ["p0_20", "p20_50", "p50_70", "p70_90", "p90_98", "p99", "p100"]
CAT_SET = set(CADICA_CATS)
CADICA_BASE_LABELS = Path("data/cadica_base/labels")

# Colors for original categories (BGR)
CAT_COLORS = {
    "p0_20":  (180, 220, 180),  # light green — mild
    "p20_50": (0, 200, 200),    # yellow
    "p50_70": (0, 165, 255),    # orange
    "p70_90": (0, 100, 255),    # dark orange
    "p90_98": (0, 0, 255),      # red
    "p99":    (0, 0, 180),      # dark red
    "p100":   (128, 0, 128),    # purple — total occlusion
}
PRED_COLOR = (255, 100, 50)  # blue-ish for predictions


def parse_original_label(label_path: Path):
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


def draw_boxes(img, gt_xyxy, gt_cats, pred_boxes, pred_scores, conf_thresh):
    """Draw GT (colored by category) and predictions (blue) on image."""
    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    # Draw GT boxes
    for i in range(gt_xyxy.shape[0]):
        x1, y1, x2, y2 = gt_xyxy[i].astype(int)
        cat = gt_cats[i] if i < len(gt_cats) else "unknown"
        color = CAT_COLORS.get(cat, (0, 255, 0))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"GT:{cat}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Draw predictions
    for i in range(pred_boxes.shape[0]):
        if pred_scores[i] < conf_thresh:
            continue
        x1, y1, x2, y2 = pred_boxes[i].astype(int)
        score = pred_scores[i]
        cv2.rectangle(vis, (x1, y1), (x2, y2), PRED_COLOR, 2)
        cv2.putText(vis, f"{score:.2f}", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, PRED_COLOR, 1)

    return vis


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="rfdetr_temporal/runs/cadica_temporal_v1/best.pth")
    p.add_argument("--dataset", default="data/cadica_split_90_10")
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--conf", type=float, default=0.15, help="Confidence threshold for drawing preds")
    p.add_argument("--n", type=int, default=32, help="Number of sample images to save")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.checkpoint).parent

    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg = Config(**{k: v for k, v in cfg_dict.items() if hasattr(Config, k)})
    else:
        cfg = Config()
    cfg.data_root = Path(args.dataset)
    cfg.T = args.T

    model = TemporalRFDETR(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    _, postprocess = _build_criterion(cfg)

    ds = TemporalStenosisDataset("valid", cfg)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=0, collate_fn=collate_fn, pin_memory=False)

    # Collect all results first
    all_vis = []
    centre = cfg.T // 2
    img_dir = Path(args.dataset) / "valid" / "images"

    for idx, (images, targets_list, fnames_batch) in enumerate(loader):
        images = images.to(device)
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

        scores = results[0]["scores"].cpu().numpy()
        boxes = results[0]["boxes"].cpu().numpy()

        gt_cxcywh = centre_targets[0]["boxes"].cpu().numpy()
        if gt_cxcywh.shape[0] > 0:
            cx, cy, w, h = gt_cxcywh[:, 0], gt_cxcywh[:, 1], gt_cxcywh[:, 2], gt_cxcywh[:, 3]
            gt_xyxy = np.column_stack([
                (cx - w/2) * cfg.img_size, (cy - h/2) * cfg.img_size,
                (cx + w/2) * cfg.img_size, (cy + h/2) * cfg.img_size,
            ])
        else:
            gt_xyxy = np.zeros((0, 4), dtype=np.float32)

        centre_fname = fnames_batch[0][centre]
        gt_cats = match_gt_to_categories(gt_xyxy, centre_fname)

        # Load original image for visualization
        img_path = img_dir / centre_fname
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        has_gt = gt_xyxy.shape[0] > 0
        has_pred = np.any(scores >= args.conf)
        all_vis.append((img, gt_xyxy, gt_cats, boxes, scores, centre_fname, has_gt, has_pred))

    print(f"Total val windows: {len(all_vis)}")

    # Select diverse samples: prioritize images with GT, mix categories
    random.seed(args.seed)
    with_gt = [v for v in all_vis if v[6]]
    without_gt = [v for v in all_vis if not v[6]]

    # Pick mostly images with GT
    n_gt = min(len(with_gt), int(args.n * 0.8))
    n_empty = min(len(without_gt), args.n - n_gt)
    selected = random.sample(with_gt, n_gt) + random.sample(without_gt, n_empty)
    random.shuffle(selected)
    selected = selected[:args.n]

    # Save individual images
    out_dir = run_dir / "vis_predictions"
    out_dir.mkdir(exist_ok=True)

    for i, (img, gt_xyxy, gt_cats, boxes, scores, fname, _, _) in enumerate(selected):
        vis = draw_boxes(img, gt_xyxy, gt_cats, boxes, scores, args.conf)
        cv2.imwrite(str(out_dir / f"{i:03d}_{Path(fname).stem}.jpg"), vis)

    # Make a grid
    grid_size = 512
    ncols = 4
    nrows = (len(selected) + ncols - 1) // ncols
    grid = np.zeros((nrows * grid_size, ncols * grid_size, 3), dtype=np.uint8)

    for i, (img, gt_xyxy, gt_cats, boxes, scores, fname, _, _) in enumerate(selected):
        vis = draw_boxes(img, gt_xyxy, gt_cats, boxes, scores, args.conf)
        vis = cv2.resize(vis, (grid_size, grid_size))
        r, c = divmod(i, ncols)
        grid[r*grid_size:(r+1)*grid_size, c*grid_size:(c+1)*grid_size] = vis

    grid_path = run_dir / "predictions_grid.jpg"
    cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Saved {len(selected)} images to {out_dir}/")
    print(f"Saved grid to {grid_path}")

    # Legend
    print(f"\nLegend: GT boxes colored by category, predictions in blue (conf >= {args.conf})")
    for cat, color in CAT_COLORS.items():
        print(f"  {cat}: RGB{color[::-1]}")


if __name__ == "__main__":
    main()
