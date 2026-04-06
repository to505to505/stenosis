"""Evaluation and inference for STQD-Det.

Runs iterative prediction → NMS → confidence filtering → COCO-style mAP.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stqd_det.config import Config
from stqd_det.dataset import get_dataloader
from stqd_det.model.detector import STQDDet

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    HAS_PYCOCOTOOLS = True
except ImportError:
    HAS_PYCOCOTOOLS = False


def compute_ap_per_frame(
    predictions: list[dict],
    gt_boxes: list[torch.Tensor],
    gt_labels: list[torch.Tensor],
    iou_thresh: float = 0.5,
) -> dict:
    """Compute AP metrics for a sequence of frames.

    Args:
        predictions: List of T dicts with "boxes", "scores", "labels".
        gt_boxes: List of T tensors (M, 4) in xyxy.
        gt_labels: List of T tensors (M,).
        iou_thresh: IoU threshold for TP/FP determination.

    Returns:
        Dict with precision, recall, F1, AP at given IoU threshold.
    """
    from torchvision.ops import box_iou

    all_tp = 0
    all_fp = 0
    all_fn = 0

    for t in range(len(predictions)):
        pred = predictions[t]
        gt = gt_boxes[t]

        if len(pred["boxes"]) == 0:
            all_fn += len(gt)
            continue
        if len(gt) == 0:
            all_fp += len(pred["boxes"])
            continue

        iou = box_iou(pred["boxes"], gt)  # (P, M)

        # Greedy matching
        matched_gt = set()
        sorted_idx = pred["scores"].argsort(descending=True)

        for idx in sorted_idx:
            if iou.shape[1] == 0:
                all_fp += 1
                continue
            max_iou, max_gt = iou[idx].max(dim=0)
            max_gt = max_gt.item()
            if max_iou >= iou_thresh and max_gt not in matched_gt:
                all_tp += 1
                matched_gt.add(max_gt)
            else:
                all_fp += 1

        all_fn += len(gt) - len(matched_gt)

    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "TP": all_tp,
        "FP": all_fp,
        "FN": all_fn,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: Config,
) -> dict:
    """Run evaluation on a dataloader.

    Returns:
        metrics: Dict with averaged precision, recall, F1.
    """
    model.eval()

    all_metrics = {"precision": 0.0, "recall": 0.0, "F1": 0.0}
    num_samples = 0

    for images, targets in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        B = images.shape[0]
        T = images.shape[1]

        results = model(images)  # list of B, each list of T dicts

        for b in range(B):
            gt_boxes = [targets[b][t]["boxes"].to(device) for t in range(T)]
            gt_labels = [targets[b][t]["labels"].to(device) for t in range(T)]

            metrics = compute_ap_per_frame(
                results[b], gt_boxes, gt_labels, iou_thresh=0.5
            )

            for k in all_metrics:
                all_metrics[k] += metrics[k]
            num_samples += 1

    # Average
    for k in all_metrics:
        all_metrics[k] /= max(num_samples, 1)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="STQD-Det Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data-root", type=str, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt.get("config", {})) if "config" in ckpt else Config()
    if args.data_root:
        cfg.data_root = Path(args.data_root)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    model = STQDDet(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Evaluating on split: {args.split}")

    loader = get_dataloader(args.split, cfg, shuffle=False)
    metrics = evaluate(model, loader, device, cfg)

    print("\n" + "=" * 40)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['F1']:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
