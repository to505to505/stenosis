"""Validation evaluation for PSSTT.

Centre-frame AP@0.3 / AP@0.5 / F1, micro-pooled over the val split's
sliding windows whose context is fully populated (no boundary padding).
Mirrors :mod:`rfdetr_video.evaluate` so val numbers are directly
comparable.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep

from .config import Config


def _gt_xyxy_from_targets(target: Dict[str, torch.Tensor]) -> np.ndarray:
    """Targets are already xyxy-pixel from our dataset wrapper."""
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return boxes.detach().cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, cfg: Config, device) -> Dict[str, float]:
    """Run validation and return centre-frame metrics."""
    model.eval()
    T = cfg.T
    centre = T // 2

    dets_centre: List[Dict[str, np.ndarray]] = []
    gts_centre: List[np.ndarray] = []

    for images, targets_list, fnames in loader:
        images = images.to(device, non_blocking=True)
        B = images.shape[0]
        with torch.amp.autocast("cuda", enabled=cfg.amp):
            out = model(images, targets=None)
        centre_preds = out["centre"]  # list[B] of {boxes, scores, labels}

        for b in range(B):
            pred = centre_preds[b]
            boxes = pred["boxes"].detach().cpu().numpy().astype(np.float32)
            scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
            gt_xyxy = _gt_xyxy_from_targets(targets_list[b][centre])
            det = {"boxes": boxes, "scores": scores}
            # Only count interior windows (all T filenames distinct) for parity
            # with rfdetr_video evaluation.
            if len(set(fnames[b])) == T:
                dets_centre.append(det)
                gts_centre.append(gt_xyxy)

    ap30 = evaluate_map(dets_centre, gts_centre, 0.3)
    ap50 = evaluate_map(dets_centre, gts_centre, 0.5)
    f1, prec, rec, conf = f1_confidence_sweep(dets_centre, gts_centre)
    return {
        "AP@0.3": ap30,
        "AP@0.5": ap50,
        "F1": f1,
        "precision": prec,
        "recall": rec,
        "best_conf": conf,
        "n_windows": float(len(dets_centre)),
    }
