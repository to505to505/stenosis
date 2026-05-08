"""Evaluation utilities for Video RF-DETR.

Reuses the metric helpers from :mod:`rfdetr_temporal.evaluate` and runs
the video forward pass at inference. Reports both per-frame mAP
(averaged over all T frames in each window) and centre-frame mAP
(parity with the previous evaluator) so we can compare against existing
single-frame and temporal baselines.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.cuda.amp import autocast

from rfdetr_temporal.evaluate import (
    evaluate_map,
    f1_confidence_sweep,
)

from .config import Config


def _gt_xyxy_from_cxcywh(boxes_cxcywh: np.ndarray, img_size: int) -> np.ndarray:
    if boxes_cxcywh.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    return np.column_stack([
        (cx - w / 2) * img_size,
        (cy - h / 2) * img_size,
        (cx + w / 2) * img_size,
        (cy + h / 2) * img_size,
    ])


@torch.no_grad()
def evaluate(model, loader, criterion, postprocess, cfg: Config, device):
    """Run evaluation.

    Headline metrics (no prefix) use micro-pooled AP over centre frames
    whose windows contain **no boundary padding** (i.e. the sequence had
    at least T frames so every frame in the window is distinct).  Windows
    that were padded because the sequence is shorter than T are skipped —
    this gives a fair comparison with single-frame detectors that see each
    frame exactly once in its full context.

    ``all/`` prefixed metrics cover all T positions, no filtering.
    """
    model.eval()
    criterion.eval()

    centre = cfg.T // 2
    detections_centre, gts_centre = [], []
    detections_all, gts_all = [], []
    total_loss = 0.0
    n_batches = 0

    for images, targets_list, fnames in loader:
        images = images.to(device)
        B = images.shape[0]
        T = cfg.T

        # Build flat target list of length B*T for SetCriterion.
        targets_flat = []
        for sample in targets_list:
            for t in range(T):
                t_dict = sample[t]
                targets_flat.append({
                    "boxes": t_dict["boxes"].to(device),
                    "labels": t_dict["labels"].to(device),
                    "orig_size": torch.tensor(
                        [cfg.img_size, cfg.img_size], device=device,
                    ),
                })

        with autocast(enabled=cfg.amp):
            out = model(images, query_mode="student")
            # Flatten predictions to (B*T, Q, *) for the criterion.
            pred_flat = {
                "pred_logits": out["pred_logits"].reshape(B * T, *out["pred_logits"].shape[2:]),
                "pred_boxes": out["pred_boxes"].reshape(B * T, *out["pred_boxes"].shape[2:]),
            }
            if "enc_outputs" in out:
                pred_flat["enc_outputs"] = out["enc_outputs"]
            if "aux_outputs" in out:
                pred_flat["aux_outputs"] = out["aux_outputs"]
            loss_dict = criterion(pred_flat, targets_flat)
            loss = sum(
                loss_dict[k] * criterion.weight_dict[k]
                for k in loss_dict if k in criterion.weight_dict
            )
        total_loss += loss.item()
        n_batches += 1

        orig_sizes = torch.stack([t["orig_size"] for t in targets_flat])
        results = postprocess(pred_flat, orig_sizes)

        for b in range(B):
            for t in range(T):
                idx = b * T + t
                scores = results[idx]["scores"].cpu().numpy()
                boxes = results[idx]["boxes"].cpu().numpy()
                gt_xyxy = _gt_xyxy_from_cxcywh(
                    targets_list[b][t]["boxes"].cpu().numpy(), cfg.img_size,
                )
                det = {"boxes": boxes, "scores": scores}
                detections_all.append(det)
                gts_all.append(gt_xyxy)
                if t == centre:
                    # Only count interior windows: all T filenames must be
                    # distinct (padding replicates the last frame filename).
                    if len(set(fnames[b])) == T:
                        detections_centre.append(det)
                        gts_centre.append(gt_xyxy)

    iou_thrs = np.arange(0.5, 1.0, 0.05)

    def _metric_block(dets, gts, prefix=""):
        ap30 = evaluate_map(dets, gts, 0.3)
        ap50 = evaluate_map(dets, gts, 0.5)
        f1, prec, rec, conf = f1_confidence_sweep(dets, gts)
        return {
            f"{prefix}AP@0.3": ap30,
            f"{prefix}AP@0.5": ap50,
            f"{prefix}F1": f1,
            f"{prefix}precision": prec,
            f"{prefix}recall": rec,
            f"{prefix}best_conf": conf,
        }

    metrics = _metric_block(detections_centre, gts_centre)
    metrics.update(_metric_block(detections_all, gts_all, prefix="all/"))
    metrics["val_loss"] = total_loss / max(n_batches, 1)
    return metrics
