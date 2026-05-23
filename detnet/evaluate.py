"""Validation/test evaluation for Stenosis-DetNet.

Centre-frame AP@0.3 / AP@0.5 / F1, micro-pooled over the split's sliding
windows whose context is fully populated (no boundary padding).

Mirrors :mod:`psstt.evaluate` so val numbers are directly comparable.
Optionally applies SCA (:mod:`detnet.sca`) on the per-window per-frame
outputs and reports a second set of metrics from the SCA-filtered
centre-frame boxes.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep

from .config import Config
from .sca import FrameDetections, SCAConfig, apply_sca


def _gt_xyxy_from_targets(target: Dict[str, torch.Tensor]) -> np.ndarray:
    boxes = target["boxes"]
    if boxes.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return boxes.detach().cpu().numpy().astype(np.float32)


def _sca_config_from_cfg(cfg: Config) -> SCAConfig:
    return SCAConfig(
        t_iou=cfg.sca_t_iou,
        t_frame=cfg.sca_t_frame,
        t_distance=cfg.sca_t_distance,
        t_sim=cfg.sca_t_sim,
        interpolate_missing=cfg.sca_interpolate_missing,
    )


def _denorm_frame(image_chw: torch.Tensor, cfg: Config) -> np.ndarray:
    """Convert a normalised (3, H, W) tensor back to (H, W) uint8 for SSIM."""
    mean = torch.tensor(cfg.pixel_mean).view(3, 1, 1)
    std = torch.tensor(cfg.pixel_std).view(3, 1, 1)
    img = image_chw.detach().cpu() * std + mean
    img = img.clamp(0, 1).numpy()
    # Frames are grayscale broadcast to 3 channels; take channel 0.
    return (img[0] * 255.0).astype(np.uint8)


@torch.no_grad()
def evaluate(
    model,
    loader,
    cfg: Config,
    device,
    apply_sca_pp: bool = False,
) -> Dict[str, float]:
    """Run validation and return centre-frame metrics.

    If ``apply_sca_pp`` is True, SCA is applied on each window's per-frame
    outputs and the **centre frame** of the SCA-filtered result is used.
    Otherwise, the model's raw centre-frame predictions are reported.
    """
    model.eval()
    T = cfg.T
    centre = T // 2
    sca_cfg = _sca_config_from_cfg(cfg)

    dets_centre: List[Dict[str, np.ndarray]] = []
    gts_centre: List[np.ndarray] = []

    for images, targets_list, fnames in loader:
        images_dev = images.to(device, non_blocking=True)
        B = images_dev.shape[0]
        with torch.amp.autocast("cuda", enabled=cfg.amp):
            out = model(images_dev, targets=None)

        if apply_sca_pp and "all_frames" in out:
            all_frames = out["all_frames"]  # list[B] of list[T]
            for b in range(B):
                if len(set(fnames[b])) != T:
                    continue
                per_frame: List[FrameDetections] = []
                for t in range(T):
                    pred = all_frames[b][t]
                    img_np = _denorm_frame(images[b, t], cfg)
                    per_frame.append(FrameDetections(
                        boxes=pred["boxes"].detach().cpu().numpy().astype(np.float32),
                        scores=pred["scores"].detach().cpu().numpy().astype(np.float32),
                        image=img_np,
                    ))
                fused = apply_sca(per_frame, sca_cfg)
                gt_xyxy = _gt_xyxy_from_targets(targets_list[b][centre])
                dets_centre.append({
                    "boxes": fused[centre].boxes,
                    "scores": fused[centre].scores,
                })
                gts_centre.append(gt_xyxy)
        else:
            centre_preds = out["centre"]
            for b in range(B):
                pred = centre_preds[b]
                boxes = pred["boxes"].detach().cpu().numpy().astype(np.float32)
                scores = pred["scores"].detach().cpu().numpy().astype(np.float32)
                gt_xyxy = _gt_xyxy_from_targets(targets_list[b][centre])
                if len(set(fnames[b])) == T:
                    dets_centre.append({"boxes": boxes, "scores": scores})
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
