"""Unified cadica_50plus_new evaluation — temporal + video runs.

Computes AP30, AP50, AP75, AP5095, F1 in both MACRO (mean/video) and
MICRO (pooled across all centre frames) for each listed run.
Covers both TemporalRFDETR and VideoRFDETR checkpoints.

Usage:
    conda activate new_seg_final
    PYTHONPATH=rf-detr/src python _eval_cadica_unified.py
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/home/dsa/stenosis")
DATA_DIR = ROOT / "data" / "cadica_50plus_new"
IMG_DIR  = DATA_DIR / "images"
LBL_DIR  = DATA_DIR / "labels"
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_5095 = np.arange(0.5, 1.0, 0.05)

RUNS = [
    # (label, arch, run_dir)
    ("temporal_nodistill",      "temporal",
     ROOT / "rfdetr_temporal" / "runs" / "temporal_v1"),
    ("temporal_crrcd_cons_new", "temporal",
     ROOT / "rfdetr_temporal" / "runs" / "temporal_small_t5_k0_distill_aligned_crrcd_cons_new"),
    ("video_nodistill_v6_etf",  "video",
     ROOT / "rfdetr_video" / "runs" / "stfs_nodistill_v6_etf"),
    ("video_crrcd_v6_etf",      "video",
     ROOT / "rfdetr_video" / "runs" / "stfs_crrcd_v6_etf"),
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def _load_cfg_temporal(run_dir: Path):
    from rfdetr_temporal.config import Config
    with open(run_dir / "config.json") as f:
        raw = json.load(f)
    cfg = Config()
    for k, v in raw.items():
        if hasattr(cfg, k):
            cur = getattr(cfg, k)
            if isinstance(cur, Path) and v is not None:
                v = Path(v)
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def _load_cfg_video(run_dir: Path):
    from rfdetr_video.config import Config
    with open(run_dir / "config.json") as f:
        raw = json.load(f)
    cfg = Config()
    for k, v in raw.items():
        if hasattr(cfg, k):
            cur = getattr(cfg, k)
            if isinstance(cur, Path) and v is not None:
                v = Path(v)
            try:
                setattr(cfg, k, v)
            except Exception:
                pass
    return cfg


def _to_tensor(img: np.ndarray, size: int,
               mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    arr = np.stack([img, img, img], axis=0)
    t = torch.from_numpy(arr)
    return (t - mean) / std


def _yolo_xyxy(lbl_path: Path, w: int, h: int) -> np.ndarray:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lab = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    if lab.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy = lab[:, 1] * w, lab[:, 2] * h
    bw, bh = lab[:, 3] * w, lab[:, 4] * h
    return np.column_stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]).astype(np.float32)


def _compute_all_metrics(all_dets, all_gts, rows):
    """Returns (macro_dict, micro_dict)."""
    from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep
    macro = {m: float(np.mean([r[m] for r in rows]))
             for m in ("AP30", "AP50", "AP75", "AP5095", "F1")}
    mic_ap30   = evaluate_map(all_dets, all_gts, 0.3)
    mic_ap50   = evaluate_map(all_dets, all_gts, 0.5)
    mic_ap75   = evaluate_map(all_dets, all_gts, 0.75)
    mic_ap5095 = float(np.mean([evaluate_map(all_dets, all_gts, t) for t in IOU_5095]))
    mic_f1, mic_p, mic_r, mic_thr = f1_confidence_sweep(all_dets, all_gts)
    micro = {"AP30": mic_ap30, "AP50": mic_ap50, "AP75": mic_ap75,
             "AP5095": mic_ap5095, "F1": mic_f1, "P": mic_p, "R": mic_r, "thr": mic_thr}
    return macro, micro


# ─── temporal eval ────────────────────────────────────────────────────────────

def eval_temporal(label: str, run_dir: Path):
    from rfdetr_temporal.config import Config
    from rfdetr_temporal.dataset import build_sequence_index
    from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep
    from rfdetr_temporal.model import TemporalRFDETR, _build_criterion

    print(f"\n{'='*70}\n[{label}]  {run_dir}\n{'='*70}")
    cfg = _load_cfg_temporal(run_dir)
    print(f"  T={cfg.T}  img_size={cfg.img_size}")

    # Point rfdetr_checkpoint at best.pth so shape-filtered init runs (not
    # internet download).  The full weights are loaded below anyway.
    cfg.rfdetr_checkpoint = str(run_dir / "best.pth")
    model = TemporalRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    msg = model.load_state_dict(sd, strict=False)
    print(f"  loaded  missing={len(msg.missing_keys)}")
    model.eval()

    _criterion, postprocess = _build_criterion(cfg)
    sequences = build_sequence_index(IMG_DIR)
    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(cfg.pixel_std,  dtype=torch.float32).view(3, 1, 1)
    centre = cfg.T // 2

    per_video_dets: dict = defaultdict(list)
    per_video_gts: dict  = defaultdict(list)
    t0 = time.time()

    for vi, (pid, sid, paths) in enumerate(sequences):
        n = len(paths)
        if n == 0:
            continue
        windows = ([list(paths) + [paths[-1]] * (cfg.T - n)] if n < cfg.T
                   else [paths[s:s+cfg.T] for s in range(n - cfg.T + 1)])
        for win in windows:
            frames, orig_h, orig_w = [], None, None
            for p in win:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if orig_h is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_tensor(img, cfg.img_size, mean_t, std_t))
            ft = torch.stack(frames, 0).unsqueeze(0).to(DEVICE)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                out = model(ft)
            orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
            res = postprocess(out, orig_size)[0]
            boxes  = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            centre_path = win[centre]
            gt = _yolo_xyxy(LBL_DIR / (centre_path.stem + ".txt"), orig_w, orig_h)
            key = f"{pid}_v{sid}"
            per_video_dets[key].append({"boxes": boxes, "scores": scores})
            per_video_gts[key].append(gt)
        if (vi + 1) % 40 == 0 or (vi + 1) == len(sequences):
            print(f"  {vi+1}/{len(sequences)}  {time.time()-t0:.0f}s")

    rows = []
    for vid in sorted(per_video_dets):
        dets, gts = per_video_dets[vid], per_video_gts[vid]
        rows.append({
            "video": vid,
            "AP30":   evaluate_map(dets, gts, 0.3),
            "AP50":   evaluate_map(dets, gts, 0.5),
            "AP75":   evaluate_map(dets, gts, 0.75),
            "AP5095": float(np.mean([evaluate_map(dets, gts, t) for t in IOU_5095])),
            "F1":     f1_confidence_sweep(dets, gts)[0],
        })

    all_dets = [d for v in per_video_dets.values() for d in v]
    all_gts  = [g for v in per_video_gts.values()  for g in v]
    macro, micro = _compute_all_metrics(all_dets, all_gts, rows)
    return macro, micro


# ─── video eval ───────────────────────────────────────────────────────────────

def eval_video(label: str, run_dir: Path):
    from rfdetr_video.model import VideoRFDETR, build_criterion
    from rfdetr_temporal.dataset import build_sequence_index
    from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep

    print(f"\n{'='*70}\n[{label}]  {run_dir}\n{'='*70}")
    cfg = _load_cfg_video(run_dir)
    print(f"  T={cfg.T}  img_size={cfg.img_size}")

    cfg.rfdetr_checkpoint = str(run_dir / "best.pth")
    model = VideoRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    msg = model.load_state_dict(sd, strict=False)
    print(f"  loaded  missing={len(msg.missing_keys)}")
    model.eval()

    _criterion, postprocess = build_criterion(cfg)
    sequences = build_sequence_index(IMG_DIR)
    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(cfg.pixel_std,  dtype=torch.float32).view(3, 1, 1)
    centre = cfg.T // 2

    per_video_dets: dict = defaultdict(list)
    per_video_gts: dict  = defaultdict(list)
    t0 = time.time()

    for vi, (pid, sid, paths) in enumerate(sequences):
        n = len(paths)
        if n == 0:
            continue
        windows = ([list(paths) + [paths[-1]] * (cfg.T - n)] if n < cfg.T
                   else [paths[s:s+cfg.T] for s in range(n - cfg.T + 1)])
        for win in windows:
            frames, orig_h, orig_w = [], None, None
            for p in win:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if orig_h is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_tensor(img, cfg.img_size, mean_t, std_t))
            ft = torch.stack(frames, 0).unsqueeze(0).to(DEVICE)   # (1, T, 3, H, W)
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
                out = model(ft, query_mode="student")
            centre_logits = out["pred_logits"][:, centre]   # (1, Q, K)
            centre_boxes  = out["pred_boxes"][:, centre]
            orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
            res = postprocess({"pred_logits": centre_logits,
                               "pred_boxes":  centre_boxes}, orig_size)[0]
            boxes  = res["boxes"].cpu().numpy()
            scores = res["scores"].cpu().numpy()
            centre_path = win[centre]
            gt = _yolo_xyxy(LBL_DIR / (centre_path.stem + ".txt"), orig_w, orig_h)
            key = f"{pid}_v{sid}"
            per_video_dets[key].append({"boxes": boxes, "scores": scores})
            per_video_gts[key].append(gt)
        if (vi + 1) % 40 == 0 or (vi + 1) == len(sequences):
            print(f"  {vi+1}/{len(sequences)}  {time.time()-t0:.0f}s")

    rows = []
    for vid in sorted(per_video_dets):
        dets, gts = per_video_dets[vid], per_video_gts[vid]
        rows.append({
            "video": vid,
            "AP30":   evaluate_map(dets, gts, 0.3),
            "AP50":   evaluate_map(dets, gts, 0.5),
            "AP75":   evaluate_map(dets, gts, 0.75),
            "AP5095": float(np.mean([evaluate_map(dets, gts, t) for t in IOU_5095])),
            "F1":     f1_confidence_sweep(dets, gts)[0],
        })

    all_dets = [d for v in per_video_dets.values() for d in v]
    all_gts  = [g for v in per_video_gts.values()  for g in v]
    macro, micro = _compute_all_metrics(all_dets, all_gts, rows)
    return macro, micro


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    results = []
    for label, arch, run_dir in RUNS:
        if not run_dir.exists():
            print(f"SKIP {label} — {run_dir} not found")
            continue
        if arch == "temporal":
            macro, micro = eval_temporal(label, run_dir)
        else:
            macro, micro = eval_video(label, run_dir)
        results.append((label, arch, macro, micro))
        print(f"\n  MACRO  AP30={macro['AP30']:.4f}  AP50={macro['AP50']:.4f}  "
              f"F1={macro['F1']:.4f}")
        print(f"  MICRO  AP30={micro['AP30']:.4f}  AP50={micro['AP50']:.4f}  "
              f"F1={micro['F1']:.4f}")

    print(f"\n\n{'='*90}")
    print("COMPARISON TABLE — cadica_50plus_new")
    print(f"{'='*90}")
    fmt = f"  {{:<35s}}  {{:>6s}} {{:>6s}}  {{:>6s}} {{:>6s}}  {{:>6s}}"
    print(fmt.format("run", "mac30", "mac50", "mic30", "mic50", "micF1"))
    print("  " + "-"*87)
    for label, arch, macro, micro in results:
        print(fmt.format(
            label,
            f"{macro['AP30']:.4f}", f"{macro['AP50']:.4f}",
            f"{micro['AP30']:.4f}", f"{micro['AP50']:.4f}",
            f"{micro['F1']:.4f}",
        ))

    out = ROOT / "_eval_cadica_unified_results.json"
    payload = [
        {"label": lbl, "arch": arch, "macro": mac, "micro": mic}
        for lbl, arch, mac, mic in results
    ]
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
