"""Compare two Temporal RF-DETR runs on cadica_50plus_new (per-video metrics).

Runs evaluated:
    A = rfdetr_temporal/runs/temporal_v1                       (no distillation)
    B = rfdetr_temporal/runs/temporal_small_t5_k0_distill      (KD-DETR distill)

For each model we:
    1. Load the run's `config.json` (uses its own img_size / T).
    2. Build TemporalRFDETR and load best.pth (state_dict).
    3. Walk cadica_50plus_new sequences (one (patient, video) = one sequence),
       build sliding windows of T frames, predict on the centre frame.
    4. Aggregate per-video:  AP@0.5, AP@0.75, AP@0.5:0.95, best-F1.
    5. Report per-video table and overall (macro mean across videos and
       micro pooling of all detections).

Centre-frame predictions come from `model(images, query_mode='student')`
(default) — the distillation branch is not used at inference.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from rfdetr_temporal.config import Config
from rfdetr_temporal.dataset import build_sequence_index, build_windows, load_yolo_labels
from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep
from rfdetr_temporal.model import TemporalRFDETR, _build_criterion

ROOT = Path("/home/dsa/stenosis")
DATA_DIR = ROOT / "data" / "cadica_50plus_new"
IMG_DIR = DATA_DIR / "images"
LBL_DIR = DATA_DIR / "labels"

RUNS = [
    ("temporal_v1",                    ROOT / "rfdetr_temporal" / "runs" / "temporal_v1"),
    ("temporal_small_t5_k0_distill",   ROOT / "rfdetr_temporal" / "runs" / "temporal_small_t5_k0_distill"),
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_cfg(run_dir: Path) -> Config:
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


def _load_model(run_dir: Path, cfg: Config) -> TemporalRFDETR:
    model = TemporalRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    msg = model.load_state_dict(sd, strict=False)
    print(f"  loaded best.pth  missing={len(msg.missing_keys)}  "
          f"unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model


def _to_tensor(img: np.ndarray, size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    arr = np.stack([img, img, img], axis=0)  # (3, H, W)
    t = torch.from_numpy(arr)
    return (t - mean) / std


def _yolo_xyxy_in_pixels(lbl_path: Path, w: int, h: int) -> np.ndarray:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lab = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    if lab.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx = lab[:, 1] * w
    cy = lab[:, 2] * h
    bw = lab[:, 3] * w
    bh = lab[:, 4] * h
    return np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]).astype(np.float32)


def _predict_centre(model: TemporalRFDETR, frames: torch.Tensor, postprocess,
                    orig_w: int, orig_h: int):
    """frames: (T, 3, H, W) on CPU; returns (boxes_xyxy_pixels, scores) numpy."""
    images = frames.unsqueeze(0).to(DEVICE)        # (1, T, 3, H, W)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        out = model(images)                        # query_mode default = "student"
    orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
    res = postprocess(out, orig_size)[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


def _eval_on_dataset(name: str, run_dir: Path):
    print(f"\n{'='*70}\n[{name}]  {run_dir}\n{'='*70}")
    cfg = _load_cfg(run_dir)
    print(f"  cfg: T={cfg.T}  img_size={cfg.img_size}  num_classes={cfg.num_classes}")

    model = _load_model(run_dir, cfg)
    _criterion, postprocess = _build_criterion(cfg)

    sequences = build_sequence_index(IMG_DIR)
    print(f"  found {len(sequences)} videos in cadica_50plus_new")

    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

    # Per-video accumulators of centre-frame predictions and GT.
    per_video_dets: dict = defaultdict(list)
    per_video_gts:  dict = defaultdict(list)
    centre = cfg.T // 2

    total_centre_frames = 0
    t0 = time.time()
    for vi, (pid, sid, paths) in enumerate(sequences):
        n = len(paths)
        if n == 0:
            continue

        # Build all sliding windows of length T (with edge padding when n<T).
        windows = []
        if n < cfg.T:
            padded = list(paths) + [paths[-1]] * (cfg.T - n)
            windows.append(padded)
        else:
            for s in range(n - cfg.T + 1):
                windows.append(paths[s:s + cfg.T])

        for win in windows:
            # Load all T frames at original resolution → resize → tensor.
            frames = []
            orig_h = orig_w = None
            for p in win:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(p)
                if orig_h is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_tensor(img, cfg.img_size, mean_t, std_t))
            frames_t = torch.stack(frames, dim=0)   # (T, 3, H, W)

            # Predict on the centre frame.
            boxes, scores = _predict_centre(model, frames_t, postprocess, orig_w, orig_h)
            centre_path = win[centre]
            gt = _yolo_xyxy_in_pixels(LBL_DIR / (centre_path.stem + ".txt"), orig_w, orig_h)

            key = f"{pid}_v{sid}"
            per_video_dets[key].append({"boxes": boxes, "scores": scores})
            per_video_gts[key].append(gt)
            total_centre_frames += 1

        if (vi + 1) % 25 == 0 or (vi + 1) == len(sequences):
            print(f"  videos {vi + 1:3d}/{len(sequences)}  "
                  f"frames={total_centre_frames}  "
                  f"elapsed={time.time() - t0:.1f}s")

    # ── Per-video metrics ──────────────────────────────────────────
    rows = []
    iou_thrs_5095 = np.arange(0.5, 1.0, 0.05)
    for vid in sorted(per_video_dets.keys()):
        dets = per_video_dets[vid]
        gts  = per_video_gts[vid]
        n_frames = len(dets)
        n_gt = int(sum(g.shape[0] for g in gts))

        ap50 = evaluate_map(dets, gts, 0.5)
        ap75 = evaluate_map(dets, gts, 0.75)
        ap5095 = float(np.mean([evaluate_map(dets, gts, t) for t in iou_thrs_5095]))
        f1, prec, rec, conf = f1_confidence_sweep(dets, gts)

        rows.append({
            "video": vid, "n_frames": n_frames, "n_gt": n_gt,
            "AP50": ap50, "AP75": ap75, "AP5095": ap5095,
            "F1": f1, "P": prec, "R": rec, "thr": conf,
        })

    # ── Aggregates ─────────────────────────────────────────────────
    macro = {
        "AP50":   float(np.mean([r["AP50"]   for r in rows])),
        "AP75":   float(np.mean([r["AP75"]   for r in rows])),
        "AP5095": float(np.mean([r["AP5095"] for r in rows])),
        "F1":     float(np.mean([r["F1"]     for r in rows])),
    }

    # Micro: pool all centre-frame predictions / GT across all videos.
    all_dets = [d for v in per_video_dets.values() for d in v]
    all_gts  = [g for v in per_video_gts.values()  for g in v]
    micro_ap50   = evaluate_map(all_dets, all_gts, 0.5)
    micro_ap75   = evaluate_map(all_dets, all_gts, 0.75)
    micro_ap5095 = float(np.mean([evaluate_map(all_dets, all_gts, t) for t in iou_thrs_5095]))
    micro_f1, micro_p, micro_r, micro_thr = f1_confidence_sweep(all_dets, all_gts)
    micro = {
        "AP50": micro_ap50, "AP75": micro_ap75, "AP5095": micro_ap5095,
        "F1": micro_f1, "P": micro_p, "R": micro_r, "thr": micro_thr,
    }

    return {
        "name": name,
        "run_dir": str(run_dir),
        "cfg": {"T": cfg.T, "img_size": cfg.img_size},
        "n_videos": len(rows),
        "n_centre_frames": total_centre_frames,
        "rows": rows,
        "macro_per_video": macro,
        "micro_pooled": micro,
    }


def _print_per_video_table(report: dict):
    print(f"\n[{report['name']}] per-video metrics ({report['n_videos']} videos):")
    print(f"  {'video':<14s} {'frames':>6s} {'gt':>4s}  "
          f"{'AP50':>6s} {'AP75':>6s} {'AP5095':>6s}  "
          f"{'F1':>6s} {'P':>6s} {'R':>6s}")
    for r in report["rows"]:
        print(f"  {r['video']:<14s} {r['n_frames']:>6d} {r['n_gt']:>4d}  "
              f"{r['AP50']:>6.3f} {r['AP75']:>6.3f} {r['AP5095']:>6.3f}  "
              f"{r['F1']:>6.3f} {r['P']:>6.3f} {r['R']:>6.3f}")


def _print_summary(report_a: dict, report_b: dict):
    print("\n" + "=" * 80)
    print("OVERALL COMPARISON  (cadica_50plus_new)")
    print("=" * 80)
    for label, key in [("MACRO mean across videos", "macro_per_video"),
                       ("MICRO pooled across all centre frames", "micro_pooled")]:
        print(f"\n  {label}:")
        a = report_a[key]; b = report_b[key]
        head = f"{'metric':<10s} {'A: ' + report_a['name']:>40s} {'B: ' + report_b['name']:>40s}  Δ(B-A)"
        print(f"    {head}")
        for m in ["AP50", "AP75", "AP5095", "F1"]:
            va = a.get(m, float('nan')); vb = b.get(m, float('nan'))
            print(f"    {m:<10s} {va:>40.4f} {vb:>40.4f}  {vb - va:+.4f}")

    # Per-video win/lose table on AP50.
    a_rows = {r["video"]: r for r in report_a["rows"]}
    b_rows = {r["video"]: r for r in report_b["rows"]}
    common = sorted(set(a_rows) & set(b_rows))
    wins = sum(1 for v in common if b_rows[v]["AP50"] > a_rows[v]["AP50"] + 1e-6)
    losses = sum(1 for v in common if b_rows[v]["AP50"] < a_rows[v]["AP50"] - 1e-6)
    ties = len(common) - wins - losses
    print(f"\n  Per-video AP50 head-to-head:  B wins={wins}  ties={ties}  "
          f"A wins={losses}  (n={len(common)})")


def main():
    reports = []
    for name, run_dir in RUNS:
        rep = _eval_on_dataset(name, run_dir)
        reports.append(rep)
        _print_per_video_table(rep)
        torch.cuda.empty_cache()

    _print_summary(reports[0], reports[1])

    out_path = ROOT / "_compare_temporal_cadica50plus_results.json"
    with open(out_path, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\nSaved JSON → {out_path}")


if __name__ == "__main__":
    main()
