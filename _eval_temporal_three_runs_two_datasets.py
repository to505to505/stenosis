"""Evaluate three Temporal RF-DETR runs on two datasets and save per-run results.txt.

Runs:
    - temporal_v1
    - temporal_v1_repeat
    - temporal_small_t5_k0_distill_aligned

Datasets:
    - data/cadica_50plus_new
    - data/dataset2_split/test

Metrics per video: AP@0.3, AP@0.5, AP@0.75, AP@0.5:0.95, best-F1 (P, R, thr).
Aggregates: macro mean across videos, micro pooled across all centre frames.

Each run's results.txt is written into rfdetr_temporal/runs/<name>/results.txt.
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
from rfdetr_temporal.dataset import build_sequence_index
from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep
from rfdetr_temporal.model import TemporalRFDETR, _build_criterion

ROOT = Path("/home/dsa/stenosis")

RUNS = [
    "temporal_v1",
    "temporal_v1_repeat",
    "temporal_small_t5_k0_distill_aligned",
]

DATASETS = [
    ("cadica_50plus_new", ROOT / "data" / "cadica_50plus_new" / "images",
                          ROOT / "data" / "cadica_50plus_new" / "labels"),
    ("dataset2_split_test", ROOT / "data" / "dataset2_split" / "test" / "images",
                            ROOT / "data" / "dataset2_split" / "test" / "labels"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_5095 = np.arange(0.5, 1.0, 0.05)


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
    arr = np.stack([img, img, img], axis=0)
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


def _predict_centre(model, frames, postprocess, orig_w, orig_h):
    images = frames.unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        out = model(images)
    orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
    res = postprocess(out, orig_size)[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


def eval_run_on_dataset(model, cfg, postprocess, ds_name, img_dir, lbl_dir):
    print(f"\n  ── dataset: {ds_name}  ({img_dir})")
    sequences = build_sequence_index(img_dir)
    print(f"     {len(sequences)} videos")

    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

    per_video_dets = defaultdict(list)
    per_video_gts = defaultdict(list)
    centre = cfg.T // 2

    total = 0
    t0 = time.time()
    for vi, (pid, sid, paths) in enumerate(sequences):
        n = len(paths)
        if n == 0:
            continue

        if n < cfg.T:
            windows = [list(paths) + [paths[-1]] * (cfg.T - n)]
        else:
            windows = [paths[s:s + cfg.T] for s in range(n - cfg.T + 1)]

        for win in windows:
            frames = []
            orig_h = orig_w = None
            for p in win:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(p)
                if orig_h is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_tensor(img, cfg.img_size, mean_t, std_t))
            frames_t = torch.stack(frames, dim=0)
            boxes, scores = _predict_centre(model, frames_t, postprocess, orig_w, orig_h)
            centre_path = win[centre]
            gt = _yolo_xyxy_in_pixels(lbl_dir / (centre_path.stem + ".txt"), orig_w, orig_h)
            key = f"{pid}_v{sid}"
            per_video_dets[key].append({"boxes": boxes, "scores": scores})
            per_video_gts[key].append(gt)
            total += 1

        if (vi + 1) % 50 == 0 or (vi + 1) == len(sequences):
            print(f"     videos {vi + 1:3d}/{len(sequences)}  frames={total}  "
                  f"elapsed={time.time() - t0:.1f}s")

    rows = []
    for vid in sorted(per_video_dets):
        dets = per_video_dets[vid]
        gts = per_video_gts[vid]
        ap30 = evaluate_map(dets, gts, 0.3)
        ap50 = evaluate_map(dets, gts, 0.5)
        ap75 = evaluate_map(dets, gts, 0.75)
        ap5095 = float(np.mean([evaluate_map(dets, gts, t) for t in IOU_5095]))
        f1, p_, r_, thr = f1_confidence_sweep(dets, gts)
        rows.append({
            "video": vid,
            "n_frames": len(dets),
            "n_gt": int(sum(g.shape[0] for g in gts)),
            "AP30": ap30, "AP50": ap50, "AP75": ap75, "AP5095": ap5095,
            "F1": f1, "P": p_, "R": r_, "thr": thr,
        })

    macro = {
        m: float(np.mean([r[m] for r in rows]))
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1")
    }

    all_dets = [d for v in per_video_dets.values() for d in v]
    all_gts = [g for v in per_video_gts.values() for g in v]
    micro_ap30 = evaluate_map(all_dets, all_gts, 0.3)
    micro_ap50 = evaluate_map(all_dets, all_gts, 0.5)
    micro_ap75 = evaluate_map(all_dets, all_gts, 0.75)
    micro_ap5095 = float(np.mean([evaluate_map(all_dets, all_gts, t) for t in IOU_5095]))
    micro_f1, micro_p, micro_r, micro_thr = f1_confidence_sweep(all_dets, all_gts)
    micro = {
        "AP30": micro_ap30, "AP50": micro_ap50, "AP75": micro_ap75, "AP5095": micro_ap5095,
        "F1": micro_f1, "P": micro_p, "R": micro_r, "thr": micro_thr,
    }

    return {
        "dataset": ds_name,
        "n_videos": len(rows),
        "n_centre_frames": total,
        "rows": rows,
        "macro_per_video": macro,
        "micro_pooled": micro,
    }


def format_report(run_name: str, run_dir: Path, cfg: Config, reports: list[dict]) -> str:
    lines = []
    lines.append(f"Run: {run_name}")
    lines.append(f"Path: {run_dir}")
    lines.append(f"Config: T={cfg.T}  img_size={cfg.img_size}  num_classes={cfg.num_classes}")
    lines.append("")
    for rep in reports:
        lines.append("=" * 80)
        lines.append(f"Dataset: {rep['dataset']}")
        lines.append(f"  videos={rep['n_videos']}  centre_frames={rep['n_centre_frames']}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Per-video metrics:")
        lines.append(f"  {'video':<14s} {'frames':>6s} {'gt':>4s}  "
                     f"{'AP30':>6s} {'AP50':>6s} {'AP75':>6s} {'AP5095':>6s}  "
                     f"{'F1':>6s} {'P':>6s} {'R':>6s} {'thr':>6s}")
        for r in rep["rows"]:
            lines.append(
                f"  {r['video']:<14s} {r['n_frames']:>6d} {r['n_gt']:>4d}  "
                f"{r['AP30']:>6.3f} {r['AP50']:>6.3f} {r['AP75']:>6.3f} {r['AP5095']:>6.3f}  "
                f"{r['F1']:>6.3f} {r['P']:>6.3f} {r['R']:>6.3f} {r['thr']:>6.3f}"
            )
        lines.append("")
        lines.append("MACRO mean across videos:")
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1"):
            lines.append(f"  {m:<8s} {rep['macro_per_video'][m]:.4f}")
        lines.append("")
        lines.append("MICRO pooled across all centre frames:")
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1", "P", "R"):
            lines.append(f"  {m:<8s} {rep['micro_pooled'][m]:.4f}")
        lines.append(f"  thr      {rep['micro_pooled']['thr']:.4f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    summary = {}
    for run_name in RUNS:
        run_dir = ROOT / "rfdetr_temporal" / "runs" / run_name
        print(f"\n{'#' * 80}\n# {run_name}\n# {run_dir}\n{'#' * 80}")
        cfg = _load_cfg(run_dir)
        print(f"  cfg: T={cfg.T}  img_size={cfg.img_size}")
        model = _load_model(run_dir, cfg)
        _criterion, postprocess = _build_criterion(cfg)

        reports = []
        for ds_name, img_dir, lbl_dir in DATASETS:
            rep = eval_run_on_dataset(model, cfg, postprocess, ds_name, img_dir, lbl_dir)
            reports.append(rep)

        text = format_report(run_name, run_dir, cfg, reports)
        out_txt = run_dir / "results.txt"
        out_txt.write_text(text)
        print(f"\n  wrote → {out_txt}")

        out_json = run_dir / "results.json"
        out_json.write_text(json.dumps({
            "run": run_name,
            "cfg": {"T": cfg.T, "img_size": cfg.img_size},
            "datasets": reports,
        }, indent=2))
        print(f"  wrote → {out_json}")

        summary[run_name] = {r["dataset"]: {
            "macro": r["macro_per_video"],
            "micro": r["micro_pooled"],
            "n_videos": r["n_videos"],
            "n_centre_frames": r["n_centre_frames"],
        } for r in reports}

        del model
        torch.cuda.empty_cache()

    # Final compact comparison.
    print("\n" + "=" * 100)
    print("SUMMARY (MACRO mean across videos)")
    print("=" * 100)
    for ds_name, *_ in DATASETS:
        print(f"\nDataset: {ds_name}")
        print(f"  {'run':<40s} {'AP30':>7s} {'AP50':>7s} {'AP75':>7s} {'AP5095':>7s} {'F1':>7s}")
        for run_name in RUNS:
            mac = summary[run_name][ds_name]["macro"]
            print(f"  {run_name:<40s} "
                  f"{mac['AP30']:>7.4f} {mac['AP50']:>7.4f} {mac['AP75']:>7.4f} "
                  f"{mac['AP5095']:>7.4f} {mac['F1']:>7.4f}")
    print("\n" + "=" * 100)
    print("SUMMARY (MICRO pooled)")
    print("=" * 100)
    for ds_name, *_ in DATASETS:
        print(f"\nDataset: {ds_name}")
        print(f"  {'run':<40s} {'AP30':>7s} {'AP50':>7s} {'AP75':>7s} {'AP5095':>7s} {'F1':>7s}")
        for run_name in RUNS:
            mic = summary[run_name][ds_name]["micro"]
            print(f"  {run_name:<40s} "
                  f"{mic['AP30']:>7.4f} {mic['AP50']:>7.4f} {mic['AP75']:>7.4f} "
                  f"{mic['AP5095']:>7.4f} {mic['F1']:>7.4f}")


if __name__ == "__main__":
    main()
