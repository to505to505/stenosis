"""Evaluate temporal_small_t5_k0_distill_aligned_crrcd on two datasets.

Datasets:
    - data/cadica_50plus_new
    - data/dataset2_split/test

Writes results.txt and results.json into the run directory.
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
RUN_NAME = "temporal_small_t5_k0_distill_aligned_crrcd"
RUN_DIR = ROOT / "rfdetr_temporal" / "runs" / RUN_NAME

DATASETS = [
    ("cadica_50plus_new",
     ROOT / "data" / "cadica_50plus_new" / "images",
     ROOT / "data" / "cadica_50plus_new" / "labels"),
    ("dataset2_split_test",
     ROOT / "data" / "dataset2_split" / "test" / "images",
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
        "AP30": micro_ap30, "AP50": micro_ap50, "AP75": micro_ap75,
        "AP5095": micro_ap5095, "F1": micro_f1, "P": micro_p,
        "R": micro_r, "thr": micro_thr,
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
    print(f"\n{'#' * 80}\n# {RUN_NAME}\n# {RUN_DIR}\n{'#' * 80}")
    cfg = _load_cfg(RUN_DIR)
    print(f"  cfg: T={cfg.T}  img_size={cfg.img_size}  num_classes={cfg.num_classes}")

    model = _load_model(RUN_DIR, cfg)
    _criterion, postprocess = _build_criterion(cfg)

    reports = []
    for ds_name, img_dir, lbl_dir in DATASETS:
        rep = eval_run_on_dataset(model, cfg, postprocess, ds_name, img_dir, lbl_dir)
        reports.append(rep)

    text = format_report(RUN_NAME, RUN_DIR, cfg, reports)
    out_txt = RUN_DIR / "results.txt"
    out_txt.write_text(text)
    print(f"\n  wrote → {out_txt}")

    out_json = RUN_DIR / "results.json"
    out_json.write_text(json.dumps({
        "run": RUN_NAME,
        "cfg": {"T": cfg.T, "img_size": cfg.img_size},
        "datasets": reports,
    }, indent=2))
    print(f"  wrote → {out_json}")

    print("\n" + "=" * 80)
    for rep in reports:
        print(f"\n{rep['dataset']}  (macro)")
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1"):
            print(f"  {m:<8s} {rep['macro_per_video'][m]:.4f}")
        print(f"{rep['dataset']}  (micro)")
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1", "P", "R"):
            print(f"  {m:<8s} {rep['micro_pooled'][m]:.4f}")


if __name__ == "__main__":
    main()
