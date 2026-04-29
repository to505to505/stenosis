"""Evaluate the CPC-augmented run AND compare temporal consistency vs the
no-CPC baseline.

For each of the two runs:
    - temporal_small_t5_k0_distill_aligned_crrcd      (no CPC, baseline)
    - temporal_small_t5_k0_distill_aligned_crrcd_cpc  (with CPC)

we run inference on:
    - data/cadica_50plus_new
    - data/dataset2_split/test

and produce:

  1. results.txt + results.json for the CPC run (same format as
     `_eval_crrcd_run.py` so it is directly comparable).
  2. _compare_temporal_cpc_consistency.txt + .json containing
     frame-to-frame *consistency* metrics for both runs side-by-side.

Consistency metrics (computed at each model's own best-F1 threshold τ
selected on the dataset, so each model is judged at its operating point):

    flicker_rate          fraction of consecutive centre-frame pairs whose
                          "detection present?" flag flips
    mean_count_diff       mean |#dets(t+1) − #dets(t)|
    matched_iou           greedy-IoU mean IoU between matched boxes in
                          consecutive frames (only frames with ≥1 det
                          on both sides count)
    matched_disp_norm     mean Euclidean centre displacement of those
                          matched pairs, normalised by image diagonal

Lower flicker_rate / mean_count_diff / matched_disp_norm and higher
matched_iou ⇒ more temporally consistent predictions.
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
RUNS_DIR = ROOT / "rfdetr_temporal" / "runs"

RUN_BASELINE = "temporal_small_t5_k0_distill_aligned_crrcd"
RUN_CPC = "temporal_small_t5_k0_distill_aligned_crrcd_cpc"

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
CONSISTENCY_IOU_MATCH = 0.3        # greedy match threshold for flicker tracking


# ───────────────────────── helpers (lifted from _eval_crrcd_run.py) ──
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
    print(f"  loaded {run_dir.name}/best.pth  "
          f"missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}")
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
    return np.column_stack(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]
    ).astype(np.float32)


def _predict_centre(model, frames, postprocess, orig_w, orig_h):
    images = frames.unsqueeze(0).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
        out = model(images)
    orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
    res = postprocess(out, orig_size)[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


# ───────────────────────── inference per (model, dataset) ────────────
def predict_run_on_dataset(model, cfg, postprocess, ds_name, img_dir, lbl_dir):
    print(f"\n  ── inference: {ds_name}  ({img_dir})")
    sequences = build_sequence_index(img_dir)
    print(f"     {len(sequences)} videos")

    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

    per_video_dets = defaultdict(list)
    per_video_gts = defaultdict(list)
    per_video_meta = {}
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

        key = f"{pid}_v{sid}"
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
            per_video_dets[key].append({"boxes": boxes, "scores": scores})
            per_video_gts[key].append(gt)
            per_video_meta[key] = (orig_h, orig_w)
            total += 1

        if (vi + 1) % 50 == 0 or (vi + 1) == len(sequences):
            print(f"     videos {vi + 1:3d}/{len(sequences)}  frames={total}  "
                  f"elapsed={time.time() - t0:.1f}s")

    return {
        "per_video_dets": dict(per_video_dets),
        "per_video_gts": dict(per_video_gts),
        "per_video_meta": per_video_meta,
        "n_centre_frames": total,
    }


# ───────────────────────── results.txt computation ───────────────────
def compute_full_report(preds):
    per_video_dets = preds["per_video_dets"]
    per_video_gts = preds["per_video_gts"]
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
        "n_videos": len(rows),
        "n_centre_frames": preds["n_centre_frames"],
        "rows": rows,
        "macro_per_video": macro,
        "micro_pooled": micro,
    }


def format_full_report(run_name: str, run_dir: Path, cfg: Config, reports: list[dict]) -> str:
    lines = [
        f"Run: {run_name}",
        f"Path: {run_dir}",
        f"Config: T={cfg.T}  img_size={cfg.img_size}  num_classes={cfg.num_classes}",
        "",
    ]
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


# ───────────────────────── consistency metrics ───────────────────────
def _filter_by_score(det: dict, thr: float) -> np.ndarray:
    """Return (N, 4) xyxy boxes with score ≥ thr, sorted by descending score."""
    s = det["scores"]
    if s.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    m = s >= thr
    boxes = det["boxes"][m]
    sc = s[m]
    if boxes.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    order = np.argsort(-sc)
    return boxes[order].astype(np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(ix2 - ix1, 0.0)
    ih = np.maximum(iy2 - iy1, 0.0)
    inter = iw * ih
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    return inter / np.maximum(union, 1e-6)


def _greedy_match(iou: np.ndarray, iou_thr: float) -> list[tuple[int, int, float]]:
    """Greedy IoU matching, return list of (i, j, iou) pairs."""
    pairs: list[tuple[int, int, float]] = []
    if iou.size == 0:
        return pairs
    flat = [(iou[i, j], i, j)
            for i in range(iou.shape[0]) for j in range(iou.shape[1])
            if iou[i, j] >= iou_thr]
    flat.sort(reverse=True)
    used_i, used_j = set(), set()
    for v, i, j in flat:
        if i in used_i or j in used_j:
            continue
        used_i.add(i); used_j.add(j)
        pairs.append((i, j, float(v)))
    return pairs


def compute_consistency(preds: dict, thr: float) -> dict:
    per_video_dets = preds["per_video_dets"]
    per_video_meta = preds["per_video_meta"]

    flick_rates = []
    count_diffs = []
    matched_ious = []
    matched_disps = []
    n_videos_used = 0
    n_pairs_total = 0
    n_pairs_with_match = 0

    for vid, dets in per_video_dets.items():
        if len(dets) < 2:
            continue
        oh, ow = per_video_meta[vid]
        diag = float(np.hypot(ow, oh)) or 1.0

        filt = [_filter_by_score(d, thr) for d in dets]
        present = np.array([b.shape[0] > 0 for b in filt])
        counts = np.array([b.shape[0] for b in filt])

        flips = (present[:-1] != present[1:]).astype(np.float32)
        flick_rates.append(float(flips.mean()))
        count_diffs.append(float(np.abs(counts[:-1].astype(np.int32)
                                        - counts[1:].astype(np.int32)).mean()))

        vid_ious = []
        vid_disps = []
        for a, b in zip(filt[:-1], filt[1:]):
            n_pairs_total += 1
            if a.shape[0] == 0 or b.shape[0] == 0:
                continue
            iou = _iou_matrix(a, b)
            pairs = _greedy_match(iou, CONSISTENCY_IOU_MATCH)
            if not pairs:
                continue
            n_pairs_with_match += 1
            for i, j, v in pairs:
                vid_ious.append(v)
                ca = ((a[i, 0] + a[i, 2]) * 0.5, (a[i, 1] + a[i, 3]) * 0.5)
                cb = ((b[j, 0] + b[j, 2]) * 0.5, (b[j, 1] + b[j, 3]) * 0.5)
                disp = float(np.hypot(ca[0] - cb[0], ca[1] - cb[1])) / diag
                vid_disps.append(disp)
        if vid_ious:
            matched_ious.append(float(np.mean(vid_ious)))
            matched_disps.append(float(np.mean(vid_disps)))
        n_videos_used += 1

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "threshold": float(thr),
        "n_videos_used": n_videos_used,
        "n_consecutive_pairs": int(n_pairs_total),
        "n_pairs_with_match": int(n_pairs_with_match),
        "flicker_rate":     _mean(flick_rates),
        "mean_count_diff":  _mean(count_diffs),
        "matched_iou":      _mean(matched_ious),
        "matched_disp_norm": _mean(matched_disps),
    }


# ───────────────────────── orchestration ─────────────────────────────
def run_one(run_name: str, write_results_txt: bool):
    run_dir = RUNS_DIR / run_name
    print(f"\n{'#' * 80}\n# {run_name}\n# {run_dir}\n{'#' * 80}")
    cfg = _load_cfg(run_dir)
    print(f"  cfg: T={cfg.T}  img_size={cfg.img_size}  cpc_enabled={getattr(cfg, 'cpc_enabled', False)}")

    model = _load_model(run_dir, cfg)
    _criterion, postprocess = _build_criterion(cfg)

    out = {"run": run_name, "cfg": {"T": cfg.T, "img_size": cfg.img_size}, "datasets": []}
    full_reports = []
    for ds_name, img_dir, lbl_dir in DATASETS:
        preds = predict_run_on_dataset(model, cfg, postprocess, ds_name, img_dir, lbl_dir)
        rep = compute_full_report(preds)
        rep["dataset"] = ds_name
        full_reports.append(rep)

        thr = rep["micro_pooled"]["thr"]
        cons = compute_consistency(preds, thr)
        out["datasets"].append({
            "dataset": ds_name,
            "n_videos": rep["n_videos"],
            "n_centre_frames": rep["n_centre_frames"],
            "macro_per_video": rep["macro_per_video"],
            "micro_pooled": rep["micro_pooled"],
            "consistency_at_micro_thr": cons,
        })

    if write_results_txt:
        text = format_full_report(run_name, run_dir, cfg, full_reports)
        (run_dir / "results.txt").write_text(text)
        (run_dir / "results.json").write_text(json.dumps(
            {"run": run_name, "cfg": {"T": cfg.T, "img_size": cfg.img_size},
             "datasets": full_reports}, indent=2))
        print(f"  wrote → {run_dir / 'results.txt'}")
        print(f"  wrote → {run_dir / 'results.json'}")

    # free GPU before next model
    del model
    torch.cuda.empty_cache()
    return out


def format_consistency_compare(base: dict, cpc: dict) -> str:
    L = []
    L.append("Temporal consistency comparison")
    L.append(f"  baseline:  {base['run']}")
    L.append(f"  with CPC:  {cpc['run']}")
    L.append("")
    L.append("Each metric is computed at the model's own best-F1 threshold "
             "on that dataset (so the model is judged at its own operating "
             "point). Lower is better for flicker_rate, mean_count_diff, "
             "matched_disp_norm; higher is better for matched_iou.")
    L.append("")

    by_ds_base = {d["dataset"]: d for d in base["datasets"]}
    by_ds_cpc = {d["dataset"]: d for d in cpc["datasets"]}

    for ds_name in [d["dataset"] for d in base["datasets"]]:
        b = by_ds_base[ds_name]
        c = by_ds_cpc[ds_name]
        L.append("=" * 80)
        L.append(f"Dataset: {ds_name}")
        L.append(f"  videos={b['n_videos']}  centre_frames={b['n_centre_frames']}")
        L.append("=" * 80)
        L.append("")
        L.append("Detection accuracy (micro-pooled, for context):")
        L.append(f"  {'metric':<12s} {'baseline':>10s} {'cpc':>10s} {'Δ (cpc-base)':>14s}")
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1"):
            vb = b["micro_pooled"][m]
            vc = c["micro_pooled"][m]
            L.append(f"  {m:<12s} {vb:>10.4f} {vc:>10.4f} {vc - vb:>+14.4f}")
        L.append(f"  {'thr':<12s} {b['micro_pooled']['thr']:>10.4f} "
                 f"{c['micro_pooled']['thr']:>10.4f} "
                 f"{c['micro_pooled']['thr'] - b['micro_pooled']['thr']:>+14.4f}")
        L.append("")
        L.append("Consistency metrics:")
        L.append(f"  {'metric':<22s} {'baseline':>10s} {'cpc':>10s} {'Δ (cpc-base)':>14s} {'better?':>10s}")
        cb = b["consistency_at_micro_thr"]
        cc = c["consistency_at_micro_thr"]
        order = [
            ("flicker_rate",      "lower"),
            ("mean_count_diff",   "lower"),
            ("matched_iou",       "higher"),
            ("matched_disp_norm", "lower"),
        ]
        for m, want in order:
            vb = cb[m]; vc = cc[m]
            d = vc - vb
            if np.isnan(vb) or np.isnan(vc):
                tag = "n/a"
            elif (want == "lower" and d < 0) or (want == "higher" and d > 0):
                tag = "CPC ✓"
            elif d == 0:
                tag = "tie"
            else:
                tag = "BASE"
            L.append(f"  {m:<22s} {vb:>10.4f} {vc:>10.4f} {d:>+14.4f} {tag:>10s}")
        L.append(f"  {'(used videos)':<22s} {cb['n_videos_used']:>10d} {cc['n_videos_used']:>10d}")
        L.append(f"  {'(consec. pairs)':<22s} {cb['n_consecutive_pairs']:>10d} {cc['n_consecutive_pairs']:>10d}")
        L.append(f"  {'(pairs with match)':<22s} {cb['n_pairs_with_match']:>10d} {cc['n_pairs_with_match']:>10d}")
        L.append("")

    return "\n".join(L) + "\n"


def main():
    base = run_one(RUN_BASELINE, write_results_txt=False)
    cpc  = run_one(RUN_CPC,      write_results_txt=True)

    text = format_consistency_compare(base, cpc)
    out_txt = ROOT / "_compare_temporal_cpc_consistency.txt"
    out_json = ROOT / "_compare_temporal_cpc_consistency.json"
    out_txt.write_text(text)
    out_json.write_text(json.dumps({"baseline": base, "cpc": cpc}, indent=2))
    print(f"\n  wrote → {out_txt}")
    print(f"  wrote → {out_json}")
    print("\n" + text)


if __name__ == "__main__":
    main()
