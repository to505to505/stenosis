"""Compute STFS-compatible metrics for rfdetr_large_arcade2x_704_reg.

Loads already-saved per-frame predictions (preds_*.json) and computes
AP30/AP50/AP75/AP5095/F1/Frag per video and globally, in the same format
as _eval_stfs_ablations.py.  Results saved to the run directory.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rfdetr_temporal.dataset import build_sequence_index
from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep

ROOT     = Path("/home/dsa/stenosis")
RUN_DIR  = ROOT / "rfdetr_runs" / "rfdetr_large_arcade2x_704_reg"
RUN_LABEL = "rfdetr_large_arcade2x_704_reg (single-frame)"

DATASETS = [
    ("cadica_50plus_new",
     ROOT / "data" / "cadica_50plus_new" / "images",
     ROOT / "data" / "cadica_50plus_new" / "labels",
     RUN_DIR / "preds_cadica_50plus_new.json"),
    ("dataset2_split_test",
     ROOT / "data" / "dataset2_split" / "test" / "images",
     ROOT / "data" / "dataset2_split" / "test" / "labels",
     RUN_DIR / "preds_dataset2_split_test.json"),
]

IOU_5095  = np.arange(0.5, 1.0, 0.05)
LINK_IOU  = 0.3   # same as _eval_stfs_ablations.py
MATCH_IOU = 0.5   # same as _eval_stfs_ablations.py


# ─────────────────────── helpers (mirrors _eval_stfs_ablations.py) ───────────

def _yolo_xyxy_in_pixels(lbl_path: Path, w: int, h: int) -> np.ndarray:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    rows = []
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                     (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_area = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    b_area = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = a_area[:, None] + b_area[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def build_gt_tracks(gts_per_frame):
    tracks = []
    active = []
    for f_idx, gt in enumerate(gts_per_frame):
        if gt.shape[0] == 0:
            active = []
            continue
        if not active:
            for box in gt:
                tracks.append([(f_idx, box.copy())])
                active.append(len(tracks) - 1)
            continue
        last_boxes = np.stack([tracks[ti][-1][1] for ti in active], axis=0)
        ious = _iou_matrix(last_boxes, gt)
        new_active = []
        used_gt = set()
        flat = [(ious[i, j], i, j)
                for i in range(ious.shape[0])
                for j in range(ious.shape[1])
                if ious[i, j] >= LINK_IOU]
        flat.sort(reverse=True)
        used_track_local = set()
        for iou_v, i, j in flat:
            if i in used_track_local or j in used_gt:
                continue
            ti = active[i]
            tracks[ti].append((f_idx, gt[j].copy()))
            new_active.append(ti)
            used_track_local.add(i)
            used_gt.add(j)
        for j in range(gt.shape[0]):
            if j not in used_gt:
                tracks.append([(f_idx, gt[j].copy())])
                new_active.append(len(tracks) - 1)
        active = new_active
    return tracks


def compute_video_frag(dets_per_frame, gts_per_frame, score_thr):
    tracks = build_gt_tracks(gts_per_frame)
    frag_total = 0
    track_frames_total = 0
    for trk in tracks:
        if len(trk) < 3:
            track_frames_total += len(trk)
            continue
        status = []
        for f_idx, box in trk:
            det = dets_per_frame[f_idx]
            keep = det["scores"] >= score_thr
            if not keep.any():
                status.append(0)
                continue
            ious = _iou_matrix(box[None, :], det["boxes"][keep])[0]
            status.append(1 if (ious >= MATCH_IOU).any() else 0)
        seen_one = False
        in_gap = False
        for s in status:
            if s == 1:
                if in_gap and seen_one:
                    frag_total += 1
                    in_gap = False
                seen_one = True
                in_gap = False
            else:
                if seen_one:
                    in_gap = True
        track_frames_total += len(trk)
    return frag_total, track_frames_total


# ──────────────────────────── per-dataset eval ───────────────────────────────

def eval_on_dataset(ds_name, img_dir, lbl_dir, preds_json):
    print(f"  ── dataset: {ds_name}")
    preds = json.loads(preds_json.read_text())
    sequences = build_sequence_index(img_dir)

    per_video_dets: dict = defaultdict(list)
    per_video_gts:  dict = defaultdict(list)
    total = 0
    t0 = time.time()

    from PIL import Image as PILImage

    for vi, (pid, sid, paths) in enumerate(sequences):
        # Read image size once per video (all frames assumed same size)
        with PILImage.open(paths[0]) as _img:
            orig_w, orig_h = _img.size

        for p in paths:
            entry = preds.get(p.name, {"boxes": [], "scores": []})
            boxes  = np.array(entry["boxes"],  dtype=np.float32).reshape(-1, 4)
            scores = np.array(entry["scores"], dtype=np.float32)

            gt = _yolo_xyxy_in_pixels(lbl_dir / (p.stem + ".txt"), orig_w, orig_h)
            key = f"{pid}_v{sid}"
            per_video_dets[key].append({"boxes": boxes, "scores": scores})
            per_video_gts[key].append(gt)
            total += 1

        if (vi + 1) % 10 == 0 or (vi + 1) == len(sequences):
            print(f"     videos {vi + 1:3d}/{len(sequences)}  frames={total}  "
                  f"elapsed={time.time() - t0:.1f}s")

    rows = []
    for vid in sorted(per_video_dets):
        dets = per_video_dets[vid]
        gts  = per_video_gts[vid]
        ap30   = evaluate_map(dets, gts, 0.3)
        ap50   = evaluate_map(dets, gts, 0.5)
        ap75   = evaluate_map(dets, gts, 0.75)
        ap5095 = float(np.mean([evaluate_map(dets, gts, t) for t in IOU_5095]))
        f1, p_, r_, thr = f1_confidence_sweep(dets, gts)
        frag, trk_frames = compute_video_frag(dets, gts, score_thr=thr)
        rows.append({
            "video": vid, "n_frames": len(dets),
            "n_gt": int(sum(g.shape[0] for g in gts)),
            "AP30": ap30, "AP50": ap50, "AP75": ap75, "AP5095": ap5095,
            "F1": f1, "P": p_, "R": r_, "thr": thr,
            "Frag": int(frag), "FragRate": (frag / trk_frames) if trk_frames > 0 else 0.0,
            "track_frames": int(trk_frames),
        })

    macro = {
        m: float(np.mean([r[m] for r in rows]))
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1", "FragRate")
    }
    macro["Frag"] = float(np.mean([r["Frag"] for r in rows]))

    all_dets = [d for v in per_video_dets.values() for d in v]
    all_gts  = [g for v in per_video_gts.values()  for g in v]
    micro_ap30   = evaluate_map(all_dets, all_gts, 0.3)
    micro_ap50   = evaluate_map(all_dets, all_gts, 0.5)
    micro_ap75   = evaluate_map(all_dets, all_gts, 0.75)
    micro_ap5095 = float(np.mean([evaluate_map(all_dets, all_gts, t) for t in IOU_5095]))
    micro_f1, micro_p, micro_r, micro_thr = f1_confidence_sweep(all_dets, all_gts)
    total_frag       = int(sum(r["Frag"] for r in rows))
    total_trk_frames = int(sum(r["track_frames"] for r in rows))
    micro = {
        "AP30": micro_ap30, "AP50": micro_ap50, "AP75": micro_ap75,
        "AP5095": micro_ap5095, "F1": micro_f1, "P": micro_p, "R": micro_r,
        "thr": micro_thr, "Frag": total_frag,
        "FragRate": (total_frag / total_trk_frames) if total_trk_frames > 0 else 0.0,
        "track_frames": total_trk_frames,
    }

    return {"dataset": ds_name, "n_videos": len(rows), "n_centre_frames": total,
            "rows": rows, "macro_per_video": macro, "micro_pooled": micro}


# ──────────────────────────── formatting ─────────────────────────────────────

def _block(rep):
    mac = rep["macro_per_video"]
    mic = rep["micro_pooled"]
    lines = [
        f"  Dataset : {rep['dataset']}  ({rep['n_videos']} videos, "
        f"{rep['n_centre_frames']} frames)",
        f"  Model   : {RUN_LABEL}",
        "",
        f"  MACRO (mean/video)  AP30={mac['AP30']:.4f}  AP50={mac['AP50']:.4f}  "
        f"AP75={mac['AP75']:.4f}  AP5095={mac['AP5095']:.4f}  "
        f"F1={mac['F1']:.4f}  Frag={mac['Frag']:.2f}  FragRate={mac['FragRate']:.4f}",
        f"  MICRO (pooled)      AP30={mic['AP30']:.4f}  AP50={mic['AP50']:.4f}  "
        f"AP75={mic['AP75']:.4f}  AP5095={mic['AP5095']:.4f}  "
        f"F1={mic['F1']:.4f}  P={mic['P']:.4f}  R={mic['R']:.4f}  thr={mic['thr']:.4f}  "
        f"Frag={mic['Frag']}  FragRate={mic['FragRate']:.4f}",
    ]
    return "\n".join(lines)


# ──────────────────────────────── main ───────────────────────────────────────

def main():
    print(f"\n{'#' * 80}")
    print(f"# STFS-compatible metrics – {RUN_LABEL}")
    print(f"# {RUN_DIR}")
    print(f"{'#' * 80}\n")

    all_reps = []
    parts = []
    for ds_name, img_dir, lbl_dir, preds_json in DATASETS:
        rep = eval_on_dataset(ds_name, img_dir, lbl_dir, preds_json)
        all_reps.append(rep)
        blk = _block(rep)
        parts.append(blk)
        print(blk)
        print()

    # comparison table
    table_lines = ["\n" + "=" * 100,
                   "RESULTS  (MICRO pooled metrics)",
                   "=" * 100]
    header = (f"  {'Dataset':<30s}  {'AP30':>6}  {'AP50':>6}  {'AP75':>6}  "
              f"{'F1':>6}  {'P':>6}  {'R':>6}  {'Frag':>5}  {'FragRt':>7}")
    table_lines.append(header)
    table_lines.append("  " + "-" * (len(header) - 2))
    for rep in all_reps:
        mic = rep["micro_pooled"]
        table_lines.append(
            f"  {rep['dataset']:<30s}  {mic['AP30']:>6.4f}  {mic['AP50']:>6.4f}  "
            f"{mic['AP75']:>6.4f}  {mic['F1']:>6.4f}  {mic['P']:>6.4f}  "
            f"{mic['R']:>6.4f}  {mic['Frag']:>5d}  {mic['FragRate']:>7.4f}"
        )
    table_str = "\n".join(table_lines) + "\n"
    print(table_str)

    full_text = "\n".join(parts) + "\n" + table_str

    out_txt  = RUN_DIR / "ablation_results.txt"
    out_json = RUN_DIR / "ablation_results.json"
    out_txt.write_text(full_text)
    print(f"  wrote → {out_txt}")
    out_json.write_text(json.dumps({"model": RUN_LABEL, "datasets": all_reps}, indent=2,
                                   default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))
    print(f"  wrote → {out_json}")


if __name__ == "__main__":
    main()
