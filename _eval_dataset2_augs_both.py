"""Evaluate rfdetr_runs/dataset2_augs (RFDETRSmall, single-frame) on
cadica_50plus_new and dataset2_split/test.

Produces per-video AP30/50/75/5095, F1, Fragmentation — identical metric
definitions and output format to rfdetr_video/_eval_video_frag.py /
stfs_nodistill_v1/results.json, so the results are directly comparable.

Writes:
    rfdetr_runs/dataset2_augs/results.txt
    rfdetr_runs/dataset2_augs/results.json
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from rfdetr import RFDETRSmall
# rfdetr_temporal.dataset imports albumentations at module level — inline what we need instead
from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep

# ── filename parsers (copied from rfdetr_temporal/dataset.py) ────────────────
_FNAME_RE   = re.compile(r"^(\d+_\d+)_(\d+)_(\d+)_bmp_jpg\.rf\.[0-9a-f]+\.jpg$")
_CADICA_RE  = re.compile(r"^(p\d+)_v(\d+)_(\d+)\.(?:png|jpg)$")

def _parse_filename(fname: str) -> Optional[Tuple[str, int, int]]:
    m = _FNAME_RE.match(fname)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = _CADICA_RE.match(fname)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None

def build_sequence_index(image_dir: Path) -> List[Tuple[str, int, List[Path]]]:
    groups: Dict[Tuple[str, int], List[Tuple[int, Path]]] = defaultdict(list)
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".png"):
            continue
        parsed = _parse_filename(p.name)
        if parsed is None:
            continue
        pid, sid, fnum = parsed
        groups[(pid, sid)].append((fnum, p))
    sequences = []
    for (pid, sid), frames in sorted(groups.items()):
        frames.sort(key=lambda x: x[0])
        sequences.append((pid, sid, [p for _, p in frames]))
    return sequences

ROOT     = Path("/home/dsa/stenosis")
MODEL_NAME = "dataset2_augs"
CKPT     = ROOT / "rfdetr_runs" / MODEL_NAME / "checkpoint_best_total.pth"
OUT_DIR  = ROOT / "rfdetr_runs" / MODEL_NAME
RESOLUTION = 512
CONF_EVAL  = 0.01   # low threshold → get all detections for AP sweep

DATASETS = [
    (
        "cadica_50plus_new",
        ROOT / "data" / "cadica_50plus_new" / "images",
        ROOT / "data" / "cadica_50plus_new" / "labels",
    ),
    (
        "dataset2_split_test",
        ROOT / "data" / "dataset2_split" / "test" / "images",
        ROOT / "data" / "dataset2_split" / "test" / "labels",
    ),
]

IOU_5095  = np.arange(0.5, 1.0, 0.05)
LINK_IOU  = 0.3   # IoU to link consecutive GT boxes into a track
MATCH_IOU = 0.5   # IoU to count a detection as a hit in frag metric


# ─────────────────────────────── helpers ─────────────────────────────────────

def _yolo_xyxy_in_pixels(lbl_path: Path, w: int, h: int) -> np.ndarray:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lab = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    if lab.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx = lab[:, 1] * w;  cy = lab[:, 2] * h
    bw = lab[:, 3] * w;  bh = lab[:, 4] * h
    return np.column_stack([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2]).astype(np.float32)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_area = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    b_area = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clip(0).prod(-1)
    union = a_area[:, None] + b_area[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def build_gt_tracks(gts_per_frame: list) -> list:
    tracks: list = []
    active: list = []
    for f_idx, gt in enumerate(gts_per_frame):
        if gt.shape[0] == 0:
            active = []
            continue
        if not active:
            for box in gt:
                tracks.append([(f_idx, box.copy())])
                active.append(len(tracks) - 1)
            continue
        last_boxes = np.stack([tracks[ti][-1][1] for ti in active])
        ious = _iou_matrix(last_boxes, gt)
        flat = sorted(
            [(ious[i, j], i, j)
             for i in range(ious.shape[0]) for j in range(ious.shape[1])
             if ious[i, j] >= LINK_IOU],
            reverse=True,
        )
        used_tr, used_gt = set(), set()
        new_active = []
        for _, i, j in flat:
            if i in used_tr or j in used_gt:
                continue
            ti = active[i]
            tracks[ti].append((f_idx, gt[j].copy()))
            new_active.append(ti)
            used_tr.add(i); used_gt.add(j)
        for j in range(gt.shape[0]):
            if j in used_gt:
                continue
            tracks.append([(f_idx, gt[j].copy())])
            new_active.append(len(tracks) - 1)
        active = new_active
    return tracks


def compute_video_frag(dets_per_frame: list, gts_per_frame: list, score_thr: float):
    tracks = build_gt_tracks(gts_per_frame)
    frag_total = 0
    trk_frames  = 0
    for trk in tracks:
        trk_frames += len(trk)
        if len(trk) < 3:
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
        seen, in_gap = False, False
        for s in status:
            if s:
                if in_gap and seen:
                    frag_total += 1
                    in_gap = False
                seen = True; in_gap = False
            else:
                if seen:
                    in_gap = True
    return frag_total, trk_frames


# ─────────────────────────── per-dataset eval ────────────────────────────────

def eval_on_dataset(model: RFDETRSmall, ds_name: str, img_dir: Path, lbl_dir: Path) -> dict:
    print(f"\n  ── dataset: {ds_name}  ({img_dir})")
    sequences = build_sequence_index(img_dir)
    print(f"     {len(sequences)} videos")

    per_video_dets: dict = defaultdict(list)
    per_video_gts:  dict = defaultdict(list)

    total = 0
    t0 = time.time()
    for vi, (pid, sid, paths) in enumerate(sequences):
        for p in paths:
            img_pil = Image.open(p).convert("RGB")
            orig_w, orig_h = img_pil.size
            det = model.predict(img_pil, threshold=CONF_EVAL)
            if det is not None and len(det) > 0:
                boxes  = np.array(det.xyxy,        dtype=np.float32)
                scores = np.array(det.confidence,  dtype=np.float32)
            else:
                boxes  = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros(0,      dtype=np.float32)
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
            "video": vid,
            "n_frames": len(dets),
            "n_gt": int(sum(g.shape[0] for g in gts)),
            "AP30": ap30, "AP50": ap50, "AP75": ap75, "AP5095": ap5095,
            "F1": f1, "P": p_, "R": r_, "thr": thr,
            "Frag": int(frag),
            "FragRate": (frag / trk_frames) if trk_frames > 0 else 0.0,
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
    mf1, mp, mr, mthr = f1_confidence_sweep(all_dets, all_gts)
    total_frag       = int(sum(r["Frag"] for r in rows))
    total_trk_frames = int(sum(r["track_frames"] for r in rows))
    micro = {
        "AP30": micro_ap30, "AP50": micro_ap50, "AP75": micro_ap75,
        "AP5095": micro_ap5095, "F1": mf1, "P": mp, "R": mr, "thr": mthr,
        "Frag": total_frag,
        "FragRate": (total_frag / total_trk_frames) if total_trk_frames > 0 else 0.0,
        "track_frames": total_trk_frames,
    }

    return {
        "dataset": ds_name,
        "n_videos": len(rows),
        "n_centre_frames": total,
        "rows": rows,
        "macro_per_video": macro,
        "micro_pooled": micro,
    }


# ─────────────────────────── report formatting ───────────────────────────────

def format_report(reports: list) -> str:
    lines = [
        f"Run: {MODEL_NAME}",
        f"Path: {OUT_DIR}",
        f"Config: T=1 (single-frame)  img_size={RESOLUTION}  num_classes=1",
        "",
    ]
    for rep in reports:
        lines += [
            "=" * 90,
            f"Dataset: {rep['dataset']}",
            f"  videos={rep['n_videos']}  frames={rep['n_centre_frames']}",
            "=" * 90, "",
            "Per-video metrics:",
            f"  {'video':<14s} {'frames':>6s} {'gt':>4s}  "
            f"{'AP30':>6s} {'AP50':>6s} {'AP75':>6s} {'AP5095':>6s}  "
            f"{'F1':>6s} {'P':>6s} {'R':>6s} {'thr':>6s}  "
            f"{'Frag':>5s} {'FragRt':>7s}",
        ]
        for r in rep["rows"]:
            lines.append(
                f"  {r['video']:<14s} {r['n_frames']:>6d} {r['n_gt']:>4d}  "
                f"{r['AP30']:>6.3f} {r['AP50']:>6.3f} {r['AP75']:>6.3f} {r['AP5095']:>6.3f}  "
                f"{r['F1']:>6.3f} {r['P']:>6.3f} {r['R']:>6.3f} {r['thr']:>6.3f}  "
                f"{r['Frag']:>5d} {r['FragRate']:>7.4f}"
            )
        mac = rep["macro_per_video"]
        mic = rep["micro_pooled"]
        lines += ["", "MACRO mean across videos:"]
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1", "Frag", "FragRate"):
            v = mac[m]
            lines.append(f"  {m:<8s} {v:.3f}" if m == "Frag" else f"  {m:<8s} {v:.4f}")
        lines += ["", "MICRO pooled across all frames:"]
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1", "P", "R"):
            lines.append(f"  {m:<8s} {mic[m]:.4f}")
        lines.append(f"  thr      {mic['thr']:.4f}")
        lines.append(f"  Frag     {mic['Frag']:d}   (over {mic['track_frames']} track-frames)")
        lines.append(f"  FragRate {mic['FragRate']:.4f}")
        lines.append("")
    return "\n".join(lines) + "\n"


# ─────────────────────────────── main ────────────────────────────────────────

def main():
    print(f"\n{'#' * 80}\n# {MODEL_NAME}\n# {OUT_DIR}\n{'#' * 80}")
    print(f"  checkpoint: {CKPT}")

    model = RFDETRSmall(
        pretrain_weights=str(CKPT),
        resolution=RESOLUTION,
        num_classes=1,
    )

    reports = []
    for ds_name, img_dir, lbl_dir in DATASETS:
        rep = eval_on_dataset(model, ds_name, img_dir, lbl_dir)
        reports.append(rep)

    text = format_report(reports)
    (OUT_DIR / "results.txt").write_text(text)
    print(f"\n  wrote → {OUT_DIR / 'results.txt'}")

    (OUT_DIR / "results.json").write_text(json.dumps({
        "run": MODEL_NAME,
        "cfg": {"T": 1, "img_size": RESOLUTION},
        "datasets": reports,
    }, indent=2))
    print(f"  wrote → {OUT_DIR / 'results.json'}")

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for rep in reports:
        mac = rep["macro_per_video"]
        mic = rep["micro_pooled"]
        print(f"\nDataset: {rep['dataset']}")
        print(f"  MACRO  AP30={mac['AP30']:.4f}  AP50={mac['AP50']:.4f}  "
              f"AP5095={mac['AP5095']:.4f}  F1={mac['F1']:.4f}  "
              f"Frag(mean/video)={mac['Frag']:.2f}  FragRate={mac['FragRate']:.4f}")
        print(f"  MICRO  AP30={mic['AP30']:.4f}  AP50={mic['AP50']:.4f}  "
              f"AP5095={mic['AP5095']:.4f}  F1={mic['F1']:.4f}  "
              f"Frag(total)={mic['Frag']}  FragRate={mic['FragRate']:.4f}")


if __name__ == "__main__":
    main()
