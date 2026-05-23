"""Intuitive metrics for video_overfit_R1 on cadica_50plus_new + dataset2_split_test.

Reuses the inference helpers from _eval_stfs_ablations.py but reports a richer,
easier-to-read set of metrics:

  Frame-level (match IoU = 0.3):
    AP@0.3
    Best-F1 sweep: F1, P, R, best_score_thr, TP, FP, FN
    Fixed-threshold sweep: at score 0.3 / 0.5 / 0.7 → P, R, F1
    Mean localisation quality on TPs: mean IoU
    Frame coverage counts: total, with-GT, with-detection, with-TP

  Track-level (track linking IoU = 0.3, match IoU = 0.3):
    Number of GT tracks (= unique stenosis instances)
    Tracks ever detected (>=1 frame hit)
    Tracks well-detected (>=50% frames hit)
    Tracks excellently-detected (>=80% frames hit)
    Tracks completely missed (0 frames hit)
    Mean per-track recall (frac of frames hit)
    Mean longest-correct-streak ratio per track
    Mean fragmentations per track

  Video-level:
    Total videos
    Videos with >=1 correct detection
    Mean per-video F1 @ IoU 0.3

Outputs:
  rfdetr_video/runs/video_overfit_R1/intuitive_metrics.txt
  rfdetr_video/runs/video_overfit_R1/intuitive_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from rfdetr_video.config import Config
from rfdetr_video.model import VideoRFDETR, build_criterion
from rfdetr_temporal.dataset import build_sequence_index
from rfdetr_temporal.evaluate import evaluate_map

ROOT = Path("/home/dsa/stenosis")
RUN_NAME = "video_overfit_R1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match thresholds — user explicitly asked for IoU=0.3
MATCH_IOU_FRAME = 0.3   # box-vs-GT match in P/R/F1 sweep
LINK_IOU_TRACK = 0.3    # GT-to-GT linking across frames to build tracks
MATCH_IOU_TRACK = 0.3   # det-vs-track-GT match when scoring frames of a track
FIXED_SCORE_THRS = [0.2, 0.3, 0.5, 0.7]
TRACK_FIXED_THRS = [0.2, 0.3]

DATASETS = [
    ("cadica_50plus_new",
     ROOT / "data" / "cadica_50plus_new" / "images",
     ROOT / "data" / "cadica_50plus_new" / "labels"),
    ("dataset2_split_test",
     ROOT / "data" / "dataset2_split" / "test" / "images",
     ROOT / "data" / "dataset2_split" / "test" / "labels"),
]


# ─────────────────────────── load model + cfg ────────────────────────────────

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


def _load_model(run_dir: Path, cfg: Config) -> VideoRFDETR:
    model = VideoRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    msg = model.load_state_dict(sd, strict=False)
    print(f"  loaded best.pth  missing={len(msg.missing_keys)}  "
          f"unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model


# ─────────────────────────────── inference ───────────────────────────────────

def _to_tensor(img, size, mean, std):
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    arr = np.stack([img, img, img], axis=0)
    t = torch.from_numpy(arr)
    return (t - mean) / std


def _yolo_xyxy_in_pixels(lbl_path, w, h):
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


@torch.no_grad()
def _predict_centre(model, frames, postprocess, orig_w, orig_h, centre):
    images = frames.unsqueeze(0).to(DEVICE)
    with torch.amp.autocast("cuda", enabled=True):
        out = model(images, query_mode="student")
    centre_out = {
        "pred_logits": out["pred_logits"][:, centre],
        "pred_boxes":  out["pred_boxes"][:, centre],
    }
    orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
    res = postprocess(centre_out, orig_size)[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


# ───────────────────────────── IoU helpers ───────────────────────────────────

def _iou_matrix(a, b):
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


# ───────── frame-level metrics: F1 sweep / fixed thr / counts ────────────────

def _per_frame_match(boxes, gts, score_thr, iou_thr):
    """Greedy 1-1 matching of det boxes vs GTs for one frame at given score thr.

    Returns (tp, fp, fn, tp_ious[list of float]).
    """
    if gts.shape[0] == 0:
        keep = boxes["scores"] >= score_thr
        return 0, int(keep.sum()), 0, []
    keep = boxes["scores"] >= score_thr
    if not np.any(keep):
        return 0, 0, gts.shape[0], []
    dets_kept = boxes["boxes"][keep]
    iou = _iou_matrix(dets_kept, gts)
    # Greedy match in score-desc order (we already filtered keep, preserve order
    # by score because postprocess returns sorted by score? Not guaranteed —
    # sort explicitly):
    scores_kept = boxes["scores"][keep]
    order = np.argsort(-scores_kept)
    iou = iou[order]
    matched = np.zeros(gts.shape[0], dtype=bool)
    tp = fp = 0
    tp_ious = []
    for i in range(iou.shape[0]):
        j = int(np.argmax(iou[i]))
        if iou[i, j] >= iou_thr and not matched[j]:
            matched[j] = True
            tp += 1
            tp_ious.append(float(iou[i, j]))
        else:
            fp += 1
    fn = int((~matched).sum())
    return tp, fp, fn, tp_ious


def f1_sweep_at_iou(all_dets, all_gts, iou_thr, n=101):
    """Sweep score threshold; report best-F1 + counts."""
    # Precompute per-frame sorted scores + iou matrices
    pre = []
    total_gt = 0
    for det, gt in zip(all_dets, all_gts):
        total_gt += gt.shape[0]
        scores = det["scores"]
        boxes = det["boxes"]
        if len(scores) == 0:
            pre.append((scores, None, gt.shape[0]))
            continue
        order = np.argsort(-scores)
        s_sorted = scores[order]
        b_sorted = boxes[order]
        if gt.shape[0] == 0:
            pre.append((s_sorted, None, 0))
            continue
        iou_mat = _iou_matrix(b_sorted, gt)
        pre.append((s_sorted, iou_mat, gt.shape[0]))

    if total_gt == 0:
        return {"F1": 0, "P": 0, "R": 0, "thr": 0, "TP": 0, "FP": 0, "FN": 0}

    best = {"F1": -1.0}
    for thr in np.linspace(0.0, 1.0, n):
        tp = fp = fn = 0
        for s_sorted, iou_mat, n_gt in pre:
            if n_gt == 0:
                fp += int(np.sum(s_sorted >= thr)) if len(s_sorted) > 0 else 0
                continue
            if len(s_sorted) == 0:
                fn += n_gt
                continue
            keep = s_sorted >= thr
            if not np.any(keep):
                fn += n_gt
                continue
            matched = np.zeros(n_gt, dtype=bool)
            for iou_row in iou_mat[keep]:
                j = int(np.argmax(iou_row))
                if iou_row[j] >= iou_thr and not matched[j]:
                    matched[j] = True
                    tp += 1
                else:
                    fp += 1
            fn += int(np.sum(~matched))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-8)
        if f1 > best["F1"]:
            best = {"F1": f1, "P": p, "R": r, "thr": float(thr),
                    "TP": tp, "FP": fp, "FN": fn}
    return best


def prf_at_fixed_thr(all_dets, all_gts, score_thr, iou_thr):
    tp = fp = fn = 0
    tp_ious_all = []
    n_frames_with_gt = 0
    n_frames_with_det = 0
    n_frames_with_tp = 0
    for det, gt in zip(all_dets, all_gts):
        tp_i, fp_i, fn_i, ious_i = _per_frame_match(det, gt, score_thr, iou_thr)
        tp += tp_i
        fp += fp_i
        fn += fn_i
        tp_ious_all.extend(ious_i)
        if gt.shape[0] > 0:
            n_frames_with_gt += 1
        if (det["scores"] >= score_thr).any():
            n_frames_with_det += 1
        if tp_i > 0:
            n_frames_with_tp += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-8)
    return {
        "thr": score_thr, "iou": iou_thr,
        "P": p, "R": r, "F1": f1,
        "TP": tp, "FP": fp, "FN": fn,
        "mean_TP_IoU": float(np.mean(tp_ious_all)) if tp_ious_all else 0.0,
        "n_frames_with_gt": n_frames_with_gt,
        "n_frames_with_det": n_frames_with_det,
        "n_frames_with_TP": n_frames_with_tp,
    }


# ───────────────────────── track-level metrics ───────────────────────────────

def build_gt_tracks(gts_per_frame, link_iou=LINK_IOU_TRACK):
    """Greedy: link GT boxes across consecutive frames by IoU >= link_iou."""
    tracks = []   # each: list of (f_idx, box)
    active = []   # indices into tracks of currently-active tracks
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
                if ious[i, j] >= link_iou]
        flat.sort(reverse=True)
        used_track_local = set()
        for _iou, i, j in flat:
            if i in used_track_local or j in used_gt:
                continue
            ti = active[i]
            tracks[ti].append((f_idx, gt[j].copy()))
            new_active.append(ti)
            used_track_local.add(i)
            used_gt.add(j)
        for j in range(gt.shape[0]):
            if j in used_gt:
                continue
            tracks.append([(f_idx, gt[j].copy())])
            new_active.append(len(tracks) - 1)
        active = new_active
    return tracks


def track_metrics(per_video_dets, per_video_gts, score_thr, match_iou=MATCH_IOU_TRACK):
    """Compute track-level stats given a score threshold."""
    n_tracks = 0
    tracks_ever_found = 0
    tracks_well_found = 0      # >=50% frames hit
    tracks_excellently_found = 0  # >=80% frames hit
    tracks_missed = 0
    per_track_recalls = []
    longest_streak_ratios = []
    frags_per_track = []
    for vid in sorted(per_video_dets):
        dets = per_video_dets[vid]
        gts = per_video_gts[vid]
        tracks = build_gt_tracks(gts)
        for trk in tracks:
            n_tracks += 1
            statuses = []
            for f_idx, box in trk:
                det = dets[f_idx]
                keep = det["scores"] >= score_thr
                if not keep.any():
                    statuses.append(0)
                    continue
                ious = _iou_matrix(box[None, :], det["boxes"][keep])[0]
                statuses.append(1 if (ious >= match_iou).any() else 0)
            n_hits = sum(statuses)
            recall = n_hits / len(statuses)
            per_track_recalls.append(recall)
            if n_hits > 0:
                tracks_ever_found += 1
            if recall >= 0.5:
                tracks_well_found += 1
            if recall >= 0.8:
                tracks_excellently_found += 1
            if n_hits == 0:
                tracks_missed += 1

            # Longest correct streak
            best_streak = 0
            cur = 0
            for s in statuses:
                if s == 1:
                    cur += 1
                    best_streak = max(best_streak, cur)
                else:
                    cur = 0
            longest_streak_ratios.append(best_streak / len(statuses))

            # Fragmentations: count 1→0→1 transitions (gaps between hits)
            # Only count gaps INSIDE the part of the track from first to last hit.
            first_hit = next((i for i, s in enumerate(statuses) if s == 1), None)
            last_hit = next((len(statuses) - 1 - i for i, s in enumerate(reversed(statuses)) if s == 1), None)
            frags = 0
            if first_hit is not None and last_hit is not None and last_hit > first_hit:
                in_gap = False
                for s in statuses[first_hit:last_hit + 1]:
                    if s == 0:
                        in_gap = True
                    elif in_gap:
                        frags += 1
                        in_gap = False
            frags_per_track.append(frags)

    if n_tracks == 0:
        return {
            "n_tracks": 0, "tracks_ever_found": 0, "tracks_well_found": 0,
            "tracks_excellently_found": 0, "tracks_missed": 0,
            "mean_per_track_recall": 0.0, "mean_longest_streak_ratio": 0.0,
            "mean_frags_per_track": 0.0, "score_thr": score_thr,
        }
    return {
        "n_tracks": n_tracks,
        "tracks_ever_found": tracks_ever_found,
        "tracks_well_found": tracks_well_found,
        "tracks_excellently_found": tracks_excellently_found,
        "tracks_missed": tracks_missed,
        "frac_ever_found": tracks_ever_found / n_tracks,
        "frac_well_found": tracks_well_found / n_tracks,
        "frac_excellently_found": tracks_excellently_found / n_tracks,
        "frac_missed": tracks_missed / n_tracks,
        "mean_per_track_recall": float(np.mean(per_track_recalls)),
        "mean_longest_streak_ratio": float(np.mean(longest_streak_ratios)),
        "mean_frags_per_track": float(np.mean(frags_per_track)),
        "score_thr": score_thr,
    }


# ───────────────────────── per-dataset evaluation ────────────────────────────

def eval_on_dataset(model, cfg, postprocess, ds_name, img_dir, lbl_dir):
    sequences = build_sequence_index(img_dir)
    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(cfg.pixel_std,  dtype=torch.float32).view(3, 1, 1)
    centre = cfg.T // 2

    per_video_dets = defaultdict(list)
    per_video_gts  = defaultdict(list)
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
            boxes, scores = _predict_centre(
                model, frames_t, postprocess, orig_w, orig_h, centre,
            )
            centre_path = win[centre]
            gt = _yolo_xyxy_in_pixels(
                lbl_dir / (centre_path.stem + ".txt"), orig_w, orig_h,
            )
            key = f"{pid}_v{sid}"
            per_video_dets[key].append({"boxes": boxes, "scores": scores})
            per_video_gts[key].append(gt)
            total += 1

        if (vi + 1) % 20 == 0 or (vi + 1) == len(sequences):
            print(f"     videos {vi + 1:3d}/{len(sequences)}  frames={total}  "
                  f"elapsed={time.time() - t0:.1f}s")

    # ----- Pooled (micro) frame-level metrics -----
    all_dets = [d for v in per_video_dets.values() for d in v]
    all_gts  = [g for v in per_video_gts.values()  for g in v]
    n_frames = len(all_dets)
    n_gt_total = int(sum(g.shape[0] for g in all_gts))

    ap30 = float(evaluate_map(all_dets, all_gts, 0.3))
    ap50 = float(evaluate_map(all_dets, all_gts, 0.5))

    best_at_iou30 = f1_sweep_at_iou(all_dets, all_gts, iou_thr=MATCH_IOU_FRAME)
    fixed = [prf_at_fixed_thr(all_dets, all_gts, thr, MATCH_IOU_FRAME)
             for thr in FIXED_SCORE_THRS]

    # ----- Mean per-video F1 @ IoU 0.3 -----
    per_video_f1 = []
    n_videos_with_correct = 0
    for vid in sorted(per_video_dets):
        dets = per_video_dets[vid]
        gts = per_video_gts[vid]
        b = f1_sweep_at_iou(dets, gts, iou_thr=MATCH_IOU_FRAME, n=51)
        per_video_f1.append(b["F1"])
        if b["TP"] > 0:
            n_videos_with_correct += 1
    macro_f1 = float(np.mean(per_video_f1)) if per_video_f1 else 0.0

    # ----- Track-level (at best-F1 thr from sweep at IoU=0.3) -----
    trk = track_metrics(per_video_dets, per_video_gts,
                        score_thr=best_at_iou30["thr"],
                        match_iou=MATCH_IOU_TRACK)

    # ----- Track-level at softer fixed thresholds for comparison -----
    trk_fixed = [track_metrics(per_video_dets, per_video_gts,
                               score_thr=thr, match_iou=MATCH_IOU_TRACK)
                 for thr in TRACK_FIXED_THRS]

    return {
        "dataset": ds_name,
        "n_videos": len(per_video_dets),
        "n_frames": n_frames,
        "n_gt_total": n_gt_total,
        "AP@0.3": ap30,
        "AP@0.5": ap50,
        "best_F1_at_IoU0.3": best_at_iou30,
        "fixed_thr_at_IoU0.3": fixed,
        "macro_per_video_F1_at_IoU0.3": macro_f1,
        "n_videos_with_correct": n_videos_with_correct,
        "tracks_at_best_thr": trk,
        "tracks_at_fixed_thr": trk_fixed,
    }


# ──────────────────────────── formatting ─────────────────────────────────────

def _fmt_block(rep):
    L = []
    L.append(f"Dataset : {rep['dataset']}")
    L.append(f"  Coverage      : {rep['n_videos']} videos, "
             f"{rep['n_frames']} centre frames, {rep['n_gt_total']} GT boxes")
    L.append(f"  AP@0.3        : {rep['AP@0.3']:.4f}")
    L.append(f"  AP@0.5        : {rep['AP@0.5']:.4f}")
    b = rep["best_F1_at_IoU0.3"]
    L.append(f"  Best-F1 sweep @ IoU=0.3:")
    L.append(f"    F1={b['F1']:.4f}  P={b['P']:.4f}  R={b['R']:.4f}  "
             f"score_thr={b['thr']:.3f}")
    L.append(f"    TP={b['TP']}  FP={b['FP']}  FN={b['FN']}")
    L.append(f"  Fixed score threshold @ IoU=0.3:")
    for f in rep["fixed_thr_at_IoU0.3"]:
        L.append(f"    thr={f['thr']:.2f}  P={f['P']:.4f}  R={f['R']:.4f}  "
                 f"F1={f['F1']:.4f}  (TP={f['TP']}  FP={f['FP']}  FN={f['FN']})  "
                 f"mean TP IoU={f['mean_TP_IoU']:.3f}")
    L.append(f"  Macro per-video F1 @ IoU=0.3: {rep['macro_per_video_F1_at_IoU0.3']:.4f}")
    L.append(f"  Videos with >=1 correct det : "
             f"{rep['n_videos_with_correct']}/{rep['n_videos']}")
    t = rep["tracks_at_best_thr"]
    L.append(f"  Track-level (score thr = best-F1 = {t['score_thr']:.3f}, match IoU=0.3):")
    L.append(f"    Total GT tracks (stenosis instances) : {t['n_tracks']}")
    L.append(f"    Tracks ever detected (>=1 hit)        : "
             f"{t['tracks_ever_found']}  ({t['frac_ever_found']:.2%})")
    L.append(f"    Tracks well-detected (>=50% frames)   : "
             f"{t['tracks_well_found']}  ({t['frac_well_found']:.2%})")
    L.append(f"    Tracks excellently-detected (>=80%)   : "
             f"{t['tracks_excellently_found']}  ({t['frac_excellently_found']:.2%})")
    L.append(f"    Tracks completely missed              : "
             f"{t['tracks_missed']}  ({t['frac_missed']:.2%})")
    L.append(f"    Mean per-track recall                 : "
             f"{t['mean_per_track_recall']:.4f}")
    L.append(f"    Mean longest-streak ratio per track   : "
             f"{t['mean_longest_streak_ratio']:.4f}")
    L.append(f"    Mean fragmentations per track         : "
             f"{t['mean_frags_per_track']:.4f}")
    for ts in rep["tracks_at_fixed_thr"]:
        L.append(f"  Track-level (fixed thr = {ts['score_thr']:.2f}, match IoU=0.3):")
        L.append(f"    Total GT tracks                       : {ts['n_tracks']}")
        L.append(f"    Tracks ever detected (>=1 hit)        : "
                 f"{ts['tracks_ever_found']}  ({ts['frac_ever_found']:.2%})")
        L.append(f"    Tracks well-detected (>=50% frames)   : "
                 f"{ts['tracks_well_found']}  ({ts['frac_well_found']:.2%})")
        L.append(f"    Tracks excellently-detected (>=80%)   : "
                 f"{ts['tracks_excellently_found']}  ({ts['frac_excellently_found']:.2%})")
        L.append(f"    Tracks completely missed              : "
                 f"{ts['tracks_missed']}  ({ts['frac_missed']:.2%})")
        L.append(f"    Mean per-track recall                 : "
                 f"{ts['mean_per_track_recall']:.4f}")
        L.append(f"    Mean longest-streak ratio per track   : "
                 f"{ts['mean_longest_streak_ratio']:.4f}")
        L.append(f"    Mean fragmentations per track         : "
                 f"{ts['mean_frags_per_track']:.4f}")
    return "\n".join(L)


# ────────────────────────────────── main ─────────────────────────────────────

def main(run_name=RUN_NAME):
    run_dir = ROOT / "rfdetr_video" / "runs" / run_name
    print(f"\n{'#' * 80}")
    print(f"# Intuitive metrics – {run_name}")
    print(f"# {run_dir}")
    print(f"{'#' * 80}\n")

    base_cfg = _load_cfg(run_dir)
    print(f"  Base cfg: T={base_cfg.T}  img_size={base_cfg.img_size}")
    model = _load_model(run_dir, base_cfg)
    _criterion, postprocess = build_criterion(base_cfg)

    reports = []
    for ds_name, img_dir, lbl_dir in DATASETS:
        print(f"\n── dataset: {ds_name}")
        rep = eval_on_dataset(model, base_cfg, postprocess,
                              ds_name, img_dir, lbl_dir)
        reports.append(rep)
        print()
        print(_fmt_block(rep))

    # ── Save text ──
    parts = [f"Intuitive metrics – {run_name}",
             f"Run dir: {run_dir}",
             f"Match IoU (frame & track) = 0.3, link IoU (track build) = 0.3",
             "=" * 80]
    for rep in reports:
        parts.append(_fmt_block(rep))
        parts.append("-" * 80)
    out_txt = run_dir / "intuitive_metrics.txt"
    out_txt.write_text("\n".join(parts))
    print(f"\n  wrote → {out_txt}")

    out_json = run_dir / "intuitive_metrics.json"
    out_json.write_text(json.dumps({"run": run_name, "reports": reports}, indent=2))
    print(f"  wrote → {out_json}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RUN_NAME)
