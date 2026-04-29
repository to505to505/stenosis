"""Evaluate the consistency-trained Temporal RF-DETR run on cadica_50plus_new and
dataset2_split/test. Produces the standard per-video metrics (AP30/50/75/5095,
best-F1 with P/R/thr) plus the MOT-style Fragmentation metric (Frag) which
quantifies temporal flicker of detections.

Frag definition (per video):
    1. Build GT trajectories by greedy IoU-linking GT boxes across consecutive
       centre frames (link if IoU >= 0.3, otherwise start a new track).
    2. Pick the best-F1 confidence threshold from the standard sweep at IoU 0.5.
    3. For each GT track, build a binary "matched" sequence: at each frame the
       GT box belongs to, mark 1 if any prediction with score >= thr has
       IoU >= 0.5 with that GT box, else 0.
    4. Frag = number of (1, 0+, 1) gap patterns summed across all tracks.
       FragRate = Frag / total_track_frames (interpretable: gaps per frame).

Outputs:
    rfdetr_temporal/runs/<run>/results.txt   (human readable)
    rfdetr_temporal/runs/<run>/results.json  (machine readable)
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

RUN_NAME = "temporal_small_t5_k0_distill_aligned_crrcd_consistency"

DATASETS = [
    ("cadica_50plus_new", ROOT / "data" / "cadica_50plus_new" / "images",
                          ROOT / "data" / "cadica_50plus_new" / "labels"),
    ("dataset2_split_test", ROOT / "data" / "dataset2_split" / "test" / "images",
                            ROOT / "data" / "dataset2_split" / "test" / "labels"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_5095 = np.arange(0.5, 1.0, 0.05)
LINK_IOU = 0.3   # IoU threshold for greedy GT-track linking
MATCH_IOU = 0.5  # IoU threshold for det↔GT matching when computing Frag


# ────────────────────────────── helpers ──────────────────────────────────

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


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes."""
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


# ───────────────────── GT tracks + Fragmentation ────────────────────────

def build_gt_tracks(gts_per_frame: list[np.ndarray]) -> list[list[tuple[int, np.ndarray]]]:
    """Greedy IoU-linking of GT boxes across consecutive frames.
    Returns a list of tracks, each a list of (frame_idx, box_xyxy) tuples.
    """
    tracks: list[list[tuple[int, np.ndarray]]] = []
    # active_tracks: list of indices into ``tracks`` that are still alive
    # (last update == previous frame).
    active: list[int] = []
    last_frame_of: dict[int, int] = {}

    for f_idx, gt in enumerate(gts_per_frame):
        if gt.shape[0] == 0:
            # Keep active alive only across single empty frames? Standard MOT
            # ends a track on first miss; we follow that convention since we
            # are measuring detection flicker, not occlusion gaps in GT.
            active = []
            continue

        if not active:
            for box in gt:
                tracks.append([(f_idx, box.copy())])
                last_frame_of[len(tracks) - 1] = f_idx
                active.append(len(tracks) - 1)
            continue

        # Match active tracks (their last box) to current GT
        last_boxes = np.stack(
            [tracks[ti][-1][1] for ti in active], axis=0
        )
        ious = _iou_matrix(last_boxes, gt)            # (T_active, N_gt)
        new_active: list[int] = []
        used_gt = set()
        # Greedy: take highest IoU pairs first
        flat = [(ious[i, j], i, j)
                for i in range(ious.shape[0])
                for j in range(ious.shape[1])
                if ious[i, j] >= LINK_IOU]
        flat.sort(reverse=True)
        used_track_local = set()
        for iou, i, j in flat:
            if i in used_track_local or j in used_gt:
                continue
            ti = active[i]
            tracks[ti].append((f_idx, gt[j].copy()))
            last_frame_of[ti] = f_idx
            new_active.append(ti)
            used_track_local.add(i)
            used_gt.add(j)
        # Unmatched GT → new tracks
        for j in range(gt.shape[0]):
            if j in used_gt:
                continue
            tracks.append([(f_idx, gt[j].copy())])
            last_frame_of[len(tracks) - 1] = f_idx
            new_active.append(len(tracks) - 1)
        active = new_active
    return tracks


def compute_video_frag(
    dets_per_frame: list[dict],
    gts_per_frame: list[np.ndarray],
    score_thr: float,
) -> tuple[int, int]:
    """Returns (frag_count, total_track_frames).

    Frag = sum over GT tracks of the number of (1, 0+, 1) gap patterns in the
    per-frame "is matched by a kept prediction" binary sequence.
    """
    tracks = build_gt_tracks(gts_per_frame)
    frag_total = 0
    track_frames_total = 0
    for trk in tracks:
        if len(trk) < 3:
            # Need at least 3 frames to have an internal gap pattern (1,0,1)
            track_frames_total += len(trk)
            continue
        status: list[int] = []
        for f_idx, box in trk:
            det = dets_per_frame[f_idx]
            keep = det["scores"] >= score_thr
            if not keep.any():
                status.append(0)
                continue
            kept_boxes = det["boxes"][keep]
            ious = _iou_matrix(box[None, :], kept_boxes)[0]
            status.append(1 if (ious >= MATCH_IOU).any() else 0)
        # Count (1, 0+, 1) gaps inside the status sequence.
        # Equivalent to number of transitions 0→1 that are preceded by a 1
        # somewhere earlier.
        seen_one = False
        in_gap = False
        for s in status:
            if s == 1:
                if in_gap and seen_one:
                    frag_total += 1
                    in_gap = False
                seen_one = True
                in_gap = False
            else:  # s == 0
                if seen_one:
                    in_gap = True
        track_frames_total += len(trk)
    return frag_total, track_frames_total


# ─────────────────────────── per-dataset eval ────────────────────────────

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
    all_gts = [g for v in per_video_gts.values() for g in v]
    micro_ap30 = evaluate_map(all_dets, all_gts, 0.3)
    micro_ap50 = evaluate_map(all_dets, all_gts, 0.5)
    micro_ap75 = evaluate_map(all_dets, all_gts, 0.75)
    micro_ap5095 = float(np.mean([evaluate_map(all_dets, all_gts, t) for t in IOU_5095]))
    micro_f1, micro_p, micro_r, micro_thr = f1_confidence_sweep(all_dets, all_gts)
    # Pooled Frag: sum of per-video Frag and track_frames using each video's
    # best-F1 threshold (consistent with macro definition).
    total_frag = int(sum(r["Frag"] for r in rows))
    total_trk_frames = int(sum(r["track_frames"] for r in rows))
    micro = {
        "AP30": micro_ap30, "AP50": micro_ap50, "AP75": micro_ap75, "AP5095": micro_ap5095,
        "F1": micro_f1, "P": micro_p, "R": micro_r, "thr": micro_thr,
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


def format_report(run_name: str, run_dir: Path, cfg: Config, reports: list[dict]) -> str:
    lines = []
    lines.append(f"Run: {run_name}")
    lines.append(f"Path: {run_dir}")
    lines.append(f"Config: T={cfg.T}  img_size={cfg.img_size}  num_classes={cfg.num_classes}")
    lines.append("")
    for rep in reports:
        lines.append("=" * 90)
        lines.append(f"Dataset: {rep['dataset']}")
        lines.append(f"  videos={rep['n_videos']}  centre_frames={rep['n_centre_frames']}")
        lines.append("=" * 90)
        lines.append("")
        lines.append("Per-video metrics:")
        lines.append(
            f"  {'video':<14s} {'frames':>6s} {'gt':>4s}  "
            f"{'AP30':>6s} {'AP50':>6s} {'AP75':>6s} {'AP5095':>6s}  "
            f"{'F1':>6s} {'P':>6s} {'R':>6s} {'thr':>6s}  "
            f"{'Frag':>5s} {'FragRt':>7s}"
        )
        for r in rep["rows"]:
            lines.append(
                f"  {r['video']:<14s} {r['n_frames']:>6d} {r['n_gt']:>4d}  "
                f"{r['AP30']:>6.3f} {r['AP50']:>6.3f} {r['AP75']:>6.3f} {r['AP5095']:>6.3f}  "
                f"{r['F1']:>6.3f} {r['P']:>6.3f} {r['R']:>6.3f} {r['thr']:>6.3f}  "
                f"{r['Frag']:>5d} {r['FragRate']:>7.4f}"
            )
        lines.append("")
        lines.append("MACRO mean across videos:")
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1", "Frag", "FragRate"):
            v = rep["macro_per_video"][m]
            if m == "Frag":
                lines.append(f"  {m:<8s} {v:.3f}")
            else:
                lines.append(f"  {m:<8s} {v:.4f}")
        lines.append("")
        lines.append("MICRO pooled across all centre frames:")
        for m in ("AP30", "AP50", "AP75", "AP5095", "F1", "P", "R"):
            lines.append(f"  {m:<8s} {rep['micro_pooled'][m]:.4f}")
        lines.append(f"  thr      {rep['micro_pooled']['thr']:.4f}")
        lines.append(f"  Frag     {rep['micro_pooled']['Frag']:d}   "
                     f"(over {rep['micro_pooled']['track_frames']} track-frames)")
        lines.append(f"  FragRate {rep['micro_pooled']['FragRate']:.4f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main():
    run_dir = ROOT / "rfdetr_temporal" / "runs" / RUN_NAME
    print(f"\n{'#' * 80}\n# {RUN_NAME}\n# {run_dir}\n{'#' * 80}")
    cfg = _load_cfg(run_dir)
    print(f"  cfg: T={cfg.T}  img_size={cfg.img_size}")
    model = _load_model(run_dir, cfg)
    _criterion, postprocess = _build_criterion(cfg)

    reports = []
    for ds_name, img_dir, lbl_dir in DATASETS:
        rep = eval_run_on_dataset(model, cfg, postprocess, ds_name, img_dir, lbl_dir)
        reports.append(rep)

    text = format_report(RUN_NAME, run_dir, cfg, reports)
    out_txt = run_dir / "results.txt"
    out_txt.write_text(text)
    print(f"\n  wrote → {out_txt}")

    out_json = run_dir / "results.json"
    out_json.write_text(json.dumps({
        "run": RUN_NAME,
        "cfg": {"T": cfg.T, "img_size": cfg.img_size},
        "datasets": reports,
    }, indent=2))
    print(f"  wrote → {out_json}")

    # Compact summary
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
