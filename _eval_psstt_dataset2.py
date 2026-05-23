"""Test-time evaluation of a PSSTT run on dataset2_split (test split).

Mirrors the layout of :mod:`_eval_stfs_ablations` so MICRO-pooled metrics
(AP30/AP50/AP75/AP5095/F1/P/R/Frag) drop in next to RF-DETR-video numbers
for direct comparison.

Outputs (written to ``psstt/runs/<RUN_NAME>/``):
  ablation_results.txt
  ablation_results.json
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

from psstt.config import Config
from psstt.model import VideoFasterRCNN
from rfdetr_temporal.dataset import build_sequence_index
from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep

ROOT = Path("/home/dsa/stenosis")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_5095 = np.arange(0.5, 1.0, 0.05)
LINK_IOU = 0.3
MATCH_IOU = 0.5

DATASETS = [
    ("dataset2_split_test",
     ROOT / "data" / "dataset2_split" / "test" / "images",
     ROOT / "data" / "dataset2_split" / "test" / "labels"),
]


# ─────────────────────────────── helpers ────────────────────────────────────

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


def _load_model(run_dir: Path, cfg: Config) -> VideoFasterRCNN:
    # We never need COCO weights download here — the run checkpoint
    # already has the trained state. Disable the pretrained download
    # to keep this script offline-safe.
    cfg.pretrained_coco = False
    model = VideoFasterRCNN(cfg).to(DEVICE)
    ckpt_path = run_dir / "best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    msg = model.load_state_dict(sd, strict=False)
    print(f"  loaded best.pth  missing={len(msg.missing_keys)}  "
          f"unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model


def _to_tensor(img: np.ndarray, size: int,
               mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
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


@torch.no_grad()
def _predict_centre(
    model: VideoFasterRCNN,
    frames: torch.Tensor,            # (T, 3, H, W) — already model-input size
    orig_w: int, orig_h: int,
):
    """Return centre-frame (boxes_xyxy, scores) rescaled back to original image size."""
    images = frames.unsqueeze(0).to(DEVICE)         # (1, T, 3, H, W)
    with torch.amp.autocast("cuda", enabled=True):
        out = model(images, targets=None)
    centre = out["centre"][0]
    boxes = centre["boxes"].detach().cpu().numpy().astype(np.float32)
    scores = centre["scores"].detach().cpu().numpy().astype(np.float32)
    # Rescale from model input space (cfg.img_size) to original frame size.
    inp_h = inp_w = int(model.cfg.img_size)
    if boxes.shape[0] > 0:
        sx = orig_w / inp_w
        sy = orig_h / inp_h
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
    return boxes, scores


# ───────────────────── IoU + fragmentation helpers ──────────────────────────

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
        for iou, i, j in flat:
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


# ───────────────────────── per-dataset eval ─────────────────────────────────

def eval_on_dataset(model, cfg, ds_name, img_dir, lbl_dir):
    sequences = build_sequence_index(img_dir)
    mean_t = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std_t  = torch.tensor(cfg.pixel_std,  dtype=torch.float32).view(3, 1, 1)
    centre = cfg.T // 2

    per_video_dets: dict = defaultdict(list)
    per_video_gts:  dict = defaultdict(list)
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
            boxes, scores = _predict_centre(model, frames_t, orig_w, orig_h)
            centre_path = win[centre]
            gt = _yolo_xyxy_in_pixels(
                lbl_dir / (centre_path.stem + ".txt"), orig_w, orig_h,
            )
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


def _block(rep: dict) -> str:
    mac = rep["macro_per_video"]
    mic = rep["micro_pooled"]
    return "\n".join([
        f"  Dataset : {rep['dataset']}  ({rep['n_videos']} videos, "
        f"{rep['n_centre_frames']} centre frames)",
        "",
        f"  MACRO (mean/video)  AP30={mac['AP30']:.4f}  AP50={mac['AP50']:.4f}  "
        f"AP75={mac['AP75']:.4f}  AP5095={mac['AP5095']:.4f}  "
        f"F1={mac['F1']:.4f}  Frag={mac['Frag']:.2f}  FragRate={mac['FragRate']:.4f}",
        f"  MICRO (pooled)      AP30={mic['AP30']:.4f}  AP50={mic['AP50']:.4f}  "
        f"AP75={mic['AP75']:.4f}  AP5095={mic['AP5095']:.4f}  "
        f"F1={mic['F1']:.4f}  P={mic['P']:.4f}  R={mic['R']:.4f}  thr={mic['thr']:.4f}  "
        f"Frag={mic['Frag']}  FragRate={mic['FragRate']:.4f}",
    ])


def main(run_name: str):
    run_dir = ROOT / "psstt" / "runs" / run_name
    print(f"\n{'#' * 80}\n# PSSTT Evaluation – {run_name}\n# {run_dir}\n{'#' * 80}\n")
    base_cfg = _load_cfg(run_dir)
    print(f"  Base cfg: T={base_cfg.T}  img_size={base_cfg.img_size}")
    model = _load_model(run_dir, base_cfg)

    text_parts = [
        f"PSSTT Evaluation – {run_name}",
        f"Run dir: {run_dir}",
    ]
    reps = []
    for ds_name, img_dir, lbl_dir in DATASETS:
        print(f"\n  ── dataset: {ds_name}")
        rep = eval_on_dataset(model, model.cfg, ds_name, img_dir, lbl_dir)
        reps.append(rep)
        print(_block(rep))
        text_parts.append("\n" + "=" * 80)
        text_parts.append(f"DATASET: {ds_name}")
        text_parts.append("=" * 80)
        text_parts.append(_block(rep))

    out_txt = run_dir / "ablation_results.txt"
    out_txt.write_text("\n".join(text_parts) + "\n")
    print(f"\n  wrote → {out_txt}")

    out_json = run_dir / "ablation_results.json"
    out_json.write_text(json.dumps({
        "run": run_name,
        "base_cfg": {"T": base_cfg.T, "img_size": base_cfg.img_size},
        "datasets": reps,
    }, indent=2))
    print(f"  wrote → {out_json}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python _eval_psstt_dataset2.py <run_name>")
        sys.exit(1)
    main(sys.argv[1])
