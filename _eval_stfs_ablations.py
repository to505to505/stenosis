"""STFS Ablation tests for stfs_nodistill_v4_tempdropout_new.

Three isolated tests modifying only the track_queries parameters (no retraining):

  Baseline  – original config
  Test 1    – iou_gate = 0.0      (disable IoU gate, check motion-break hypothesis)
  Test 2    – score_thresh = 0.05  (lower seed threshold, check weak-frame loss)
  Test 3    – iou_weight=0.0, l1_weight=7.0  (pure-L1+cls cost matrix, à la Stenosis-DetNet)

Outputs (written to runs/<RUN_NAME>/):
  ablation_results.txt
  ablation_results.json
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from rfdetr_video.config import Config
from rfdetr_video.model import VideoRFDETR, build_criterion
from rfdetr_temporal.dataset import build_sequence_index
from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep

ROOT = Path("/home/dsa/stenosis")
RUN_NAME = "stfs_nodistill_v4_tempdropout_new"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_5095 = np.arange(0.5, 1.0, 0.05)
LINK_IOU = 0.3
MATCH_IOU = 0.5

DATASETS = [
    ("cadica_50plus_new",
     ROOT / "data" / "cadica_50plus_new" / "images",
     ROOT / "data" / "cadica_50plus_new" / "labels"),
    ("dataset2_split_test",
     ROOT / "data" / "dataset2_split" / "test" / "images",
     ROOT / "data" / "dataset2_split" / "test" / "labels"),
]

# ─────────────────────── Ablation configurations ────────────────────────────

ABLATIONS = [
    {
        "name": "baseline",
        "label": "Baseline (original config)",
        "overrides": {},
    },
    {
        "name": "test1_iou_gate_0",
        "label": "Test 1: iou_gate=0.0 (disable IoU gate)",
        "overrides": {"stfs_match_iou_thresh": 0.0},
    },
    {
        "name": "test2_score_thresh_005",
        "label": "Test 2: score_thresh=0.05 (lower seed threshold)",
        "overrides": {"stfs_track_score_thresh": 0.05},
    },
    {
        "name": "test3_l1_only_cost",
        "label": "Test 3: iou_weight=0.0, l1_weight=7.0 (pure L1+cls cost)",
        "overrides": {"stfs_iou_weight": 0.0, "stfs_l1_weight": 7.0},
    },
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


def _load_model(run_dir: Path, cfg: Config) -> VideoRFDETR:
    model = VideoRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    msg = model.load_state_dict(sd, strict=False)
    print(f"  loaded best.pth  missing={len(msg.missing_keys)}  "
          f"unexpected={len(msg.unexpected_keys)}")
    model.eval()
    return model


def _apply_overrides(cfg: Config, overrides: dict) -> None:
    for k, v in overrides.items():
        setattr(cfg, k, v)


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
def _predict_centre(model, frames, postprocess, orig_w, orig_h, centre):
    """frames: (T, 3, H, W) → boxes, scores for the centre frame."""
    images = frames.unsqueeze(0).to(DEVICE)   # (1, T, 3, H, W)
    with torch.amp.autocast("cuda", enabled=True):
        out = model(images, query_mode="student")
    centre_out = {
        "pred_logits": out["pred_logits"][:, centre],
        "pred_boxes":  out["pred_boxes"][:, centre],
    }
    orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
    res = postprocess(centre_out, orig_size)[0]
    return res["boxes"].cpu().numpy(), res["scores"].cpu().numpy()


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

def eval_on_dataset(model, cfg, postprocess, ds_name, img_dir, lbl_dir):
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


# ──────────────────────────── formatting ────────────────────────────────────

def _block(rep: dict, abl_label: str) -> str:
    mac = rep["macro_per_video"]
    mic = rep["micro_pooled"]
    lines = [
        f"  Dataset : {rep['dataset']}  ({rep['n_videos']} videos, "
        f"{rep['n_centre_frames']} centre frames)",
        f"  Ablation: {abl_label}",
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


def _comparison_table(all_results: list) -> str:
    """Print a concise side-by-side comparison for each dataset."""
    # Gather dataset names
    ds_names = list({rep["dataset"] for _, _, reps in all_results for rep in reps})
    lines = ["\n" + "=" * 100,
             "COMPARISON TABLE  (MICRO pooled metrics)",
             "=" * 100]
    for ds in sorted(ds_names):
        lines.append(f"\nDataset: {ds}")
        header = f"  {'Ablation':<42s}  {'AP30':>6}  {'AP50':>6}  {'AP75':>6}  " \
                 f"{'F1':>6}  {'P':>6}  {'R':>6}  {'Frag':>5}  {'FragRt':>7}"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        for abl_name, abl_label, reps in all_results:
            rep = next((r for r in reps if r["dataset"] == ds), None)
            if rep is None:
                continue
            mic = rep["micro_pooled"]
            lines.append(
                f"  {abl_label:<42s}  {mic['AP30']:>6.4f}  {mic['AP50']:>6.4f}  "
                f"{mic['AP75']:>6.4f}  {mic['F1']:>6.4f}  {mic['P']:>6.4f}  "
                f"{mic['R']:>6.4f}  {mic['Frag']:>5d}  {mic['FragRate']:>7.4f}"
            )
    return "\n".join(lines) + "\n"


# ────────────────────────────────── main ────────────────────────────────────

def main():
    run_dir = ROOT / "rfdetr_video" / "runs" / RUN_NAME
    print(f"\n{'#' * 80}")
    print(f"# STFS Ablation Tests – {RUN_NAME}")
    print(f"# {run_dir}")
    print(f"{'#' * 80}\n")

    base_cfg = _load_cfg(run_dir)
    print(f"  Base cfg: T={base_cfg.T}  img_size={base_cfg.img_size}")
    print(f"  Base STFS: iou_gate={base_cfg.stfs_match_iou_thresh}  "
          f"score_thresh={base_cfg.stfs_track_score_thresh}  "
          f"iou_w={base_cfg.stfs_iou_weight}  l1_w={base_cfg.stfs_l1_weight}  "
          f"cls_w={base_cfg.stfs_cls_weight}  min_track_len={base_cfg.stfs_min_track_len}\n")

    # Load model once (weights are shared; only cfg pointers change at inference).
    model = _load_model(run_dir, base_cfg)
    _criterion, postprocess = build_criterion(base_cfg)

    all_results = []   # list of (abl_name, abl_label, [rep_per_dataset])
    full_text_parts = []

    for abl in ABLATIONS:
        abl_name  = abl["name"]
        abl_label = abl["label"]
        overrides = abl["overrides"]

        # Apply overrides to the live cfg (model reads cfg at runtime from self.cfg).
        _apply_overrides(model.cfg, overrides)

        print(f"\n{'─' * 70}")
        print(f"  Running: {abl_label}")
        if overrides:
            for k, v in overrides.items():
                print(f"    override: {k} = {v}")
        print()

        reps = []
        for ds_name, img_dir, lbl_dir in DATASETS:
            print(f"  ── dataset: {ds_name}")
            rep = eval_on_dataset(model, model.cfg, postprocess,
                                  ds_name, img_dir, lbl_dir)
            reps.append(rep)
            print(_block(rep, abl_label))

        all_results.append((abl_name, abl_label, reps))

        # Build per-ablation section for the text report.
        part_lines = [f"\n{'=' * 80}", f"ABLATION: {abl_label}",
                      f"overrides: {overrides if overrides else '(none)'}",
                      "=" * 80]
        for rep in reps:
            part_lines.append(_block(rep, abl_label))
        full_text_parts.append("\n".join(part_lines))

        # Restore cfg to baseline after each ablation.
        _apply_overrides(model.cfg, {k: getattr(base_cfg, k) for k in overrides})

    # ── Comparison table ────────────────────────────────────────────
    comparison = _comparison_table(all_results)
    print(comparison)

    # ── Save text report ────────────────────────────────────────────
    header = (
        f"STFS Ablation Tests – {RUN_NAME}\n"
        f"Run dir: {run_dir}\n"
        f"Base STFS params: "
        f"iou_gate={base_cfg.stfs_match_iou_thresh}  "
        f"score_thresh={base_cfg.stfs_track_score_thresh}  "
        f"iou_w={base_cfg.stfs_iou_weight}  l1_w={base_cfg.stfs_l1_weight}  "
        f"cls_w={base_cfg.stfs_cls_weight}\n"
    )
    full_text = header + "\n".join(full_text_parts) + comparison
    out_txt = run_dir / "ablation_results.txt"
    out_txt.write_text(full_text)
    print(f"  wrote → {out_txt}")

    # ── Save JSON ───────────────────────────────────────────────────
    json_out = {
        "run": RUN_NAME,
        "base_cfg": {
            "stfs_match_iou_thresh": base_cfg.stfs_match_iou_thresh,
            "stfs_track_score_thresh": base_cfg.stfs_track_score_thresh,
            "stfs_iou_weight": base_cfg.stfs_iou_weight,
            "stfs_l1_weight": base_cfg.stfs_l1_weight,
            "stfs_cls_weight": base_cfg.stfs_cls_weight,
            "stfs_min_track_len": base_cfg.stfs_min_track_len,
        },
        "ablations": [
            {
                "name": abl_name,
                "label": abl_label,
                "overrides": dict(abl["overrides"]),
                "datasets": reps,
            }
            for (abl_name, abl_label, reps), abl in zip(all_results, ABLATIONS)
        ],
    }
    out_json = run_dir / "ablation_results.json"
    out_json.write_text(json.dumps(json_out, indent=2))
    print(f"  wrote → {out_json}")


if __name__ == "__main__":
    main()
