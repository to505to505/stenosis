"""Cadica eval-only ablations for rfdetr_video.

This script isolates where the video distillation signal disappears:

  final                 normal inference: ETF -> first pass -> STFS -> refine
  first_pass            pre-STFS / pre-refinement predictions from the same run
  refine_no_stfs        refinement layer with identity STFS injection
  no_etf_final          normal STFS + refine, but ETF disabled at eval
  no_etf_first_pass     first pass with ETF disabled at eval
  no_etf_refine_no_stfs refinement with both ETF and STFS injection disabled

The runs are existing checkpoints only; no training or package code edits.

Usage:
    PYTHONPATH=rf-detr/src PYTHONUNBUFFERED=1 \
      /home/dsa/miniconda3/envs/stenosis/bin/python -u _eval_cadica_video_ablate.py
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch

ROOT = Path("/home/dsa/stenosis")
DATA_DIR = ROOT / "data" / "cadica_50plus_new"
IMG_DIR = DATA_DIR / "images"
LBL_DIR = DATA_DIR / "labels"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_5095 = np.arange(0.5, 1.0, 0.05)

RUNS = [
    ("video_nodistill_v6_etf", ROOT / "rfdetr_video" / "runs" / "stfs_nodistill_v6_etf"),
    ("video_crrcd_v6_etf", ROOT / "rfdetr_video" / "runs" / "stfs_crrcd_v6_etf"),
]


@dataclass
class TrackStats:
    windows: int = 0
    total_tracks: int = 0
    total_injected_slots: int = 0
    total_centre_injected_slots: int = 0
    windows_with_any_injection: int = 0
    windows_with_centre_injection: int = 0
    best_t_counts: Counter = field(default_factory=Counter)

    def record(self, tracks_per_batch, shape) -> None:
        B, T, Q, _D = shape
        centre = T // 2
        self.windows += B
        any_in_window = [False] * B
        centre_in_window = [False] * B
        for b, tracks in enumerate(tracks_per_batch):
            self.total_tracks += len(tracks)
            for tr in tracks:
                best_q = tr.slots[tr.best_t]
                if best_q < 0:
                    continue
                self.best_t_counts[int(tr.best_t)] += 1
                for t, slot in enumerate(tr.slots):
                    if slot >= 0:
                        continue
                    self.total_injected_slots += 1
                    any_in_window[b] = True
                    if t == centre:
                        self.total_centre_injected_slots += 1
                        centre_in_window[b] = True
        self.windows_with_any_injection += sum(any_in_window)
        self.windows_with_centre_injection += sum(centre_in_window)

    def summary(self) -> dict:
        windows = max(self.windows, 1)
        return {
            "windows": int(self.windows),
            "avg_tracks_per_window": self.total_tracks / windows,
            "avg_injected_slots_per_window": self.total_injected_slots / windows,
            "avg_centre_injected_slots_per_window": self.total_centre_injected_slots / windows,
            "window_any_injection_rate": self.windows_with_any_injection / windows,
            "window_centre_injection_rate": self.windows_with_centre_injection / windows,
            "best_t_counts": {str(k): int(v) for k, v in sorted(self.best_t_counts.items())},
        }


def _load_cfg_video(run_dir: Path):
    from rfdetr_video.config import Config

    with open(run_dir / "config.json") as f:
        raw = json.load(f)
    cfg = Config()
    for key, value in raw.items():
        if hasattr(cfg, key):
            cur = getattr(cfg, key)
            if isinstance(cur, Path) and value is not None:
                value = Path(value)
            try:
                setattr(cfg, key, value)
            except Exception:
                pass
    return cfg


def _to_tensor(img: np.ndarray, size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    arr = np.stack([img, img, img], axis=0)
    return (torch.from_numpy(arr) - mean) / std


def _yolo_xyxy(lbl_path: Path, width: int, height: int) -> np.ndarray:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    lab = np.loadtxt(lbl_path, dtype=np.float32).reshape(-1, 5)
    if lab.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    cx, cy = lab[:, 1] * width, lab[:, 2] * height
    bw, bh = lab[:, 3] * width, lab[:, 4] * height
    return np.column_stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]).astype(np.float32)


def _compute_metrics(all_dets, all_gts, per_video_dets, per_video_gts):
    from rfdetr_temporal.evaluate import evaluate_map, f1_confidence_sweep

    rows = []
    for video in sorted(per_video_dets):
        dets, gts = per_video_dets[video], per_video_gts[video]
        rows.append({
            "video": video,
            "AP30": evaluate_map(dets, gts, 0.3),
            "AP50": evaluate_map(dets, gts, 0.5),
            "AP75": evaluate_map(dets, gts, 0.75),
            "AP5095": float(np.mean([evaluate_map(dets, gts, threshold) for threshold in IOU_5095])),
            "F1": f1_confidence_sweep(dets, gts)[0],
        })
    macro = {metric: float(np.mean([row[metric] for row in rows])) for metric in ("AP30", "AP50", "AP75", "AP5095", "F1")}
    mic_f1, mic_p, mic_r, mic_thr = f1_confidence_sweep(all_dets, all_gts)
    micro = {
        "AP30": evaluate_map(all_dets, all_gts, 0.3),
        "AP50": evaluate_map(all_dets, all_gts, 0.5),
        "AP75": evaluate_map(all_dets, all_gts, 0.75),
        "AP5095": float(np.mean([evaluate_map(all_dets, all_gts, threshold) for threshold in IOU_5095])),
        "F1": mic_f1,
        "P": mic_p,
        "R": mic_r,
        "thr": mic_thr,
    }
    return macro, micro


def _identity_inject(
    query_embed,
    refpoint,
    tracks_per_batch,
    *,
    alpha,
    aggregator=None,
    shifter=None,
    return_shift_candidates: bool = False,
):
    if return_shift_candidates:
        inject_mask = torch.zeros(
            query_embed.shape[:3], dtype=torch.bool, device=query_embed.device,
        )
        return query_embed, refpoint, None, inject_mask
    return query_embed, refpoint


@contextmanager
def patched_inject(mode: str, stats: TrackStats | None = None):
    import rfdetr_video.model as video_model_mod

    original = video_model_mod.inject_features


    def wrapper(
        query_embed,
        refpoint,
        tracks_per_batch,
        *,
        alpha,
        aggregator=None,
        shifter=None,
        return_shift_candidates: bool = False,
    ):
        if stats is not None:
            stats.record(tracks_per_batch, query_embed.shape)
        if mode == "identity":
            return _identity_inject(
                query_embed,
                refpoint,
                tracks_per_batch,
                alpha=alpha,
                aggregator=aggregator,
                shifter=shifter,
                return_shift_candidates=return_shift_candidates,
            )
        return original(
            query_embed,
            refpoint,
            tracks_per_batch,
            alpha=alpha,
            aggregator=aggregator,
            shifter=shifter,
            return_shift_candidates=return_shift_candidates,
        )

    video_model_mod.inject_features = wrapper
    try:
        yield
    finally:
        video_model_mod.inject_features = original


def _append_detection(bucket, postprocess, logits, boxes, orig_h, orig_w, gt, key):
    orig_size = torch.tensor([[orig_h, orig_w]], device=DEVICE)
    res = postprocess({"pred_logits": logits, "pred_boxes": boxes}, orig_size)[0]
    bucket["dets_by_video"][key].append({
        "boxes": res["boxes"].cpu().numpy(),
        "scores": res["scores"].cpu().numpy(),
    })
    bucket["gts_by_video"][key].append(gt)
    bucket["all_dets"].append(bucket["dets_by_video"][key][-1])
    bucket["all_gts"].append(gt)


def _empty_bucket() -> dict:
    return {
        "dets_by_video": defaultdict(list),
        "gts_by_video": defaultdict(list),
        "all_dets": [],
        "all_gts": [],
    }


def evaluate_run(label: str, run_dir: Path) -> dict:
    from rfdetr_temporal.dataset import build_sequence_index
    from rfdetr_video.model import VideoRFDETR, build_criterion

    print(f"\n{'=' * 80}\n[{label}] {run_dir}\n{'=' * 80}")
    cfg = _load_cfg_video(run_dir)
    cfg.rfdetr_checkpoint = str(run_dir / "best.pth")
    print(f"  T={cfg.T}  img_size={cfg.img_size}  etf={cfg.etf_enabled}")

    model = VideoRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    print(f"  loaded missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()

    _criterion, postprocess = build_criterion(cfg)
    sequences = build_sequence_index(IMG_DIR)
    mean = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)
    centre = cfg.T // 2

    variants = {
        "final": _empty_bucket(),
        "first_pass": _empty_bucket(),
        "refine_no_stfs": _empty_bucket(),
        "no_etf_final": _empty_bucket(),
        "no_etf_first_pass": _empty_bucket(),
        "no_etf_refine_no_stfs": _empty_bucket(),
    }
    stats = {
        "normal": TrackStats(),
        "no_etf": TrackStats(),
    }

    original_etf = model.etf
    t0 = time.time()
    for seq_idx, (pid, sid, paths) in enumerate(sequences):
        n = len(paths)
        if n == 0:
            continue
        windows = ([list(paths) + [paths[-1]] * (cfg.T - n)] if n < cfg.T else [paths[start:start + cfg.T] for start in range(n - cfg.T + 1)])
        for window in windows:
            frames, orig_h, orig_w = [], None, None
            for path in window:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if orig_h is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_tensor(img, cfg.img_size, mean, std))
            ft = torch.stack(frames, 0).unsqueeze(0).to(DEVICE)
            key = f"{pid}_v{sid}"
            centre_path = window[centre]
            gt = _yolo_xyxy(LBL_DIR / (centre_path.stem + ".txt"), orig_w, orig_h)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                model.etf = original_etf
                with patched_inject("normal", stats["normal"]):
                    out = model(ft, query_mode="student")
                _append_detection(variants["final"], postprocess, out["pred_logits"][:, centre], out["pred_boxes"][:, centre], orig_h, orig_w, gt, key)
                _append_detection(variants["first_pass"], postprocess, out["first_pass"]["pred_logits"][:, centre], out["first_pass"]["pred_boxes"][:, centre], orig_h, orig_w, gt, key)

                with patched_inject("identity"):
                    out_no_stfs = model(ft, query_mode="student")
                _append_detection(variants["refine_no_stfs"], postprocess, out_no_stfs["pred_logits"][:, centre], out_no_stfs["pred_boxes"][:, centre], orig_h, orig_w, gt, key)

                model.etf = None
                with patched_inject("normal", stats["no_etf"]):
                    out_no_etf = model(ft, query_mode="student")
                _append_detection(variants["no_etf_final"], postprocess, out_no_etf["pred_logits"][:, centre], out_no_etf["pred_boxes"][:, centre], orig_h, orig_w, gt, key)
                _append_detection(variants["no_etf_first_pass"], postprocess, out_no_etf["first_pass"]["pred_logits"][:, centre], out_no_etf["first_pass"]["pred_boxes"][:, centre], orig_h, orig_w, gt, key)

                with patched_inject("identity"):
                    out_no_etf_no_stfs = model(ft, query_mode="student")
                _append_detection(variants["no_etf_refine_no_stfs"], postprocess, out_no_etf_no_stfs["pred_logits"][:, centre], out_no_etf_no_stfs["pred_boxes"][:, centre], orig_h, orig_w, gt, key)
                model.etf = original_etf

        if (seq_idx + 1) % 40 == 0 or (seq_idx + 1) == len(sequences):
            print(f"  {seq_idx + 1}/{len(sequences)}  {time.time() - t0:.0f}s")

    model.etf = original_etf

    result = {"label": label, "run_dir": str(run_dir), "variants": {}, "track_stats": {}}
    for variant, bucket in variants.items():
        macro, micro = _compute_metrics(
            bucket["all_dets"], bucket["all_gts"], bucket["dets_by_video"], bucket["gts_by_video"],
        )
        result["variants"][variant] = {"macro": macro, "micro": micro}
        print(
            f"  {variant:<22s} MICRO AP30={micro['AP30']:.4f} "
            f"AP50={micro['AP50']:.4f} F1={micro['F1']:.4f} | "
            f"MACRO AP30={macro['AP30']:.4f} AP50={macro['AP50']:.4f}"
        )
    result["track_stats"] = {key: stat.summary() for key, stat in stats.items()}
    print("  track stats:")
    for key, stat in result["track_stats"].items():
        print(f"    {key}: {json.dumps(stat)}")
    del model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    results = []
    for label, run_dir in RUNS:
        results.append(evaluate_run(label, run_dir))

    print(f"\n{'=' * 110}")
    print("VIDEO ABLATION SUMMARY — cadica_50plus_new MICRO")
    print(f"{'=' * 110}")
    header = "  {:<25s} {:<22s} {:>7s} {:>7s} {:>7s}"
    print(header.format("run", "variant", "AP30", "AP50", "F1"))
    print("  " + "-" * 105)
    for run in results:
        for variant, metrics in run["variants"].items():
            micro = metrics["micro"]
            print(header.format(run["label"], variant, f"{micro['AP30']:.4f}", f"{micro['AP50']:.4f}", f"{micro['F1']:.4f}"))

    out = ROOT / "_eval_cadica_video_ablate_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()