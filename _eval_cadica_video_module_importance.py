"""Eval-only module-importance ablations for one Video RF-DETR run.

This is intentionally separate from ``_eval_cadica_video_ablate.py`` so the
historical comparison table stays stable. It evaluates a single checkpoint on a
chosen video dataset and isolates the runtime contribution of ETF, refinement,
STFS injection, STFS aggregator, STFS shifter, and the 5-candidate sparse
refinement branch.

Usage:
    PYTHONPATH=rf-detr/src PYTHONUNBUFFERED=1 \
      /home/dsa/miniconda3/envs/stenosis/bin/python -u \
      _eval_cadica_video_module_importance.py \
      --run-dir rfdetr_video/runs/stfs_crrcd_v6_etf_postref_centreKD_stfsAlign_stfsShifter
"""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import cv2
import torch

from _eval_cadica_video_ablate import (
    DEVICE,
    IMG_DIR as CADICA_IMG_DIR,
    LBL_DIR as CADICA_LBL_DIR,
    ROOT,
    TrackStats,
    _append_detection,
    _compute_metrics,
    _empty_bucket,
    _identity_inject,
    _load_cfg_video,
    _to_tensor,
    _yolo_xyxy,
)

KEEP = object()


@contextmanager
def patched_runtime(
    model,
    *,
    etf_enabled: bool = True,
    inject_mode: str = "normal",
    aggregator_override: Any = KEEP,
    shifter_override: Any = KEEP,
    disable_candidates: bool = False,
    stats: TrackStats | None = None,
):
    """Temporarily ablate runtime modules without changing checkpoint weights."""
    import rfdetr_video.model as video_model_mod

    original_etf = model.etf
    original_aggregator = model.stfs_aggregator
    original_shifter = model.stfs_shifter
    original_inject = video_model_mod.inject_features

    model.etf = original_etf if etf_enabled else None
    if aggregator_override is not KEEP:
        model.stfs_aggregator = aggregator_override
    if shifter_override is not KEEP:
        model.stfs_shifter = shifter_override

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
        if inject_mode == "identity":
            return _identity_inject(
                query_embed,
                refpoint,
                tracks_per_batch,
                alpha=alpha,
                aggregator=aggregator,
                shifter=shifter,
                return_shift_candidates=return_shift_candidates,
            )
        if inject_mode != "normal":
            raise ValueError(f"unknown inject_mode={inject_mode!r}")

        result = original_inject(
            query_embed,
            refpoint,
            tracks_per_batch,
            alpha=alpha,
            aggregator=aggregator,
            shifter=shifter,
            return_shift_candidates=return_shift_candidates,
        )
        if disable_candidates and return_shift_candidates:
            enriched_emb, enriched_ref, _shift_candidates, inject_mask = result
            return enriched_emb, enriched_ref, None, inject_mask
        return result

    video_model_mod.inject_features = wrapper
    try:
        yield
    finally:
        video_model_mod.inject_features = original_inject
        model.etf = original_etf
        model.stfs_aggregator = original_aggregator
        model.stfs_shifter = original_shifter


def _run_student(model, frames, **runtime_kwargs):
    with patched_runtime(model, **runtime_kwargs):
        return model(frames, query_mode="student")


def _metric_row(metrics: dict) -> str:
    micro = metrics["micro"]
    return (
        f"AP30={micro['AP30']:.4f}  AP50={micro['AP50']:.4f}  "
        f"F1={micro['F1']:.4f}  P={micro['P']:.4f}  R={micro['R']:.4f}"
    )


def _delta(base: dict, ablated: dict, key: str) -> float:
    return float(base["micro"][key] - ablated["micro"][key])


def _write_summary(result: dict, path: Path) -> None:
    variants = result["variants"]
    final = variants["final"]
    pairs = [
        ("ETF", "no_etf_final"),
        ("ETF on first pass", "no_etf_first_pass", "first_pass"),
        ("refinement only", "first_pass", "refine_no_stfs"),
        ("STFS net", "refine_no_stfs"),
        ("STFS aggregator", "stfs_no_aggregator"),
        ("STFS shifter", "stfs_no_shifter"),
        ("5-candidate refinement", "stfs_no_candidate_refine"),
        ("aggregator+shifter stack", "stfs_legacy_no_agg_no_shift"),
    ]

    lines = [
        f"Video RF-DETR module-importance ablation: {result['label']}",
        f"Run dir: {result['run_dir']}",
        f"Dataset: {result['dataset']}",
        f"Images: {result['img_dir']}",
        f"Labels: {result['lbl_dir']}",
        "",
        f"MICRO pooled metrics on {result['dataset']}",
        "",
        f"{'variant':<32s} {'AP30':>8s} {'AP50':>8s} {'F1':>8s} {'P':>8s} {'R':>8s}",
        "-" * 78,
    ]
    for name, metrics in variants.items():
        micro = metrics["micro"]
        lines.append(
            f"{name:<32s} {micro['AP30']:8.4f} {micro['AP50']:8.4f} "
            f"{micro['F1']:8.4f} {micro['P']:8.4f} {micro['R']:8.4f}"
        )

    lines.extend(["", "Importance deltas (positive means the module helps final/AP metric)", ""])
    lines.append(f"{'module':<32s} {'dAP30':>8s} {'dAP50':>8s} {'dF1':>8s}")
    lines.append("-" * 62)
    for item in pairs:
        if len(item) == 2:
            label, ablated_name = item
            base = final
        else:
            label, ablated_name, base_name = item
            base = variants[base_name]
        ablated = variants[ablated_name]
        lines.append(
            f"{label:<32s} {_delta(base, ablated, 'AP30'):8.4f} "
            f"{_delta(base, ablated, 'AP50'):8.4f} {_delta(base, ablated, 'F1'):8.4f}"
        )

    lines.extend(["", "Track stats", ""])
    for key, stats in result["track_stats"].items():
        lines.append(f"{key}: {json.dumps(stats, sort_keys=True)}")

    path.write_text("\n".join(lines) + "\n")


def evaluate_run(
    label: str,
    run_dir: Path,
    *,
    dataset_name: str = "cadica_50plus_new",
    img_dir: Path = CADICA_IMG_DIR,
    lbl_dir: Path = CADICA_LBL_DIR,
    max_sequences: int | None = None,
) -> dict:
    from rfdetr_temporal.dataset import build_sequence_index
    from rfdetr_video.model import VideoRFDETR, build_criterion

    print(f"\n{'=' * 80}\n[{label}] {run_dir}\n{'=' * 80}")
    cfg = _load_cfg_video(run_dir)
    cfg.rfdetr_checkpoint = str(run_dir / "best.pth")
    print(
        "  "
        f"T={cfg.T} img_size={cfg.img_size} etf={cfg.etf_enabled} "
        f"aggregator={cfg.stfs_aggregator_enabled} shifter={cfg.stfs_shifter_enabled} "
        f"postrefKD={getattr(cfg, 'distill_through_refine', False)} "
        f"centreKD={getattr(cfg, 'distill_centre_frame_only', False)}"
    )

    model = VideoRFDETR(cfg).to(DEVICE)
    ckpt = torch.load(run_dir / "best.pth", map_location="cpu", weights_only=False)
    msg = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    print(f"  loaded missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()

    _criterion, postprocess = build_criterion(cfg)
    sequences = build_sequence_index(img_dir)
    if max_sequences is not None:
        sequences = sequences[:max_sequences]
        print(f"  max_sequences={max_sequences}")
    print(f"  dataset={dataset_name} images={img_dir} labels={lbl_dir}")

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
        "stfs_no_aggregator": _empty_bucket(),
        "stfs_no_shifter": _empty_bucket(),
        "stfs_no_candidate_refine": _empty_bucket(),
        "stfs_legacy_no_agg_no_shift": _empty_bucket(),
    }
    stats = {
        "normal": TrackStats(),
        "no_etf": TrackStats(),
    }

    t0 = time.time()
    for seq_idx, (pid, sid, paths) in enumerate(sequences):
        n = len(paths)
        if n == 0:
            continue
        if n < cfg.T:
            windows = [list(paths) + [paths[-1]] * (cfg.T - n)]
        else:
            windows = [paths[start:start + cfg.T] for start in range(n - cfg.T + 1)]

        for window in windows:
            frames, orig_h, orig_w = [], None, None
            for path in window:
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    raise FileNotFoundError(path)
                if orig_h is None:
                    orig_h, orig_w = img.shape[:2]
                frames.append(_to_tensor(img, cfg.img_size, mean, std))

            ft = torch.stack(frames, 0).unsqueeze(0).to(DEVICE)
            key = f"{pid}_v{sid}"
            centre_path = window[centre]
            gt = _yolo_xyxy(lbl_dir / (centre_path.stem + ".txt"), orig_w, orig_h)

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=DEVICE.type == "cuda"):
                out = _run_student(model, ft, stats=stats["normal"])
                _append_detection(
                    variants["final"], postprocess,
                    out["pred_logits"][:, centre], out["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )
                _append_detection(
                    variants["first_pass"], postprocess,
                    out["first_pass"]["pred_logits"][:, centre],
                    out["first_pass"]["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )

                out_no_stfs = _run_student(model, ft, inject_mode="identity")
                _append_detection(
                    variants["refine_no_stfs"], postprocess,
                    out_no_stfs["pred_logits"][:, centre], out_no_stfs["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )

                out_no_etf = _run_student(model, ft, etf_enabled=False, stats=stats["no_etf"])
                _append_detection(
                    variants["no_etf_final"], postprocess,
                    out_no_etf["pred_logits"][:, centre], out_no_etf["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )
                _append_detection(
                    variants["no_etf_first_pass"], postprocess,
                    out_no_etf["first_pass"]["pred_logits"][:, centre],
                    out_no_etf["first_pass"]["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )

                out_no_etf_no_stfs = _run_student(
                    model, ft, etf_enabled=False, inject_mode="identity",
                )
                _append_detection(
                    variants["no_etf_refine_no_stfs"], postprocess,
                    out_no_etf_no_stfs["pred_logits"][:, centre],
                    out_no_etf_no_stfs["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )

                out_no_aggregator = _run_student(model, ft, aggregator_override=None)
                _append_detection(
                    variants["stfs_no_aggregator"], postprocess,
                    out_no_aggregator["pred_logits"][:, centre],
                    out_no_aggregator["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )

                out_no_shifter = _run_student(model, ft, shifter_override=None)
                _append_detection(
                    variants["stfs_no_shifter"], postprocess,
                    out_no_shifter["pred_logits"][:, centre], out_no_shifter["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )

                out_no_candidate = _run_student(model, ft, disable_candidates=True)
                _append_detection(
                    variants["stfs_no_candidate_refine"], postprocess,
                    out_no_candidate["pred_logits"][:, centre], out_no_candidate["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )

                out_legacy = _run_student(
                    model, ft, aggregator_override=None, shifter_override=None,
                )
                _append_detection(
                    variants["stfs_legacy_no_agg_no_shift"], postprocess,
                    out_legacy["pred_logits"][:, centre], out_legacy["pred_boxes"][:, centre],
                    orig_h, orig_w, gt, key,
                )

        if (seq_idx + 1) % 20 == 0 or (seq_idx + 1) == len(sequences):
            print(f"  {seq_idx + 1}/{len(sequences)} sequences  {time.time() - t0:.0f}s")

    result = {
        "label": label,
        "run_dir": str(run_dir),
        "dataset": dataset_name,
        "img_dir": str(img_dir),
        "lbl_dir": str(lbl_dir),
        "variants": {},
        "track_stats": {},
    }
    for variant, bucket in variants.items():
        macro, micro = _compute_metrics(
            bucket["all_dets"], bucket["all_gts"],
            bucket["dets_by_video"], bucket["gts_by_video"],
        )
        result["variants"][variant] = {"macro": macro, "micro": micro}
        print(f"  {variant:<32s} {_metric_row(result['variants'][variant])}")

    result["track_stats"] = {key: stat.summary() for key, stat in stats.items()}
    print("  track stats:")
    for key, stat in result["track_stats"].items():
        print(f"    {key}: {json.dumps(stat, sort_keys=True)}")

    del model
    torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=ROOT / "rfdetr_video" / "runs" / "stfs_crrcd_v6_etf_postref_centreKD_stfsAlign_stfsShifter",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--dataset-name", default="cadica_50plus_new")
    parser.add_argument("--img-dir", type=Path, default=CADICA_IMG_DIR)
    parser.add_argument("--lbl-dir", type=Path, default=CADICA_LBL_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--max-sequences", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    img_dir = args.img_dir if args.img_dir.is_absolute() else ROOT / args.img_dir
    lbl_dir = args.lbl_dir if args.lbl_dir.is_absolute() else ROOT / args.lbl_dir
    label = args.label or run_dir.name
    safe_label = label.replace("/", "_")
    safe_dataset = args.dataset_name.replace("/", "_")
    if args.output is not None:
        output = args.output
    elif args.dataset_name == "cadica_50plus_new":
        output = ROOT / f"_eval_cadica_video_module_importance_{safe_label}.json"
    else:
        output = ROOT / f"_eval_{safe_dataset}_video_module_importance_{safe_label}.json"
    summary_output = args.summary_output or output.with_suffix(".txt")

    result = evaluate_run(
        label,
        run_dir,
        dataset_name=args.dataset_name,
        img_dir=img_dir,
        lbl_dir=lbl_dir,
        max_sequences=args.max_sequences,
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    _write_summary(result, summary_output)
    print(f"\nSaved -> {output}")
    print(f"Saved -> {summary_output}")


if __name__ == "__main__":
    main()