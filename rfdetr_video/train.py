"""Training loop for Video RF-DETR.

Mirrors the argparse surface of :mod:`rfdetr_temporal.train` for the
flags that still apply (``--num-general-queries``, ``--crrcd*``,
``--consistency-weight``). Removed flags (``--neighborhood-k``,
``--cpc*``, ``--temporal-dropout*``, ``--consistency-{offset,top-k,
kl-weight,l1-weight}``, ``--temporal-layers``) are no longer relevant.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from rfdetr.utilities.dynamic_batch_resize import (
    choose_dynamic_batch_size,
    resize_tensor_batch,
)

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

from .config import Config, resolve_distill_frame_indices
from .dataset import get_video_dataloader
from .model import VideoRFDETR, build_criterion
from .evaluate import evaluate
from .consistency import num_consistency_loss
from .distill import (
    VideoFrozenRFDETRTeacher,
    distillation_loss,
    CRRCDLoss,
)


# ─────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────
def write_best_txt(run_dir: Path, best_metrics: dict, best_epoch: int, cfg: Config):
    with open(run_dir / "best.txt", "w") as f:
        f.write("Video RF-DETR Best Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Best mAP30:    {best_metrics.get('AP@0.3', 0):.5f}\n")
        f.write(f"Best mAP50:    {best_metrics.get('AP@0.5', 0):.5f}\n")
        f.write(f"Best epoch:    {best_epoch}\n")
        f.write("\n--- Metrics ---\n")
        for k, v in sorted(best_metrics.items()):
            f.write(f"{k:35s}  {v}\n")
        f.write("\n--- Config ---\n")
        for k, v in sorted(asdict(cfg).items()):
            f.write(f"{str(k):35s}  {v}\n")


def save_train_csv(run_dir: Path, history: list):
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(run_dir / "train.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_lr(optimizer, step: int, warmup_iters: int, base_lrs: list):
    if step >= warmup_iters:
        return
    alpha = step / max(warmup_iters, 1)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * alpha


def _flatten_targets(
    targets_list: List[List[Dict[str, torch.Tensor]]], device, img_size: int,
) -> List[Dict[str, torch.Tensor]]:
    """[B][T] of dicts → flat list of length B*T."""
    flat = []
    for sample in targets_list:
        for t_dict in sample:
            flat.append({
                "boxes": t_dict["boxes"].to(device),
                "labels": t_dict["labels"].to(device),
                "orig_size": torch.tensor(
                    [img_size, img_size], device=device,
                ),
            })
    return flat


def _flatten_predictions(out: dict, B: int, T: int) -> dict:
    """Reshape (B, T, Q, *) → (B*T, Q, *) for SetCriterion."""
    pl = out["pred_logits"].reshape(B * T, *out["pred_logits"].shape[2:])
    pb = out["pred_boxes"].reshape(B * T, *out["pred_boxes"].shape[2:])
    flat = {"pred_logits": pl, "pred_boxes": pb}
    if "aux_outputs" in out:
        flat["aux_outputs"] = out["aux_outputs"]      # already (BT, Q, *)
    if "enc_outputs" in out:
        flat["enc_outputs"] = out["enc_outputs"]      # already (BT, …)
    return flat


def _dynamic_batch_resize_config(cfg: Config) -> dict | None:
    if not cfg.dynamic_batch_resize_enabled:
        return None
    return {
        "min_size": cfg.dynamic_batch_resize_min_size,
        "max_size": cfg.dynamic_batch_resize_max_size,
        "step": cfg.dynamic_batch_resize_step,
        "p": cfg.dynamic_batch_resize_p,
    }


def _select_distill_frames(
    images: torch.Tensor,
    teacher_frames: torch.Tensor,
    frame_indices: List[int] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if frame_indices is None:
        return images, teacher_frames
    return (
        images[:, frame_indices].contiguous(),
        teacher_frames[:, frame_indices].contiguous(),
    )


# ─────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────
def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    run_dir = Path(cfg.output_dir)
    if cfg.run_name:
        run_dir = run_dir / cfg.run_name
    else:
        run_dir = run_dir / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    use_teacher_frame = bool(cfg.distill_enabled)
    train_loader = get_video_dataloader(
        "train", cfg, shuffle=True, with_teacher_frame=use_teacher_frame,
    )
    val_loader = get_video_dataloader("valid", cfg, shuffle=False)

    model = VideoRFDETR(cfg).to(device)
    criterion, postprocess = build_criterion(cfg)
    criterion = criterion.to(device)

    teacher = None
    crrcd_module: CRRCDLoss | None = None
    if cfg.distill_enabled:
        teacher = VideoFrozenRFDETRTeacher(cfg).to(device).eval()
        model.register_teacher_queries(
            teacher.refpoint_embed_weight, teacher.query_feat_weight,
        )
        Q_t = int(teacher.refpoint_embed_weight.shape[0])
        print(
            f"[KD-DETR] Teacher queries registered: Q_specific={Q_t}, "
            f"general_enabled={cfg.distill_general_enabled}, "
            f"Q_general={cfg.distill_num_general_queries}"
        )
        if cfg.crrcd_enabled:
            crrcd_module = CRRCDLoss(
                hidden_dim=int(teacher.hidden_dim),
                relation_dim=int(cfg.crrcd_relation_dim),
                frm_hidden_dim=int(cfg.crrcd_hidden_dim),
                num_fg=int(cfg.crrcd_num_fg),
                num_bg=int(cfg.crrcd_num_bg),
                num_negatives=int(cfg.crrcd_num_negatives),
                temperature=float(cfg.crrcd_temperature),
            ).to(device)
            model.crrcd = crrcd_module
            print(
                f"[CRRCD] Enabled — K_fg={cfg.crrcd_num_fg}, "
                f"K_bg={cfg.crrcd_num_bg}, n_neg={cfg.crrcd_num_negatives}, "
                f"τ={cfg.crrcd_temperature}, β={cfg.crrcd_loss_weight}"
            )

    distill_frame_indices = None
    if cfg.distill_enabled:
        distill_frame_indices = resolve_distill_frame_indices(cfg.T, cfg)
        distill_scope = (
            "all frames" if distill_frame_indices is None
            else f"frame_indices={distill_frame_indices}"
        )
        print(
            f"[KD] distill_centre_frame_only={cfg.distill_centre_frame_only}, "
            f"distill_frame_offsets={cfg.distill_frame_offsets}, "
            f"scope={distill_scope}"
        )
    print(
        f"[Video] etf_enabled={cfg.etf_enabled} "
        f"consistency_enabled={cfg.consistency_enabled}"
    )
    dynamic_resize_config = _dynamic_batch_resize_config(cfg)
    if dynamic_resize_config is not None:
        print(
            "[Video] DynamicBatchResize enabled: "
            f"{cfg.dynamic_batch_resize_min_size}-{cfg.dynamic_batch_resize_max_size}, "
            f"step={cfg.dynamic_batch_resize_step}, p={cfg.dynamic_batch_resize_p}"
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params:,} / {n_total:,} total")

    param_groups = model.get_param_groups()
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.lr_step_milestones), gamma=cfg.lr_gamma,
    )
    scaler = GradScaler(enabled=cfg.amp)

    with open(run_dir / "config.json", "w") as f:
        cfg_dict = asdict(cfg)
        for k, v in cfg_dict.items():
            if isinstance(v, Path):
                cfg_dict[k] = str(v)
        json.dump(cfg_dict, f, indent=2)

    if cfg.wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or run_dir.name,
                config=asdict(cfg),
            )
        except ImportError:
            print("[WARN] wandb not installed, disabling")
            cfg.wandb_enabled = False

    best_map30 = 0.0
    best_metrics: dict = {}
    best_epoch = 0
    history: list = []
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        criterion.train()
        epoch_losses: dict = {}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if use_teacher_frame:
                images, targets_list, teacher_frames, _ = batch
                teacher_frames = teacher_frames.to(device, non_blocking=True)
            else:
                images, targets_list, _ = batch
                teacher_frames = None
            images = images.to(device, non_blocking=True)
            current_img_size = cfg.img_size
            dynamic_resize_size = None
            if dynamic_resize_config is not None:
                rng = random.Random(global_step)
                dynamic_resize_size = choose_dynamic_batch_size(
                    dynamic_resize_config,
                    divisor=32,
                    rng=rng,
                )
                if dynamic_resize_size is not None:
                    images = resize_tensor_batch(images, dynamic_resize_size)
                    current_img_size = int(dynamic_resize_size)
            B, T = images.shape[:2]

            targets_flat = _flatten_targets(targets_list, device, current_img_size)
            warmup_lr(optimizer, global_step, cfg.warmup_iters, base_lrs)

            with autocast(enabled=cfg.amp):
                # ── Branch 1: video forward, multi-frame det loss ──
                out = model(images, query_mode="student")
                pred_flat = _flatten_predictions(out, B, T)
                loss_dict = criterion(pred_flat, targets_flat)
                weight_dict = criterion.weight_dict
                loss = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict if k in weight_dict
                )

                # ── Multi-frame count consistency (L_num) ─────────
                if cfg.consistency_enabled:
                    loss_num = num_consistency_loss(
                        out["pred_logits"],
                        threshold=float(cfg.consistency_threshold),
                        soft_temp=float(cfg.consistency_soft_temp),
                    )
                    loss = loss + cfg.consistency_weight * loss_num
                    loss_dict["loss_num"] = loss_num.detach()

                # ── Branch 2: CRRCD + KD specific (per-frame) ─────
                if cfg.distill_enabled:
                    kd_images, kd_teacher_frames = _select_distill_frames(
                        images, teacher_frames, distill_frame_indices,
                    )
                    with torch.no_grad():
                        t_out = teacher.forward_video(kd_teacher_frames)
                    student_kd_spec = model(
                        kd_images,
                        query_mode="teacher",
                        decoder_inputs={
                            "tgt": t_out["decoder_tgt"],            # (BT, Q, D)
                            "refpoints": t_out["decoder_refpoints"],
                        },
                    )
                    student_hs_spec = model._captured_decoder_hs
                    distill_spec = distillation_loss(
                        student_kd_spec, t_out, cfg,
                    )
                    loss = loss + cfg.distill_loss_weight * distill_spec["loss_distill"]
                    for k, v in distill_spec.items():
                        loss_dict[f"spec/{k}"] = v.detach()

                    if (
                        crrcd_module is not None
                        and student_hs_spec is not None
                        and "decoder_hs" in t_out
                    ):
                        loss_rcd = crrcd_module(
                            teacher_hs=t_out["decoder_hs"],
                            student_hs=student_hs_spec,
                            weights=t_out["foreground_weight"],
                        )
                        loss = loss + cfg.crrcd_loss_weight * loss_rcd
                        loss_dict["spec/loss_crrcd"] = loss_rcd.detach()

                # ── Branch 3: KD general (per-frame) ─────────────
                if cfg.distill_enabled and cfg.distill_general_enabled:
                    Q_g = int(cfg.distill_num_general_queries)
                    gen_q = model.sample_general_queries(
                        Q_g, device=device, dtype=images.dtype,
                    )
                    gen_images, gen_teacher_frames = _select_distill_frames(
                        images, teacher_frames, distill_frame_indices,
                    )
                    with torch.no_grad():
                        t_out_gen = teacher.forward_video_general(
                            gen_teacher_frames,
                            gen_q["refpoint"], gen_q["query_feat"],
                            min_weight=cfg.distill_general_min_weight,
                        )
                    student_kd_gen = model(
                        gen_images,
                        query_mode="general",
                        general_queries=gen_q,
                        decoder_inputs={
                            "tgt": t_out_gen["decoder_tgt"],
                            "refpoints": t_out_gen["decoder_refpoints"],
                        },
                    )
                    distill_gen = distillation_loss(
                        student_kd_gen, t_out_gen, cfg,
                    )
                    gen_w = cfg.distill_loss_weight * cfg.distill_general_loss_weight
                    loss = loss + gen_w * distill_gen["loss_distill"]
                    for k, v in distill_gen.items():
                        loss_dict[f"gen/{k}"] = v.detach()

                loss_scaled = loss / cfg.grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1

            for k, v in loss_dict.items():
                epoch_losses.setdefault(k, 0.0)
                epoch_losses[k] += float(v.item()) if torch.is_tensor(v) else float(v)

            if (batch_idx + 1) % cfg.log_interval == 0:
                print(
                    f"  [{epoch+1}/{cfg.epochs}] step {batch_idx+1}/"
                    f"{len(train_loader)}  loss={loss.item():.4f}"
                )

            if cfg.wandb_enabled and global_step % cfg.log_interval == 0:
                import wandb
                step_log = {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                if dynamic_resize_size is not None:
                    step_log["train/dynamic_batch_resize_size"] = dynamic_resize_size
                for k, v in loss_dict.items():
                    step_log[f"train/{k}"] = float(v.item()) if torch.is_tensor(v) else float(v)
                wandb.log(step_log, step=global_step)

        scheduler.step()

        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        train_loss = sum(
            epoch_losses.get(k, 0) * criterion.weight_dict.get(k, 0)
            for k in epoch_losses if k in criterion.weight_dict
        )

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.epochs}  train_loss={train_loss:.4f}  time={dt:.1f}s")

        if (epoch + 1) % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, criterion, postprocess, cfg, device)
            record = {"epoch": epoch + 1, "train_loss": train_loss, **metrics}
            history.append(record)
            print(
                f"  val — mAP30={metrics['AP@0.3']:.4f}  "
                f"mAP50={metrics['AP@0.5']:.4f}  "

                f"F1={metrics['F1']:.4f}  "
                f"all/mAP30={metrics.get('all/AP@0.3', 0):.4f}  "
                f"val_loss={metrics.get('val_loss', 0):.4f}"
            )
            if cfg.wandb_enabled:
                import wandb
                log_dict = {"epoch": epoch + 1, "train_loss": train_loss}
                for k, v in metrics.items():
                    log_dict[f"val/{k}"] = v
                wandb.log(log_dict, step=global_step)
            if metrics["AP@0.3"] > best_map30:
                best_map30 = metrics["AP@0.3"]
                best_metrics = metrics.copy()
                best_epoch = epoch + 1
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1, **metrics},
                    run_dir / "best.pth",
                )
                write_best_txt(run_dir, best_metrics, best_epoch, cfg)
                print(f"  ★ New best micro mAP@0.3={best_map30:.4f}")
            with open(run_dir / "history.json", "w") as _f:
                json.dump(history, _f, indent=2)
            save_train_csv(run_dir, history)

        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1},
            run_dir / "last.pth",
        )

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_train_csv(run_dir, history)
    write_best_txt(run_dir, best_metrics, best_epoch, cfg)
    print(f"[INFO] Best metrics saved to {run_dir / 'best.txt'}")

    if cfg.wandb_enabled:
        import wandb
        for k, v in best_metrics.items():
            wandb.run.summary[f"best/{k}"] = v
        wandb.run.summary["best/epoch"] = best_epoch
        wandb.finish()

    print(f"\nTraining complete. Best micro mAP@0.3={best_map30:.4f}")
    print(f"Outputs saved to {run_dir}")

    # ── Free VRAM ─────────────────────────────────────────────────────
    del train_loader, val_loader
    del model, criterion, postprocess, optimizer, scaler, scheduler
    if teacher is not None:
        del teacher
    if crrcd_module is not None:
        del crrcd_module
    torch.cuda.empty_cache()

    return run_dir


# ─────────────────────────────────────────────────────────────────────
#  Argparse
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train Video RF-DETR (multi-frame consistency + optional ETF). "
            "Removed flags vs rfdetr_temporal: --neighborhood-k, "
            "--temporal-layers, --cpc*, --temporal-dropout*, "
            "--consistency-{offset,top-k,kl-weight,l1-weight}."
        )
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--dataset", type=str, default="data/dataset2_split")
    p.add_argument("--checkpoint", type=str,
                   default="rfdetr_runs/dataset2_augs/checkpoint_best_total.pth")
    p.add_argument("--output-dir", type=str, default="rfdetr_video/runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--freeze-decoder", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--img-size", type=int, default=None)

    # KD / CRRCD
    p.add_argument("--distill", action="store_true")
    p.add_argument("--no-general", action="store_true")
    p.add_argument("--num-general-queries", type=int, default=None)
    p.add_argument("--distill-teacher-ckpt", type=str, default=None)
    p.add_argument("--crrcd", action="store_true",
                   help="Per-frame CRRCD (requires --distill).")
    p.add_argument("--crrcd-weight", type=float, default=None)
    p.add_argument("--crrcd-num-fg", type=int, default=None)
    p.add_argument("--crrcd-num-bg", type=int, default=None)
    p.add_argument("--crrcd-num-negatives", type=int, default=None)
    p.add_argument("--crrcd-temperature", type=float, default=None)
    distill_frame_group = p.add_mutually_exclusive_group()
    distill_frame_group.add_argument(
        "--distill-centre-frame-only", action="store_true",
        help="Restrict KD/CRRCD branches to the centre frame "
            "of each window (T_kd=1).",
    )
    distill_frame_group.add_argument(
        "--distill-frame-offsets", type=int, nargs="+", default=None,
        metavar="OFFSET",
        help="Restrict KD/CRRCD to frame offsets relative to the centre. "
            "Use '-1 1' to distill only the left and right centre neighbours.",
    )

    # Consistency
    p.add_argument("--consistency-weight", type=float, default=None)
    p.add_argument("--consistency-threshold", type=float, default=None)
    p.add_argument("--no-consistency", action="store_true")

    # Early Temporal Fusion (ETF)
    p.add_argument(
        "--etf", action="store_true",
        help="Enable Early Temporal Fusion: temporal self-attention on "
             "backbone feature maps before the RF-DETR encoder.",
    )
    p.add_argument("--etf-heads", type=int, default=None,
                   help="Number of attention heads in the ETF layer (default 8).")
    p.add_argument("--etf-dropout", type=float, default=None,
                   help="Attention dropout for ETF (default 0.0).")
    p.add_argument("--etf-spatial-radius", type=int, default=None,
                   help="Spatial radius for ETF key/value tokens. Use 0 for "
                        "temporal-only attention, 1 for a 3x3 window.")

    # Temporal Dropout
    p.add_argument("--temporal-dropout", action="store_true",
                   help="Enable train-only temporal frame masking.")
    p.add_argument("--temporal-dropout-prob", type=float, default=None)
    p.add_argument("--temporal-dropout-centre-p", type=float, default=None)
    p.add_argument("--temporal-dropout-neighbour-p", type=float, default=None)
    p.add_argument("--temporal-dropout-radius", type=int, default=None)
    p.add_argument("--temporal-dropout-noise-std", type=float, default=None)

    # Dynamic batch-level resize augmentation
    p.add_argument("--dynamic-batch-resize", action="store_true",
                   help="Resize each train batch to one random square size.")
    p.add_argument("--dynamic-batch-resize-min-size", type=int, default=None)
    p.add_argument("--dynamic-batch-resize-max-size", type=int, default=None)
    p.add_argument("--dynamic-batch-resize-step", type=int, default=None)
    p.add_argument("--dynamic-batch-resize-p", type=float, default=None)
    return p.parse_args()


def _maybe(d, k, v, cast=lambda x: x):
    if v is not None:
        d[k] = cast(v)


if __name__ == "__main__":
    args = parse_args()
    cfg_kwargs = dict(
        data_root=Path(args.dataset),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        rfdetr_checkpoint=args.checkpoint,
        freeze_decoder=args.freeze_decoder,
        wandb_enabled=not args.no_wandb,
        run_name=args.run_name,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum,
        distill_enabled=bool(args.distill),
        distill_general_enabled=bool(args.distill) and not args.no_general,
        consistency_enabled=not args.no_consistency,
    )
    _maybe(cfg_kwargs, "img_size", args.img_size, int)
    _maybe(cfg_kwargs, "distill_num_general_queries", args.num_general_queries, int)
    _maybe(cfg_kwargs, "distill_teacher_ckpt", args.distill_teacher_ckpt, str)
    if args.crrcd:
        cfg_kwargs["crrcd_enabled"] = True
    if args.distill_centre_frame_only:
        cfg_kwargs["distill_centre_frame_only"] = True
    _maybe(
        cfg_kwargs,
        "distill_frame_offsets",
        args.distill_frame_offsets,
        lambda values: tuple(int(value) for value in values),
    )
    _maybe(cfg_kwargs, "crrcd_loss_weight", args.crrcd_weight, float)
    _maybe(cfg_kwargs, "crrcd_num_fg", args.crrcd_num_fg, int)
    _maybe(cfg_kwargs, "crrcd_num_bg", args.crrcd_num_bg, int)
    _maybe(cfg_kwargs, "crrcd_num_negatives", args.crrcd_num_negatives, int)
    _maybe(cfg_kwargs, "crrcd_temperature", args.crrcd_temperature, float)
    _maybe(cfg_kwargs, "consistency_weight", args.consistency_weight, float)
    _maybe(cfg_kwargs, "consistency_threshold", args.consistency_threshold, float)
    if args.etf:
        cfg_kwargs["etf_enabled"] = True
    _maybe(cfg_kwargs, "etf_heads", args.etf_heads, int)
    _maybe(cfg_kwargs, "etf_dropout", args.etf_dropout, float)
    _maybe(cfg_kwargs, "etf_spatial_radius", args.etf_spatial_radius, int)
    if args.temporal_dropout:
        cfg_kwargs["temporal_dropout_enabled"] = True
    _maybe(cfg_kwargs, "temporal_dropout_prob", args.temporal_dropout_prob, float)
    _maybe(cfg_kwargs, "temporal_dropout_centre_p", args.temporal_dropout_centre_p, float)
    _maybe(cfg_kwargs, "temporal_dropout_neighbour_p", args.temporal_dropout_neighbour_p, float)
    _maybe(cfg_kwargs, "temporal_dropout_radius", args.temporal_dropout_radius, int)
    _maybe(cfg_kwargs, "temporal_dropout_noise_std", args.temporal_dropout_noise_std, float)
    if args.dynamic_batch_resize:
        cfg_kwargs["dynamic_batch_resize_enabled"] = True
    _maybe(cfg_kwargs, "dynamic_batch_resize_min_size", args.dynamic_batch_resize_min_size, int)
    _maybe(cfg_kwargs, "dynamic_batch_resize_max_size", args.dynamic_batch_resize_max_size, int)
    _maybe(cfg_kwargs, "dynamic_batch_resize_step", args.dynamic_batch_resize_step, int)
    _maybe(cfg_kwargs, "dynamic_batch_resize_p", args.dynamic_batch_resize_p, float)
    cfg = Config(**cfg_kwargs)
    train(cfg)
