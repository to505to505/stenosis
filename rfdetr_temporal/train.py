"""Training loop for Temporal RF-DETR.

Custom training loop (no PyTorch Lightning) since we wrap the RF-DETR
forward pass with temporal feature fusion.

Usage:
    python -m rfdetr_temporal.train
    python -m rfdetr_temporal.train --epochs 30 --batch-size 4
    python -m rfdetr_temporal.train --no-wandb --freeze-decoder
"""

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

from .config import Config
from .dataset import get_dataloader
from .model import TemporalRFDETR, _build_criterion
from .evaluate import evaluate
from .distill import FrozenRFDETRTeacher, distillation_loss, CRRCDLoss
from .consistency_loss import temporal_consistency_loss


def write_best_txt(run_dir: Path, best_metrics: dict, best_epoch: int, cfg: Config):
    """Write best.txt with best metrics and training config."""
    with open(run_dir / "best.txt", "w") as f:
        f.write("Temporal RF-DETR Best Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Best mAP50:    {best_metrics.get('AP@0.5', 0):.5f}\n")
        f.write(f"Best epoch:    {best_epoch}\n")

        f.write("\n--- Metrics ---\n")
        for k, v in sorted(best_metrics.items()):
            f.write(f"{k:35s}  {v}\n")

        f.write("\n--- Config ---\n")
        for k, v in sorted(asdict(cfg).items()):
            f.write(f"{str(k):35s}  {v}\n")


def save_train_csv(run_dir: Path, history: list):
    """Save training history as train.csv."""
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


def train(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # ── output dir ──────────────────────────────────────────────────
    run_dir = Path(cfg.output_dir)
    if cfg.run_name:
        run_dir = run_dir / cfg.run_name
    else:
        run_dir = run_dir / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    # ── data ────────────────────────────────────────────────────────
    use_teacher_frame = bool(cfg.distill_enabled)
    use_paired = bool(cfg.consistency_enabled)
    if use_paired:
        centre = cfg.T // 2
        assert centre + int(cfg.consistency_offset) < cfg.T, (
            f"consistency_offset={cfg.consistency_offset} too large for T={cfg.T}: "
            f"centre+offset must be < T (centre={centre})"
        )
    train_loader = get_dataloader(
        "train", cfg, shuffle=True,
        with_teacher_frame=use_teacher_frame,
        with_paired_window=use_paired,
    )
    val_loader   = get_dataloader("valid", cfg, shuffle=False)

    # ── model ───────────────────────────────────────────────────────
    model = TemporalRFDETR(cfg).to(device)
    criterion, postprocess = _build_criterion(cfg)
    criterion = criterion.to(device)

    # ── KD-DETR teacher (specific + general sampling) ──────────────
    teacher = None
    crrcd_module: CRRCDLoss | None = None
    if cfg.distill_enabled:
        teacher = FrozenRFDETRTeacher(cfg).to(device).eval()
        model.register_teacher_queries(
            teacher.refpoint_embed_weight, teacher.query_feat_weight,
        )
        Q_t = int(teacher.refpoint_embed_weight.shape[0])
        print(
            f"[KD-DETR] Teacher queries registered: Q_specific={Q_t}, "
            f"general_enabled={cfg.distill_general_enabled}, "
            f"Q_general={cfg.distill_num_general_queries}"
        )

        # ── CRRCD relational distillation (optional) ─────────────
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
            # Attach to the student so its params are captured by
            # ``model.get_param_groups()`` and saved with the checkpoint.
            model.crrcd = crrcd_module
            print(
                f"[CRRCD] Enabled — K_fg={cfg.crrcd_num_fg}, "
                f"K_bg={cfg.crrcd_num_bg}, n_neg={cfg.crrcd_num_negatives}, "
                f"τ={cfg.crrcd_temperature}, β={cfg.crrcd_loss_weight}"
            )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total  = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params:,} / {n_total:,} total")

    # ── optimizer + scheduler ───────────────────────────────────────
    param_groups = model.get_param_groups()
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg.lr_step_milestones),
        gamma=cfg.lr_gamma,
    )

    scaler = GradScaler(enabled=cfg.amp)

    # ── save config ──────────────────────────────────────────────────
    with open(run_dir / "config.json", "w") as f:
        cfg_dict = asdict(cfg)
        # Convert Path objects to strings for JSON serialization
        for k, v in cfg_dict.items():
            if isinstance(v, Path):
                cfg_dict[k] = str(v)
        json.dump(cfg_dict, f, indent=2)

    # ── wandb ───────────────────────────────────────────────────────
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

    # ── training loop ───────────────────────────────────────────────
    best_map50 = 0.0
    best_metrics = {}
    best_epoch = 0
    history = []
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        criterion.train()
        epoch_losses = {}
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Unpack batch — paired-window mode has two windows per sample.
            images_b = None
            targets_list_b = None
            if use_paired:
                if use_teacher_frame:
                    (images, targets_list, images_b, targets_list_b,
                     teacher_centre, _fnames) = batch
                    teacher_centre = teacher_centre.to(device, non_blocking=True)
                else:
                    (images, targets_list, images_b, targets_list_b,
                     _fnames) = batch
                    teacher_centre = None
                images_b = images_b.to(device, non_blocking=True)
            elif use_teacher_frame:
                images, targets_list, teacher_centre, _fnames = batch
                teacher_centre = teacher_centre.to(device, non_blocking=True)
            else:
                images, targets_list, _fnames = batch
                teacher_centre = None
            images = images.to(device, non_blocking=True)

            # Centre-frame targets for loss computation
            centre = cfg.T // 2

            def _build_centre_targets(tl):
                out = []
                for sample_targets in tl:
                    t = sample_targets[centre]
                    out.append({
                        "boxes": t["boxes"].to(device),
                        "labels": t["labels"].to(device),
                        "orig_size": torch.tensor(
                            [cfg.img_size, cfg.img_size], device=device
                        ),
                    })
                return out

            centre_targets = _build_centre_targets(targets_list)
            centre_targets_b = (
                _build_centre_targets(targets_list_b) if use_paired else None
            )

            # Warmup
            warmup_lr(optimizer, global_step, cfg.warmup_iters, base_lrs)

            with autocast(enabled=cfg.amp):
                # ── Branch 1: detection (student's own queries) ────
                outputs = model(images, query_mode="student")
                # Pull CPC out of the prediction dict so SetCriterion only
                # sees standard keys (pred_logits/pred_boxes/aux/enc_outputs).
                loss_cpc = outputs.pop("loss_cpc", None) if isinstance(outputs, dict) else None
                loss_dict = criterion(outputs, centre_targets)
                weight_dict = criterion.weight_dict
                loss = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict if k in weight_dict
                )

                # ── CPC temporal regulariser (training only) ──────
                if cfg.cpc_enabled and loss_cpc is not None:
                    loss = loss + cfg.cpc_weight * loss_cpc
                    loss_dict["loss_cpc"] = loss_cpc.detach()

                # ── Branch 2 (KD-DETR specific sampling) ──────────
                if cfg.distill_enabled:
                    with torch.no_grad():
                        teacher_out_spec = teacher(teacher_centre)
                    # Inject the teacher's *final* decoder inputs (tgt +
                    # post-topk refpoints) so the student's deformable
                    # decoder samples at the teacher's spatial locations.
                    student_kd_spec = model(
                        images,
                        query_mode="teacher",
                        decoder_inputs={
                            "tgt": teacher_out_spec["decoder_tgt"],
                            "refpoints": teacher_out_spec["decoder_refpoints"],
                        },
                    )
                    # Capture the student's last-layer decoder hidden state
                    # *immediately* after the Branch-2 forward, before any
                    # subsequent forward overwrites it.
                    student_hs_spec = model._captured_decoder_hs
                    distill_spec = distillation_loss(
                        student_kd_spec, teacher_out_spec, cfg,
                    )
                    loss = loss + cfg.distill_loss_weight * distill_spec["loss_distill"]
                    for k, v in distill_spec.items():
                        loss_dict[f"spec/{k}"] = v.detach()

                    # ── CRRCD relational contrastive distillation ───
                    if (
                        crrcd_module is not None
                        and student_hs_spec is not None
                        and "decoder_hs" in teacher_out_spec
                    ):
                        loss_rcd = crrcd_module(
                            teacher_hs=teacher_out_spec["decoder_hs"],
                            student_hs=student_hs_spec,
                            weights=teacher_out_spec["foreground_weight"],
                        )
                        loss = loss + cfg.crrcd_loss_weight * loss_rcd
                        loss_dict["spec/loss_crrcd"] = loss_rcd.detach()

                # ── Branch 3 (KD-DETR general sampling) ───────────
                if cfg.distill_enabled and cfg.distill_general_enabled:
                    Q_g = int(cfg.distill_num_general_queries)
                    gen_q = model.sample_general_queries(
                        Q_g, device=device, dtype=images.dtype,
                    )
                    with torch.no_grad():
                        teacher_out_gen = teacher.forward_general(
                            teacher_centre,
                            gen_q["refpoint"],
                            gen_q["query_feat"],
                            min_weight=cfg.distill_general_min_weight,
                        )
                    student_kd_gen = model(
                        images,
                        query_mode="general",
                        general_queries=gen_q,
                        decoder_inputs={
                            "tgt": teacher_out_gen["decoder_tgt"],
                            "refpoints": teacher_out_gen["decoder_refpoints"],
                        },
                    )
                    distill_gen = distillation_loss(
                        student_kd_gen, teacher_out_gen, cfg,
                    )
                    gen_w = cfg.distill_loss_weight * cfg.distill_general_loss_weight
                    loss = loss + gen_w * distill_gen["loss_distill"]
                    for k, v in distill_gen.items():
                        loss_dict[f"gen/{k}"] = v.detach()

                # ── Sliding-Window Temporal Consistency ──────────
                # Window B: standard detection loss on its own centre frame
                # (no KD branches — KD operates on window A only).
                # Then a third forward of A querying centre+1 produces
                # predictions for the same physical frame as B's centre,
                # which we Hungarian-match against B's centre predictions.
                if cfg.consistency_enabled and images_b is not None:
                    outputs_b = model(images_b, query_mode="student")
                    loss_cpc_b = (
                        outputs_b.pop("loss_cpc", None)
                        if isinstance(outputs_b, dict) else None
                    )
                    loss_dict_b = criterion(outputs_b, centre_targets_b)
                    loss_b = sum(
                        loss_dict_b[k] * weight_dict[k]
                        for k in loss_dict_b if k in weight_dict
                    )
                    loss = loss + loss_b
                    for k, v in loss_dict_b.items():
                        loss_dict[f"B/{k}"] = v.detach()
                    if cfg.cpc_enabled and loss_cpc_b is not None:
                        loss = loss + cfg.cpc_weight * loss_cpc_b
                        loss_dict["B/loss_cpc"] = loss_cpc_b.detach()

                    # Third forward: A queried at centre+offset → predicts
                    # B's centre frame. GT-anchored matching against
                    # outputs_b via the criterion's HungarianMatcher.
                    outputs_a_at_next = model(
                        images, query_mode="student",
                        predict_frame=centre + int(cfg.consistency_offset),
                    )
                    # Strip auxiliary keys CPC may have left behind (CPC
                    # is gated off when predict_frame != centre, but be safe).
                    if isinstance(outputs_a_at_next, dict):
                        outputs_a_at_next.pop("loss_cpc", None)
                    loss_cons = temporal_consistency_loss(
                        outputs_a_at_next, outputs_b,
                        targets=centre_targets_b,
                        matcher=criterion.matcher,
                        kl_weight=float(cfg.consistency_kl_weight),
                        box_l1_weight=float(cfg.consistency_l1_weight),
                    )
                    loss = loss + cfg.consistency_weight * loss_cons
                    loss_dict["loss_cons"] = loss_cons.detach()

                loss_scaled = loss / cfg.grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1

            # Accumulate losses
            for k, v in loss_dict.items():
                epoch_losses.setdefault(k, 0.0)
                epoch_losses[k] += float(v.item())

            if (batch_idx + 1) % cfg.log_interval == 0:
                avg = loss.item()
                print(f"  [{epoch+1}/{cfg.epochs}] step {batch_idx+1}/{len(train_loader)}  loss={avg:.4f}")

            # Per-step wandb logging
            if cfg.wandb_enabled and global_step % cfg.log_interval == 0:
                import wandb
                step_log = {"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}
                for k, v in loss_dict.items():
                    step_log[f"train/{k}"] = v.item()
                wandb.log(step_log, step=global_step)

        scheduler.step()

        # Average epoch losses
        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        train_loss = sum(
            epoch_losses.get(k, 0) * criterion.weight_dict.get(k, 0)
            for k in epoch_losses if k in criterion.weight_dict
        )

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.epochs}  train_loss={train_loss:.4f}  time={dt:.1f}s")

        # ── validation ──────────────────────────────────────────────
        if (epoch + 1) % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, criterion, postprocess, cfg, device)
            record = {"epoch": epoch + 1, "train_loss": train_loss, **metrics}
            history.append(record)

            print(
                f"  val — mAP50={metrics['AP@0.5']:.4f}  "
                f"mAP50-95={metrics['AP@0.5:0.95']:.4f}  "
                f"F1={metrics['F1']:.4f}  "
                f"val_loss={metrics.get('val_loss', 0):.4f}"
            )

            if cfg.wandb_enabled:
                import wandb
                log_dict = {"epoch": epoch + 1, "train_loss": train_loss}
                for k, v in metrics.items():
                    log_dict[f"val/{k}"] = v
                wandb.log(log_dict, step=global_step)

            # Save best
            if metrics["AP@0.5"] > best_map50:
                best_map50 = metrics["AP@0.5"]
                best_metrics = metrics.copy()
                best_epoch = epoch + 1
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        **metrics,
                    },
                    run_dir / "best.pth",
                )
                write_best_txt(run_dir, best_metrics, best_epoch, cfg)
                print(f"  ★ New best mAP50={best_map50:.4f}")

            # Incremental save after every validated epoch
            with open(run_dir / "history.json", "w") as _f:
                json.dump(history, _f, indent=2)
            save_train_csv(run_dir, history)

        # Save latest
        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1},
            run_dir / "last.pth",
        )

    # ── final save (ensures files are up-to-date after last epoch) ──
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_train_csv(run_dir, history)
    write_best_txt(run_dir, best_metrics, best_epoch, cfg)
    print(f"[INFO] Best metrics saved to {run_dir / 'best.txt'}")

    if cfg.wandb_enabled:
        import wandb
        # Log best metrics as wandb summary
        for k, v in best_metrics.items():
            wandb.run.summary[f"best/{k}"] = v
        wandb.run.summary["best/epoch"] = best_epoch
        wandb.finish()

    print(f"\nTraining complete. Best mAP50={best_map50:.4f}")
    print(f"Outputs saved to {run_dir}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train Temporal RF-DETR")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--neighborhood-k", type=int, default=0)
    p.add_argument("--dataset", type=str, default="data/dataset2_split")
    p.add_argument("--checkpoint", type=str,
                   default="rfdetr_runs/dataset2_augs/checkpoint_best_total.pth")
    p.add_argument("--output-dir", type=str, default="rfdetr_temporal/runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--freeze-decoder", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--temporal-layers", type=int, default=2)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--distill", action="store_true",
                   help="Enable KD-DETR distillation (specific sampling).")
    p.add_argument("--no-general", action="store_true",
                   help="Disable KD-DETR general sampling branch.")
    p.add_argument("--num-general-queries", type=int, default=None)
    p.add_argument("--distill-teacher-ckpt", type=str, default=None)
    p.add_argument("--crrcd", action="store_true",
                   help="Enable CRRCD relational contrastive distillation "
                        "(requires --distill).")
    p.add_argument("--crrcd-weight", type=float, default=None,
                   help="β coefficient for the CRRCD loss term.")
    p.add_argument("--crrcd-num-fg", type=int, default=None)
    p.add_argument("--crrcd-num-bg", type=int, default=None)
    p.add_argument("--crrcd-num-negatives", type=int, default=None)
    p.add_argument("--crrcd-temperature", type=float, default=None)
    p.add_argument("--cpc", action="store_true",
                   help="Enable Contrastive Predictive Coding temporal regulariser.")
    p.add_argument("--cpc-weight", type=float, default=None,
                   help="Weight for the CPC loss term (default: 1.0).")
    p.add_argument("--consistency", action="store_true",
                   help="Enable Sliding-Window Temporal Consistency loss "
                        "(uses paired-window dataloader; adds ~2× student "
                        "forwards per step).")
    p.add_argument("--consistency-weight", type=float, default=None,
                   help="Weight for the consistency loss (default: 0.5).")
    p.add_argument("--consistency-top-k", type=int, default=None,
                   help="(deprecated, ignored — GT-anchored matching no longer "
                        "uses top-K filtering).")
    p.add_argument("--consistency-offset", type=int, default=None,
                   help="Frame offset between paired windows (default: 1). "
                        "Must satisfy centre + offset < T, e.g. T=5 → offset∈{1,2}.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_kwargs = dict(
        data_root=Path(args.dataset),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        neighborhood_k=args.neighborhood_k,
        rfdetr_checkpoint=args.checkpoint,
        freeze_decoder=args.freeze_decoder,
        wandb_enabled=not args.no_wandb,
        run_name=args.run_name,
        temporal_attn_layers=args.temporal_layers,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum,
        distill_enabled=bool(args.distill),
        distill_general_enabled=bool(args.distill) and not args.no_general,
    )
    if args.img_size is not None:
        cfg_kwargs["img_size"] = int(args.img_size)
    if args.num_general_queries is not None:
        cfg_kwargs["distill_num_general_queries"] = int(args.num_general_queries)
    if args.distill_teacher_ckpt is not None:
        cfg_kwargs["distill_teacher_ckpt"] = args.distill_teacher_ckpt
    if args.crrcd:
        cfg_kwargs["crrcd_enabled"] = True
    if args.crrcd_weight is not None:
        cfg_kwargs["crrcd_loss_weight"] = float(args.crrcd_weight)
    if args.crrcd_num_fg is not None:
        cfg_kwargs["crrcd_num_fg"] = int(args.crrcd_num_fg)
    if args.crrcd_num_bg is not None:
        cfg_kwargs["crrcd_num_bg"] = int(args.crrcd_num_bg)
    if args.crrcd_num_negatives is not None:
        cfg_kwargs["crrcd_num_negatives"] = int(args.crrcd_num_negatives)
    if args.crrcd_temperature is not None:
        cfg_kwargs["crrcd_temperature"] = float(args.crrcd_temperature)
    if args.cpc:
        cfg_kwargs["cpc_enabled"] = True
    if args.cpc_weight is not None:
        cfg_kwargs["cpc_weight"] = float(args.cpc_weight)
    if args.consistency:
        cfg_kwargs["consistency_enabled"] = True
    if args.consistency_weight is not None:
        cfg_kwargs["consistency_weight"] = float(args.consistency_weight)
    if args.consistency_top_k is not None:
        cfg_kwargs["consistency_top_k"] = int(args.consistency_top_k)
    if args.consistency_offset is not None:
        cfg_kwargs["consistency_offset"] = int(args.consistency_offset)
    cfg = Config(**cfg_kwargs)
    train(cfg)
