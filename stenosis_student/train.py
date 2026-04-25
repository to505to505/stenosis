"""Training loop for the stenosis student.

Mirrors the structure of ``rfdetr_temporal/train.py``: AMP, gradient
accumulation, linear warm-up, multi-step LR decay, optional W&B logging.

Usage:
    python -m stenosis_student.train
    python -m stenosis_student.train --epochs 30 --batch-size 4
    python -m stenosis_student.train --no-wandb --img-size 640
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

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_SILENT", "true")

from .config import Config
from .dataset import get_dataloader
from .distill_losses import cosine_distill_loss
from .evaluate import evaluate
from .model import StenosisStudent
from .temporal_consistency import temporal_consistency_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_lr(optimizer, step: int, warmup_iters: int, base_lrs):
    if step >= warmup_iters:
        return
    alpha = step / max(warmup_iters, 1)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * alpha


def write_best_txt(run_dir: Path, best_metrics: dict, best_epoch: int, cfg: Config):
    with open(run_dir / "best.txt", "w") as f:
        f.write("Stenosis Student — Best Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Best mAP50:  {best_metrics.get('AP@0.5', 0):.5f}\n")
        f.write(f"Best epoch:  {best_epoch}\n\n")
        f.write("--- Metrics ---\n")
        for k, v in sorted(best_metrics.items()):
            f.write(f"{k:30s}  {v}\n")
        f.write("\n--- Config ---\n")
        for k, v in sorted(asdict(cfg).items()):
            f.write(f"{str(k):30s}  {v}\n")


def save_train_csv(run_dir: Path, history: list):
    if not history:
        return
    fieldnames = list(history[0].keys())
    with open(run_dir / "train.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(history)


def train(cfg: Config) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    run_dir = Path(cfg.output_dir)
    run_dir = run_dir / (cfg.run_name or f"run_{int(time.time())}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    train_loader = get_dataloader("train", cfg, shuffle=True)
    val_loader = get_dataloader("valid", cfg, shuffle=False)

    model = StenosisStudent(cfg).to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} total")

    # ── Stage-4: optional frozen teacher ───────────────────────────────
    teacher = None
    if cfg.distill_enabled:
        from .teacher import FrozenRFDETRTeacher
        teacher = FrozenRFDETRTeacher(cfg).to(device).eval()
        teacher_stride = float(cfg.stage_strides[
            cfg.fpn_stage_indices[cfg.distill_student_level_idx]
        ])
        print(f"[Teacher] frozen, hidden_dim={teacher.hidden_dim}, "
              f"target student stride={int(teacher_stride)}")

    # Stride of the student FPN level used by the temporal-consistency loss.
    consistency_stride = float(cfg.stage_strides[
        cfg.fpn_stage_indices[cfg.temporal_consistency_level_idx]
    ])

    optimizer = torch.optim.AdamW(
        model.get_param_groups(), weight_decay=cfg.weight_decay,
    )
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.lr_step_milestones), gamma=cfg.lr_gamma,
    )
    scaler = GradScaler(enabled=cfg.amp)

    cfg_dict = asdict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, Path):
            cfg_dict[k] = str(v)
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    if cfg.wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or run_dir.name,
                config=cfg_dict,
            )
        except ImportError:
            print("[WARN] wandb not installed, disabling")
            cfg.wandb_enabled = False

    best_map50 = 0.0
    best_metrics: dict = {}
    best_epoch = 0
    history: list = []
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses: dict = {}
        t0 = time.time()

        for batch_idx, (images, centre_clean, targets, _fnames) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            if centre_clean is not None:
                centre_clean = centre_clean.to(device, non_blocking=True)
            gpu_targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()}
                           for t in targets]

            warmup_lr(optimizer, global_step, cfg.warmup_iters, base_lrs)

            need_extras = cfg.distill_enabled or cfg.temporal_consistency_enabled
            with autocast(enabled=cfg.amp):
                if need_extras:
                    out = model(images, gpu_targets, return_extras=True)
                    losses = out["head_outputs"]
                else:
                    losses = model(images, gpu_targets)
                    out = None

                loss_det = losses["loss"]
                loss_distill = images.new_zeros(())
                loss_temporal = images.new_zeros(())

                if cfg.distill_enabled and teacher is not None:
                    if centre_clean is None:
                        raise RuntimeError(
                            "distill_enabled=True but dataloader returned no "
                            "centre_clean tensor. Make sure the dataset is "
                            "constructed with the same Config."
                        )
                    with torch.no_grad():
                        teacher_feat = teacher(centre_clean)
                    student_feat = out["student_distill_feat"]
                    loss_distill = cosine_distill_loss(student_feat, teacher_feat)

                if cfg.temporal_consistency_enabled:
                    mff = out["multi_frame_fpn_level"]
                    if mff is not None:
                        loss_temporal = temporal_consistency_loss(
                            mff, gpu_targets, cfg, stride=consistency_stride,
                        )

                loss_total = (
                    cfg.det_loss_weight * loss_det
                    + cfg.distill_loss_weight * loss_distill
                    + cfg.temporal_consistency_weight * loss_temporal
                )
                loss = loss_total / cfg.grad_accum_steps
            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            # Aggregate per-batch losses (always include the stage-4 columns
            # so train.csv has a stable schema even when toggles are off).
            step_losses = {
                "loss": float(loss_total.item()),
                "loss_det": float(loss_det.item()),
                "loss_cls": float(losses["loss_cls"].item()),
                "loss_box": float(losses["loss_box"].item()),
                "loss_ctr": float(losses["loss_ctr"].item()),
                "loss_distill": float(loss_distill.item()),
                "loss_temporal": float(loss_temporal.item()),
            }
            for k, v in step_losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            if (batch_idx + 1) % cfg.log_interval == 0:
                print(f"  [{epoch+1}/{cfg.epochs}] step {batch_idx+1}/{len(train_loader)}  "
                      f"loss={step_losses['loss']:.4f}  "
                      f"det={step_losses['loss_det']:.4f}  "
                      f"distill={step_losses['loss_distill']:.4f}  "
                      f"temp={step_losses['loss_temporal']:.4f}")
                if cfg.wandb_enabled:
                    import wandb
                    log = {"train/lr": optimizer.param_groups[0]["lr"]}
                    for k, v in step_losses.items():
                        log[f"train/{k}"] = v
                    wandb.log(log, step=global_step)

        scheduler.step()
        n_batches = max(len(train_loader), 1)
        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.epochs}  train_loss={epoch_losses.get('loss', 0):.4f}  "
              f"time={dt:.1f}s")

        if (epoch + 1) % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, cfg, device)
            record = {"epoch": epoch + 1,
                      "train_loss": epoch_losses.get("loss", 0.0),
                      **metrics}
            history.append(record)
            print(f"  val — mAP50={metrics['AP@0.5']:.4f}  "
                  f"mAP50-95={metrics['AP@0.5:0.95']:.4f}  "
                  f"F1={metrics['F1']:.4f}  val_loss={metrics['val_loss']:.4f}")

            if cfg.wandb_enabled:
                import wandb
                log = {"epoch": epoch + 1, "train_loss": epoch_losses.get("loss", 0.0)}
                for k, v in metrics.items():
                    log[f"val/{k}"] = v
                wandb.log(log, step=global_step)

            if metrics["AP@0.5"] > best_map50:
                best_map50 = metrics["AP@0.5"]
                best_metrics = metrics.copy()
                best_epoch = epoch + 1
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1, **metrics},
                    run_dir / "best.pth",
                )
                print(f"  ★ New best mAP50={best_map50:.4f}")

        torch.save({"model": model.state_dict(), "epoch": epoch + 1},
                   run_dir / "last.pth")

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

    print(f"\nTraining complete. Best mAP50={best_map50:.4f}")
    print(f"Outputs saved to {run_dir}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train Stenosis Student (ConvNeXt-V2 + TSM + FCOS)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--T", type=int, default=9)
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--dataset", type=str, default="data/dataset2_split")
    p.add_argument("--output-dir", type=str, default="stenosis_student/runs")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--hf-model", type=str, default=None)

    # ── Stage-4 toggles ──────────────────────────────────────────────
    p.add_argument("--distill", action="store_true",
                   help="Enable feature distillation from a frozen RF-DETR teacher.")
    p.add_argument("--distill-ckpt", type=str, default=None,
                   help="Override path to the teacher checkpoint.")
    p.add_argument("--temporal-dropout", action="store_true",
                   help="Enable asymmetric centre-frame Cutout (Temporal Dropout).")
    p.add_argument("--temporal-dropout-prob", type=float, default=None)
    p.add_argument("--temporal-consistency", action="store_true",
                   help="Enable InfoNCE temporal-consistency loss.")
    p.add_argument("--lambda-det", type=float, default=None)
    p.add_argument("--lambda-distill", type=float, default=None)
    p.add_argument("--lambda-temporal", type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg_kwargs = dict(
        data_root=Path(args.dataset),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lr_backbone=args.lr_backbone,
        T=args.T,
        centre_index=args.T // 2,
        img_size=args.img_size,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum,
        wandb_enabled=not args.no_wandb,
        run_name=args.run_name,
        amp=not args.no_amp,
        distill_enabled=args.distill,
        temporal_dropout_enabled=args.temporal_dropout,
        temporal_consistency_enabled=args.temporal_consistency,
    )
    if args.hf_model:
        cfg_kwargs["hf_model_id"] = args.hf_model
    if args.distill_ckpt:
        cfg_kwargs["distill_teacher_ckpt"] = args.distill_ckpt
    if args.temporal_dropout_prob is not None:
        cfg_kwargs["temporal_dropout_prob"] = args.temporal_dropout_prob
    if args.lambda_det is not None:
        cfg_kwargs["det_loss_weight"] = args.lambda_det
    if args.lambda_distill is not None:
        cfg_kwargs["distill_loss_weight"] = args.lambda_distill
    if args.lambda_temporal is not None:
        cfg_kwargs["temporal_consistency_weight"] = args.lambda_temporal
    cfg = Config(**cfg_kwargs)
    train(cfg)
