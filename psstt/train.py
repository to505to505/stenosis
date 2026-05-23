"""Training loop for PSSTT.

SGD per the paper (lr=0.02, momentum=0.9, wd=1e-4) with a 500-iter linear
warmup, then either cosine or multistep decay. Reuses
:mod:`rfdetr_video.ema`, :mod:`rfdetr_video.schedule`, and
:mod:`rfdetr_video.selection` so model selection / early stopping behave
identically to the RF-DETR-video runs we compare against.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext as _nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

from rfdetr_video.ema import ModelEMA
from rfdetr_video.schedule import build_scheduler
from rfdetr_video.selection import (
    EarlyStopper,
    SmoothedTracker,
    composite_selection_score,
)

from .config import Config
from .dataset import get_dataloader
from .evaluate import evaluate
from .model import VideoFasterRCNN


# ─────────────────────────────────────────────────────────────────────
#  Helpers (mirror rfdetr_video.train)
# ─────────────────────────────────────────────────────────────────────
def write_best_txt(run_dir: Path, best_metrics: dict, best_epoch: int, cfg: Config):
    with open(run_dir / "best.txt", "w") as f:
        f.write("PSSTT Best Results\n")
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


# ─────────────────────────────────────────────────────────────────────
#  Train
# ─────────────────────────────────────────────────────────────────────
def train(cfg: Config) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    run_dir = Path(cfg.output_dir)
    if cfg.run_name:
        run_dir = run_dir / cfg.run_name
    else:
        run_dir = run_dir / f"run_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {run_dir}")

    train_loader = get_dataloader("train", cfg, shuffle=True)
    val_loader = get_dataloader("valid", cfg, shuffle=False)

    model = VideoFasterRCNN(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_params:,} / {n_total:,} total")
    print(f"[PSSTT] supervise_all_frames={cfg.supervise_all_frames}")

    param_groups = model.get_param_groups()
    optimizer = torch.optim.SGD(
        param_groups,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_enabled else None

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

    best_sel = float("-inf")
    best_metrics: dict = {}
    best_epoch = 0
    history: list = []
    global_step = 0
    sel_tracker = SmoothedTracker(cfg.selection_smooth_k)
    early_stopper = (
        EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)
        if cfg.early_stop_enabled else None
    )

    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses: Dict[str, float] = {}
        t0 = time.time()
        stop_early = False

        for batch_idx, batch in enumerate(train_loader):
            images, targets_list, _ = batch
            images = images.to(device, non_blocking=True)
            # targets are already on CPU per-frame dicts; model moves them.

            warmup_lr(optimizer, global_step, cfg.warmup_iters, base_lrs)

            with torch.amp.autocast("cuda", enabled=cfg.amp):
                loss_dict = model(images, targets=targets_list)
                loss = sum(loss_dict.values())
                loss_scaled = loss / cfg.grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)

            global_step += 1

            for k, v in loss_dict.items():
                epoch_losses.setdefault(k, 0.0)
                epoch_losses[k] += float(v.detach().item())

            if (batch_idx + 1) % cfg.log_interval == 0:
                comp = " ".join(
                    f"{k.replace('loss_', '')}={float(v.detach().item()):.3f}"
                    for k, v in loss_dict.items()
                )
                print(
                    f"  [{epoch+1}/{cfg.epochs}] step {batch_idx+1}/"
                    f"{len(train_loader)}  loss={loss.item():.4f}  ({comp})"
                )

            if cfg.wandb_enabled and global_step % cfg.log_interval == 0:
                import wandb
                step_log = {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                }
                for k, v in loss_dict.items():
                    step_log[f"train/{k}"] = float(v.detach().item())
                wandb.log(step_log, step=global_step)

        scheduler.step()

        n_batches = len(train_loader)
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        train_loss = sum(epoch_losses.values())
        dt = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.epochs}  train_loss={train_loss:.4f}  time={dt:.1f}s")

        if (epoch + 1) % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, cfg, device)
            if ema is not None:
                with ema.applied_to(model):
                    ema_metrics = evaluate(model, val_loader, cfg, device)
            else:
                ema_metrics = metrics
            sel = composite_selection_score(ema_metrics, cfg.selection_weights)
            sel_smoothed = sel_tracker.add(sel)
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **metrics,
                **{f"ema/{k}": v for k, v in ema_metrics.items()},
                "sel": sel,
                "sel_smoothed": sel_smoothed,
            }
            history.append(record)
            print(
                f"  val — mAP30={metrics['AP@0.3']:.4f} "
                f"| ema mAP30={ema_metrics['AP@0.3']:.4f} "
                f"ema mAP50={ema_metrics['AP@0.5']:.4f} "
                f"ema F1={ema_metrics['F1']:.4f} "
                f"| sel={sel:.4f} sel_smoothed={sel_smoothed:.4f}"
            )
            if cfg.wandb_enabled:
                import wandb
                log_dict = {"epoch": epoch + 1, "train_loss": train_loss,
                            "sel": sel, "sel_smoothed": sel_smoothed}
                for k, v in metrics.items():
                    log_dict[f"val/{k}"] = v
                for k, v in ema_metrics.items():
                    log_dict[f"val_ema/{k}"] = v
                wandb.log(log_dict, step=global_step)

            if sel_smoothed > best_sel:
                best_sel = sel_smoothed
                best_metrics = {**ema_metrics, "sel_smoothed": sel_smoothed}
                best_epoch = epoch + 1
                ema_ctx = ema.applied_to(model) if ema is not None else _nullcontext()
                with ema_ctx:
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch + 1,
                         **ema_metrics},
                        run_dir / "best.pth",
                    )
                write_best_txt(run_dir, best_metrics, best_epoch, cfg)
                print(f"  ★ New best smoothed sel={best_sel:.4f} (epoch {epoch + 1})")
            with open(run_dir / "history.json", "w") as _f:
                json.dump(history, _f, indent=2)
            save_train_csv(run_dir, history)
            if early_stopper is not None and early_stopper.update(sel_smoothed):
                print(
                    f"  ⨯ Early stop — no smoothed-sel improvement for "
                    f"{cfg.early_stop_patience} evals"
                )
                stop_early = True

        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1},
            run_dir / "last.pth",
        )
        if ema is not None:
            with ema.applied_to(model):
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1},
                    run_dir / "last_ema.pth",
                )
        if stop_early:
            break

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    save_train_csv(run_dir, history)
    if best_metrics:
        write_best_txt(run_dir, best_metrics, best_epoch, cfg)
        print(f"[INFO] Best metrics saved to {run_dir / 'best.txt'}")
    else:
        print("[WARN] No evaluation ran — no best.pth/best.txt written")

    if cfg.wandb_enabled:
        import wandb
        for k, v in best_metrics.items():
            wandb.run.summary[f"best/{k}"] = v
        wandb.run.summary["best/epoch"] = best_epoch
        wandb.finish()

    print(
        f"\nTraining complete. Best EMA mAP@0.3="
        f"{best_metrics.get('AP@0.3', 0):.4f} "
        f"(smoothed sel={best_sel:.4f}, epoch {best_epoch})"
    )
    print(f"Outputs saved to {run_dir}")

    del train_loader, val_loader, model, optimizer, scaler, scheduler
    torch.cuda.empty_cache()
    return run_dir


# ─────────────────────────────────────────────────────────────────────
#  Argparse
# ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train PSSTT on dataset2_split")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--centre-only", action="store_true",
                   help="Train only the centre frame's predictions (memory saver)")
    return p.parse_args()


def cfg_from_args(args) -> Config:
    cfg = Config()
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.grad_accum is not None:
        cfg.grad_accum_steps = args.grad_accum
    if args.lr is not None:
        cfg.lr = args.lr
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.T is not None:
        cfg.T = args.T
    if args.img_size is not None:
        cfg.img_size = args.img_size
    if args.no_wandb:
        cfg.wandb_enabled = False
    if args.no_pretrained:
        cfg.pretrained_coco = False
    if args.centre_only:
        cfg.supervise_all_frames = False
    return cfg


def main():
    args = parse_args()
    cfg = cfg_from_args(args)
    train(cfg)


if __name__ == "__main__":
    main()
