"""Training loop for Temporal RF-DETR.

Custom training loop (no PyTorch Lightning) since we wrap the RF-DETR
forward pass with temporal feature fusion.

Usage:
    python -m rfdetr_temporal.train
    python -m rfdetr_temporal.train --epochs 30 --batch-size 4
    python -m rfdetr_temporal.train --no-wandb --freeze-decoder
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rf-detr" / "src"))

from .config import Config
from .dataset import get_dataloader
from .model import TemporalRFDETR, _build_criterion
from .evaluate import evaluate


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
    train_loader = get_dataloader("train", cfg, shuffle=True)
    val_loader   = get_dataloader("valid", cfg, shuffle=False)

    # ── model ───────────────────────────────────────────────────────
    model = TemporalRFDETR(cfg).to(device)
    criterion, postprocess = _build_criterion(cfg)
    criterion = criterion.to(device)

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

    # ── wandb ───────────────────────────────────────────────────────
    if cfg.wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name or run_dir.name,
                config=vars(cfg),
            )
        except ImportError:
            print("[WARN] wandb not installed, disabling")
            cfg.wandb_enabled = False

    # ── training loop ───────────────────────────────────────────────
    best_map50 = 0.0
    history = []
    global_step = 0

    for epoch in range(cfg.epochs):
        model.train()
        criterion.train()
        epoch_losses = {}
        t0 = time.time()

        for batch_idx, (images, targets_list, _fnames) in enumerate(train_loader):
            # images: (B, T, 3, H, W)
            images = images.to(device)

            # Centre-frame targets for loss computation
            centre = cfg.T // 2
            centre_targets = []
            for sample_targets in targets_list:
                t = sample_targets[centre]
                centre_targets.append({
                    "boxes": t["boxes"].to(device),
                    "labels": t["labels"].to(device),
                    "orig_size": torch.tensor(
                        [cfg.img_size, cfg.img_size], device=device
                    ),
                })

            # Warmup
            warmup_lr(optimizer, global_step, cfg.warmup_iters, base_lrs)

            with autocast(enabled=cfg.amp):
                outputs = model(images)
                loss_dict = criterion(outputs, centre_targets)
                weight_dict = criterion.weight_dict
                loss = sum(
                    loss_dict[k] * weight_dict[k]
                    for k in loss_dict if k in weight_dict
                )
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
                epoch_losses[k] += v.item()

            if (batch_idx + 1) % cfg.log_interval == 0:
                avg = loss.item()
                print(f"  [{epoch+1}/{cfg.epochs}] step {batch_idx+1}/{len(train_loader)}  loss={avg:.4f}")

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
                wandb.log({"epoch": epoch + 1, "train_loss": train_loss, **metrics})

            # Save best
            if metrics["AP@0.5"] > best_map50:
                best_map50 = metrics["AP@0.5"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        **metrics,
                    },
                    run_dir / "best.pth",
                )
                print(f"  ★ New best mAP50={best_map50:.4f}")

        # Save latest
        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1},
            run_dir / "last.pth",
        )

    # ── save history ────────────────────────────────────────────────
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    if cfg.wandb_enabled:
        import wandb
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
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(
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
        temporal_attn_layers=args.temporal_layers,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum,
    )
    train(cfg)
