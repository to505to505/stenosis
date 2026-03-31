"""Training script for the Spatio-Temporal Stenosis Detector."""

import argparse
import os
import sys
import time
from pathlib import Path

import warnings

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Ensure package imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stenosis_temporal.config import Config
from stenosis_temporal.dataset import get_dataloader
from stenosis_temporal.evaluate import run_evaluation
from stenosis_temporal.model.detector import StenosisTemporalDetector


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup followed by multi-step LR decay."""

    def __init__(self, optimizer, warmup_iters, milestones, gamma=0.1, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.milestones = sorted(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # This method works with iteration count stored in self.last_epoch
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / max(self.warmup_iters, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            factor = 1.0
            for m in self.milestones:
                if self.last_epoch >= m:
                    factor *= self.gamma
            return [base_lr * factor for base_lr in self.base_lrs]


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    cfg: Config,
    writer: SummaryWriter,
    global_step: int,
) -> int:
    model.train()
    running_losses = {}
    t0 = time.time()
    accum = cfg.grad_accum_steps

    optimizer.zero_grad()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        with autocast('cuda', enabled=cfg.amp):
            losses = model(images, targets)

        total_loss = losses["total_loss"] / accum
        scaler.scale(total_loss).backward()

        # Accumulate losses (un-scaled for logging)
        for k, v in losses.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()

        if (batch_idx + 1) % accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()
            global_step += 1

        if (batch_idx + 1) % (cfg.log_interval * accum) == 0:
            elapsed = time.time() - t0
            n_logged = cfg.log_interval * accum
            avg = {k: v / n_logged for k, v in running_losses.items()}
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{epoch}][{batch_idx+1}/{len(loader)}]  "
                f"loss={avg['total_loss']:.4f}  "
                f"rpn_cls={avg.get('rpn_objectness_loss',0):.4f}  "
                f"rpn_box={avg.get('rpn_box_loss',0):.4f}  "
                f"det_cls={avg.get('det_cls_loss',0):.4f}  "
                f"det_reg={avg.get('det_reg_loss',0):.4f}  "
                f"lr={lr:.6f}  "
                f"time={elapsed:.1f}s"
            )
            for k, v in avg.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            running_losses = {}
            t0 = time.time()

    # Handle leftover accumulated gradients
    if len(loader) % accum != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        global_step += 1

    return global_step





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="GPU device index")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    cfg = Config()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Output directory
    run_dir = cfg.output_dir / "train"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))

    print("=" * 60)
    print("Spatio-Temporal Stenosis Detector — Training")
    print("=" * 60)

    # Data
    print("Loading data...")
    train_loader = get_dataloader("train", cfg, shuffle=True)
    val_loader = get_dataloader("valid", cfg, shuffle=False)

    # Model
    print("Building model...")
    model = StenosisTemporalDetector(cfg).to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {num_params:.1f}M")

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # LR scheduler: warmup + multi-step (milestones in iterations)
    iters_per_epoch = len(train_loader)
    step_milestones_iters = [m * iters_per_epoch for m in cfg.lr_step_milestones]
    scheduler = WarmupMultiStepLR(
        optimizer,
        warmup_iters=cfg.warmup_iters,
        milestones=step_milestones_iters,
        gamma=cfg.lr_gamma,
    )

    scaler = GradScaler('cuda', enabled=cfg.amp)

    start_epoch = 0
    global_step = 0
    best_map = 0.0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_map = ckpt.get("best_map", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n--- Epoch {epoch+1}/{cfg.epochs} ---")
        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch + 1, cfg, writer, global_step,
        )

        # Validation — mAP evaluation
        if (epoch + 1) % cfg.eval_interval == 0:
            print("Evaluating (mAP)...")
            val_metrics = run_evaluation(model, val_loader, device, cfg)
            ap50 = val_metrics["AP@0.5"]
            ap5095 = val_metrics["AP@0.5:0.95"]
            print(
                f"  mAP@0.5={ap50:.4f}  mAP@0.5:0.95={ap5095:.4f}  "
                f"dets={val_metrics['num_detections']}  gt={val_metrics['num_gt']}"
            )
            writer.add_scalar("val/mAP@0.5", ap50, global_step)
            writer.add_scalar("val/mAP@0.5:0.95", ap5095, global_step)

            # Save best by mAP@0.5:0.95
            if ap5095 > best_map:
                best_map = ap5095
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch,
                     "AP@0.5": ap50, "AP@0.5:0.95": ap5095},
                    run_dir / "best.pt",
                )
                print(f"  Saved best model (mAP@0.5:0.95={ap5095:.4f})")

        # Periodic checkpoint
        ckpt_path = run_dir / "last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_map": best_map,
            },
            ckpt_path,
        )

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()
