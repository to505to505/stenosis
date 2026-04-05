"""Training script for the Spatio-Temporal Stenosis Detector."""

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import warnings

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Suppress wandb noise
os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

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
    wandb_run=None,
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

            # W&B
            if wandb_run is not None:
                import wandb
                wb_log = {f"train/{k}": v for k, v in avg.items()}
                wb_log["train/lr"] = lr
                wandb.log(wb_log, step=global_step)

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
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--name", type=str, default=None, help="Run name")
    args = parser.parse_args()

    cfg = Config()
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.no_wandb:
        cfg.wandb_enabled = False
    if args.name:
        cfg.run_name = args.name

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    # Seed
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Run name & output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"temporal_{timestamp}"
    run_dir = cfg.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))

    # W&B init
    wandb_run = None
    if cfg.wandb_enabled:
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg.wandb_project,
                name=run_name,
                config=asdict(cfg),
                dir=str(run_dir),
            )
            print(f"[wandb] Logging to project={cfg.wandb_project}, run={run_name}")
        except Exception as e:
            print(f"[wandb] Init failed: {e}. Continuing without W&B.")
            wandb_run = None

    # Metrics CSV
    csv_path = run_dir / "metrics.csv"
    csv_fields = [
        "epoch", "train/total_loss", "train/rpn_objectness_loss", "train/rpn_box_loss",
        "train/det_cls_loss", "train/det_reg_loss", "train/lr",
        "val/mAP_50", "val/mAP_75", "val/mAP_50_95", "val/mAR",
        "val/F1", "val/precision", "val/recall",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    print("=" * 60)
    print("Spatio-Temporal Stenosis Detector — Training")
    print(f"Run: {run_name}")
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
    try:
        for epoch in range(start_epoch, cfg.epochs):
            print(f"\n--- Epoch {epoch+1}/{cfg.epochs} ---")
            global_step = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                device, epoch + 1, cfg, writer, global_step, wandb_run,
            )

            # Collect epoch-level train losses for CSV (last logged averages)
            epoch_row = {"epoch": epoch + 1, "train/lr": optimizer.param_groups[0]["lr"]}

            # Validation — mAP evaluation
            if (epoch + 1) % cfg.eval_interval == 0:
                print("Evaluating...")
                val_metrics = run_evaluation(model, val_loader, device, cfg)
                ap50 = val_metrics["AP@0.5"]
                ap75 = val_metrics["AP@0.75"]
                ap5095 = val_metrics["AP@0.5:0.95"]
                mar = val_metrics["mAR"]
                f1 = val_metrics["F1"]
                precision = val_metrics["precision"]
                recall = val_metrics["recall"]

                print(
                    f"  mAP@0.5={ap50:.4f}  mAP@0.75={ap75:.4f}  "
                    f"mAP@0.5:0.95={ap5095:.4f}  mAR={mar:.4f}\n"
                    f"  F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  "
                    f"dets={val_metrics['num_detections']}  gt={val_metrics['num_gt']}"
                )

                # TensorBoard
                writer.add_scalar("val/mAP_50", ap50, global_step)
                writer.add_scalar("val/mAP_75", ap75, global_step)
                writer.add_scalar("val/mAP_50_95", ap5095, global_step)
                writer.add_scalar("val/mAR", mar, global_step)
                writer.add_scalar("val/F1", f1, global_step)
                writer.add_scalar("val/precision", precision, global_step)
                writer.add_scalar("val/recall", recall, global_step)

                # W&B
                if wandb_run is not None:
                    import wandb
                    wandb.log({
                        "val/mAP_50": ap50,
                        "val/mAP_75": ap75,
                        "val/mAP_50_95": ap5095,
                        "val/mAR": mar,
                        "val/F1": f1,
                        "val/precision": precision,
                        "val/recall": recall,
                        "epoch": epoch + 1,
                    }, step=global_step)

                # CSV row
                epoch_row.update({
                    "val/mAP_50": ap50,
                    "val/mAP_75": ap75,
                    "val/mAP_50_95": ap5095,
                    "val/mAR": mar,
                    "val/F1": f1,
                    "val/precision": precision,
                    "val/recall": recall,
                })

                # Save best by mAP@0.5:0.95
                if ap5095 > best_map:
                    best_map = ap5095
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch,
                         "AP@0.5": ap50, "AP@0.75": ap75,
                         "AP@0.5:0.95": ap5095, "mAR": mar,
                         "F1": f1, "precision": precision, "recall": recall},
                        run_dir / "best.pt",
                    )
                    print(f"  ✓ Saved best model (mAP@0.5:0.95={ap5095:.4f})")

            csv_writer.writerow(epoch_row)
            csv_file.flush()

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

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    finally:
        writer.close()
        csv_file.close()

        # Write best.txt summary
        _write_best_txt(run_dir, cfg, best_map, run_name)

        if wandb_run is not None:
            import wandb
            wandb.finish()

        print(f"\nTraining complete. Best mAP@0.5:0.95: {best_map:.4f}")
        print(f"Checkpoints saved to: {run_dir}")


def _write_best_txt(run_dir: Path, cfg: Config, best_map: float, run_name: str):
    """Write best.txt with best metrics + config (matches RF-DETR format)."""
    best_ckpt = run_dir / "best.pt"
    best_metrics = {}
    if best_ckpt.exists():
        ckpt = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
        for k in ["AP@0.5", "AP@0.75", "AP@0.5:0.95", "mAR", "F1", "precision", "recall", "epoch"]:
            if k in ckpt:
                best_metrics[k] = ckpt[k]

    best_txt = run_dir / "best.txt"
    with open(best_txt, "w") as f:
        f.write(f"Stenosis Temporal Detector — Best Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Run:           {run_name}\n")
        f.write(f"Best mAP50-95: {best_map:.5f}\n")
        f.write(f"Best epoch:    {best_metrics.get('epoch', 'N/A')}\n")

        f.write("\n--- Metrics ---\n")
        for k, v in sorted(best_metrics.items()):
            if k != "epoch":
                f.write(f"  {k:25s}  {v:.5f}\n" if isinstance(v, float) else f"  {k:25s}  {v}\n")

        f.write("\n--- Config ---\n")
        cfg_dict = asdict(cfg)
        for k, v in sorted(cfg_dict.items()):
            f.write(f"  {k:30s}  {v}\n")

    print(f"[INFO] Best metrics saved to {best_txt}")


if __name__ == "__main__":
    main()
