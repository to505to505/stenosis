"""Training script for Stenosis-DetNet.

SGD optimizer, lr=0.002, momentum=0.9, 50 epochs with early stopping (patience=5).
"""

import argparse
import csv
import os
import sys
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stenosis_detnet.config import Config
from stenosis_detnet.dataset import get_dataloader
from stenosis_detnet.evaluate import run_evaluation
from stenosis_detnet.model.detector import StenosisDetNet


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
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

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        optimizer.zero_grad()
        with autocast('cuda', enabled=cfg.amp):
            losses = model(images, targets)

        total_loss = losses["total_loss"]
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        for k, v in losses.items():
            running_losses[k] = running_losses.get(k, 0.0) + v.item()

        global_step += 1

        if (batch_idx + 1) % cfg.log_interval == 0:
            elapsed = time.time() - t0
            n_logged = cfg.log_interval
            avg = {k: v / n_logged for k, v in running_losses.items()}
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [{epoch}][{batch_idx+1}/{len(loader)}]  "
                f"loss={avg['total_loss']:.4f}  "
                f"ga_loc={avg.get('loss_ga_loc',0):.4f}  "
                f"ga_shape={avg.get('loss_ga_shape',0):.4f}  "
                f"rpn_cls={avg.get('loss_rpn_cls',0):.4f}  "
                f"rpn_box={avg.get('loss_rpn_bbox',0):.4f}  "
                f"det_cls={avg.get('det_cls_loss',0):.4f}  "
                f"det_reg={avg.get('det_reg_loss',0):.4f}  "
                f"lr={lr:.6f}  "
                f"time={elapsed:.1f}s"
            )
            for k, v in avg.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            writer.add_scalar("train/lr", lr, global_step)

            if wandb_run is not None:
                import wandb
                wb_log = {f"train/{k}": v for k, v in avg.items()}
                wb_log["train/lr"] = lr
                wandb.log(wb_log, step=global_step)

            running_losses = {}
            t0 = time.time()

    return global_step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = Config()
    if args.wandb_project:
        cfg.wandb_project = args.wandb_project
    if args.no_wandb:
        cfg.wandb_enabled = False
    if args.name:
        cfg.run_name = args.name
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"detnet_{timestamp}"
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

    # CSV
    csv_path = run_dir / "metrics.csv"
    csv_fields = [
        "epoch", "train/total_loss", "train/lr",
        "val/mAP_50", "val/mAP_75", "val/mAP_50_95", "val/mAR",
        "val/F1", "val/precision", "val/recall",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    print("=" * 60)
    print("Stenosis-DetNet — Training")
    print(f"Run: {run_name}")
    print(f"SGD lr={cfg.lr}, momentum={cfg.momentum}, epochs={cfg.epochs}")
    print(f"Early stopping patience={cfg.early_stopping_patience}")
    print("=" * 60)

    # Data
    print("Loading data...")
    train_loader = get_dataloader("train", cfg, shuffle=True)
    val_loader = get_dataloader("valid", cfg, shuffle=False)

    # Model
    print("Building model...")
    model = StenosisDetNet(cfg).to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {num_params:.1f}M")

    # Optimizer: SGD with specified hyperparameters
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    scaler = GradScaler('cuda', enabled=cfg.amp)

    start_epoch = 0
    global_step = 0
    best_map = 0.0
    patience_counter = 0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_map = ckpt.get("best_map", 0.0)
        patience_counter = ckpt.get("patience_counter", 0)
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    try:
        for epoch in range(start_epoch, cfg.epochs):
            print(f"\n--- Epoch {epoch+1}/{cfg.epochs} ---")
            global_step = train_one_epoch(
                model, train_loader, optimizer, scaler,
                device, epoch + 1, cfg, writer, global_step, wandb_run,
            )

            epoch_row = {"epoch": epoch + 1, "train/lr": optimizer.param_groups[0]["lr"]}

            # Validation
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

                writer.add_scalar("val/mAP_50", ap50, global_step)
                writer.add_scalar("val/mAP_75", ap75, global_step)
                writer.add_scalar("val/mAP_50_95", ap5095, global_step)
                writer.add_scalar("val/mAR", mar, global_step)
                writer.add_scalar("val/F1", f1, global_step)

                if wandb_run is not None:
                    import wandb
                    wandb.log({
                        "val/mAP_50": ap50, "val/mAP_75": ap75,
                        "val/mAP_50_95": ap5095, "val/mAR": mar,
                        "val/F1": f1, "val/precision": precision,
                        "val/recall": recall, "epoch": epoch + 1,
                    }, step=global_step)

                epoch_row.update({
                    "val/mAP_50": ap50, "val/mAP_75": ap75,
                    "val/mAP_50_95": ap5095, "val/mAR": mar,
                    "val/F1": f1, "val/precision": precision,
                    "val/recall": recall,
                })

                # Early stopping check
                if ap50 > best_map:
                    best_map = ap50
                    patience_counter = 0
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch,
                         "AP@0.5": ap50, "AP@0.75": ap75,
                         "AP@0.5:0.95": ap5095, "mAR": mar,
                         "F1": f1, "precision": precision, "recall": recall},
                        run_dir / "best.pt",
                    )
                    print(f"  ✓ Saved best model (mAP@0.5={ap50:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{cfg.early_stopping_patience})")

                    if patience_counter >= cfg.early_stopping_patience:
                        print(f"\n  ⏹ Early stopping triggered after {epoch+1} epochs.")
                        break

            csv_writer.writerow(epoch_row)
            csv_file.flush()

            # Checkpoint
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_map": best_map,
                    "patience_counter": patience_counter,
                },
                run_dir / "last.pt",
            )

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")
    finally:
        writer.close()
        csv_file.close()

        if wandb_run is not None:
            import wandb
            wandb.finish()

        print(f"\nTraining complete. Best mAP@0.5: {best_map:.4f}")
        print(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()
