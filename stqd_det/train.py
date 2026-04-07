"""Training script for STQD-Det.

AdamW optimizer, lr=2.5e-5, weight_decay=1e-4.
Linear warmup for first 500 iterations.
Batch size 1 (each mini-batch = 9 consecutive frames).
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

from stqd_det.config import Config
from stqd_det.dataset import get_dataloader
from stqd_det.model.detector import STQDDet


def build_lr_scheduler(optimizer, warmup_iters: int, total_iters: int):
    """Linear warmup then constant LR."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_iters:
            return float(current_step) / float(max(1, warmup_iters))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


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

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)

        optimizer.zero_grad()
        with autocast("cuda", enabled=cfg.amp):
            losses = model(images, targets)

        total_loss = losses["total_loss"]

        # Always run the full scaler pipeline so GradScaler can detect
        # inf/nan grads and reduce the AMP scale factor automatically.
        # Without this, non-finite loss cascades indefinitely.
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
        scaler.step(optimizer)   # skips optimizer.step if inf grads detected
        scaler.update()          # halves scale factor on inf, preventing cascade
        scheduler.step()

        if not torch.isfinite(total_loss):
            print(
                f"WARNING: Non-finite loss at step {global_step}, "
                f"scale→{scaler.get_scale():.0f}"
            )
            global_step += 1
            continue

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
                f"cls={avg.get('loss_cls', 0):.4f}  "
                f"l1={avg.get('loss_l1', 0):.4f}  "
                f"giou={avg.get('loss_giou', 0):.4f}  "
                f"consist={avg.get('loss_consistency', 0):.4f}  "
                f"lr={lr:.2e}  time={elapsed:.1f}s"
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
    parser = argparse.ArgumentParser(description="STQD-Det Training")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--num-proposals", type=int, default=None,
                        help="Number of noise proposals per frame (default: 100)")
    parser.add_argument("--decoder-layers", type=int, default=None,
                        help="Number of decoder layers (default: 6)")
    parser.add_argument("--img-size", type=int, default=None,
                        help="Image size H=W (default: 512)")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Disable gradient checkpointing")
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
    if args.data_root:
        cfg.data_root = Path(args.data_root)
    if args.num_proposals:
        cfg.num_proposals = args.num_proposals
    if args.decoder_layers:
        cfg.decoder_layers = args.decoder_layers
    if args.img_size:
        cfg.img_h = args.img_size
        cfg.img_w = args.img_size
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = cfg.run_name or f"stqd_det_{timestamp}"
    run_dir = cfg.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))

    # W&B
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
        except Exception as e:
            print(f"[wandb] Init failed: {e}. Continuing without W&B.")
            wandb_run = None

    # CSV logging
    csv_path = run_dir / "metrics.csv"
    csv_fields = [
        "epoch", "train/total_loss", "train/loss_cls", "train/loss_l1",
        "train/loss_giou", "train/loss_consistency", "train/lr",
        "val/loss", "val/mAP_50", "val/mAP_75",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    csv_writer.writeheader()

    print("=" * 60)
    print("STQD-Det — Spatio-Temporal Quantum Diffusion Detector")
    print(f"Run: {run_name}")
    print(f"AdamW lr={cfg.lr}, warmup={cfg.warmup_iters} iters, epochs={cfg.epochs}")
    print(f"Batch size={cfg.batch_size}, T={cfg.T} frames")
    print(f"Proposals: {cfg.num_proposals}, Decoder layers: {cfg.decoder_layers}")
    print("=" * 60)

    # Data
    print("Loading data...")
    train_loader = get_dataloader("train", cfg, shuffle=True)
    val_loader = get_dataloader("valid", cfg, shuffle=False)

    total_iters = len(train_loader) * cfg.epochs

    # Model
    print("Building model...")
    model = STQDDet(cfg).to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {num_params:.1f}M")

    # GPU warmup: run a dummy forward+backward pass to trigger cudnn
    # benchmark autotuning and CUDA kernel JIT for all conv shapes upfront.
    # Without this, the first ~150 training iterations are 4-6× slower.
    if device.type == "cuda":
        print("  GPU warmup (cuDNN autotuning)...", flush=True)
        model.train()
        _warmup_B, _warmup_T = 1, cfg.T
        _warmup_imgs = torch.randn(
            _warmup_B, _warmup_T, cfg.in_channels, cfg.img_h, cfg.img_w,
            device=device,
        )
        _warmup_targets = [
            [{"boxes": torch.zeros(0, 4, device=device),
              "labels": torch.zeros(0, dtype=torch.long, device=device)}
             for _ in range(_warmup_T)]
        ]
        with autocast("cuda", enabled=cfg.amp):
            _warmup_losses = model(_warmup_imgs, _warmup_targets)
        _warmup_losses["total_loss"].backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        del _warmup_imgs, _warmup_targets, _warmup_losses
        torch.cuda.empty_cache()
        print("  GPU warmup done.", flush=True)

    # Optimizer: AdamW per paper spec
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = build_lr_scheduler(optimizer, cfg.warmup_iters, total_iters)
    scaler = GradScaler("cuda", enabled=cfg.amp)

    # Suppress harmless LambdaLR init warning (it calls step() internally
    # before any optimizer.step() — this is expected and benign).
    warnings.filterwarnings(
        "ignore",
        message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
        category=UserWarning,
    )

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")
    patience_counter = 0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, cfg.epochs):
        print(f"\n{'─' * 40} Epoch {epoch} {'─' * 40}")

        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch, cfg, writer, global_step, wandb_run,
        )

        # Validation (simple loss-based for now)
        if (epoch + 1) % cfg.eval_interval == 0:
            model.eval()
            val_loss = 0.0
            val_count = 0
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    model.train()  # need training mode for loss computation
                    losses = model(images, targets)
                    val_loss += losses["total_loss"].item()
                    val_count += 1
            model.train()

            avg_val_loss = val_loss / max(val_count, 1)
            print(f"  Val loss: {avg_val_loss:.4f}")
            writer.add_scalar("val/loss", avg_val_loss, global_step)

            # CSV logging
            lr = optimizer.param_groups[0]["lr"]
            csv_writer.writerow({
                "epoch": epoch,
                "train/total_loss": "",
                "train/loss_cls": "",
                "train/loss_l1": "",
                "train/loss_giou": "",
                "train/loss_consistency": "",
                "train/lr": f"{lr:.2e}",
                "val/mAP_50": "",
                "val/mAP_75": "",
                "val/loss": f"{avg_val_loss:.4f}",
            })
            csv_file.flush()

            # Early stopping
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                patience_counter = 0
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_loss": best_loss,
                    "config": asdict(cfg),
                }, run_dir / "best.pt")
                # Write best.txt
                with open(run_dir / "best.txt", "w") as f_best:
                    f_best.write(f"epoch: {epoch}\n")
                    f_best.write(f"global_step: {global_step}\n")
                    f_best.write(f"best_val_loss: {best_loss:.6f}\n")
                print(f"  Saved best model (loss={best_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        # Save last.pt every epoch (for easy resume)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_loss": best_loss,
            "config": asdict(cfg),
        }, run_dir / "last.pt")

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_loss": best_loss,
                "config": asdict(cfg),
            }, run_dir / f"epoch_{epoch}.pt")

    csv_file.close()
    writer.close()
    print(f"\nTraining complete. Best val loss: {best_loss:.4f}")
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    main()
