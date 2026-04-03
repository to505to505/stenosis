"""Training script for Faster R-CNN with ViT-Small (VasoMIM) backbone."""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from detect_model import build_faster_rcnn
from detect_dataset import get_dataloader


def cosine_lr(optimizer, base_lr, warmup_iters, total_iters, min_lr=1e-6):
    """Returns a LambdaLR scheduler with linear warmup + cosine decay."""
    def lr_lambda(step):
        if step < warmup_iters:
            return step / max(warmup_iters, 1)
        progress = (step - warmup_iters) / max(total_iters - warmup_iters, 1)
        return max(min_lr / base_lr, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, writer, global_step, amp_enabled):
    model.train()
    running = {}
    t0 = time.time()
    log_every = 50

    for i, (images, targets) in enumerate(loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast("cuda", enabled=amp_enabled):
            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        global_step += 1

        # Accumulate for logging
        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v.item()
        running["total"] = running.get("total", 0.0) + total_loss.item()

        if (i + 1) % log_every == 0:
            elapsed = time.time() - t0
            avg = {k: v / log_every for k, v in running.items()}
            lr = optimizer.param_groups[0]["lr"]
            parts = "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
            print(f"  [{epoch}][{i+1}/{len(loader)}]  {parts}  lr={lr:.2e}  {elapsed:.1f}s")
            for k, v in avg.items():
                writer.add_scalar(f"train/{k}", v, global_step)
            writer.add_scalar("train/lr", lr, global_step)
            running = {}
            t0 = time.time()

    return global_step


@torch.no_grad()
def validate(model, loader, device, amp_enabled):
    """Quick validation — returns average loss dict."""
    model.train()  # keep in train mode to get losses
    totals = {}
    count = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with autocast("cuda", enabled=amp_enabled):
            loss_dict = model(images, targets)
        for k, v in loss_dict.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser("ViTDet Faster R-CNN Training")
    parser.add_argument("--data_root", type=str, default="/home/dsa/stenosis/data/dataset2_split")
    parser.add_argument("--weights", type=str, default="weights/vit_small_encoder_512.pth",
                        help="Path to VasoMIM pretrained encoder weights")
    parser.add_argument("--output_dir", type=str, default="runs/detect")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--freeze_backbone", type=int, default=0,
                        help="Freeze backbone for first N epochs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Output
    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir / "tb_logs"))

    print("=" * 60)
    print("Faster R-CNN  —  ViT-Small (VasoMIM) backbone")
    print("=" * 60)

    # Data
    print("Loading data...")
    train_loader = get_dataloader(args.data_root, "train", args.img_size,
                                  args.batch_size, args.num_workers, shuffle=True)
    val_loader = get_dataloader(args.data_root, "valid", args.img_size,
                                args.batch_size, args.num_workers, shuffle=False)

    # Model
    print("Building model...")
    model = build_faster_rcnn(
        num_classes=2,
        pretrained_path=args.weights,
        img_size=args.img_size,
    )
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {num_params:.1f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR schedule
    iters_per_epoch = len(train_loader)
    total_iters = args.epochs * iters_per_epoch
    warmup_iters = args.warmup_epochs * iters_per_epoch
    scheduler = cosine_lr(optimizer, args.lr, warmup_iters, total_iters)

    scaler = GradScaler("cuda", enabled=args.amp)

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # Backbone freezing
        if epoch < args.freeze_backbone:
            for p in model.backbone.vit.parameters():
                p.requires_grad = False
            if epoch == 0:
                print("  Backbone frozen")
        elif epoch == args.freeze_backbone and args.freeze_backbone > 0:
            for p in model.backbone.vit.parameters():
                p.requires_grad = True
            print("  Backbone unfrozen")

        global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, epoch + 1, writer, global_step, args.amp,
        )

        # Validation
        val_losses = validate(model, val_loader, device, args.amp)
        val_total = sum(val_losses.values())
        parts = "  ".join(f"{k}={v:.4f}" for k, v in val_losses.items())
        print(f"  val: {parts}  total={val_total:.4f}")
        for k, v in val_losses.items():
            writer.add_scalar(f"val/{k}", v, global_step)

        # Save
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": min(best_val_loss, val_total),
            "args": vars(args),
        }

        torch.save(ckpt, run_dir / "last.pt")
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  ★ New best val loss: {val_total:.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, run_dir / f"epoch_{epoch+1}.pt")

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {run_dir}")


if __name__ == "__main__":
    main()
