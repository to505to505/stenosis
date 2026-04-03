"""Train Faster R-CNN (ViT-Small + SimpleFPN) on dataset2_split."""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.utils.data
from PIL import Image
from torchvision import tv_tensors
from torchvision.transforms import v2 as T

from detect_model import build_faster_rcnn


# ---------------------------------------------------------------------------
# Dataset — YOLO-format labels → Faster R-CNN targets
# ---------------------------------------------------------------------------

class YOLODetectionDataset(torch.utils.data.Dataset):
    """Reads images + YOLO-format .txt labels and returns Faster R-CNN targets."""

    def __init__(self, img_dir: str, label_dir: str, img_size: int = 512,
                 transforms=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.transforms = transforms

        self.img_files = sorted(
            p for p in self.img_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
        )
        print(f"[Dataset] {len(self.img_files)} images from {img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Load YOLO label: class cx cy w h (normalized)
        label_path = self.label_dir / (img_path.stem + ".txt")
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    # Convert YOLO normalized → absolute xyxy
                    x1 = (cx - w / 2) * orig_w
                    y1 = (cy - h / 2) * orig_h
                    x2 = (cx + w / 2) * orig_w
                    y2 = (cy + h / 2) * orig_h
                    # Clamp
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(orig_w, x2)
                    y2 = min(orig_h, y2)
                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(cls + 1)  # YOLO class 0 → RCNN class 1 (0=background)

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        img = tv_tensors.Image(img)
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY",
                                          canvas_size=(orig_h, orig_w))
        target = {"boxes": boxes, "labels": labels,
                  "image_id": idx}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def get_train_transforms(img_size: int = 512):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToDtype(torch.float32, scale=True),
    ])


def get_val_transforms(img_size: int = 512):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToDtype(torch.float32, scale=True),
    ])


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


# ---------------------------------------------------------------------------
# Evaluation helpers (COCO-style AP via torchvision)
# ---------------------------------------------------------------------------

def evaluate_map(model, data_loader, device):
    """Compute mAP@[.5:.95] and mAP@.5 using torchvision MeanAveragePrecision."""
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    metric = MeanAveragePrecision(iou_type="bbox")
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            # Move to CPU for metrics
            preds = []
            gts = []
            for out, tgt in zip(outputs, targets):
                preds.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu(),
                })
                gts.append({
                    "boxes": tgt["boxes"].cpu() if isinstance(tgt["boxes"], torch.Tensor) else tgt["boxes"],
                    "labels": tgt["labels"].cpu() if isinstance(tgt["labels"], torch.Tensor) else tgt["labels"],
                })
            metric.update(preds, gts)
    result = metric.compute()
    return result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    model.train()
    running = {}
    count = 0
    t0 = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            running[k] = running.get(k, 0.0) + v.item()
        count += 1

        if (i + 1) % print_freq == 0:
            elapsed = time.time() - t0
            avg = {k: v / count for k, v in running.items()}
            total_avg = sum(avg.values())
            parts = " | ".join(f"{k}: {v:.4f}" for k, v in avg.items())
            print(f"  [{epoch}][{i+1}/{len(data_loader)}] "
                  f"loss: {total_avg:.4f} ({parts}) "
                  f"[{elapsed:.0f}s]")

    avg = {k: v / count for k, v in running.items()}
    total_avg = sum(avg.values())
    return total_avg, avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../data/dataset2_split")
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--lr-backbone-factor", type=float, default=0.1,
                        help="LR multiplier for backbone (ViT) params")
    parser.add_argument("--pretrained", default="weights/vit_small_encoder_512.pth")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="runs/detect")
    parser.add_argument("--eval-freq", type=int, default=1,
                        help="Evaluate every N epochs")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Save args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Data
    data_dir = Path(args.data_dir)
    train_ds = YOLODetectionDataset(
        data_dir / "train" / "images", data_dir / "train" / "labels",
        img_size=args.img_size, transforms=get_train_transforms(args.img_size),
    )
    val_ds = YOLODetectionDataset(
        data_dir / "valid" / "images", data_dir / "valid" / "labels",
        img_size=args.img_size, transforms=get_val_transforms(args.img_size),
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    print(f"Train: {len(train_ds)} images, {len(train_loader)} batches")
    print(f"Val:   {len(val_ds)} images, {len(val_loader)} batches")

    # Model
    model = build_faster_rcnn(
        num_classes=2,  # background + stenosis
        pretrained_path=args.pretrained,
        img_size=args.img_size,
    )
    model.to(device)

    # Separate LR for backbone vs head
    backbone_params = []
    head_params = []
    for name, p in model.named_parameters():
        if "backbone.vit" in name:
            backbone_params.append(p)
        else:
            head_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.lr_backbone_factor},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    print(f"Backbone params: {sum(p.numel() for p in backbone_params)/1e6:.1f}M "
          f"(lr={args.lr * args.lr_backbone_factor:.1e})")
    print(f"Head params:     {sum(p.numel() for p in head_params)/1e6:.1f}M "
          f"(lr={args.lr:.1e})")

    start_epoch = 0
    best_map = 0.0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_map = ckpt.get("best_map", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_map={best_map:.4f}")

    # Training
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss, train_losses = train_one_epoch(
            model, optimizer, train_loader, device, epoch,
        )
        scheduler.step()

        elapsed = time.time() - t0
        lr_bb = optimizer.param_groups[0]["lr"]
        lr_hd = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch}/{args.epochs-1} — "
              f"train_loss: {train_loss:.4f} — "
              f"lr: bb={lr_bb:.2e} hd={lr_hd:.2e} — "
              f"{elapsed:.0f}s")

        # Save latest checkpoint
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_map": best_map,
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))

        # Evaluate
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            print(f"  Evaluating on val ({len(val_ds)} images)...")
            try:
                result = evaluate_map(model, val_loader, device)
                map50 = result["map_50"].item()
                map5095 = result["map"].item()
                print(f"  mAP@.5: {map50:.4f}  mAP@[.5:.95]: {map5095:.4f}")

                if map50 > best_map:
                    best_map = map50
                    torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
                    print(f"  → New best! (mAP@.5 = {best_map:.4f})")
            except Exception as e:
                print(f"  Eval failed: {e}")

    print(f"\nTraining complete. Best mAP@.5: {best_map:.4f}")
    print(f"Checkpoints saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
