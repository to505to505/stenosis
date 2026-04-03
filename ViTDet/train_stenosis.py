"""Train ViTDet (ViTAE-Small + Faster R-CNN) for stenosis detection.

Logs losses and validation metrics (mAP50, mAP50-95, precision, recall)
to Weights & Biases.

Usage:
    python ViTDet/train_stenosis.py                              # defaults
    python ViTDet/train_stenosis.py --cfg ViTDet/train_cfg.yaml
    python ViTDet/train_stenosis.py --epochs 50 --batch 4
    python ViTDet/train_stenosis.py --no-pretrained
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import yaml
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool

os.environ["WANDB_CONSOLE"] = "off"
os.environ["WANDB_SILENT"] = "true"

import wandb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vitae_backbone import ViTAE_S, build_vitae_small, load_vitae_pretrained

WEIGHTS_PATH = ROOT / "weights" / "ViTAE-S-GPU.pth"
DEFAULT_CFG = ROOT / "train_cfg.yaml"


# ─── Dataset ──────────────────────────────────────────────────────────────────


class StenosisYOLODataset(data.Dataset):
    """Read YOLO-format dataset for Faster R-CNN training.

    Expects:
        root/train/images/*.png
        root/train/labels/*.txt   (cls cx cy w h  — normalized)
    """

    def __init__(self, img_dir, label_dir, transforms=None, img_size=512):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.transforms = transforms
        self.img_size = img_size
        self.img_files = sorted(self.img_dir.glob("*.png"))
        if not self.img_files:
            self.img_files = sorted(self.img_dir.glob("*.jpg"))
        assert len(self.img_files) > 0, f"No images found in {img_dir}"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        label_path = self.label_dir / (img_path.stem + ".txt")

        # Load image as tensor [C, H, W] float32 in [0, 1]
        img = torchvision.io.read_image(str(img_path))  # [C, H, W] uint8
        _, orig_h, orig_w = img.shape
        img = img.float() / 255.0

        # Resize to target size
        if orig_h != self.img_size or orig_w != self.img_size:
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        # Ensure 3 channels (grayscale → RGB)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        # Parse YOLO labels
        boxes = []
        labels = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        # Convert normalized xywh → absolute xyxy
                        x1 = (cx - w / 2) * self.img_size
                        y1 = (cy - h / 2) * self.img_size
                        x2 = (cx + w / 2) * self.img_size
                        y2 = (cy + h / 2) * self.img_size
                        # Clamp to image bounds
                        x1 = max(0.0, min(x1, self.img_size - 1))
                        y1 = max(0.0, min(y1, self.img_size - 1))
                        x2 = max(0.0, min(x2, self.img_size))
                        y2 = max(0.0, min(y2, self.img_size))
                        if x2 > x1 and y2 > y1:
                            boxes.append([x1, y1, x2, y2])
                            labels.append(cls + 1)  # +1 because 0 = background in Faster RCNN

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


class RandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = img.flip(-1)
            w = img.shape[-1]
            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes
        return img, target


class RandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            img = img.flip(-2)
            h = img.shape[-2]
            boxes = target["boxes"]
            boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
            target["boxes"] = boxes
        return img, target


class RandomBrightness:
    def __init__(self, delta=0.3):
        self.delta = delta

    def __call__(self, img, target):
        factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.delta
        img = (img * factor).clamp(0, 1)
        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


def build_transforms(cfg, train=True):
    transforms = []
    if train:
        if cfg.get("hflip", 0) > 0:
            transforms.append(RandomHFlip(cfg["hflip"]))
        if cfg.get("vflip", 0) > 0:
            transforms.append(RandomVFlip(cfg["vflip"]))
        if cfg.get("brightness", 0) > 0:
            transforms.append(RandomBrightness(cfg["brightness"]))
    return Compose(transforms) if transforms else None


# ─── Model ────────────────────────────────────────────────────────────────────


class ViTAEBackboneWithFPN(nn.Module):
    """ViTAE-Small backbone + FPN for Faster R-CNN."""

    def __init__(self, vitae, fpn_out_channels=256):
        super().__init__()
        self.body = vitae
        in_channels = vitae.out_channels  # 384 for ViTAE-Small
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[in_channels] * 4,
            out_channels=fpn_out_channels,
            extra_blocks=LastLevelMaxPool(),
        )
        self.out_channels = fpn_out_channels

    def forward(self, x):
        features = self.body(x)
        return self.fpn(features)


def build_model(num_classes, img_size=512, pretrained_path=None):
    """Build Faster R-CNN with ViTAE-Small backbone."""
    vitae = build_vitae_small(img_size=img_size, pretrained=pretrained_path)

    backbone = ViTAEBackboneWithFPN(vitae, fpn_out_channels=256)

    # Anchor generator for 5 FPN levels (4 from backbone FPN + 1 extra from LastLevelMaxPool)
    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * 5
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # +1 for background class
    model = FasterRCNN(
        backbone,
        num_classes=num_classes + 1,
        rpn_anchor_generator=anchor_generator,
        min_size=img_size,
        max_size=img_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        box_detections_per_img=100,
    )
    return model


# ─── Evaluation ───────────────────────────────────────────────────────────────


def compute_iou(box1, box2):
    """Compute IoU between two sets of boxes [N,4] and [M,4]."""
    x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-6)


def evaluate_map(model, dataloader, device, iou_thresholds=None, score_thresh=0.01):
    """Compute mAP@50, mAP@50:95, precision, recall."""
    if iou_thresholds is None:
        iou_thresholds = torch.arange(0.5, 1.0, 0.05).tolist()

    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for out, tgt in zip(outputs, targets):
                all_preds.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu(),
                })
                all_targets.append({
                    "boxes": tgt["boxes"],
                    "labels": tgt["labels"],
                })

    # Get all classes
    all_classes = set()
    for t in all_targets:
        all_classes.update(t["labels"].tolist())
    for p in all_preds:
        all_classes.update(p["labels"].tolist())
    all_classes = sorted(all_classes)
    if not all_classes:
        return {"mAP50": 0.0, "mAP50-95": 0.0, "precision": 0.0, "recall": 0.0}

    # Per-class, per-threshold AP
    aps = {iou_t: [] for iou_t in iou_thresholds}

    total_tp50 = 0
    total_fp50 = 0
    total_fn50 = 0

    for cls in all_classes:
        # Gather all predictions and targets for this class
        pred_scores = []
        pred_matched = []  # per IoU threshold
        n_gt = 0

        for pred, tgt in zip(all_preds, all_targets):
            gt_mask = tgt["labels"] == cls
            gt_boxes = tgt["boxes"][gt_mask]
            n_gt += len(gt_boxes)

            pred_mask = (pred["labels"] == cls) & (pred["scores"] >= score_thresh)
            p_boxes = pred["boxes"][pred_mask]
            p_scores = pred["scores"][pred_mask]

            if len(p_boxes) == 0:
                continue

            # Sort by score
            order = p_scores.argsort(descending=True)
            p_boxes = p_boxes[order]
            p_scores = p_scores[order]

            if len(gt_boxes) == 0:
                for s in p_scores:
                    pred_scores.append(s.item())
                    pred_matched.append({iou_t: False for iou_t in iou_thresholds})
                continue

            ious = compute_iou(p_boxes, gt_boxes)

            for i in range(len(p_boxes)):
                pred_scores.append(p_scores[i].item())
                matched = {}
                for iou_t in iou_thresholds:
                    if ious.shape[1] > 0:
                        max_iou, max_idx = ious[i].max(dim=0)
                        # Check all gt boxes for this prediction
                        best_iou = ious[i].max().item()
                        matched[iou_t] = best_iou >= iou_t
                    else:
                        matched[iou_t] = False
                pred_matched.append(matched)

        if n_gt == 0:
            continue

        # Sort by score
        if not pred_scores:
            for iou_t in iou_thresholds:
                aps[iou_t].append(0.0)
            total_fn50 += n_gt
            continue

        indices = np.argsort(-np.array(pred_scores))

        for iou_t in iou_thresholds:
            tp = np.zeros(len(indices))
            fp = np.zeros(len(indices))

            # Re-do matching properly per image (greedy)
            # Simplified: use accumulated TP/FP
            tp_count = 0
            fp_count = 0
            for rank, idx in enumerate(indices):
                if pred_matched[idx][iou_t]:
                    tp[rank] = 1
                    tp_count += 1
                else:
                    fp[rank] = 1
                    fp_count += 1

            # Cap TP at n_gt
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / n_gt
            precisions = cum_tp / (cum_tp + cum_fp + 1e-6)

            # AP via all-points interpolation
            mrec = np.concatenate(([0.0], recalls, [1.0]))
            mpre = np.concatenate(([1.0], precisions, [0.0]))
            for i in range(len(mpre) - 1, 0, -1):
                mpre[i - 1] = max(mpre[i - 1], mpre[i])
            idx = np.where(mrec[1:] != mrec[:-1])[0]
            ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
            aps[iou_t].append(ap)

            if abs(iou_t - 0.5) < 1e-6:
                total_tp50 += int(cum_tp[-1]) if len(cum_tp) > 0 else 0
                total_fp50 += int(cum_fp[-1]) if len(cum_fp) > 0 else 0
                total_fn50 += n_gt - (int(cum_tp[-1]) if len(cum_tp) > 0 else 0)

    # Average over classes
    mAP50 = np.mean(aps[0.5]) if aps[0.5] else 0.0
    mAP_per_thresh = [np.mean(aps[t]) if aps[t] else 0.0 for t in iou_thresholds]
    mAP5095 = np.mean(mAP_per_thresh)

    precision = total_tp50 / (total_tp50 + total_fp50 + 1e-6)
    recall = total_tp50 / (total_tp50 + total_fn50 + 1e-6)

    return {
        "mAP50": float(mAP50),
        "mAP50-95": float(mAP5095),
        "precision": float(precision),
        "recall": float(recall),
    }


# ─── Training ─────────────────────────────────────────────────────────────────


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, optimizer, dataloader, device, scaler, epoch):
    model.train()
    total_loss = 0.0
    loss_components = defaultdict(float)
    n_batches = 0

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast("cuda", enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Skip batch if loss is NaN (e.g. all-empty annotations)
        if not torch.isfinite(losses):
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += losses.item()
        for k, v in loss_dict.items():
            loss_components[k] += v.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    return avg_loss, avg_components


def main():
    parser = argparse.ArgumentParser(description="Train ViTDet ViTAE-Small stenosis detector")
    parser.add_argument("--cfg", type=str, default=str(DEFAULT_CFG))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default="vitdet-stenosis")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.cfg) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    for key in ("data", "epochs", "batch", "imgsz", "device", "lr0", "name"):
        val = getattr(args, key)
        if val is not None:
            if key == "data":
                cfg["data_root"] = val
            else:
                cfg[key] = val

    # Resolve paths
    data_root = Path(cfg["data_root"])
    if not data_root.is_absolute():
        data_root = ROOT.parent / data_root
    class_names = cfg["class_names"]
    num_classes = len(class_names)
    img_size = cfg.get("imgsz", 512)
    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch", 4)
    device = torch.device(cfg.get("device", "cuda:0"))
    lr0 = cfg.get("lr0", 0.0001)
    patience = cfg.get("patience", 50)
    seed = cfg.get("seed", 42)
    use_amp = cfg.get("amp", True)
    freeze_epochs = cfg.get("freeze_backbone_epochs", 10)
    backbone_lr_scale = cfg.get("backbone_lr_scale", 0.1)
    save_period = cfg.get("save_period", 10)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Setup save directory
    project = cfg.get("project", "ViTDet/arcade")
    name = cfg.get("name", "train")
    if not Path(project).is_absolute():
        project = str((ROOT.parent / project).resolve())
    save_dir = Path(project) / name
    # Auto-increment name
    if save_dir.exists():
        i = 2
        while (Path(project) / f"{name}{i}").exists():
            i += 1
        save_dir = Path(project) / f"{name}{i}"
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    print(f"[INFO] Save directory: {save_dir}")
    print(f"[INFO] Classes: {class_names} (nc={num_classes})")
    print(f"[INFO] Image size: {img_size}")
    print(f"[INFO] Device: {device}")

    # Build datasets
    train_transforms = build_transforms(cfg, train=True)
    train_ds = StenosisYOLODataset(
        data_root / "train" / "images", data_root / "train" / "labels",
        transforms=train_transforms, img_size=img_size)
    val_ds = StenosisYOLODataset(
        data_root / "val" / "images", data_root / "val" / "labels",
        img_size=img_size)

    train_loader = data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=cfg.get("workers", 4), collate_fn=collate_fn,
        pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=cfg.get("workers", 4), collate_fn=collate_fn,
        pin_memory=True)

    print(f"[INFO] Train: {len(train_ds)} images, Val: {len(val_ds)} images")

    # Build model
    pretrained_path = None
    if not args.no_pretrained:
        pretrained_path = Path(args.weights) if args.weights else WEIGHTS_PATH
        if not pretrained_path.exists():
            print(f"[WARNING] Pretrained weights not found at {pretrained_path}")
            pretrained_path = None

    model = build_model(num_classes, img_size=img_size, pretrained_path=pretrained_path)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Parameters: {total_params / 1e6:.1f}M total, {trainable_params / 1e6:.1f}M trainable")

    # Freeze backbone initially if pretrained
    if pretrained_path and freeze_epochs > 0:
        for param in model.backbone.body.parameters():
            param.requires_grad = False
        trainable_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Backbone frozen for first {freeze_epochs} epochs "
              f"({trainable_after_freeze / 1e6:.1f}M trainable)")

    # Optimizer — separate param groups for backbone and head
    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.body") and p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=lr0, weight_decay=cfg.get("weight_decay", 0.05))

    # LR scheduler
    warmup_epochs = cfg.get("warmup_epochs", 5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # WandB init
    os.environ["WANDB_PROJECT"] = args.wandb_project
    run_name = f"{save_dir.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "architecture": "ViTDet (ViTAE-Small + Faster R-CNN)",
            "backbone": "ViTAE-Small (384d, 3 stages)",
            "pretrained": pretrained_path is not None,
            "num_classes": num_classes,
            "class_names": class_names,
            "img_size": img_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr0": lr0,
            "optimizer": cfg.get("optimizer", "AdamW"),
            "weight_decay": cfg.get("weight_decay", 0.05),
            "freeze_backbone_epochs": freeze_epochs,
            "backbone_lr_scale": backbone_lr_scale,
            **{k: v for k, v in cfg.items() if k not in ("class_names", "data_root")},
        },
        dir=str(save_dir),
    )

    # Save args
    with open(save_dir / "args.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    # Training loop
    best_fitness = 0.0
    best_epoch = 0
    best_metrics = {}
    epochs_no_improve = 0
    results_log = []

    print(f"\n{'='*60}")
    print(f"Starting training for {epochs} epochs")
    print(f"{'='*60}\n")

    try:
        for epoch in range(epochs):
            t0 = time.time()

            # Unfreeze backbone after freeze_epochs
            if pretrained_path and freeze_epochs > 0 and epoch == freeze_epochs:
                print(f"\n[INFO] Epoch {epoch}: Unfreezing backbone")
                for param in model.backbone.body.parameters():
                    param.requires_grad = True
                # Add backbone params to optimizer with lower LR
                backbone_params = list(model.backbone.body.parameters())
                optimizer.add_param_group({
                    "params": backbone_params,
                    "lr": lr0 * backbone_lr_scale,
                    "initial_lr": lr0 * backbone_lr_scale,
                })
                n_unfrozen = sum(p.numel() for p in backbone_params)
                print(f"  Unfroze {n_unfrozen / 1e6:.1f}M backbone params (lr={lr0 * backbone_lr_scale})")

            # Train
            avg_loss, loss_components = train_one_epoch(
                model, optimizer, train_loader, device, scaler, epoch)

            # Step LR
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # Validate
            metrics = evaluate_map(model, val_loader, device)

            # Fitness = 0.1 * mAP50 + 0.9 * mAP50-95 (same as ultralytics)
            fitness = 0.1 * metrics["mAP50"] + 0.9 * metrics["mAP50-95"]

            elapsed = time.time() - t0

            # Log
            log_dict = {
                "epoch": epoch,
                "train/loss": avg_loss,
                "train/lr": current_lr,
                "val/mAP50": metrics["mAP50"],
                "val/mAP50-95": metrics["mAP50-95"],
                "val/precision": metrics["precision"],
                "val/recall": metrics["recall"],
                "val/fitness": fitness,
            }
            for k, v in loss_components.items():
                log_dict[f"train/{k}"] = v

            wandb.log(log_dict, step=epoch)

            results_log.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "lr": current_lr,
                **{f"train_{k}": v for k, v in loss_components.items()},
                **{f"val_{k}": v for k, v in metrics.items()},
                "fitness": fitness,
                "time_s": elapsed,
            })

            # Print progress
            print(f"Epoch {epoch:3d}/{epochs-1}  "
                  f"loss={avg_loss:.4f}  "
                  f"mAP50={metrics['mAP50']:.4f}  "
                  f"mAP50-95={metrics['mAP50-95']:.4f}  "
                  f"P={metrics['precision']:.4f}  "
                  f"R={metrics['recall']:.4f}  "
                  f"lr={current_lr:.6f}  "
                  f"[{elapsed:.1f}s]")

            # Save best
            if fitness > best_fitness:
                best_fitness = fitness
                best_epoch = epoch
                best_metrics = metrics.copy()
                epochs_no_improve = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "fitness": fitness,
                    "metrics": metrics,
                }, weights_dir / "best.pt")
                print(f"  ✓ New best (fitness={fitness:.4f})")
            else:
                epochs_no_improve += 1

            # Save periodic checkpoints
            if save_period > 0 and epoch % save_period == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "fitness": fitness,
                    "metrics": metrics,
                }, weights_dir / f"epoch{epoch}.pt")

            # Save last
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "fitness": fitness,
                "metrics": metrics,
            }, weights_dir / "last.pt")

            # Early stopping
            if patience > 0 and epochs_no_improve >= patience:
                print(f"\n[INFO] Early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user")

    # ── Post-training ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")

    # 1. Save results.csv
    if results_log:
        csv_path = save_dir / "results.csv"
        keys = results_log[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results_log)
        # Copy to train.csv
        shutil.copy2(csv_path, save_dir / "train.csv")
        print(f"[INFO] Training history saved to {csv_path}")

    # 2. Write best.txt
    best_txt = save_dir / "best.txt"
    with open(best_txt, "w") as f:
        f.write("ViTDet (ViTAE-Small + Faster R-CNN) Best Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best fitness:    {best_fitness:.6f}\n")
        f.write(f"Best epoch:      {best_epoch}\n\n")
        f.write("--- Metrics ---\n")
        for k, v in sorted(best_metrics.items()):
            f.write(f"{k:30s}  {v:.6f}\n")
        f.write("\n--- Config ---\n")
        for k, v in sorted(cfg.items()):
            f.write(f"{k:30s}  {v}\n")
    print(f"[INFO] Best metrics saved to {best_txt}")

    # 3. Log final metrics to wandb
    if wandb.run:
        wandb.summary.update({
            "best_fitness": best_fitness,
            "best_epoch": best_epoch,
            "best_mAP50": best_metrics.get("mAP50", 0),
            "best_mAP50-95": best_metrics.get("mAP50-95", 0),
            "best_precision": best_metrics.get("precision", 0),
            "best_recall": best_metrics.get("recall", 0),
        })
        wandb.finish()

    # 4. List checkpoints
    ckpts = sorted(weights_dir.glob("*.pt"))
    print(f"[INFO] Checkpoints ({len(ckpts)}): {[c.name for c in ckpts]}")
    print(f"[INFO] All results saved to {save_dir}")


if __name__ == "__main__":
    main()
