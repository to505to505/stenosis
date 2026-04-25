"""Train a ResNet-50 binary stenosis-presence classifier on Dataset A
(stenosis_arcade) only. The backbone will later be used as a frozen feature
extractor for domain-gap analysis against video-frame datasets B and C.

Binary label: 1 if the YOLO label file for the image contains at least one
bounding box, else 0.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

ARCADE_ROOT = Path('/home/dsa/stenosis/data/stenosis_arcade')
OUT_DIR = Path('/home/dsa/stenosis/domain_gap/checkpoints')
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _yolo_label_is_positive(lbl_path: Path) -> int:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return 0
    for line in lbl_path.read_text().splitlines():
        if line.strip():
            return 1
    return 0


class ArcadeBinaryDataset(Dataset):
    def __init__(self, split: str, transform):
        self.img_dir = ARCADE_ROOT / split / 'images'
        self.lbl_dir = ARCADE_ROOT / split / 'labels'
        self.items = []
        for p in sorted(self.img_dir.iterdir()):
            if p.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            label = _yolo_label_is_positive(self.lbl_dir / f'{p.stem}.txt')
            self.items.append((p, label))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), label


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_model(num_classes: int = 2) -> nn.Module:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss_sum += criterion(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tf, eval_tf = build_transforms()
    train_ds = ArcadeBinaryDataset('train', train_tf)
    val_ds = ArcadeBinaryDataset('val', eval_tf)

    pos = sum(y for _, y in train_ds.items)
    print(f'train: {len(train_ds)} (pos={pos}, neg={len(train_ds) - pos})')
    print(f'val:   {len(val_ds)}')

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = build_model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    best_acc = -1.0
    best_path = OUT_DIR / 'resnet50_arcade_best.pth'
    last_path = OUT_DIR / 'resnet50_arcade_last.pth'

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}')
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            running += loss.item() * y.numel()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        sched.step()

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f'epoch {epoch}: train_loss={running / len(train_ds):.4f} '
              f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, last_path)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, best_path)
            print(f'  -> new best val_acc={best_acc:.4f}, saved to {best_path}')

    print(f'done. best val_acc={best_acc:.4f}')


if __name__ == '__main__':
    main()
