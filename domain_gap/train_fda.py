"""Train a ResNet-50 binary stenosis-presence classifier on the FDA-aligned
mirror of Dataset A (``data/stenosis_arcade_fda_B``).

This is a thin wrapper around ``domain_gap.train`` that swaps the dataset
root and the checkpoint path. The label semantics and split layout are
identical to the original Dataset A, so we reuse the same Dataset class.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from domain_gap import train as base
from domain_gap.train import build_model, build_transforms, evaluate

FDA_ROOT = Path('/home/dsa/stenosis/data/stenosis_arcade_fda_B')
OUT_DIR = Path('/home/dsa/stenosis/domain_gap/checkpoints')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Repoint the ArcadeBinaryDataset to the FDA-aligned root.
    base.ARCADE_ROOT = FDA_ROOT

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tf, eval_tf = build_transforms()
    train_ds = base.ArcadeBinaryDataset('train', train_tf)
    val_ds = base.ArcadeBinaryDataset('val', eval_tf)

    pos = sum(y for _, y in train_ds.items)
    print(f'[FDA-A] train: {len(train_ds)} (pos={pos}, neg={len(train_ds) - pos})')
    print(f'[FDA-A] val:   {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = build_model().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    best_acc = -1.0
    best_path = OUT_DIR / 'resnet50_arcade_fda_best.pth'
    last_path = OUT_DIR / 'resnet50_arcade_fda_last.pth'

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

        torch.save({'model': model.state_dict(), 'epoch': epoch,
                    'val_acc': val_acc}, last_path)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({'model': model.state_dict(), 'epoch': epoch,
                        'val_acc': val_acc}, best_path)
            print(f'  -> new best val_acc={best_acc:.4f}, saved to {best_path}')

    print(f'done. best val_acc={best_acc:.4f}')


if __name__ == '__main__':
    main()
