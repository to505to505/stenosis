"""Train a DANN ResNet-50: source labels from Dataset A
(``stenosis_arcade``), unlabeled target from Dataset B
(``cadica_50plus_new``). The 2048-D bottleneck is fed through a
Gradient Reversal Layer into a small domain classifier so the backbone
becomes domain-invariant w.r.t. (A, B).

Mirrors the optimiser / schedule / AMP setup of
``domain_gap/train.py`` so the resulting checkpoint is directly
comparable with the baseline / MixStyle / FDA results in
``cluster_distances_*.txt``.

Total loss: ``L = L_cls + L_dom``, where:
    L_cls  = CE(label_logits[source], y_source)
    L_dom  = BCE(dom_logits([source; target]),
                 [0]*B_s + [1]*B_t)

Adversarial strength follows the original DANN schedule
``alpha = 2 / (1 + exp(-10 * p)) - 1``, where ``p in [0, 1]`` is the
training progress measured per *step* (smoother than per-epoch).
"""
from __future__ import annotations

import argparse
import itertools
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from domain_gap.dann_resnet import build_dann_resnet50

ROOT = Path('/home/dsa/stenosis')
ARCADE_ROOT = ROOT / 'data' / 'stenosis_arcade'
CADICA_IMG_DIR = ROOT / 'data' / 'cadica_50plus_new' / 'images'
OUT_DIR = ROOT / 'domain_gap' / 'checkpoints'
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CADICA_RE = re.compile(r'^(p\d+_v\d+)_(\d+)\.png$')


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def _yolo_label_is_positive(lbl_path: Path) -> int:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return 0
    for line in lbl_path.read_text().splitlines():
        if line.strip():
            return 1
    return 0


class ArcadeBinaryDataset(Dataset):
    """Source: stenosis_arcade with binary stenosis-presence labels."""

    def __init__(self, split: str, transform):
        self.img_dir = ARCADE_ROOT / split / 'images'
        self.lbl_dir = ARCADE_ROOT / split / 'labels'
        self.items: list[tuple[Path, int]] = []
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


class CadicaUnlabeledDataset(Dataset):
    """Target: cadica_50plus_new frames, decimated per sequence."""

    def __init__(self, transform, stride: int = 2):
        seqs: dict[str, list[Path]] = defaultdict(list)
        for p in CADICA_IMG_DIR.iterdir():
            m = CADICA_RE.match(p.name)
            if m:
                seqs[m.group(1)].append(p)
        self.paths: list[Path] = []
        for _, frames in seqs.items():
            frames = sorted(frames)
            self.paths.extend(frames[:: max(1, stride)])
        self.paths.sort()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        # Dummy label - never used (target is unlabeled).
        return self.transform(img), 0


def build_transforms():
    """Standard moderate augmentations - same set used by
    ``train.py``.  The *same* transform is applied to source and target
    so the domain classifier cannot exploit augmentation differences.
    """
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


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_source(model, loader, device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)  # alpha=None -> only cls logits
        loss_sum += criterion(logits, y).item()
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def dann_alpha(step: int, total_steps: int, gamma: float = 10.0) -> float:
    """``alpha = 2 / (1 + exp(-gamma * p)) - 1``, p in [0, 1]."""
    p = min(1.0, max(0.0, step / max(1, total_steps)))
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--accum-steps', type=int, default=1,
                    help='gradient accumulation steps (effective batch = '
                         'batch_size * accum_steps)')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--domain-lr-mult', type=float, default=10.0,
                    help='LR multiplier for the domain classifier head')
    ap.add_argument('--alpha-gamma', type=float, default=10.0,
                    help='gamma in DANN alpha schedule (lower = slower ramp)')
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--num-workers', type=int, default=6)
    ap.add_argument('--target-stride', type=int, default=2,
                    help='per-sequence decimation stride for cadica frames')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_tf, eval_tf = build_transforms()
    src_train_ds = ArcadeBinaryDataset('train', train_tf)
    src_val_ds = ArcadeBinaryDataset('val', eval_tf)
    tgt_train_ds = CadicaUnlabeledDataset(train_tf, stride=args.target_stride)

    pos = sum(y for _, y in src_train_ds.items)
    print(f'source train: {len(src_train_ds)} (pos={pos}, '
          f'neg={len(src_train_ds) - pos})')
    print(f'source val:   {len(src_val_ds)}')
    print(f'target train: {len(tgt_train_ds)}  '
          f'(cadica, stride={args.target_stride})')

    src_loader = DataLoader(
        src_train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        src_val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    def make_target_loader():
        return DataLoader(
            tgt_train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )

    steps_per_epoch = len(src_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f'steps/epoch={steps_per_epoch}  total_steps={total_steps}')

    model = build_dann_resnet50(num_classes=2, pretrained=True).to(device)

    domain_param_ids = {id(p) for p in model.domain_head.parameters()}
    backbone_params = [p for p in model.parameters()
                       if id(p) not in domain_param_ids]
    domain_params = list(model.domain_head.parameters())
    optim = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': args.lr},
            {'params': domain_params,
             'lr': args.lr * args.domain_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    cls_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    best_acc = -1.0
    best_path = OUT_DIR / 'resnet50_arcade_dann_best.pth'
    last_path = OUT_DIR / 'resnet50_arcade_dann_last.pth'

    global_step = 0
    accum = max(1, args.accum_steps)
    for epoch in range(1, args.epochs + 1):
        model.train()
        target_iter = itertools.cycle(make_target_loader())
        run_cls, run_dom, run_n = 0.0, 0.0, 0
        last_alpha = 0.0
        last_dom_acc = 0.0
        optim.zero_grad(set_to_none=True)
        pbar = tqdm(src_loader, desc=f'epoch {epoch}/{args.epochs}')
        for it, (xs, ys) in enumerate(pbar):
            xt, _ = next(target_iter)
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)
            bs, bt = xs.size(0), xt.size(0)

            alpha = dann_alpha(global_step, total_steps,
                               gamma=args.alpha_gamma)
            last_alpha = alpha

            with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
                feat_s = model.forward_features(xs)
                feat_t = model.forward_features(xt)
                cls_logits = model.classify(feat_s)
                loss_cls = cls_criterion(cls_logits, ys)

                feat_st = torch.cat([feat_s, feat_t], dim=0)
                dom_logits = model.discriminate(feat_st, alpha)
                dom_targets = torch.cat([
                    torch.zeros(bs, device=device),
                    torch.ones(bt, device=device),
                ], dim=0)
                loss_dom = dom_criterion(dom_logits, dom_targets)

                loss = (loss_cls + loss_dom) / accum

            scaler.scale(loss).backward()
            do_step = ((it + 1) % accum == 0) or (it + 1 == len(src_loader))
            if do_step:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)

            with torch.no_grad():
                dom_pred = (dom_logits.detach() > 0).float()
                last_dom_acc = (dom_pred == dom_targets).float().mean().item()

            run_cls += loss_cls.item() * bs
            run_dom += loss_dom.item() * (bs + bt)
            run_n += bs
            global_step += 1
            pbar.set_postfix(
                cls=f'{loss_cls.item():.3f}',
                dom=f'{loss_dom.item():.3f}',
                a=f'{alpha:.3f}',
                dacc=f'{last_dom_acc:.2f}',
            )
        sched.step()

        val_loss, val_acc = evaluate_source(model, val_loader, device)
        print(
            f'epoch {epoch}: '
            f'train_cls={run_cls / max(run_n, 1):.4f} '
            f'train_dom={run_dom / max(run_n * 2, 1):.4f} '
            f'alpha={last_alpha:.3f} '
            f'last_dom_batch_acc={last_dom_acc:.3f} '
            f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}'
        )

        ckpt = {
            'model': model.state_dict(),
            'epoch': epoch,
            'val_acc': val_acc,
            'alpha_final': last_alpha,
            'args': vars(args),
        }
        torch.save(ckpt, last_path)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, best_path)
            print(f'  -> new best val_acc={best_acc:.4f}, saved to {best_path}')

    print(f'done. best val_acc={best_acc:.4f}')


if __name__ == '__main__':
    main()
