"""Extract 2048-D ResNet-50 embeddings from 4 sources for the FDA
re-evaluation t-SNE:

    A   - original Dataset A (data/stenosis_arcade)
    A*  - FDA-aligned Dataset A (data/stenosis_arcade_fda_B)
    B   - internal video (data/cadica_50plus_new)
    C   - external video (data/dataset2_split)

Backbone: ResNet-50 fine-tuned on the **FDA-aligned A** training split
(``checkpoints/resnet50_arcade_fda_best.pth``). Penultimate features.

Output: ``domain_gap/features/embeddings_fda.npz`` with arrays
``features (4N, 2048)``, ``labels (str)``, ``paths (str)``.
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
from torchvision.models import resnet50
from tqdm import tqdm

from domain_gap.extract_features import (
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    PathDataset,
    _collect_dataset_a,
    _collect_dataset_b,
    _collect_dataset_c,
    _sample,
)

ROOT = Path('/home/dsa/stenosis')
OUT_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_fda.npz'
CKPT = ROOT / 'domain_gap' / 'checkpoints' / 'resnet50_arcade_fda_best.pth'
FDA_A_ROOT = ROOT / 'data' / 'stenosis_arcade_fda_B'


def _collect_dataset_a_fda() -> list[Path]:
    """FDA-aligned mirror of Dataset A (same split layout)."""
    imgs: list[Path] = []
    for split in ('train', 'val', 'test'):
        d = FDA_A_ROOT / split / 'images'
        if d.exists():
            imgs.extend(sorted(p for p in d.iterdir()
                               if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}))
    return imgs


def build_feature_extractor(device) -> nn.Module:
    model = resnet50(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, 2)
    state = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(state['model'])
    model.fc = nn.Identity()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


@torch.no_grad()
def embed(model, loader, device) -> np.ndarray:
    feats = []
    for x, _ in tqdm(loader, desc='embed'):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            f = model(x)
        feats.append(f.float().cpu().numpy())
    return (np.concatenate(feats, axis=0) if feats
            else np.zeros((0, 2048), dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride-b', type=int, default=2)
    ap.add_argument('--stride-c', type=int, default=5)
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pool_a = _collect_dataset_a()
    pool_a_fda = _collect_dataset_a_fda()
    pool_b = _collect_dataset_b(args.stride_b)
    pool_c = _collect_dataset_c(args.stride_c)
    print(f'pool sizes: A={len(pool_a)} A*={len(pool_a_fda)} '
          f'B={len(pool_b)} C={len(pool_c)}')

    # Sample A and A* with the same RNG state so the chosen set of
    # source identities is *identical* across the two views.
    rng_a = random.Random(args.seed)
    rng_astar = random.Random(args.seed)
    rng_b = random.Random(args.seed + 1)
    rng_c = random.Random(args.seed + 2)

    sample_a = _sample(pool_a, args.n, rng_a)
    sample_a_fda = _sample(pool_a_fda, args.n, rng_astar)
    sample_b = _sample(pool_b, args.n, rng_b)
    sample_c = _sample(pool_c, args.n, rng_c)
    print(f'sampled:    A={len(sample_a)} A*={len(sample_a_fda)} '
          f'B={len(sample_b)} C={len(sample_c)}')

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_feature_extractor(device)

    feats_all, labels_all, paths_all = [], [], []
    for tag, sample in (
        ('A', sample_a),
        ('A*', sample_a_fda),
        ('B', sample_b),
        ('C', sample_c),
    ):
        loader = DataLoader(
            PathDataset(sample, tf), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers, pin_memory=True,
        )
        f = embed(model, loader, device)
        feats_all.append(f)
        labels_all.extend([tag] * len(sample))
        paths_all.extend(str(p) for p in sample)
        print(f'{tag}: features {f.shape}')

    features = np.concatenate(feats_all, axis=0)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        features=features,
        labels=np.array(labels_all),
        paths=np.array(paths_all),
    )
    print(f'saved {OUT_PATH} -> features={features.shape}')


if __name__ == '__main__':
    main()
