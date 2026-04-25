"""Extract 2048-D DANN-ResNet-50 embeddings from 3 datasets:

    A - Dataset A (data/stenosis_arcade)              -- source
    B - Dataset B (data/cadica_50plus_new)            -- target (used in DANN)
    C - Dataset C (data/dataset2_split)               -- unseen external video

Backbone: DANN-ResNet-50 trained on (A_labeled, B_unlabeled) via the GRL
(``checkpoints/resnet50_arcade_dann_best.pth``). At eval time the GRL
and domain classifier are bypassed -- only ``forward_features`` is used.

Output: ``domain_gap/features/embeddings_dann.npz`` with
``features (3000, 2048)``, ``labels``, ``paths``.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from domain_gap.dann_resnet import build_dann_resnet50
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
OUT_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_dann.npz'
CKPT_DIR = ROOT / 'domain_gap' / 'checkpoints'


def build_feature_extractor(device, ckpt_path: Path) -> torch.nn.Module:
    model = build_dann_resnet50(num_classes=2, pretrained=False)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state['model'])
    print(f'loaded {ckpt_path.name} epoch={state.get("epoch")} '
          f'val_acc={state.get("val_acc")}')
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
            f = model.forward_features(x)
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
    ap.add_argument('--ckpt', type=str, default='last',
                    choices=('best', 'last'),
                    help="which DANN checkpoint to embed with "
                         "('last' uses final-epoch weights where alpha~1)")
    args = ap.parse_args()

    pool_a = _collect_dataset_a()
    pool_b = _collect_dataset_b(args.stride_b)
    pool_c = _collect_dataset_c(args.stride_c)
    print(f'pool sizes: A={len(pool_a)} B={len(pool_b)} C={len(pool_c)}')

    # Same RNG offsets per dataset as MixStyle / FDA so the {A, B, C}
    # samples here match those runs exactly.
    rng_a = random.Random(args.seed)
    rng_b = random.Random(args.seed + 1)
    rng_c = random.Random(args.seed + 2)
    sample_a = _sample(pool_a, args.n, rng_a)
    sample_b = _sample(pool_b, args.n, rng_b)
    sample_c = _sample(pool_c, args.n, rng_c)
    print(f'sampled:    A={len(sample_a)} B={len(sample_b)} C={len(sample_c)}')

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = CKPT_DIR / f'resnet50_arcade_dann_{args.ckpt}.pth'
    model = build_feature_extractor(device, ckpt_path)

    feats_all, labels_all, paths_all = [], [], []
    for tag, sample in (('A', sample_a), ('B', sample_b), ('C', sample_c)):
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
