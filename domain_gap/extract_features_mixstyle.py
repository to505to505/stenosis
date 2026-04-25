"""Extract 2048-D MixStyle-ResNet-50 embeddings from 3 datasets:

    A - Dataset A (data/stenosis_arcade)              -- source
    B - Dataset B (data/cadica_50plus_new)            -- unseen internal video
    C - Dataset C (data/dataset2_split)               -- unseen external video

Backbone: MixStyle-ResNet-50 trained on Dataset A only
(``checkpoints/resnet50_arcade_mixstyle_best.pth``). Set to eval() so
MixStyle is a pure identity at feature time.

Output: ``domain_gap/features/embeddings_mixstyle.npz`` with
``features (3N, 2048)``, ``labels``, ``paths``.
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
from domain_gap.mixstyle_resnet import build_mixstyle_resnet50

ROOT = Path('/home/dsa/stenosis')
OUT_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_mixstyle.npz'
CKPT = ROOT / 'domain_gap' / 'checkpoints' / 'resnet50_arcade_mixstyle_best.pth'


def build_feature_extractor(device) -> torch.nn.Module:
    model = build_mixstyle_resnet50(num_classes=2, pretrained=False)
    state = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(state['model'])
    model.eval()                # disables MixStyle (acts as identity)
    for p in model.parameters():
        p.requires_grad = False
    return model.to(device)


@torch.no_grad()
def embed(model, loader, device) -> np.ndarray:
    feats = []
    for x, _ in tqdm(loader, desc='embed'):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            f = model.forward_features(x)   # 2048-D penultimate
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

    pool_a = _collect_dataset_a()
    pool_b = _collect_dataset_b(args.stride_b)
    pool_c = _collect_dataset_c(args.stride_c)
    print(f'pool sizes: A={len(pool_a)} B={len(pool_b)} C={len(pool_c)}')

    # Use the same RNG offsets per dataset as extract_features_fda.py so
    # the {A, B, C} samples here match the FDA / baseline runs exactly.
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
    model = build_feature_extractor(device)

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
