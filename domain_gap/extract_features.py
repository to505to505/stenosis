"""Extract 2048-D ResNet-50 embeddings from three datasets for domain-gap
analysis.

- Dataset A (stenosis_arcade): static 2D frames -> sampled directly.
- Dataset B (cadica_50plus_new): video sequences (pXX_vYY_*.png) -> decimate
  per sequence before sampling.
- Dataset C (dataset2_split): video-like frames grouped by filename prefix
  (<study>_<video>_<frame>) -> decimate per sequence before sampling.

Decimation strategy: keep every N-th frame per sequence (default N=5, i.e.
~1 frame per ~0.2 s at 25 fps). If fewer than 1000 frames remain after
decimation for a dataset, the stride is reduced.
"""
from __future__ import annotations

import argparse
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
from torchvision.models import resnet50
from tqdm import tqdm

ROOT = Path('/home/dsa/stenosis')
OUT_DIR = ROOT / 'domain_gap' / 'features'
CKPT = ROOT / 'domain_gap' / 'checkpoints' / 'resnet50_arcade_best.pth'
IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
N_PER_DATASET = 1000

CADICA_RE = re.compile(r'^(p\d+_v\d+)_(\d+)\.png$')
DATASET2_RE = re.compile(r'^(\d+_\d+_\d+)_(\d+)_')


def _collect_dataset_a() -> list[Path]:
    # static images: no temporal decimation needed.
    imgs = []
    for split in ('train', 'val', 'test'):
        d = ROOT / 'data' / 'stenosis_arcade' / split / 'images'
        if d.exists():
            imgs.extend(sorted(p for p in d.iterdir()
                               if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}))
    return imgs


def _decimate(sequences: dict[str, list[Path]], stride: int) -> list[Path]:
    kept = []
    for _, frames in sequences.items():
        frames = sorted(frames)
        kept.extend(frames[::max(1, stride)])
    return kept


def _collect_dataset_b(stride: int) -> list[Path]:
    img_dir = ROOT / 'data' / 'cadica_50plus_new' / 'images'
    seqs: dict[str, list[Path]] = defaultdict(list)
    for p in img_dir.iterdir():
        m = CADICA_RE.match(p.name)
        if m:
            seqs[m.group(1)].append(p)
    return _decimate(seqs, stride)


def _collect_dataset_c(stride: int) -> list[Path]:
    seqs: dict[str, list[Path]] = defaultdict(list)
    for split in ('train', 'valid', 'test'):
        d = ROOT / 'data' / 'dataset2_split' / split / 'images'
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            m = DATASET2_RE.match(p.name)
            key = m.group(1) if m else p.stem
            seqs[key].append(p)
    return _decimate(seqs, stride)


def _sample(paths: list[Path], n: int, rng: random.Random) -> list[Path]:
    if len(paths) <= n:
        return list(paths)
    return rng.sample(paths, n)


class PathDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        return self.transform(img), str(p)


def build_feature_extractor(device) -> nn.Module:
    model = resnet50(weights=None)
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, 2)
    state = torch.load(CKPT, map_location='cpu')
    model.load_state_dict(state['model'])
    model.fc = nn.Identity()  # penultimate 2048-D output.
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
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride-b', type=int, default=5,
                    help='keep every Nth frame per cadica sequence')
    ap.add_argument('--stride-c', type=int, default=5,
                    help='keep every Nth frame per dataset2 sequence')
    ap.add_argument('--n', type=int, default=N_PER_DATASET)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_a = _collect_dataset_a()
    pool_b = _collect_dataset_b(args.stride_b)
    pool_c = _collect_dataset_c(args.stride_c)
    print(f'pool sizes: A={len(pool_a)} B={len(pool_b)} C={len(pool_c)}')

    sample_a = _sample(pool_a, args.n, rng)
    sample_b = _sample(pool_b, args.n, rng)
    sample_c = _sample(pool_c, args.n, rng)
    print(f'sampled:    A={len(sample_a)} B={len(sample_b)} C={len(sample_c)}')

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_feature_extractor(device)

    labels, feats, paths = [], [], []
    for tag, sample in (('A', sample_a), ('B', sample_b), ('C', sample_c)):
        loader = DataLoader(
            PathDataset(sample, tf), batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        f = embed(model, loader, device)
        feats.append(f)
        labels.extend([tag] * len(sample))
        paths.extend(str(p) for p in sample)
        print(f'{tag}: features {f.shape}')

    features = np.concatenate(feats, axis=0)
    labels_arr = np.array(labels)
    paths_arr = np.array(paths)
    out = OUT_DIR / 'embeddings.npz'
    np.savez_compressed(out, features=features, labels=labels_arr, paths=paths_arr)
    print(f'saved {out} -> features={features.shape}')


if __name__ == '__main__':
    main()
