"""Frechet Inception Distance (FID) between FDA-Adapted Dataset A and the
two video targets:

    FID(A*, B)   - alignment quality (lower is better)
    FID(A*, C)   - zero-shot transfer to external video
    FID(A,  B), FID(A, C)  - baselines (no FDA)

Implementation
--------------
Standard FID: 2048-D Inception-V3 pool3 features (ImageNet-1k weights),
mu/sigma per dataset, Frechet distance:

    FID = ||mu_x - mu_y||^2 + Tr(Sx + Sy - 2 (Sx Sy)^{1/2})

We reuse the *exact same image paths* sampled by
``extract_features_fda.py`` (cached in ``embeddings_fda.npz``) so the
metric is computed on the same population as the t-SNE.

Output: ``domain_gap/figures/fid_fda.txt``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

ROOT = Path('/home/dsa/stenosis')
EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings_fda.npz'
OUT_PATH = ROOT / 'domain_gap' / 'figures' / 'fid_fda.txt'

# Canonical FID preprocessing for torchvision Inception-V3:
# resize to 299, ToTensor() (divides by 255), no ImageNet normalization
# (the model includes its own input transform when transform_input=True).
INC_SIZE = 299


class _PathDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((INC_SIZE, INC_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.tf(img)


def _build_inception(device) -> nn.Module:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    m = inception_v3(weights=weights, aux_logits=True)
    m.fc = nn.Identity()  # 2048-D pool3 features
    m.eval()
    for p in m.parameters():
        p.requires_grad = False
    return m.to(device)


@torch.no_grad()
def _features(model, paths: list[str], device, batch_size: int,
              num_workers: int) -> np.ndarray:
    loader = DataLoader(_PathDataset(paths), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True)
    out = []
    for x in tqdm(loader, desc='inception'):
        x = x.to(device, non_blocking=True)
        # Match TF FID: feed in [0, 1]; model.transform_input rescales to
        # the ImageNet stats internally.
        f = model(x)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _fid(mu1, s1, mu2, s2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset) @ (s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(s1) + np.trace(s2)
                 - 2 * np.trace(covmean))


def _stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=6)
    args = ap.parse_args()

    data = np.load(EMB_PATH, allow_pickle=True)
    labels = data['labels']
    paths = data['paths'].tolist()
    by_tag = {t: [paths[i] for i in range(len(paths)) if labels[i] == t]
              for t in ('A', 'A*', 'B', 'C')}
    for t, ps in by_tag.items():
        print(f'{t}: {len(ps)} paths')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _build_inception(device)

    feats = {t: _features(model, ps, device, args.batch_size,
                          args.num_workers)
             for t, ps in by_tag.items()}
    stats = {t: _stats(f) for t, f in feats.items()}

    pairs = [
        ('A',  'B'),
        ('A*', 'B'),
        ('A',  'C'),
        ('A*', 'C'),
        ('B',  'C'),
        ('A',  'A*'),
    ]
    lines = ['== Frechet Inception Distance (Inception-V3 pool3, lower = closer) ==',
             '']
    fid_vals = {}
    for a, b in pairs:
        mu1, s1 = stats[a]
        mu2, s2 = stats[b]
        v = _fid(mu1, s1, mu2, s2)
        fid_vals[(a, b)] = v
        lines.append(f'  FID({a:>2} , {b:>2}) = {v:10.3f}')

    delta_b = fid_vals[('A*', 'B')] - fid_vals[('A', 'B')]
    delta_c = fid_vals[('A*', 'C')] - fid_vals[('A', 'C')]
    lines += [
        '',
        '== Effect of FDA (negative = closer after adaptation) ==',
        f'  delta FID(., B) = {delta_b:+.3f}   (target domain)',
        f'  delta FID(., C) = {delta_c:+.3f}   (zero-shot external)',
    ]

    report = '\n'.join(lines)
    print('\n' + report)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report + '\n')
    print(f'\nsaved {OUT_PATH}')


if __name__ == '__main__':
    main()
