"""Extract 2048-D ResNet-50 embeddings for Dataset D (CD584 DICOM series)
and merge them into the existing embeddings.npz.

CD584 layout: /home/dsa/CD584/<site>/<case>/<series>/<something>.dcm
Each .dcm is a multi-frame acquisition (PhotometricInterpretation MONOCHROME2,
shape (F, H, W)). We decimate frames per series before random sampling.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pydicom

# Ensure pylibjpeg's `libjpeg` decoder plugin is registered with pydicom
# (pydicom's autodiscovery misses it in some environments).
try:
    import libjpeg  # noqa: F401
except ImportError:
    pass
try:
    import openjpeg  # noqa: F401
except ImportError:
    pass

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet50
from tqdm import tqdm

ROOT = Path('/home/dsa/stenosis')
CD584_ROOT = Path('/home/dsa/CD584')
EMB_PATH = ROOT / 'domain_gap' / 'features' / 'embeddings.npz'
CKPT = ROOT / 'domain_gap' / 'checkpoints' / 'resnet50_arcade_best.pth'

IMG_SIZE = 512
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _list_dicoms() -> list[Path]:
    return [
        p for p in CD584_ROOT.rglob('*.dcm')
        if 'Zone.Identifier' not in p.name
    ]


def _load_series_frames(dcm_path: Path) -> np.ndarray | None:
    try:
        ds = pydicom.dcmread(str(dcm_path))
        # Some CD584 files ship without PhotometricInterpretation (0028,0004).
        # Default to MONOCHROME2 so pydicom can still decode the pixel data.
        if 'PhotometricInterpretation' not in ds:
            ds.PhotometricInterpretation = 'MONOCHROME2'
        arr = ds.pixel_array
    except Exception as exc:  # pragma: no cover
        print(f'skip {dcm_path}: {exc}')
        return None
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        return None
    if getattr(ds, 'PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
        arr = arr.max() - arr
    return arr


def _decimate_and_normalize(arr: np.ndarray, stride: int) -> list[np.ndarray]:
    frames = arr[::max(1, stride)]
    out = []
    for f in frames:
        f = f.astype(np.float32)
        lo, hi = float(f.min()), float(f.max())
        if hi > lo:
            f = (f - lo) / (hi - lo)
        else:
            f = np.zeros_like(f)
        out.append((f * 255).astype(np.uint8))
    return out


def _collect_pool(stride: int) -> list[tuple[str, np.ndarray]]:
    # Return list of (series_id, 2D uint8 frame). Kept per-series so each
    # frame has a unique id for bookkeeping.
    pool: list[tuple[str, np.ndarray]] = []
    dicoms = _list_dicoms()
    print(f'CD584 DICOM files: {len(dicoms)}')
    for dcm in tqdm(dicoms, desc='dicom'):
        arr = _load_series_frames(dcm)
        if arr is None:
            continue
        frames = _decimate_and_normalize(arr, stride)
        series_id = str(dcm.relative_to(CD584_ROOT))
        for i, f in enumerate(frames):
            pool.append((f'{series_id}#f{i}', f))
    return pool


class FrameDataset(Dataset):
    def __init__(self, frames: list[tuple[str, np.ndarray]], transform):
        self.frames = frames
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int):
        fid, arr = self.frames[idx]
        img = Image.fromarray(arr, mode='L').convert('RGB')
        return self.transform(img), fid


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
def embed(model, loader, device) -> tuple[np.ndarray, list[str]]:
    feats, ids = [], []
    for x, fid in tqdm(loader, desc='embed D'):
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            f = model(x)
        feats.append(f.float().cpu().numpy())
        ids.extend(fid)
    return (np.concatenate(feats, axis=0) if feats else np.zeros((0, 2048), np.float32)), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stride', type=int, default=5,
                    help='keep every Nth frame per DICOM series')
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--num-workers', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    pool = _collect_pool(args.stride)
    print(f'CD584 pool after decimation (stride={args.stride}): {len(pool)} frames')
    if len(pool) == 0:
        raise RuntimeError('empty CD584 pool')

    sample = pool if len(pool) <= args.n else rng.sample(pool, args.n)
    print(f'sampled D: {len(sample)}')

    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_feature_extractor(device)

    loader = DataLoader(
        FrameDataset(sample, tf), batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    feats_d, ids_d = embed(model, loader, device)
    print(f'D features: {feats_d.shape}')

    data = np.load(EMB_PATH, allow_pickle=True)
    features = np.concatenate([data['features'], feats_d], axis=0)
    labels = np.concatenate([data['labels'], np.array(['D'] * len(feats_d))], axis=0)
    paths = np.concatenate([data['paths'], np.array(ids_d)], axis=0)
    np.savez_compressed(EMB_PATH, features=features, labels=labels, paths=paths)
    print(f'merged -> {features.shape}, {EMB_PATH}')


if __name__ == '__main__':
    main()
