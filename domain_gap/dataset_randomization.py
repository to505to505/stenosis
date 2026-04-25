"""Extreme Domain Randomization pipeline (Albumentations 2.x) for the
Dataset A training loader.

The intent is to *destroy* the clean 2D angiographic look of the source
domain so the backbone learns vascular *topology* rather than the
photometric / sensor style of `stenosis_arcade`. Aggressive noise,
codec / motion artefacts, lighting changes and mild geometric
distortions are all applied with high probability. Geometric distortions
are deliberately kept moderate so vessels are not torn apart.

This module exposes:

    build_train_transform(img_size=512) -> A.Compose
    build_eval_transform(img_size=512)  -> A.Compose
    AlbumentationsArcadeDataset         -> torch Dataset

The dataset returns ``(tensor, label)`` pairs where ``label`` is the
binary stenosis-presence flag (1 if the YOLO label file contains at
least one box, else 0).
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

ARCADE_ROOT = Path('/home/dsa/stenosis/data/stenosis_arcade')
IMG_SIZE = 512
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int = IMG_SIZE) -> A.Compose:
    """Extreme-randomization training pipeline.

    Probabilities sit in the 0.5-0.8 range as requested. The pipeline is
    grouped by family with ``OneOf`` blocks so we don't pile *every*
    artefact on every sample (which would make the image unreadable).
    """
    return A.Compose([
        # ---- Geometry (cheap, always-on basics) -------------------------
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # ---- Spatial distortions (topology-aware) -----------------------
        # Limits intentionally moderate -> vessels bend but do not tear.
        A.OneOf([
            A.ElasticTransform(alpha=40, sigma=6, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
            A.OpticalDistortion(distort_limit=0.2, p=1.0),
        ], p=0.6),

        # ---- Lighting / contrast / colour -------------------------------
        A.RandomBrightnessContrast(brightness_limit=0.4,
                                   contrast_limit=0.4, p=0.8),
        A.RandomGamma(gamma_limit=(60, 160), p=0.6),
        A.ColorJitter(brightness=0.3, contrast=0.3,
                      saturation=0.3, hue=0.05, p=0.5),

        # ---- Sensor / video noise --------------------------------------
        A.OneOf([
            A.GaussNoise(std_range=(0.05, 0.2), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.7, 1.3),
                                  per_channel=True, p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05),
                       intensity=(0.1, 0.5), p=1.0),
        ], p=0.7),

        # ---- Codec / motion / low-res video artefacts ------------------
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 9), p=1.0),
            A.ImageCompression(quality_range=(20, 60), p=1.0),
            A.Downscale(scale_range=(0.4, 0.75),
                        interpolation_pair={
                            'downscale': cv2.INTER_AREA,
                            'upscale': cv2.INTER_LINEAR,
                        },
                        p=1.0),
        ], p=0.7),

        # ---- Final tensorisation ---------------------------------------
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_eval_transform(img_size: int = IMG_SIZE) -> A.Compose:
    """Clean eval / feature-extraction pipeline (no augmentation)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _yolo_label_is_positive(lbl_path: Path) -> int:
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return 0
    for line in lbl_path.read_text().splitlines():
        if line.strip():
            return 1
    return 0


class AlbumentationsArcadeDataset(Dataset):
    """Dataset A binary stenosis-presence dataset using Albumentations."""

    def __init__(self, split: str, transform: A.Compose,
                 root: Path = ARCADE_ROOT):
        self.img_dir = root / split / 'images'
        self.lbl_dir = root / split / 'labels'
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
        # Albumentations expects HWC uint8 numpy.
        img = np.array(Image.open(path).convert('RGB'))
        out = self.transform(image=img)
        x = out['image']
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x, label
