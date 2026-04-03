"""Dataset for single-frame stenosis detection (YOLO format → torchvision)."""

import re
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Regex for filenames like: 14_021_1_0046_bmp_jpg.rf.<uuid>.jpg
_FNAME_RE = re.compile(
    r"^(\d+_\d+)_(\d+)_(\d+)_bmp_jpg\.rf\.[0-9a-f]+\.jpg$"
)


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO labels → absolute x1y1x2y2.  Returns (N, 5): [cls, x1, y1, x2, y2]."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    data = np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)
    cls = data[:, 0:1]
    cx = data[:, 1] * img_w
    cy = data[:, 2] * img_h
    w = data[:, 3] * img_w
    h = data[:, 4] * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.column_stack([cls, x1, y1, x2, y2])


class StenosisDetDataset(Dataset):
    """Single-frame stenosis detection dataset.

    Returns (image, target) where:
        image: (3, H, W) float tensor (grayscale repeated to 3ch)
        target: dict with 'boxes', 'labels', 'image_id'
    """

    def __init__(self, data_root: str, split: str = "train",
                 img_size: int = 512):
        super().__init__()
        self.img_size = img_size
        root = Path(data_root)
        self.img_dir = root / split / "images"
        self.lbl_dir = root / split / "labels"

        self.image_paths = sorted(self.img_dir.glob("*.jpg"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.img_dir}")
        print(f"[{split}] {len(self.image_paths)} images from {self.img_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Load grayscale, resize, repeat to 3 channels
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read {img_path}")
        oh, ow = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        # HW → 3HW (repeat grayscale to 3 channels)
        img_tensor = torch.from_numpy(img).unsqueeze(0).expand(3, -1, -1).contiguous()

        # Labels
        lbl_path = self.lbl_dir / (img_path.stem + ".txt")
        annots = load_yolo_labels(lbl_path, self.img_size, self.img_size)

        if annots.shape[0] > 0:
            boxes = torch.from_numpy(annots[:, 1:5]).float()
            # Clamp to image bounds
            boxes[:, 0::2].clamp_(0, self.img_size)
            boxes[:, 1::2].clamp_(0, self.img_size)
            # Labels: YOLO class 0 → torchvision class 1 (0 is background)
            labels = torch.from_numpy(annots[:, 0]).long() + 1
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target


def collate_fn(batch):
    """Custom collate: images as list of tensors, targets as list of dicts."""
    images, targets = zip(*batch)
    return list(images), list(targets)


def get_dataloader(data_root: str, split: str, img_size: int = 512,
                   batch_size: int = 4, num_workers: int = 4,
                   shuffle: bool = True) -> DataLoader:
    ds = StenosisDetDataset(data_root, split, img_size)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
