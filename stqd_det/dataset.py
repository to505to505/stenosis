"""Temporal sequence dataset for STQD-Det.

Builds sliding windows of T=9 consecutive frames from angiography sequences,
grouped by (patient_id, sequence_id) and sorted by frame number.
Adapted from stenosis_detnet/dataset.py.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations import ReplayCompose

from .config import Config

# Regex for filenames like: 14_021_1_0046_bmp_jpg.rf.<uuid>.jpg
_FNAME_RE = re.compile(
    r"^(\d+_\d+)_(\d+)_(\d+)_bmp_jpg\.rf\.[0-9a-f]+\.jpg$"
)


def parse_filename(fname: str) -> Optional[Tuple[str, int, int]]:
    """Extract (patient_id, sequence_id, frame_number) from filename."""
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def build_sequence_index(image_dir: Path) -> List[Tuple[str, int, List[Path]]]:
    """Group images into sequences and sort by frame number."""
    groups: Dict[Tuple[str, int], List[Tuple[int, Path]]] = defaultdict(list)
    for p in sorted(image_dir.iterdir()):
        if not p.suffix == ".jpg":
            continue
        parsed = parse_filename(p.name)
        if parsed is None:
            continue
        patient_id, seq_id, frame_num = parsed
        groups[(patient_id, seq_id)].append((frame_num, p))

    sequences = []
    for (patient_id, seq_id), frames in sorted(groups.items()):
        frames.sort(key=lambda x: x[0])
        paths = [p for _, p in frames]
        sequences.append((patient_id, seq_id, paths))
    return sequences


def build_windows(
    sequences: List[Tuple[str, int, List[Path]]], T: int
) -> List[List[Path]]:
    """Create sliding windows of T consecutive frames.

    Sequences shorter than T are padded by repeating edge frames.
    """
    windows: List[List[Path]] = []
    for _pid, _sid, paths in sequences:
        n = len(paths)
        if n == 0:
            continue
        if n < T:
            padded = list(paths) + [paths[-1]] * (T - n)
            windows.append(padded)
        else:
            for start in range(n - T + 1):
                windows.append(paths[start : start + T])
    return windows


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Load YOLO-format labels → absolute x1y1x2y2. Shape (N, 5)."""
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


def build_train_augmentation(img_h: int, img_w: int) -> ReplayCompose:
    """Build augmentation pipeline with ReplayCompose for temporal consistency."""
    return ReplayCompose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                scale=(0.85, 1.15),
                translate_percent=(-0.08, 0.08),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3
            ),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_ids"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )


class STQDDetDataset(Dataset):
    """Dataset yielding windows of T=9 consecutive grayscale frames + labels.

    Returns:
        images: (T, 1, H, W) normalized grayscale tensors.
        targets: list of T dicts, each with "boxes" (M, 4) xyxy and "labels" (M,).
    """

    def __init__(self, split: str, cfg: Config):
        self.cfg = cfg
        self.split = split
        self.img_dir = cfg.data_root / split / "images"
        self.lbl_dir = cfg.data_root / split / "labels"

        sequences = build_sequence_index(self.img_dir)
        self.windows = build_windows(sequences, cfg.T)

        self.augment = (
            build_train_augmentation(cfg.img_h, cfg.img_w)
            if split == "train" else None
        )

        self._double = split == "train"
        effective = len(self.windows) * (2 if self._double else 1)

        print(
            f"[{split}] {len(sequences)} sequences, "
            f"{sum(len(s[2]) for s in sequences)} frames, "
            f"{len(self.windows)} windows of T={cfg.T}"
            f"{f' (×2 → {effective} with augmentation)' if self._double else ''}"
        )

    def __len__(self) -> int:
        return len(self.windows) * (2 if self._double else 1)

    def __getitem__(self, idx: int):
        use_aug = False
        real_idx = idx
        if self._double:
            n = len(self.windows)
            if idx >= n:
                real_idx = idx - n
                use_aug = True

        paths = self.windows[real_idx]
        images = []
        targets = []
        saved_replay = None

        for frame_i, img_path in enumerate(paths):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            orig_h, orig_w = img.shape[:2]

            if orig_h != self.cfg.img_h or orig_w != self.cfg.img_w:
                img = cv2.resize(
                    img, (self.cfg.img_w, self.cfg.img_h),
                    interpolation=cv2.INTER_LINEAR,
                )

            lbl_name = img_path.stem + ".txt"
            lbl_path = self.lbl_dir / lbl_name
            labels = load_yolo_labels(lbl_path, self.cfg.img_w, self.cfg.img_h)
            bboxes = labels[:, 1:5].tolist()
            class_ids = labels[:, 0].astype(int).tolist()

            if use_aug and self.augment is not None:
                if frame_i == 0:
                    result = self.augment(
                        image=img, bboxes=bboxes, class_ids=class_ids,
                    )
                    saved_replay = result["replay"]
                else:
                    result = ReplayCompose.replay(
                        saved_replay, image=img, bboxes=bboxes, class_ids=class_ids,
                    )
                img = result["image"]
                bboxes = result["bboxes"]
                class_ids = result["class_ids"]

            # Normalize
            img = img.astype(np.float32) / 255.0
            img = (img - self.cfg.pixel_mean) / self.cfg.pixel_std

            img_tensor = torch.from_numpy(img).unsqueeze(0)
            images.append(img_tensor)

            if len(bboxes) > 0:
                boxes = torch.tensor(bboxes, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            cls_ids = torch.tensor(class_ids, dtype=torch.int64) if class_ids else torch.zeros(len(boxes), dtype=torch.int64)
            targets.append({"boxes": boxes, "labels": cls_ids})

        images = torch.stack(images, dim=0)  # (T, 1, H, W)
        return images, targets


def collate_fn(batch):
    """Collate into (B, T, 1, H, W) images + list of target lists."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets


def get_dataloader(split: str, cfg: Config, shuffle: bool = False) -> DataLoader:
    ds = STQDDetDataset(split, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
