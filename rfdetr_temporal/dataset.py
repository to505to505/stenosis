"""Temporal sequence dataset for RF-DETR stenosis detection.

Builds sliding windows of T consecutive frames, loads them as RGB,
and returns normalised tensors with targets in RF-DETR format
(normalised cxcywh boxes).
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

# Filename pattern: {patient_id}_{seq_id}_{frame_num}_bmp_jpg.rf.<uuid>.jpg
_FNAME_RE = re.compile(
    r"^(\d+_\d+)_(\d+)_(\d+)_bmp_jpg\.rf\.[0-9a-f]+\.jpg$"
)


def parse_filename(fname: str) -> Optional[Tuple[str, int, int]]:
    m = _FNAME_RE.match(fname)
    if m is None:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def build_sequence_index(image_dir: Path) -> List[Tuple[str, int, List[Path]]]:
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


def load_yolo_labels(label_path: Path) -> np.ndarray:
    """Load YOLO-format labels.  Returns (N, 5): [class_id, cx, cy, w, h] normalised."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)


def yolo_to_pascal(labels: np.ndarray, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """Convert YOLO normalised cxcywh to absolute pascal_voc x1y1x2y2 for augmentation."""
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)
    cls = labels[:, 0].astype(np.int32)
    cx = labels[:, 1] * img_w
    cy = labels[:, 2] * img_h
    w  = labels[:, 3] * img_w
    h  = labels[:, 4] * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.column_stack([x1, y1, x2, y2]), cls


def pascal_to_cxcywh_norm(boxes: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Convert absolute pascal_voc x1y1x2y2 back to normalised cxcywh."""
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return np.column_stack([cx, cy, w, h]).astype(np.float32)


def build_train_augmentation(img_size: int) -> ReplayCompose:
    return ReplayCompose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(-0.05, 0.05),
                border_mode=cv2.BORDER_CONSTANT,
                p=0.3,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5,
            ),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_ids"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )


class TemporalStenosisDataset(Dataset):
    """Dataset yielding windows of T consecutive RGB frames + RF-DETR-format targets."""

    def __init__(self, split: str, cfg: Config):
        self.cfg = cfg
        self.split = split
        self.img_dir = cfg.data_root / split / "images"
        self.lbl_dir = cfg.data_root / split / "labels"

        sequences = build_sequence_index(self.img_dir)
        self.windows = build_windows(sequences, cfg.T)

        self.augment = (
            build_train_augmentation(cfg.img_size) if split == "train" else None
        )

        self.mean = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.std  = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

        print(
            f"[{split}] {len(sequences)} sequences, "
            f"{sum(len(s[2]) for s in sequences)} frames, "
            f"{len(self.windows)} windows of T={cfg.T}"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        paths = self.windows[idx]
        images = []
        targets = []
        saved_replay = None

        for frame_i, img_path in enumerate(paths):
            # Load grayscale and replicate to 3 channels
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            orig_h, orig_w = img.shape[:2]

            # Resize
            if orig_h != self.cfg.img_size or orig_w != self.cfg.img_size:
                img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size),
                                 interpolation=cv2.INTER_LINEAR)

            # Load YOLO labels and convert to pascal_voc for augmentation
            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            yolo_labels = load_yolo_labels(lbl_path)
            bboxes, class_ids = yolo_to_pascal(yolo_labels, self.cfg.img_size, self.cfg.img_size)
            bboxes = bboxes.tolist()
            class_ids = class_ids.tolist()

            # Augmentation (replay same transform for all T frames)
            if self.augment is not None:
                if frame_i == 0:
                    result = self.augment(image=img, bboxes=bboxes, class_ids=class_ids)
                    saved_replay = result["replay"]
                else:
                    result = ReplayCompose.replay(
                        saved_replay, image=img, bboxes=bboxes, class_ids=class_ids
                    )
                img = result["image"]
                bboxes = result["bboxes"]
                class_ids = result["class_ids"]

            # Convert to float [0, 1] and replicate 1ch → 3ch
            img = img.astype(np.float32) / 255.0
            img_3ch = np.stack([img, img, img], axis=0)  # (3, H, W)
            img_tensor = torch.from_numpy(img_3ch)

            # ImageNet normalisation
            img_tensor = (img_tensor - self.mean) / self.std
            images.append(img_tensor)

            # Convert boxes back to normalised cxcywh for RF-DETR loss
            if len(bboxes) > 0:
                boxes_np = np.array(bboxes, dtype=np.float32)
                boxes_cxcywh = pascal_to_cxcywh_norm(boxes_np, self.cfg.img_size, self.cfg.img_size)
                boxes = torch.from_numpy(boxes_cxcywh)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(len(boxes), dtype=torch.int64)
            targets.append({"boxes": boxes, "labels": labels})

        images = torch.stack(images, dim=0)  # (T, 3, H, W)
        return images, targets, [p.name for p in paths]


def collate_fn(batch):
    """Collate into (B, T, 3, H, W) images + list[list[dict]] targets."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    filenames = [item[2] for item in batch]
    return images, targets, filenames


def get_dataloader(split: str, cfg: Config, shuffle: bool = False) -> DataLoader:
    ds = TemporalStenosisDataset(split, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
