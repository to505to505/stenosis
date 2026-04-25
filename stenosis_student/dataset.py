"""Temporal sequence dataset for the stenosis student.

Mirrors the windowing and augmentation pipeline of ``rfdetr_temporal`` but
returns FCOS-format targets (xyxy in absolute pixels) and supervises only
the centre frame of each T-frame window.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations import ReplayCompose

from .config import Config

# Filename pattern: {patient_id}_{seq_id}_{frame_num}_bmp_jpg.rf.<uuid>.jpg
_FNAME_RE = re.compile(
    r"^(\d+_\d+)_(\d+)_(\d+)_bmp_jpg\.rf\.[0-9a-f]+\.jpg$"
)
_CADICA_RE = re.compile(r"^(p\d+)_v(\d+)_(\d+)\.(?:png|jpg)$")


def parse_filename(fname: str) -> Optional[Tuple[str, int, int]]:
    m = _FNAME_RE.match(fname)
    if m is not None:
        return m.group(1), int(m.group(2)), int(m.group(3))
    m = _CADICA_RE.match(fname)
    if m is not None:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return None


def build_sequence_index(image_dir: Path) -> List[Tuple[str, int, List[Path]]]:
    groups: Dict[Tuple[str, int], List[Tuple[int, Path]]] = defaultdict(list)
    for p in sorted(image_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".png"):
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
                windows.append(paths[start:start + T])
    return windows


def load_yolo_labels(label_path: Path) -> np.ndarray:
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.loadtxt(label_path, dtype=np.float32).reshape(-1, 5)


def yolo_to_xyxy(labels: np.ndarray, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    """YOLO normalised (cx, cy, w, h) → absolute pascal_voc xyxy."""
    if len(labels) == 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int64)
    cls = labels[:, 0].astype(np.int64)
    cx = labels[:, 1] * img_w
    cy = labels[:, 2] * img_h
    w = labels[:, 3] * img_w
    h = labels[:, 4] * img_h
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.column_stack([x1, y1, x2, y2]).astype(np.float32), cls


def build_train_augmentation(_img_size: int) -> ReplayCompose:
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
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
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


class TemporalStenosisStudentDataset(Dataset):
    """Yields ``(images, centre_clean, centre_target, filenames)`` per item.

    - ``images``: ``(T, 3, H, W)`` — augmented + normalised clip; the centre
      frame may have an asymmetric Cutout applied if Temporal Dropout is on.
    - ``centre_clean``: ``(3, H, W)`` un-corrupted, normalised centre frame
      (used by the frozen teacher).  ``None`` when neither distillation nor
      temporal dropout is enabled, to save memory.
    - ``centre_target``: ``{"boxes": (n, 4) xyxy abs px, "labels": (n,) int64}``
      for the centre frame (boxes are NOT modified by Temporal Dropout).
    - ``filenames``: list of ``T`` filenames in temporal order.
    """

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
        self.std = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

        # Whether to also produce a clean centre-frame copy for distillation
        # / temporal-consistency consumers.
        self._needs_clean = bool(
            cfg.distill_enabled or cfg.temporal_dropout_enabled
        )
        # Temporal Dropout is only meaningful during training.
        self._apply_dropout = bool(
            cfg.temporal_dropout_enabled and split == "train"
        )

        n_frames = sum(len(s[2]) for s in sequences)
        print(
            f"[{split}] {len(sequences)} sequences, {n_frames} frames, "
            f"{len(self.windows)} windows of T={cfg.T}"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _normalise(self, img_uint8_or_float: np.ndarray) -> torch.Tensor:
        """Grayscale [0,1] HxW float (or uint8) → normalised (3, H, W) tensor."""
        img = img_uint8_or_float
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        img_3ch = np.stack([img, img, img], axis=0)
        t = torch.from_numpy(img_3ch)
        return (t - self.mean) / self.std

    def _maybe_apply_temporal_dropout(self, centre_img_float: np.ndarray) -> np.ndarray:
        """Optionally paint a single random Cutout square on the centre frame.

        Operates on a normalised-but-pre-mean-subtraction grayscale image
        (float32 in [0, 1], HxW).  Returns a NEW array; the input is not
        modified.  The Cutout fill value is ``cfg.temporal_dropout_fill``
        (default 0 → black before normalisation).
        """
        cfg = self.cfg
        if np.random.rand() >= cfg.temporal_dropout_prob:
            return centre_img_float
        h, w = centre_img_float.shape[:2]
        side_min = max(1, int(round(cfg.temporal_dropout_min_frac * h)))
        side_max = max(side_min, int(round(cfg.temporal_dropout_max_frac * h)))
        side = np.random.randint(side_min, side_max + 1)
        y0 = np.random.randint(0, max(h - side + 1, 1))
        x0 = np.random.randint(0, max(w - side + 1, 1))
        out = centre_img_float.copy()
        out[y0:y0 + side, x0:x0 + side] = float(cfg.temporal_dropout_fill)
        return out

    def __getitem__(self, idx: int):
        paths = self.windows[idx]
        cfg = self.cfg
        img_size = cfg.img_size
        centre_idx = cfg.centre_index

        images: List[torch.Tensor] = []
        per_frame_targets: List[Dict[str, torch.Tensor]] = []
        saved_replay = None
        centre_clean_t: Optional[torch.Tensor] = None

        for frame_i, img_path in enumerate(paths):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            h0, w0 = img.shape[:2]
            if h0 != img_size or w0 != img_size:
                img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            yolo = load_yolo_labels(lbl_path)
            bboxes_np, cls_np = yolo_to_xyxy(yolo, img_size, img_size)
            bboxes = bboxes_np.tolist()
            class_ids = cls_np.tolist()

            if self.augment is not None:
                if frame_i == 0:
                    result = self.augment(image=img, bboxes=bboxes, class_ids=class_ids)
                    saved_replay = result["replay"]
                else:
                    result = ReplayCompose.replay(
                        saved_replay, image=img, bboxes=bboxes, class_ids=class_ids,
                    )
                img = result["image"]
                bboxes = result["bboxes"]
                class_ids = result["class_ids"]

            # Convert to [0,1] grayscale once; this is the canonical pre-norm
            # representation used by both the clean copy and the dropout
            # corruption.
            img_f = img.astype(np.float32) / 255.0

            if frame_i == centre_idx:
                # Snapshot the clean (post-aug, pre-norm) centre frame BEFORE
                # any Temporal Dropout corruption.  Only kept when needed.
                if self._needs_clean:
                    centre_clean_t = self._normalise(img_f.copy())
                if self._apply_dropout:
                    img_f = self._maybe_apply_temporal_dropout(img_f)

            images.append(self._normalise(img_f))

            if len(bboxes) > 0:
                boxes_t = torch.tensor(np.asarray(bboxes, dtype=np.float32),
                                       dtype=torch.float32)
                labels_t = torch.tensor(class_ids, dtype=torch.long)
            else:
                boxes_t = torch.zeros((0, 4), dtype=torch.float32)
                labels_t = torch.zeros((0,), dtype=torch.long)
            per_frame_targets.append({"boxes": boxes_t, "labels": labels_t})

        images_t = torch.stack(images, dim=0)  # (T, 3, H, W)
        centre_target = per_frame_targets[centre_idx]
        return images_t, centre_clean_t, centre_target, [p.name for p in paths]


def collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)  # (B, T, 3, H, W)
    centre_clean_list = [item[1] for item in batch]
    if any(c is None for c in centre_clean_list):
        centre_clean = None
    else:
        centre_clean = torch.stack(centre_clean_list, dim=0)  # (B, 3, H, W)
    targets = [item[2] for item in batch]
    filenames = [item[3] for item in batch]
    return images, centre_clean, targets, filenames


def get_dataloader(split: str, cfg: Config, shuffle: bool = False) -> DataLoader:
    ds = TemporalStenosisStudentDataset(split, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == "train"),
    )
