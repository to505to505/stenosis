"""Video dataset for RF-DETR.

Returns sliding windows of ``T`` consecutive RGB frames together with
**per-frame** targets (``len(targets) == T``), so the multi-frame
detection loss can supervise every frame in the window.

Differences vs. :mod:`rfdetr_temporal.dataset`:
  • Targets list is no longer collapsed to the centre frame.
  • Optional ``with_teacher_frame`` returns *all* T frames at the HR
    teacher resolution (default 704), with the same geometric replay,
    so per-frame CRRCD distillation is spatially aligned.
  • Paired-window mode and temporal dropout are removed.
"""

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

# Reuse filename parsing + helpers from the existing temporal dataset to
# avoid divergence on dataset layout edge cases.
from rfdetr_temporal.dataset import (
    build_sequence_index,
    build_windows,
    load_yolo_labels,
    yolo_to_pascal,
    pascal_to_cxcywh_norm,
    build_train_augmentation,
    build_geometric_augmentation,
    build_photometric_augmentation,
)


class VideoStenosisDataset(Dataset):
    """Sliding-window dataset returning per-frame targets."""

    def __init__(
        self,
        split: str,
        cfg: Config,
        with_teacher_frame: bool = False,
    ):
        self.cfg = cfg
        self.split = split
        self.with_teacher_frame = with_teacher_frame
        self.img_dir = cfg.data_root / split / "images"
        self.lbl_dir = cfg.data_root / split / "labels"

        sequences = build_sequence_index(self.img_dir)
        self.windows = build_windows(sequences, cfg.T)

        if split == "train":
            if with_teacher_frame:
                self.geom_aug = build_geometric_augmentation()
                self.photo_aug = build_photometric_augmentation()
                self.augment = None
            else:
                self.augment = build_train_augmentation(cfg.img_size)
                self.geom_aug = None
                self.photo_aug = None
        else:
            self.augment = None
            self.geom_aug = None
            self.photo_aug = None

        self.centre = cfg.T // 2
        self.teacher_size = int(cfg.distill_teacher_resolution)

        self.mean = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

        print(
            f"[{split}] {len(sequences)} sequences, "
            f"{sum(len(s[2]) for s in sequences)} frames, "
            f"{len(self.windows)} windows of T={cfg.T}"
            + (" (+teacher HR T-frames)" if with_teacher_frame else "")
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _to_imagenet_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img_3ch = np.stack([img, img, img], axis=0)
        t = torch.from_numpy(img_3ch)
        return (t - self.mean) / self.std

    def _augment_one(
        self, img: np.ndarray, bboxes, class_ids, saved_replay,
    ):
        """Apply the configured augmentation pipeline; create a replay
        on the first frame and reuse it across all subsequent frames.
        """
        if self.geom_aug is not None:
            if saved_replay is None:
                result = self.geom_aug(
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
            if self.photo_aug is not None:
                img = self.photo_aug(image=img)["image"]
        elif self.augment is not None:
            if saved_replay is None:
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
        return img, bboxes, class_ids, saved_replay

    def _load_window(
        self, paths,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], object]:
        """Load T frames + per-frame targets at student resolution."""
        images = []
        targets = []
        saved_replay = None
        for img_path in paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            if img.shape[:2] != (self.cfg.img_size, self.cfg.img_size):
                img = cv2.resize(
                    img, (self.cfg.img_size, self.cfg.img_size),
                    interpolation=cv2.INTER_AREA,
                )

            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            yolo_labels = load_yolo_labels(lbl_path)
            bboxes, class_ids = yolo_to_pascal(
                yolo_labels, self.cfg.img_size, self.cfg.img_size,
            )
            bboxes = bboxes.tolist()
            class_ids = class_ids.tolist()

            img, bboxes, class_ids, saved_replay = self._augment_one(
                img, bboxes, class_ids, saved_replay,
            )

            images.append(self._to_imagenet_tensor(img))

            if len(bboxes) > 0:
                boxes_np = np.array(bboxes, dtype=np.float32)
                boxes_cxcywh = pascal_to_cxcywh_norm(
                    boxes_np, self.cfg.img_size, self.cfg.img_size,
                )
                boxes = torch.from_numpy(boxes_cxcywh)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(len(boxes), dtype=torch.int64)
            targets.append({"boxes": boxes, "labels": labels})

        frames = torch.stack(images, dim=0)  # (T, 3, H, W)
        return frames, targets, saved_replay

    def _load_teacher_window(self, paths, saved_replay) -> torch.Tensor:
        """Load all T frames at the teacher HR resolution, replaying the
        student's geometric augmentation. Returns ``(T, 3, S, S)`` where
        ``S = teacher_size``.
        """
        out = []
        for img_path in paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            if img.shape[:2] != (self.teacher_size, self.teacher_size):
                img = cv2.resize(
                    img, (self.teacher_size, self.teacher_size),
                    interpolation=cv2.INTER_AREA,
                )
            if saved_replay is not None:
                result = ReplayCompose.replay(
                    saved_replay, image=img, bboxes=[], class_ids=[],
                )
                img = result["image"]
            if img.shape[:2] != (self.teacher_size, self.teacher_size):
                img = cv2.resize(
                    img, (self.teacher_size, self.teacher_size),
                    interpolation=cv2.INTER_AREA,
                )
            out.append(self._to_imagenet_tensor(img))
        return torch.stack(out, dim=0)

    def _apply_temporal_dropout(self, frames: torch.Tensor) -> torch.Tensor:
        """Replace selected frames with Gaussian noise (train-only).
        Centre is the primary target; neighbours within ±radius are masked
        with independent probability.
        """
        cfg = self.cfg
        if not getattr(cfg, "temporal_dropout_enabled", False):
            return frames
        if self.split != "train":
            return frames
        if np.random.rand() >= float(cfg.temporal_dropout_prob):
            return frames

        T = frames.shape[0]
        centre = T // 2
        masked = []
        if np.random.rand() < float(cfg.temporal_dropout_centre_p):
            masked.append(centre)

        radius = int(cfg.temporal_dropout_radius)
        p_n = float(cfg.temporal_dropout_neighbour_p)
        if radius > 0 and p_n > 0.0:
            for delta in range(1, radius + 1):
                for cand in (centre - delta, centre + delta):
                    if 0 <= cand < T and np.random.rand() < p_n:
                        masked.append(cand)

        if not masked:
            return frames

        std = float(cfg.temporal_dropout_noise_std)
        for t_idx in masked:
            frames[t_idx] = torch.randn_like(frames[t_idx]) * std
        return frames

    def __getitem__(self, idx: int):
        paths = self.windows[idx]
        frames, targets, replay = self._load_window(paths)
        frames = self._apply_temporal_dropout(frames)
        fnames = [p.name for p in paths]
        if self.with_teacher_frame:
            teacher_frames = self._load_teacher_window(paths, replay)
            return frames, targets, teacher_frames, fnames
        return frames, targets, fnames


# ─────────────────────────────────────────────────────────────────────
#  Collate
# ─────────────────────────────────────────────────────────────────────
def collate_video(batch):
    """(B, T, 3, H, W), List[B] of List[T] of {boxes, labels}, fnames."""
    images = torch.stack([it[0] for it in batch], dim=0)
    targets = [it[1] for it in batch]
    fnames = [it[2] for it in batch]
    return images, targets, fnames


def collate_video_with_teacher(batch):
    images = torch.stack([it[0] for it in batch], dim=0)
    targets = [it[1] for it in batch]
    teacher = torch.stack([it[2] for it in batch], dim=0)  # (B, T, 3, S, S)
    fnames = [it[3] for it in batch]
    return images, targets, teacher, fnames


def get_video_dataloader(
    split: str,
    cfg: Config,
    shuffle: bool = False,
    with_teacher_frame: bool = False,
) -> DataLoader:
    ds = VideoStenosisDataset(
        split, cfg, with_teacher_frame=with_teacher_frame,
    )
    coll = collate_video_with_teacher if with_teacher_frame else collate_video
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=coll,
        pin_memory=True,
        drop_last=(split == "train"),
    )
