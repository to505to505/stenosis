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

# CADICA filename pattern: p{N}_v{M}_{frame}.png
_CADICA_RE = re.compile(
    r"^(p\d+)_v(\d+)_(\d+)\.(?:png|jpg)$"
)


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


def build_geometric_augmentation() -> ReplayCompose:
    """Geometric-only augmentation (resolution-invariant).

    Used when distillation is enabled so the same geometric replay can be
    applied to both the student frames (cfg.img_size) AND the teacher's HR
    centre frame (cfg.distill_teacher_resolution) and yield matching anatomy.
    """
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
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_ids"],
            min_area=1.0,
            min_visibility=0.2,
        ),
    )


def build_photometric_augmentation() -> A.Compose:
    """Photometric-only augmentation (student frames only)."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.3, contrast_limit=0.3, p=0.5,
            ),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        ]
    )


class TemporalStenosisDataset(Dataset):
    """Dataset yielding windows of T consecutive RGB frames + RF-DETR-format targets."""

    def __init__(
        self,
        split: str,
        cfg: Config,
        with_teacher_frame: bool = False,
        with_paired_window: bool = False,
    ):
        self.cfg = cfg
        self.split = split
        self.with_teacher_frame = with_teacher_frame
        self.with_paired_window = with_paired_window
        self.img_dir = cfg.data_root / split / "images"
        self.lbl_dir = cfg.data_root / split / "labels"

        sequences = build_sequence_index(self.img_dir)
        self.windows = build_windows(sequences, cfg.T)

        # Build paired-window index: (idx_A, idx_B) where window B = window A
        # shifted by ``consistency_offset`` frames within the same sequence.
        # Skips short / padded sequences so both items always have a true
        # neighbour.
        self.paired_windows: List[Tuple[int, int]] = []
        if with_paired_window:
            offset = int(getattr(cfg, "consistency_offset", 1))
            assert offset >= 1, f"consistency_offset must be >=1, got {offset}"
            cursor = 0
            for _pid, _sid, paths in sequences:
                n = len(paths)
                if n == 0:
                    continue
                if n < cfg.T:
                    cursor += 1                       # one padded window, no pair
                    continue
                nw = n - cfg.T + 1
                for k in range(nw - offset):
                    self.paired_windows.append((cursor + k, cursor + k + offset))
                cursor += nw

        if split == "train":
            if with_teacher_frame:
                # Split pipelines so the teacher frame can share the geometric
                # transform (no photometric perturbation on the HR target).
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
        self.std  = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

        print(
            f"[{split}] {len(sequences)} sequences, "
            f"{sum(len(s[2]) for s in sequences)} frames, "
            f"{len(self.windows)} windows of T={cfg.T}"
            + (f", {len(self.paired_windows)} paired windows" if with_paired_window else "")
        )

    def __len__(self) -> int:
        if self.with_paired_window:
            return len(self.paired_windows)
        return len(self.windows)

    def _to_imagenet_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img_3ch = np.stack([img, img, img], axis=0)  # (3, H, W)
        t = torch.from_numpy(img_3ch)
        return (t - self.mean) / self.std

    def _load_window(
        self, paths, saved_replay=None,
    ) -> Tuple[torch.Tensor, list, object]:
        """Load one window of T frames + targets, optionally re-using an
        already-saved augmentation replay so multiple windows share the same
        geometric transform (used by the paired-window mode for the overlap).

        Returns:
            frames:  (T, 3, H, W) ImageNet-normalised tensor
            targets: list of {"boxes", "labels"} dicts, one per frame
            replay:  the augmentation replay used (for re-use on the next
                     window or on the HR teacher frame)
        """
        images = []
        targets = []
        for frame_i, img_path in enumerate(paths):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            orig_h, orig_w = img.shape[:2]
            if orig_h != self.cfg.img_size or orig_w != self.cfg.img_size:
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

    def _load_teacher_centre(self, centre_path: Path, saved_replay) -> torch.Tensor:
        """Load + transform the HR teacher centre frame using the same
        geometric replay as the student window."""
        teacher_orig = cv2.imread(str(centre_path), cv2.IMREAD_GRAYSCALE)
        if teacher_orig is None:
            raise FileNotFoundError(f"Cannot read {centre_path}")
        if teacher_orig.shape[:2] != (self.teacher_size, self.teacher_size):
            teacher_orig = cv2.resize(
                teacher_orig, (self.teacher_size, self.teacher_size),
                interpolation=cv2.INTER_AREA,
            )
        if saved_replay is not None:
            result = ReplayCompose.replay(
                saved_replay, image=teacher_orig, bboxes=[], class_ids=[],
            )
            teacher_img = result["image"]
        else:
            teacher_img = teacher_orig
        if teacher_img.shape[:2] != (self.teacher_size, self.teacher_size):
            teacher_img = cv2.resize(
                teacher_img, (self.teacher_size, self.teacher_size),
                interpolation=cv2.INTER_AREA,
            )
        return self._to_imagenet_tensor(teacher_img)

    def __getitem__(self, idx: int):
        # ── Paired-window mode (sliding-window consistency) ─────────
        if self.with_paired_window:
            idx_a, idx_b = self.paired_windows[idx]
            paths_a = self.windows[idx_a]
            paths_b = self.windows[idx_b]
            # Load A first (creates the replay), then re-use the replay for B
            # so the overlapping frames share identical augmentation geometry.
            frames_a, targets_a, replay = self._load_window(paths_a, saved_replay=None)
            frames_b, targets_b, _ = self._load_window(paths_b, saved_replay=replay)
            fnames = (
                [p.name for p in paths_a],
                [p.name for p in paths_b],
            )
            if self.with_teacher_frame:
                teacher_a = self._load_teacher_centre(paths_a[self.centre], replay)
                return (
                    frames_a, targets_a, frames_b, targets_b,
                    teacher_a, fnames,
                )
            return frames_a, targets_a, frames_b, targets_b, fnames

        # ── Standard single-window mode ─────────────────────────────
        paths = self.windows[idx]
        frames, targets, replay = self._load_window(paths, saved_replay=None)
        if self.with_teacher_frame:
            teacher_tensor = self._load_teacher_centre(paths[self.centre], replay)
            return frames, targets, teacher_tensor, [p.name for p in paths]
        return frames, targets, [p.name for p in paths]


def collate_fn(batch):
    """Collate into (B, T, 3, H, W) images + list[list[dict]] targets."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    filenames = [item[2] for item in batch]
    return images, targets, filenames


def collate_fn_with_teacher(batch):
    """Collate with HR teacher centre frame: (images, targets, teacher, fnames)."""
    images = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    teacher = torch.stack([item[2] for item in batch], dim=0)
    filenames = [item[3] for item in batch]
    return images, targets, teacher, filenames


def collate_fn_paired(batch):
    """Collate paired-window items (no teacher frame).
    Returns: (images_a, targets_a, images_b, targets_b, fnames)
    """
    images_a = torch.stack([it[0] for it in batch], dim=0)
    targets_a = [it[1] for it in batch]
    images_b = torch.stack([it[2] for it in batch], dim=0)
    targets_b = [it[3] for it in batch]
    fnames = [it[4] for it in batch]
    return images_a, targets_a, images_b, targets_b, fnames


def collate_fn_paired_with_teacher(batch):
    """Collate paired-window items WITH HR teacher centre frame for window A.
    Returns: (images_a, targets_a, images_b, targets_b, teacher_a, fnames)
    """
    images_a = torch.stack([it[0] for it in batch], dim=0)
    targets_a = [it[1] for it in batch]
    images_b = torch.stack([it[2] for it in batch], dim=0)
    targets_b = [it[3] for it in batch]
    teacher_a = torch.stack([it[4] for it in batch], dim=0)
    fnames = [it[5] for it in batch]
    return images_a, targets_a, images_b, targets_b, teacher_a, fnames


def get_dataloader(
    split: str,
    cfg: Config,
    shuffle: bool = False,
    with_teacher_frame: bool = False,
    with_paired_window: bool = False,
) -> DataLoader:
    ds = TemporalStenosisDataset(
        split, cfg,
        with_teacher_frame=with_teacher_frame,
        with_paired_window=with_paired_window,
    )
    if with_paired_window:
        coll = collate_fn_paired_with_teacher if with_teacher_frame else collate_fn_paired
    else:
        coll = collate_fn_with_teacher if with_teacher_frame else collate_fn
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=coll,
        pin_memory=True,
        drop_last=(split == "train"),
    )
