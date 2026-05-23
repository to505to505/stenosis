"""Stenosis-DetNet dataset adapter.

Wraps :class:`rfdetr_video.dataset.VideoStenosisDataset` and converts each
per-frame target from cxcywh-normalised (RF-DETR convention) to xyxy-pixel
with class index 1 (torchvision convention; 0 is background).

Structurally identical to :mod:`psstt.dataset` — the dataset layout, the
sliding-window protocol, and the per-frame target convention are shared
across the two trainers.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader

from rfdetr_video.config import Config as RFVideoConfig
from rfdetr_video.dataset import VideoStenosisDataset

from .config import Config


def cxcywh_norm_to_xyxy_px(boxes_norm: torch.Tensor, img_size: int) -> torch.Tensor:
    """Convert (N, 4) normalised cxcywh boxes to (N, 4) absolute xyxy pixels."""
    if boxes_norm.numel() == 0:
        return torch.zeros((0, 4), dtype=torch.float32)
    cx, cy, w, h = boxes_norm.unbind(-1)
    x1 = (cx - w / 2) * img_size
    y1 = (cy - h / 2) * img_size
    x2 = (cx + w / 2) * img_size
    y2 = (cy + h / 2) * img_size
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp_min(0)


def _to_rfdetr_cfg(cfg: Config) -> RFVideoConfig:
    """Build a minimal RF-DETR-video config with the fields VideoStenosisDataset reads."""
    shim = RFVideoConfig()
    shim.data_root = cfg.data_root
    shim.img_size = cfg.img_size
    shim.T = cfg.T
    shim.pixel_mean = cfg.pixel_mean
    shim.pixel_std = cfg.pixel_std
    shim.batch_size = cfg.batch_size
    shim.num_workers = cfg.num_workers
    shim.distill_enabled = False
    shim.temporal_dropout_enabled = False
    shim.dynamic_batch_resize_enabled = False
    return shim


class DetNetVideoDataset(Dataset):
    """T-frame sliding-window dataset with torchvision-style per-frame targets."""

    def __init__(self, split: str, cfg: Config):
        self.cfg = cfg
        self.img_size = cfg.img_size
        self.inner = VideoStenosisDataset(split, _to_rfdetr_cfg(cfg))

    def __len__(self) -> int:
        return len(self.inner)

    def __getitem__(self, idx: int):
        frames, targets, fnames = self.inner[idx]
        tv_targets: List[Dict[str, torch.Tensor]] = []
        for t_dict in targets:
            boxes_xyxy = cxcywh_norm_to_xyxy_px(t_dict["boxes"], self.img_size)
            if boxes_xyxy.shape[0] > 0:
                keep = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (
                    boxes_xyxy[:, 3] > boxes_xyxy[:, 1]
                )
                boxes_xyxy = boxes_xyxy[keep]
            labels = torch.ones(boxes_xyxy.shape[0], dtype=torch.int64)
            tv_targets.append({"boxes": boxes_xyxy, "labels": labels})
        return frames, tv_targets, fnames


def collate_video(batch):
    """(B, T, 3, H, W); list[B] of list[T] of {boxes, labels}; list[B] of fname-tuples."""
    images = torch.stack([it[0] for it in batch], dim=0)
    targets = [it[1] for it in batch]
    fnames = [it[2] for it in batch]
    return images, targets, fnames


def get_dataloader(split: str, cfg: Config, shuffle: bool) -> DataLoader:
    ds = DetNetVideoDataset(split, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_video,
        pin_memory=True,
        drop_last=(split == "train"),
    )
