"""Centralized configuration for Temporal RF-DETR."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    data_root: Path = Path("data/dataset2_split")
    output_dir: Path = Path("rfdetr_temporal/runs")

    # ── Image ──────────────────────────────────────────────────────────
    img_size: int = 512           # RF-DETR Small resolution

    # ── Sequence ───────────────────────────────────────────────────────
    T: int = 5                    # consecutive frames per sample

    # ── RF-DETR pretrained ─────────────────────────────────────────────
    rfdetr_checkpoint: str = "rfdetr_runs/dataset2_augs/checkpoint_best_total.pth"
    freeze_backbone: bool = True  # freeze DINOv2 + projector
    freeze_decoder: bool = False  # fine-tune decoder

    # ── Temporal fusion ────────────────────────────────────────────────
    hidden_dim: int = 256         # must match RF-DETR hidden_dim
    temporal_attn_layers: int = 2 # transformer encoder layers for fusion
    temporal_nhead: int = 8       # attention heads in temporal fusion

    # ── Detection ──────────────────────────────────────────────────────
    num_classes: int = 1          # stenosis only
    num_queries: int = 300
    score_thresh: float = 0.05
    nms_thresh: float = 0.5

    # ── Normalisation (ImageNet, used by DINOv2) ──────────────────────
    pixel_mean: tuple = (0.485, 0.456, 0.406)
    pixel_std: tuple = (0.229, 0.224, 0.225)

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 1e-4
    lr_backbone: float = 1e-5    # lower LR for backbone if unfrozen
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    lr_step_milestones: tuple = (30, 40)
    lr_gamma: float = 0.1
    grad_accum_steps: int = 2

    # ── Logging ────────────────────────────────────────────────────────
    wandb_project: str = "rfdetr-temporal"
    wandb_enabled: bool = True
    run_name: Optional[str] = None
    log_interval: int = 50
    eval_interval: int = 1

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True
