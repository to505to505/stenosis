"""Centralised configuration for the Video / STFS RF-DETR refactor.

Mirrors :class:`rfdetr_temporal.config.Config` for the knobs that still
apply (CRRCD, KD, optimiser) and replaces the per-pixel temporal-fusion
options with the new STFS + multi-frame-consistency knobs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    data_root: Path = Path("data/dataset2_split")
    output_dir: Path = Path("rfdetr_video/runs")

    # ── Image / sequence ───────────────────────────────────────────────
    img_size: int = 512
    T: int = 5

    # ── RF-DETR pretrained ─────────────────────────────────────────────
    rfdetr_checkpoint: str = "rfdetr_runs/dataset2_augs/checkpoint_best_total.pth"
    freeze_backbone: bool = True
    freeze_decoder: bool = False

    # ── Detection ──────────────────────────────────────────────────────
    hidden_dim: int = 256
    num_classes: int = 1
    num_queries: int = 300
    score_thresh: float = 0.05
    nms_thresh: float = 0.5

    # ── Normalisation ──────────────────────────────────────────────────
    pixel_mean: tuple = (0.485, 0.456, 0.406)
    pixel_std: tuple = (0.229, 0.224, 0.225)

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    lr_step_milestones: tuple = (30, 40)
    lr_gamma: float = 0.1
    grad_accum_steps: int = 2

    # ── Logging ────────────────────────────────────────────────────────
    wandb_project: str = "rfdetr-video"
    wandb_enabled: bool = True
    run_name: Optional[str] = None
    log_interval: int = 50
    eval_interval: int = 1

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True

    # ── KD-DETR distillation (specific + general sampling) ────────────
    distill_enabled: bool = False
    distill_teacher_ckpt: str = (
        "rfdetr_runs/rfdetr_large_arcade2x_704_reg/checkpoint_best_total.pth"
    )
    distill_teacher_resolution: int = 704
    distill_teacher_num_classes: int = 1
    distill_num_queries: int = 300
    distill_loss_weight: float = 1.0
    distill_kl_weight: float = 2.0
    distill_l1_weight: float = 5.0
    distill_giou_weight: float = 2.0
    distill_temperature: float = 1.0
    distill_min_weight: float = 0.0
    distill_use_aux_layers: bool = False
    distill_general_enabled: bool = False
    distill_num_general_queries: int = 100
    distill_general_loss_weight: float = 0.5
    distill_general_min_weight: float = 0.1
    distill_general_query_std: float = 0.02

    # ── CRRCD: per-frame relational contrastive distillation ──────────
    crrcd_enabled: bool = False
    crrcd_loss_weight: float = 2.0
    crrcd_relation_dim: int = 256
    crrcd_hidden_dim: int = 256
    crrcd_num_fg: int = 16
    crrcd_num_bg: int = 32
    crrcd_num_negatives: int = 16
    crrcd_temperature: float = 0.1

    # ── STFS: Query-Level Spatio-Temporal Feature Sharing ─────────────
    # Per-frame Hungarian tracking cost weights: c = w_iou·(1-IoU) +
    # w_l1·L1(centres) + w_cls·(1 - class_prob).
    stfs_iou_weight: float = 2.0
    stfs_l1_weight: float = 5.0
    stfs_cls_weight: float = 2.0
    # Reject a Hungarian match if IoU falls below this threshold; the
    # corresponding frame is then marked Hypothesis-False-Negative (H-FN).
    stfs_match_iou_thresh: float = 0.1
    # Confidence threshold used to seed and update tracks.
    stfs_track_score_thresh: float = 0.3
    # Drop tracks shorter than this many confident frames (H-FP filter).
    stfs_min_track_len: int = 3
    # α in: q[t,q] ← (1-α)·q[t,q] + α·q[t*,q*] for H-FN slots.
    stfs_inject_alpha: float = 1.0

    # ── Multi-frame count consistency loss (L_num) ───────────────────
    consistency_enabled: bool = True
    consistency_weight: float = 0.5
    # Hard threshold for counting boxes; soft-count surrogate uses this
    # plus ``consistency_soft_temp`` so gradient flows.
    consistency_threshold: float = 0.3
    consistency_soft_temp: float = 0.05

    # ── Temporal Dropout (train-only frame masking) ──────────────────
    temporal_dropout_enabled: bool = False
    temporal_dropout_prob: float = 0.25
    temporal_dropout_centre_p: float = 1.0
    temporal_dropout_neighbour_p: float = 0.3
    temporal_dropout_radius: int = 1
    temporal_dropout_noise_std: float = 1.0
