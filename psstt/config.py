"""Centralised configuration for PSSTT.

Mirrors the surface of :mod:`rfdetr_video.config` for shared fields
(``T``, ``img_size``, EMA, selection, early-stop, logging) so the two
trainers can be operated with the same mental model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # Paths
    data_root: Path = Path("data/dataset2_split")
    output_dir: Path = Path("psstt/runs")
    run_name: Optional[str] = None

    # Image / sequence
    img_size: int = 512
    T: int = 5

    # Normalisation (ImageNet — matches torchvision COCO Faster R-CNN backbone)
    pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ─── Faster R-CNN backbone / RPN ───────────────────────────────────
    # Whether to load torchvision COCO pretrained weights for the full
    # Faster R-CNN (we replace the box head, but backbone + RPN benefit).
    pretrained_coco: bool = True

    # RPN proposals per frame (paper uses S=400 after NMS).
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 1000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_test: int = 400

    # ─── PSSTT / TFA ───────────────────────────────────────────────────
    num_classes: int = 1            # stenosis (FastRCNNPredictor gets 2 logits incl. bg)
    backbone_out_channels: int = 256
    roi_size: int = 7
    num_shifts: int = 4             # K=4 (up/down/left/right); +1 zero-shift = 5
    token_dim: int = 512            # D in the paper
    tfa_depth: int = 4              # L=4 Transformer layers
    tfa_heads: int = 8
    tfa_mlp_ratio: float = 4.0
    tfa_dropout: float = 0.0

    # Per-image fg/bg sampling for the box head.
    # Defaults reduced from torchvision (512) to fit a 12 GB GPU at T=5.
    box_batch_size_per_image: int = 128
    box_positive_fraction: float = 0.25
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5

    # Postprocessing thresholds (test-time).
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100

    # During training: supervise every reference frame's predictions
    # (paper-faithful). Set False to supervise only the centre frame —
    # a memory-saving fall-back at the cost of weaker supervision.
    supervise_all_frames: bool = True

    # ─── Training ──────────────────────────────────────────────────────
    epochs: int = 60
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 0.02
    lr_backbone_mult: float = 0.1   # backbone LR = lr * mult
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_accum_steps: int = 8       # effective batch = 16
    warmup_iters: int = 500
    lr_schedule: str = "cosine"     # "cosine" | "multistep"
    lr_step_milestones: Tuple[int, ...] = (40, 50)
    lr_gamma: float = 0.1
    grad_clip_max_norm: float = 10.0

    # ─── EMA + checkpoint selection + early stopping ───────────────────
    ema_enabled: bool = True
    ema_decay: float = 0.999
    selection_smooth_k: int = 3
    selection_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
    early_stop_enabled: bool = True
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.0

    # ─── Logging ───────────────────────────────────────────────────────
    wandb_project: str = "psstt"
    wandb_enabled: bool = True
    log_interval: int = 50
    eval_interval: int = 2

    # ─── Misc ──────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True
