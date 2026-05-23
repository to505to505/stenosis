"""Centralised configuration for Stenosis-DetNet.

Mirrors :mod:`psstt.config` for shared scaffolding fields so both
trainers can be operated with the same mental model. The DetNet-specific
fields are the Sequence Feature Fusion (SFF) hyper-parameters and the
Sequence Consistency Alignment (SCA) eval-time thresholds.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # Paths
    data_root: Path = Path("data/dataset2_split")
    output_dir: Path = Path("detnet/runs")
    run_name: Optional[str] = None

    # Image / sequence
    img_size: int = 512
    T: int = 5  # paper uses N=9; we mirror psstt's T=5 for fair comparison

    # Normalisation (ImageNet — matches torchvision COCO Faster-R-CNN backbone)
    pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ─── Faster R-CNN backbone / RPN ───────────────────────────────────
    # The paper replaces the RPN with Guided Anchoring; we use the standard
    # torchvision RPN as a pragmatic stand-in (Guided Anchoring is not part
    # of torchvision and would need a from-scratch port). The paper's main
    # contributions — SFF and SCA — are preserved verbatim below.
    pretrained_coco: bool = True
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 1000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_test: int = 400

    # ─── SFF (Sequence Feature Fusion) ─────────────────────────────────
    num_classes: int = 1                # stenosis (FastRCNNPredictor gets 2 logits)
    backbone_out_channels: int = 256
    roi_size: int = 7                   # 256×7×7 candidate-box feature (Table 1)
    sff_token_dim: int = 512            # D — projection target for SFF tokens
    sff_heads: int = 8                  # H — multi-head attention heads
    sff_dropout: float = 0.0
    sff_use_residual_concat: bool = True  # Eq. 3 residual-and-concat form

    # Cap the SFF token pool per (B, window) to keep self-attention tractable.
    # With box_batch_size_per_image=128 and T=5 we get 640 tokens per batch —
    # easily fits in one self-attention layer.

    # Per-image fg/bg sampling for the box head (mirrors psstt).
    box_batch_size_per_image: int = 128
    box_positive_fraction: float = 0.25
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5

    # Post-processing thresholds (test-time).
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100

    # Supervise every reference frame's predictions (paper-faithful — SCA
    # consumes per-frame detections from every frame in the window). Set
    # False to supervise only the centre frame as a memory-saving fallback.
    supervise_all_frames: bool = True

    # ─── SCA (Sequence Consistency Alignment) — eval-time only ─────────
    sca_enabled_eval: bool = True       # apply SCA on test/val sliding-window outputs
    sca_t_iou: float = 0.2              # IoU threshold for clustering (paper default)
    sca_t_frame: int = 3                # min frames a cluster must persist (paper: 5 for N=9;
                                        # scaled to ~T/2 for our T=5)
    sca_t_distance: float = 50.0        # max center-distance (px) for SSIM-based fallback
    sca_t_sim: float = 0.5              # SSIM threshold for distance-based clustering
    sca_interpolate_missing: bool = True  # fill kept clusters with linear-interp boxes

    # ─── Training ──────────────────────────────────────────────────────
    epochs: int = 60
    batch_size: int = 2
    num_workers: int = 4
    # Paper uses SGD lr=0.002 momentum=0.9 — we keep psstt's effective recipe
    # (slightly higher lr + grad accum) since it has proven to converge on
    # this dataset, then keep the paper's optimiser family.
    lr: float = 0.02
    lr_backbone_mult: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 1e-4
    grad_accum_steps: int = 8           # effective batch = 16
    warmup_iters: int = 500
    lr_schedule: str = "cosine"
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
    wandb_project: str = "detnet"
    wandb_enabled: bool = True
    log_interval: int = 50
    eval_interval: int = 2

    # ─── Misc ──────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True
