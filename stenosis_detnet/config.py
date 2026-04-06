"""Centralized configuration for Stenosis-DetNet."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    data_root: Path = Path("/workspace/stenosis/data/dataset2_split")
    output_dir: Path = Path("/workspace/stenosis/stenosis_detnet/runs")

    # ── Image ──────────────────────────────────────────────────────────
    img_h: int = 512
    img_w: int = 512
    in_channels: int = 1          # grayscale
    pixel_mean: float = 0.394     # on 0-1 scale
    pixel_std: float = 0.181      # on 0-1 scale

    # ── Sequence ───────────────────────────────────────────────────────
    T: int = 9                    # consecutive frames per sample

    # ── Backbone / FPN ─────────────────────────────────────────────────
    C: int = 256                  # FPN output channels

    # ── Guided Anchoring RPN ───────────────────────────────────────────
    S: int = 400                  # proposals per frame
    ga_loc_filter_thr: float = 0.01   # location filtering threshold
    ga_center_ratio: float = 0.2      # center region ratio for loc targets
    ga_allowed_border: int = -1
    ga_deform_groups: int = 4
    ga_square_anchor_scale: int = 4   # base scale for square anchors
    ga_approx_anchor_sizes: tuple = ((16,), (32,), (64,), (128,), (256,))
    ga_approx_anchor_ratios: tuple = ((0.5, 1.0, 2.0),) * 5
    ga_strides: tuple = (4, 8, 16, 32, 64)  # FPN level strides

    rpn_nms_thresh: float = 0.7
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 400
    rpn_post_nms_top_n_test: int = 400

    # ── RoI Align ──────────────────────────────────────────────────────
    roi_output_size: int = 7      # w_r = h_r = 7

    # ── Sequence Feature Fusion (SFF) ──────────────────────────────────
    sff_d_model: int = 512        # projection dimension
    sff_num_heads: int = 8        # H attention heads
    sff_epsilon: float = 1e-8     # denominator epsilon
    sff_dropout: float = 0.1

    # ── Classification & Regression Heads ──────────────────────────────
    num_classes: int = 2          # background + stenosis
    fc_hidden: int = 1024
    head_dropout: float = 0.5

    # ── Detection post-processing ─────────────────────────────────────
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.5

    # ── SCA Post-Processing ────────────────────────────────────────────
    sca_t_iou: float = 0.2
    sca_t_distance: float = 50.0  # pixels at 512×512
    sca_t_sim: float = 0.5        # SSIM threshold
    sca_t_frame: int = 5          # min frames for a valid lesion

    # ── Centerline ────────────────────────────────────────────────────
    frangi_scale_range: tuple = (1, 8)
    frangi_scale_step: int = 2
    frangi_threshold: float = 0.02

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 0.002
    momentum: float = 0.9
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5

    # ── Logging ────────────────────────────────────────────────────────
    wandb_project: str = "stenosis-detnet"
    wandb_enabled: bool = True
    run_name: str | None = None

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True
    log_interval: int = 50
    eval_interval: int = 1
    gradient_checkpointing: bool = False

    @property
    def sff_input_dim(self) -> int:
        """Flattened RoI feature dimension: C * roi_size * roi_size."""
        return self.C * self.roi_output_size * self.roi_output_size

    @property
    def sff_head_dim(self) -> int:
        """Dimension per attention head."""
        return self.sff_d_model // self.sff_num_heads
