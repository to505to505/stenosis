"""Centralized configuration for the Spatio-Temporal Stenosis Detector."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    data_root: Path = Path("/workspace/stenosis/data/dataset2_split")
    output_dir: Path = Path("/workspace/stenosis/stenosis_temporal/runs")

    # ── Image ──────────────────────────────────────────────────────────
    img_h: int = 512
    img_w: int = 512
    in_channels: int = 1          # grayscale
    pixel_mean: float = 103.53
    pixel_std: float = 57.12

    # ── Sequence ───────────────────────────────────────────────────────
    T: int = 5                    # consecutive frames per sample
    K: int = 4                    # spatial shift directions
    shift_fraction: float = 0.5   # shift by 0.5× proposal size

    # ── FPE ────────────────────────────────────────────────────────────
    C: int = 256                  # FPN output channels
    S: int = 128                  # RPN proposals per frame
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 128
    rpn_post_nms_top_n_test: int = 128
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    anchor_sizes: tuple = ((16,), (32,), (64,), (128,), (256,))
    anchor_ratios: tuple = ((0.5, 1.0, 2.0),) * 5

    # ── RoI Align ──────────────────────────────────────────────────────
    roi_output_size: int = 7      # w_r = h_r = 7

    # ── PSTFA (Transformer) ───────────────────────────────────────────
    D: int = 512                  # transformer input dimension
    num_transformer_layers: int = 4   # L
    nhead: int = 8
    dim_feedforward: int = 2048   # 4 × D

    # ── MTO ────────────────────────────────────────────────────────────
    num_classes: int = 2          # background + stenosis
    fc_hidden: int = 1024

    # ── Detection post-processing ─────────────────────────────────────
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100
    # IoU thresholds for assigning proposals to GT during training
    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.5

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 100
    batch_size: int = 4
    num_workers: int = 4
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    lr_step_milestones: tuple = (60, 80)
    lr_gamma: float = 0.1
    grad_accum_steps: int = 1     # 1 = no accumulation (direct batch)

    # ── Logging ─────────────────────────────────────────────────────────
    wandb_project: str = "stenosis-temporal"
    wandb_enabled: bool = True
    run_name: str | None = None   # auto-generated if None

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True
    log_interval: int = 50        # print every N iterations
    eval_interval: int = 1        # evaluate every N epochs
    proposal_chunk_size: int = 128 # process all proposals at once (S=128)
    gradient_checkpointing: bool = False  # disabled — enough VRAM headroom


    @property
    def num_tokens(self) -> int:
        """T * (K + 1) tokens per proposal."""
        return self.T * (self.K + 1)
