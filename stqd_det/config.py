"""Centralized configuration for STQD-Det."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    data_root: Path = Path("data/dataset2_split")
    output_dir: Path = Path("stqd_det/runs")

    # ── Image ──────────────────────────────────────────────────────────
    img_h: int = 512
    img_w: int = 512
    in_channels: int = 1          # grayscale
    pixel_mean: float = 0.394     # on 0-1 scale
    pixel_std: float = 0.181      # on 0-1 scale

    # ── Sequence ───────────────────────────────────────────────────────
    T: int = 9                    # consecutive frames per mini-batch

    # ── Backbone / FPN ─────────────────────────────────────────────────
    C: int = 256                  # FPN output channels
    gradient_checkpointing: bool = False

    # ── GFE (Global Feature Enhancement) ───────────────────────────────
    gfe_num_heads: int = 8
    gfe_dropout: float = 0.1
    gfe_dc_groups: int = 4        # groups for dynamic convolution

    # ── SQNB (Sequential Quantum Noise Box) ────────────────────────────
    num_proposals: int = 300      # number of noise proposals per frame
    diffusion_steps: int = 1000   # total diffusion timesteps
    snr_scale: float = 2.0        # signal-to-noise scaling factor
    box_renewal: bool = True      # renew boxes during iterative inference

    # ── Decoder ────────────────────────────────────────────────────────
    decoder_layers: int = 6       # DiffusionDet-style decoder depth
    decoder_dim: int = 256        # hidden dimension in decoder
    decoder_heads: int = 8        # attention heads in decoder
    decoder_ffn_dim: int = 2048   # feed-forward dimension
    decoder_dropout: float = 0.0

    # ── RoI Align ──────────────────────────────────────────────────────
    roi_output_size: int = 7      # w_r = h_r = 7

    # ── STFS (Spatio-Temporal Feature Sharing) ─────────────────────────
    stfs_alpha: float = 2.0       # RoI padding coefficient
    stfs_iou_weight: float = 1.0  # primary matching weight (IoU)
    stfs_dist_weight: float = 0.5 # secondary matching weight (Manhattan)
    stfs_num_heads: int = 8       # aggregator MHA heads
    stfs_ffn_dim: int = 1024      # aggregator linear block dim

    # ── Classification & Regression Heads ──────────────────────────────
    num_classes: int = 2          # number of foreground classes
    fc_hidden: int = 256

    # ── Loss weights ───────────────────────────────────────────────────
    lambda_l1: float = 5.0
    lambda_giou: float = 2.0
    lambda_num: float = 1.0       # consistency loss weight
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    beta_consistency: float = 1e-6  # small constant in L_num

    # ── Detection post-processing ──────────────────────────────────────
    score_thresh: float = 0.5     # confidence threshold at inference
    nms_thresh: float = 0.5

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 1           # exactly 1 (9-frame sequence)
    num_workers: int = 4
    lr: float = 2.5e-5            # paper spec
    weight_decay: float = 1e-4    # paper spec
    warmup_iters: int = 500       # linear warmup for first 500 iterations
    max_grad_norm: float = 5.0
    early_stopping_patience: int = 10

    # ── Logging ────────────────────────────────────────────────────────
    wandb_project: str = "stqd-det"
    wandb_enabled: bool = True
    run_name: str | None = None
    log_interval: int = 50
    eval_interval: int = 1

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True

    @property
    def top_fpn_spatial(self) -> int:
        """Spatial size of highest FPN layer (stride 32)."""
        return self.img_h // 32  # 16 for 512

    @property
    def gfe_token_dim(self) -> int:
        """Flattened feature vector dimension for GFE tokens."""
        s = self.top_fpn_spatial
        return self.C * s * s  # 256 * 16 * 16 = 65536
