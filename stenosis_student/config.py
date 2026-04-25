"""Centralised configuration for the temporal student detector."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    data_root: Path = Path("data/dataset2_split")
    output_dir: Path = Path("stenosis_student/runs")

    # ── Image / sequence ───────────────────────────────────────────────
    img_size: int = 512
    T: int = 9                         # 4 past + centre + 4 future
    centre_index: int = 4              # T // 2

    # ── Backbone ───────────────────────────────────────────────────────
    hf_model_id: str = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
    backbone_local_path: Optional[str] = None  # if set, load from disk
    freeze_stem: bool = True
    freeze_stage0: bool = True
    # Channel dims for the 4 ConvNeXt stages (standard Tiny).  Auto-detected
    # at runtime; this is the fallback / sanity check.
    stage_channels: Tuple[int, int, int, int] = (96, 192, 384, 768)
    stage_strides: Tuple[int, int, int, int] = (4, 8, 16, 32)
    # Which stages feed the FPN (P3=stage1 s8, P4=stage2 s16, P5=stage3 s32)
    fpn_stage_indices: Tuple[int, int, int] = (1, 2, 3)

    # ── TSM ────────────────────────────────────────────────────────────
    tsm_enabled: bool = True
    tsm_fold_div: int = 8              # 1/8 of channels shifted
    tsm_per_block: bool = True         # hook every block (vs every stage)

    # ── Neck (Detail-Aware Cross-Attention FPN) ───────────────────────
    fpn_dim: int = 192
    fpn_num_heads: int = 4
    fpn_attn_window: int = 0           # 0 = full attn (used only at P5/P4); P3 uses pooled K/V
    fpn_kv_pool: int = 2               # spatial pool factor for K/V at high-res levels

    # ── Head ───────────────────────────────────────────────────────────
    num_classes: int = 1
    head_num_convs: int = 4
    head_prior_prob: float = 0.01
    centre_sample_radius: float = 1.5
    # Per-level regression range (in pixels). FCOS-style assignment.  Last
    # value INF.  Tuned smaller than COCO for micro-stenosis.
    reg_ranges: Tuple[Tuple[float, float], ...] = (
        (-1.0, 64.0),       # P3 (stride 8)
        (64.0, 128.0),      # P4 (stride 16)
        (128.0, 1.0e8),     # P5 (stride 32)
    )

    # ── Loss weights ───────────────────────────────────────────────────
    cls_loss_weight: float = 1.0
    reg_loss_weight: float = 1.0
    centerness_loss_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # ── Stage-4: total-loss balance (λ₁/λ₂/λ₃) ────────────────────────
    det_loss_weight: float = 1.0          # λ₁ — multiplies the raw FCOS det loss
    distill_loss_weight: float = 0.5      # λ₂ — feature distillation
    temporal_consistency_weight: float = 0.1  # λ₃ — InfoNCE temporal consistency

    # ── Stage-4: feature distillation (frozen 2D teacher) ─────────────
    distill_enabled: bool = False
    distill_teacher_ckpt: str = "rfdetr_temporal/runs/temporal_v1/best.pth"
    distill_teacher_hidden_dim: int = 256
    # Index into the FPN output list (0=P3 s8, 1=P4 s16, 2=P5 s32). Teacher
    # RF-DETR Small projects only at P4, so 1 is the natural match.
    distill_student_level_idx: int = 1
    distill_metric: str = "cosine"        # only "cosine" implemented for now

    # ── Stage-4: asymmetric centre-frame Temporal Dropout ─────────────
    temporal_dropout_enabled: bool = False
    temporal_dropout_prob: float = 0.25   # fraction of windows that get cutout
    temporal_dropout_min_frac: float = 0.30  # square side ≥ 30 % of img
    temporal_dropout_max_frac: float = 0.50  # square side ≤ 50 % of img
    temporal_dropout_fill: float = 0.0    # pre-normalise pixel value (0 = black)

    # ── Stage-4: temporal-consistency InfoNCE ─────────────────────────
    temporal_consistency_enabled: bool = False
    temporal_consistency_temperature: float = 0.07
    temporal_consistency_neighbor_offsets: Tuple[int, ...] = (-1, 1)
    # FPN level used for box-feature pooling (same convention as distill).
    temporal_consistency_level_idx: int = 1
    temporal_consistency_pool_size: int = 7
    # If True, run consistency on the un-corrupted clip (replaces centre with
    # ``centre_clean``). Default False = use the same dropout clip — pushes the
    # model to recover identity even when the centre frame is masked.
    temporal_consistency_use_clean: bool = False

    # ── Postprocess ────────────────────────────────────────────────────
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    pre_nms_topk: int = 1000
    post_nms_topk: int = 300

    # ── Normalisation (ImageNet, used by ConvNeXt-V2 / DINOv3) ────────
    pixel_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    pixel_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ── Training ───────────────────────────────────────────────────────
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    lr_step_milestones: Tuple[int, ...] = (30, 40)
    lr_gamma: float = 0.1
    grad_accum_steps: int = 2
    grad_clip: float = 1.0

    # ── Logging ────────────────────────────────────────────────────────
    wandb_project: str = "stenosis-student"
    wandb_enabled: bool = True
    run_name: Optional[str] = None
    log_interval: int = 50
    eval_interval: int = 1

    # ── Misc ───────────────────────────────────────────────────────────
    seed: int = 42
    amp: bool = True
