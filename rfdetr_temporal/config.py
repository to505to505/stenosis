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
    T: int = 5                    # consecutive frames per sample (centre = T//2)

    # ── RF-DETR pretrained ─────────────────────────────────────────────
    # If the path exists → loaded as a fine-tuned stenosis checkpoint.
    # If it does not exist → rf-detr-nano.pth is auto-downloaded (same as
    # standard rf-detr training).
    rfdetr_checkpoint: str = "rfdetr_runs/dataset2_augs/checkpoint_best_total.pth"
    freeze_backbone: bool = True  # freeze DINOv2 + projector
    freeze_decoder: bool = False  # fine-tune decoder

    # ── Temporal fusion ────────────────────────────────────────────────
    hidden_dim: int = 256         # must match RF-DETR hidden_dim
    temporal_attn_layers: int = 2 # transformer decoder layers for fusion
    temporal_nhead: int = 8       # attention heads in temporal fusion
    neighborhood_k: int = 0       # spatial radius; 0 = per-position only (no unfold)

    # ── Distillation ───────────────────────────────────────────────────
    distill_teacher_resolution: int = 512  # HR res for teacher frame (used only with --distill)

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

    # ── KD-DETR distillation (specific sampling) ──────────────────────
    # Specific sampling: a frozen HR teacher (RF-DETR-Large @ 704) provides
    # its trained object queries as shared distillation points.  The student
    # decoder is run with these queries on the LR feature map; predictions
    # are slot-aligned to the teacher's so distillation reduces to a simple
    # weighted KL+L1+GIoU loss.
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

    # ── KD-DETR general sampling ──────────────────────────────────────
    # General sampling: a *fresh* set of random queries is drawn every
    # training step and pushed through both the student LR decoder and the
    # frozen HR teacher decoder.  Distillation on these "probe" queries
    # forces the student to match the teacher's behaviour everywhere on the
    # feature map (foreground + background context), not only on the
    # teacher-confident slots covered by specific sampling.
    distill_general_enabled: bool = False
    distill_num_general_queries: int = 100
    distill_general_loss_weight: float = 0.5
    # Floor on the per-query foreground weight in the general branch; some
    # general queries land on background and would otherwise contribute
    # ~zero gradient.  A small floor (e.g. 0.1) lets the student also learn
    # the teacher's *background* representation.
    distill_general_min_weight: float = 0.1
    # Std for the random query_feat init; refpoint_embed is initialised at
    # zero (matching LWDETR.refpoint_embed init) so two-stage proposals
    # determine the spatial coverage.
    distill_general_query_std: float = 0.02

    # ── CRRCD: Cross-Resolution Relational Contrastive Distillation ───
    # Adds a relational contrastive loss between the (frozen) HR teacher
    # decoder embeddings and the (trainable) LR student decoder embeddings
    # produced in the KD-DETR specific-sampling branch.  Two small MLPs
    # (Feature Relation Modules, FRM) model relationships between
    # foreground / background query slots:
    #   F^t  : (e^t_i, e^t_j)  →  v^t_{i,j}
    #   F^ts : (e^t_i, e^s_j)  →  v^{t,s}_{i,j}
    # A sigmoid-NCE critic forces v^{t,s} to mimic v^t at matching slot
    # pairs, structurally aligning the LR student's decoder space to the
    # HR teacher's. Requires distill_enabled=True (uses the same Branch 2
    # forward + slot alignment).
    crrcd_enabled: bool = False
    crrcd_loss_weight: float = 2.0       # β in L_total = L_det + α·L_kd + β·L_rcd
    crrcd_relation_dim: int = 256        # FRM output dim
    crrcd_hidden_dim: int = 256          # FRM hidden width (MLP)
    crrcd_num_fg: int = 16               # K_fg — top-K teacher-confident slots
    crrcd_num_bg: int = 32               # K_bg — bottom-K teacher-confident slots
    crrcd_num_negatives: int = 16        # n   — negatives per anchor (0 = use all K_bg-1)
    crrcd_temperature: float = 0.1       # τ for sigmoid critic h(u,v)=σ(cos(u,v)/τ)

    # ── CPC: Contrastive Predictive Coding (temporal regulariser) ─────
    # Adds an InfoNCE loss between the fused centre-frame context c_t and
    # the raw backbone features z_{t±1} of the neighbouring frames, with
    # a trainable bilinear projection W_k that absorbs spatial displacement
    # of the artery so the model is *not* forced to ``c_t == z_{t+1}``.
    # Computed only on the first (P4) backbone level since it matches
    # ``hidden_dim``; only active during training.
    cpc_enabled: bool = False
    cpc_weight: float = 1.0
    cpc_offsets: tuple = (-1, 1)         # which neighbour frames to predict
