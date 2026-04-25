"""Stage-4 unit tests: distillation, temporal dropout, InfoNCE consistency.

Run with:
    pytest -q stenosis_student/tests/test_stage4.py
or, for the heavy real-teacher test:
    STENOSIS_STUDENT_HEAVY=1 pytest -q stenosis_student/tests/test_stage4.py
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from stenosis_student.config import Config
from stenosis_student.distill_losses import (
    FeatureAdapter,
    cosine_distill_loss,
    info_nce_loss,
)
from stenosis_student.temporal_consistency import (
    pool_box_embeddings,
    temporal_consistency_loss,
)


# ─── 1. FeatureAdapter ──────────────────────────────────────────────────
def test_feature_adapter_shape_and_dtype():
    adapter = FeatureAdapter(in_dim=192, out_dim=256)
    x = torch.randn(2, 192, 16, 16)
    y = adapter(x)
    assert y.shape == (2, 256, 16, 16)
    assert y.dtype == x.dtype
    # 1×1 conv → exactly in*out + out parameters
    n_params = sum(p.numel() for p in adapter.parameters())
    assert n_params == 192 * 256 + 256


# ─── 2. Cosine distill loss ─────────────────────────────────────────────
def test_cosine_distill_loss_zero_for_identical():
    x = torch.randn(2, 16, 8, 8)
    loss = cosine_distill_loss(x, x.clone())
    assert torch.allclose(loss, torch.zeros(()), atol=1e-5), float(loss)


def test_cosine_distill_loss_resizes_teacher():
    s = torch.randn(2, 16, 8, 8)
    t = torch.randn(2, 16, 4, 4)  # different spatial size — must be resized
    loss = cosine_distill_loss(s, t)
    assert torch.isfinite(loss) and loss.item() > 0


def test_cosine_distill_loss_orthogonal_is_one():
    # Construct two tensors whose every spatial vector is orthogonal:
    # s aligned with channel-0 axis, t aligned with channel-1 axis.
    s = torch.zeros(1, 4, 2, 2)
    t = torch.zeros(1, 4, 2, 2)
    s[:, 0] = 1.0
    t[:, 1] = 1.0
    loss = cosine_distill_loss(s, t)
    assert torch.allclose(loss, torch.ones(()), atol=1e-5), float(loss)


# ─── 3. InfoNCE ─────────────────────────────────────────────────────────
def test_info_nce_perfect_alignment_low_loss():
    torch.manual_seed(0)
    P, D = 4, 8
    a = torch.randn(P, D)
    p = a.clone()                                      # perfect positives
    n = torch.randn(8, D)                              # random negatives
    loss = info_nce_loss(a, p, n, temperature=0.07)
    # With identical anchor/positive and random negatives, the positive
    # similarity is maximal → cross-entropy should be tiny.
    assert loss.item() < 0.5, float(loss)


def test_info_nce_decreases_with_optimisation():
    torch.manual_seed(0)
    P, D = 4, 8
    a = nn.Parameter(torch.randn(P, D))
    p = nn.Parameter(torch.randn(P, D))
    n = torch.randn(16, D)
    opt = torch.optim.SGD([a, p], lr=0.5)
    losses = []
    for _ in range(50):
        opt.zero_grad()
        l = info_nce_loss(a, p, n, temperature=0.5)
        l.backward()
        opt.step()
        losses.append(float(l.item()))
    assert losses[-1] < losses[0] * 0.5, f"{losses[0]:.3f} → {losses[-1]:.3f}"


def test_info_nce_empty_returns_zero():
    a = torch.zeros(0, 8)
    p = torch.zeros(0, 8)
    n = torch.randn(4, 8)
    loss = info_nce_loss(a, p, n)
    assert float(loss) == 0.0


# ─── 4. RoI-pool box embeddings ─────────────────────────────────────────
def test_pool_box_embeddings_shapes():
    feat = torch.randn(2, 16, 32, 32)                  # B=2, C=16, h=w=32
    boxes = torch.tensor([
        [0.0, 0.0, 64.0, 64.0],
        [32.0, 32.0, 96.0, 96.0],
        [10.0, 10.0, 50.0, 50.0],
    ])
    batch_idx = torch.tensor([0, 0, 1])
    emb = pool_box_embeddings(feat, boxes, batch_idx, stride=16, pool_size=7)
    assert emb.shape == (3, 16)
    assert emb.dtype == torch.float32


def test_pool_box_embeddings_empty():
    feat = torch.randn(1, 8, 4, 4)
    emb = pool_box_embeddings(
        feat, torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long), stride=16,
    )
    assert emb.shape == (0, 8)


# ─── 5. Temporal-consistency end-to-end ────────────────────────────────
def test_temporal_consistency_loss_finite():
    torch.manual_seed(0)
    cfg = Config(T=9, img_size=256, temporal_consistency_enabled=True)
    B, T, C, h, w = 2, 9, 16, 16, 16
    feat = torch.randn(B, T, C, h, w, requires_grad=True)
    targets = [
        {"boxes": torch.tensor([[40.0, 40.0, 120.0, 120.0],
                                 [10.0, 10.0, 80.0, 80.0]]),
         "labels": torch.zeros(2, dtype=torch.long)},
        {"boxes": torch.tensor([[60.0, 60.0, 180.0, 180.0]]),
         "labels": torch.zeros(1, dtype=torch.long)},
    ]
    loss = temporal_consistency_loss(feat, targets, cfg, stride=16.0)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


def test_temporal_consistency_loss_no_gt_is_zero():
    cfg = Config(temporal_consistency_enabled=True)
    feat = torch.randn(1, 9, 8, 4, 4, requires_grad=True)
    targets = [{"boxes": torch.zeros((0, 4)),
                "labels": torch.zeros((0,), dtype=torch.long)}]
    loss = temporal_consistency_loss(feat, targets, cfg, stride=16.0)
    assert float(loss) == 0.0


# ─── 6. Temporal Dropout dataset behaviour ─────────────────────────────
def _find_dataset_split() -> Path | None:
    """Locate any usable split (train / valid / test) under common dataset
    roots so this test can run wherever the user has data."""
    candidates = [
        Path("data/dataset2_split"),
        Path("data/dataset2_split_85_15"),
        Path("data/dataset2_split_90_10"),
    ]
    for root in candidates:
        for split in ("train", "valid", "test"):
            img_dir = root / split / "images"
            if img_dir.exists() and any(img_dir.iterdir()):
                return root
    return None


def test_temporal_dropout_only_centre_frame_modified():
    pytest.importorskip("albumentations")
    pytest.importorskip("cv2")

    root = _find_dataset_split()
    if root is None:
        pytest.skip("no dataset2_split-like data available")

    from stenosis_student.dataset import TemporalStenosisStudentDataset

    # Pick the first split with data so we can build a small dataset.
    split_to_use = None
    for s in ("train", "valid", "test"):
        if (root / s / "images").exists():
            split_to_use = s
            break
    assert split_to_use is not None

    base_cfg_kwargs = dict(
        data_root=root, T=9, img_size=128, batch_size=1, num_workers=0,
        wandb_enabled=False,
    )

    cfg_off = Config(**base_cfg_kwargs)
    ds_off = TemporalStenosisStudentDataset(split_to_use, cfg_off)
    images_off, clean_off, _t_off, _f_off = ds_off[0]
    assert clean_off is None  # no need_clean when both toggles are off

    cfg_on = Config(
        temporal_dropout_enabled=True,
        temporal_dropout_prob=1.0,                # always apply
        temporal_dropout_min_frac=0.4,
        temporal_dropout_max_frac=0.5,
        **base_cfg_kwargs,
    )
    # Force determinism so the same window is picked + same aug.
    np.random.seed(123)
    torch.manual_seed(123)
    ds_on = TemporalStenosisStudentDataset(split_to_use, cfg_on)
    np.random.seed(0)
    torch.manual_seed(0)
    images_on, clean_on, _t_on, _f_on = ds_on[0]

    centre = cfg_on.centre_index
    assert clean_on is not None and clean_on.shape == images_on[centre].shape
    # The centre frame in the corrupted clip must differ from the clean copy
    # (Cutout was applied with prob=1).
    diff = (images_on[centre] - clean_on).abs().mean().item()
    assert diff > 1e-3, f"expected centre frame to be modified, diff={diff}"


# ─── 7. Mock-backbone end-to-end with all stage-4 extras ───────────────
class _MockBackbone(nn.Module):
    def __init__(self, channels=(96, 192, 384, 768)):
        super().__init__()
        self.channels = channels
        self.stems = nn.ModuleList([
            nn.Conv2d(3, channels[i], kernel_size=4, stride=2 ** (i + 2), padding=0)
            for i in range(4)
        ])
        # Mimic what the real backbone exposes for the model code:
        self.tsm_state = None
        self.out_channels = tuple(channels[i] for i in (1, 2, 3))
        self.out_strides = (8, 16, 32)

    def forward(self, x):
        # Returns the three stages used by the FPN (matches fpn_stage_indices)
        outs = [self.stems[i](x) for i in range(4)]
        return [outs[1], outs[2], outs[3]]


def _build_student_with_mock_backbone(cfg: Config):
    from stenosis_student.model import StenosisStudent
    model = StenosisStudent.__new__(StenosisStudent)
    nn.Module.__init__(model)
    model.cfg = cfg
    model.T = cfg.T
    model.centre = cfg.centre_index
    model.backbone = _MockBackbone(channels=cfg.stage_channels)
    in_channels = list(model.backbone.out_channels)
    from stenosis_student.head import FCOSHead
    from stenosis_student.loss import FCOSLoss
    from stenosis_student.neck import DetailAwareFPN
    from stenosis_student.distill_losses import FeatureAdapter
    model.neck = DetailAwareFPN(cfg, in_channels=in_channels)
    model.head = FCOSHead(
        in_dim=cfg.fpn_dim,
        num_classes=cfg.num_classes,
        num_levels=len(in_channels),
        strides=model.backbone.out_strides,
        num_convs=cfg.head_num_convs,
        prior_prob=cfg.head_prior_prob,
    )
    model.criterion = FCOSLoss(cfg)
    model.feature_adapter = (
        FeatureAdapter(cfg.fpn_dim, cfg.distill_teacher_hidden_dim)
        if cfg.distill_enabled else None
    )
    return model


def test_model_forward_with_all_extras_and_backward():
    torch.manual_seed(0)
    cfg = Config(
        img_size=256, T=9, num_classes=1,
        distill_enabled=True,
        temporal_consistency_enabled=True,
    )
    model = _build_student_with_mock_backbone(cfg)

    B = 1
    frames = torch.randn(B, cfg.T, 3, cfg.img_size, cfg.img_size)
    targets = [{
        "boxes": torch.tensor([[80.0, 80.0, 160.0, 160.0]]),
        "labels": torch.zeros(1, dtype=torch.long),
    }]

    out = model(frames, targets, return_extras=True)
    losses = out["head_outputs"]
    assert torch.isfinite(losses["loss"])
    assert out["student_distill_feat"] is not None
    assert out["student_distill_feat"].shape[1] == cfg.distill_teacher_hidden_dim
    assert out["multi_frame_fpn_level"] is not None
    assert out["multi_frame_fpn_level"].shape[:2] == (B, cfg.T)

    # Compose a fake total loss using all three terms and check backward works.
    fake_teacher_feat = torch.randn_like(out["student_distill_feat"])
    l_distill = cosine_distill_loss(out["student_distill_feat"], fake_teacher_feat)
    l_temp = temporal_consistency_loss(
        out["multi_frame_fpn_level"], targets, cfg, stride=16.0,
    )
    total = losses["loss"] + 0.5 * l_distill + 0.1 * l_temp
    total.backward()
    # Adapter must have grads
    assert model.feature_adapter.proj.weight.grad is not None
    assert torch.isfinite(model.feature_adapter.proj.weight.grad).all()


# ─── 8. Heavy real-teacher load (opt-in) ───────────────────────────────
HEAVY = os.environ.get("STENOSIS_STUDENT_HEAVY", "0") == "1"


@pytest.mark.skipif(not HEAVY, reason="set STENOSIS_STUDENT_HEAVY=1 to enable")
def test_real_teacher_loads_and_runs():
    ckpt = Path("rfdetr_temporal/runs/temporal_v1/best.pth")
    if not ckpt.exists():
        pytest.skip(f"teacher checkpoint not present: {ckpt}")
    from stenosis_student.teacher import FrozenRFDETRTeacher
    cfg = Config(distill_enabled=True, distill_teacher_ckpt=str(ckpt))
    teacher = FrozenRFDETRTeacher(cfg).eval()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = teacher(x)
    assert out.dim() == 4
    assert out.shape[0] == 1
    assert out.shape[1] == cfg.distill_teacher_hidden_dim
    # Stride 16 → 32 spatial cells at 512.
    assert out.shape[2] == 32 and out.shape[3] == 32
