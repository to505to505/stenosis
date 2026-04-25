"""Lightweight unit tests for the stenosis student.

Run with:
    pytest -q stenosis_student/tests
or, for the slow backbone-loading test (downloads ~120 MB):
    STENOSIS_STUDENT_HEAVY=1 pytest -q stenosis_student/tests

Tests grouped here:
    1. TSM correctness (shape, zero params, recovery on non-edge frames).
    2. FCOS head shape correctness over multi-level inputs.
    3. Centre-ness target / GIoU sanity.
    4. Loss decreases on a single-batch overfit using a mock backbone.
    5. Postprocess returns valid xyxy boxes within image bounds.
    6. (heavy) Backbone forward + full model overfit on synthetic batch.
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn

from stenosis_student.config import Config
from stenosis_student.head import FCOSHead, make_locations
from stenosis_student.loss import (
    FCOSLoss,
    _centerness_target,
    _decode_distances,
    _giou_loss,
)
from stenosis_student.neck import DetailAwareFPN
from stenosis_student.postprocess import postprocess as fcos_postprocess
from stenosis_student.tsm import TSMState, install_tsm_hooks, temporal_shift


# ─── 1. TSM ──────────────────────────────────────────────────────────────
def test_tsm_shape_and_dtype():
    B, T, C, H, W = 2, 9, 32, 8, 8
    x = torch.randn(B * T, C, H, W)
    y = temporal_shift(x, T=T, fold_div=8)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_tsm_T1_noop():
    x = torch.randn(3, 16, 4, 4)
    y = temporal_shift(x, T=1, fold_div=8)
    assert torch.equal(x, y)


def test_tsm_unchanged_channels():
    """The last (C - 2*fold) channels must be untouched."""
    B, T, C, H, W = 1, 5, 32, 4, 4
    fold_div = 8
    fold = C // fold_div  # 4
    x = torch.randn(B * T, C, H, W)
    y = temporal_shift(x, T=T, fold_div=fold_div)
    assert torch.equal(x[:, 2 * fold:], y[:, 2 * fold:])


def test_tsm_forward_shift_takes_from_previous_frame():
    B, T, C, H, W = 1, 4, 16, 2, 2
    fold = C // 8  # 2
    # Make each frame's data identifiable
    x = torch.zeros(B * T, C, H, W)
    for t in range(T):
        x[t] = float(t + 1)
    y = temporal_shift(x, T=T, fold_div=8)
    # Frame 0 forward-fold must be zero (no t=-1 source)
    assert torch.all(y[0, :fold] == 0)
    # Frame 1 forward-fold must equal frame 0's data (== 1.0)
    assert torch.all(y[1, :fold] == 1.0)
    # Frame 2 backward-fold must equal frame 3's data (== 4.0)
    assert torch.all(y[2, fold:2 * fold] == 4.0)
    # Last frame backward-fold must be zero
    assert torch.all(y[T - 1, fold:2 * fold] == 0)


def test_tsm_hooks_zero_params_and_runtime_toggle():
    conv = nn.Conv2d(8, 8, 3, padding=1)
    n_before = sum(p.numel() for p in conv.parameters())
    state = install_tsm_hooks([conv], T=4, fold_div=8)
    n_after = sum(p.numel() for p in conv.parameters())
    assert n_after == n_before  # zero new parameters

    x = torch.randn(8, 8, 4, 4)  # B*T=8, T=4 → B=2
    y_on = conv(x)
    state.enabled = False
    y_off = conv(x)
    # Disabling the shift must change the output (otherwise the hook had no
    # effect at all).
    assert not torch.allclose(y_on, y_off)


# ─── 2. FCOS head shapes ────────────────────────────────────────────────
def test_fcos_head_shapes():
    cfg = Config()
    head = FCOSHead(
        in_dim=cfg.fpn_dim,
        num_classes=cfg.num_classes,
        num_levels=3,
        strides=(8, 16, 32),
    )
    feats = [
        torch.randn(2, cfg.fpn_dim, 64, 64),
        torch.randn(2, cfg.fpn_dim, 32, 32),
        torch.randn(2, cfg.fpn_dim, 16, 16),
    ]
    cls, reg, ctr = head(feats)
    assert len(cls) == 3 and len(reg) == 3 and len(ctr) == 3
    for c, r, ce, f in zip(cls, reg, ctr, feats):
        B, _, H, W = f.shape
        assert c.shape == (B, cfg.num_classes, H, W)
        assert r.shape == (B, 4, H, W)
        assert ce.shape == (B, 1, H, W)
    # Regression must be strictly positive (exp output).
    for r in reg:
        assert (r > 0).all()


def test_make_locations_centres():
    locs = make_locations([(2, 3)], strides=(8,), device=torch.device("cpu"))
    expected = torch.tensor([
        [4, 4], [12, 4], [20, 4],
        [4, 12], [12, 12], [20, 12],
    ], dtype=torch.float32)
    assert torch.equal(locs[0], expected)


# ─── 3. Loss helpers ────────────────────────────────────────────────────
def test_centerness_target_perfect_centre_is_one():
    reg = torch.tensor([[5.0, 5.0, 5.0, 5.0]])
    ct = _centerness_target(reg)
    assert torch.allclose(ct, torch.ones(1))


def test_giou_loss_zero_when_perfect_overlap():
    box = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    loss = _giou_loss(box, box)
    assert torch.allclose(loss, torch.zeros(1), atol=1e-5)


def test_decode_distances_roundtrip():
    locs = torch.tensor([[10.0, 20.0]])
    dist = torch.tensor([[3.0, 4.0, 5.0, 6.0]])
    xyxy = _decode_distances(locs, dist)
    assert torch.equal(xyxy, torch.tensor([[7.0, 16.0, 15.0, 26.0]]))


def test_loss_finite_with_targets():
    cfg = Config()
    loss_fn = FCOSLoss(cfg)
    # 2 images, 3 levels at strides 8/16/32, image 256
    sizes = [(32, 32), (16, 16), (8, 8)]
    cls = [torch.randn(2, cfg.num_classes, h, w) for h, w in sizes]
    # Positive distances
    reg = [torch.rand(2, 4, h, w) * 50 + 1 for h, w in sizes]
    ctr = [torch.randn(2, 1, h, w) for h, w in sizes]
    targets = [
        {"boxes": torch.tensor([[50.0, 60.0, 100.0, 110.0]]),
         "labels": torch.zeros(1, dtype=torch.long)},
        {"boxes": torch.zeros((0, 4)),
         "labels": torch.zeros((0,), dtype=torch.long)},
    ]
    out = loss_fn(cls, reg, ctr, targets)
    for k in ("loss_cls", "loss_box", "loss_ctr", "loss"):
        assert torch.isfinite(out[k]), f"{k} not finite: {out[k]}"


# ─── 4. Overfit on a single batch (mock backbone) ───────────────────────
class _MockBackbone(nn.Module):
    """Cheap stand-in for ConvNeXt: produces P3/P4/P5 from raw image input."""

    def __init__(self, channels=(96, 192, 384, 768)):
        super().__init__()
        self.channels = channels
        self.stems = nn.ModuleList([
            nn.Conv2d(3, channels[i], kernel_size=4, stride=2 ** (i + 2), padding=0)
            for i in range(4)
        ])

    def forward(self, x):
        outs = []
        for stem in self.stems:
            outs.append(stem(x))
        return outs


def _build_model_with_mock_backbone(cfg: Config):
    """Compose the rest of the model around a mock backbone (no HF download)."""
    import torch.nn as nn
    from stenosis_student.head import FCOSHead
    from stenosis_student.loss import FCOSLoss
    from stenosis_student.neck import DetailAwareFPN

    class MiniStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _MockBackbone(channels=cfg.stage_channels)
            in_channels = [cfg.stage_channels[i] for i in cfg.fpn_stage_indices]
            self.neck = DetailAwareFPN(cfg, in_channels=in_channels)
            self.head = FCOSHead(
                in_dim=cfg.fpn_dim,
                num_classes=cfg.num_classes,
                num_levels=3,
                strides=tuple(cfg.stage_strides[i] for i in cfg.fpn_stage_indices),
            )
            self.criterion = FCOSLoss(cfg)
            self.cfg = cfg

        def forward(self, frames, targets=None):
            B, T, C, H, W = frames.shape
            x = frames.reshape(B * T, C, H, W)
            feats = self.backbone(x)
            # take only requested stages, slice centre frame
            picked = []
            for i in cfg.fpn_stage_indices:
                f = feats[i]
                _, Cl, h, w = f.shape
                picked.append(f.reshape(B, T, Cl, h, w)[:, cfg.centre_index])
            fused = self.neck(picked)
            cls, reg, ctr = self.head(fused)
            if targets is not None:
                return self.criterion(cls, reg, ctr, targets)
            return cls, reg, ctr

    return MiniStudent()


def test_overfit_single_batch_with_mock_backbone():
    torch.manual_seed(0)
    cfg = Config(img_size=256, T=9, num_classes=1)
    model = _build_model_with_mock_backbone(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    B, T = 1, cfg.T
    frames = torch.randn(B, T, 3, cfg.img_size, cfg.img_size)
    targets = [{
        "boxes": torch.tensor([[80.0, 80.0, 160.0, 160.0]]),
        "labels": torch.zeros(1, dtype=torch.long),
    }]

    losses = []
    for _ in range(200):
        opt.zero_grad()
        out = model(frames, targets)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].item()))

    # Loss must drop substantially from its initial value (sanity check that
    # gradients flow end-to-end through neck, head, and FCOS loss).
    assert losses[-1] < 0.7 * losses[0], \
        f"loss did not decrease enough: {losses[0]:.3f} → {losses[-1]:.3f}"


# ─── 5. Postprocess ─────────────────────────────────────────────────────
def test_postprocess_outputs_valid_boxes():
    cfg = Config(img_size=256, num_classes=1, score_thresh=0.0)
    sizes = [(32, 32), (16, 16), (8, 8)]
    # Make one location very confident
    cls = [torch.full((1, 1, h, w), -10.0) for h, w in sizes]
    cls[0][0, 0, 4, 4] = 10.0
    reg = [torch.full((1, 4, h, w), 16.0) for h, w in sizes]
    ctr = [torch.full((1, 1, h, w), 5.0) for h, w in sizes]
    out = fcos_postprocess(cls, reg, ctr, cfg, image_size=cfg.img_size)
    assert len(out) == 1
    boxes = out[0]["boxes"]
    assert boxes.shape[1] == 4
    assert (boxes[:, 0] >= 0).all() and (boxes[:, 1] >= 0).all()
    assert (boxes[:, 2] <= cfg.img_size).all() and (boxes[:, 3] <= cfg.img_size).all()
    assert (boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all()
    # Highest-scoring box should have the centre at (4.5*8, 4.5*8) = (36, 36)
    top = out[0]["scores"].argmax()
    cx = 0.5 * (boxes[top, 0] + boxes[top, 2])
    cy = 0.5 * (boxes[top, 1] + boxes[top, 3])
    assert abs(cx.item() - 36.0) < 1.0 and abs(cy.item() - 36.0) < 1.0


# ─── 6. Heavy: real backbone forward (opt-in) ───────────────────────────
HEAVY = os.environ.get("STENOSIS_STUDENT_HEAVY", "0") == "1"


@pytest.mark.skipif(not HEAVY, reason="set STENOSIS_STUDENT_HEAVY=1 to enable")
def test_real_backbone_forward_shapes():
    from stenosis_student.backbone import ConvNeXtV2TinyBackbone
    cfg = Config(T=9, img_size=256)
    bb = ConvNeXtV2TinyBackbone(cfg)
    bb.eval()
    x = torch.randn(2 * cfg.T, 3, cfg.img_size, cfg.img_size)
    feats = bb(x)
    expected_strides = bb.out_strides
    for f, s in zip(feats, expected_strides):
        assert f.shape[0] == 2 * cfg.T
        assert f.shape[2] == cfg.img_size // s
        assert f.shape[3] == cfg.img_size // s


@pytest.mark.skipif(not HEAVY, reason="set STENOSIS_STUDENT_HEAVY=1 to enable")
def test_real_full_model_forward_and_loss():
    from stenosis_student.model import StenosisStudent
    cfg = Config(T=9, img_size=256, num_classes=1, batch_size=1)
    model = StenosisStudent(cfg)
    model.eval()
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size)
    targets = [{
        "boxes": torch.tensor([[80.0, 80.0, 160.0, 160.0]]),
        "labels": torch.zeros(1, dtype=torch.long),
    }]
    losses = model(frames, targets)
    assert torch.isfinite(losses["loss"])
    cls, reg, ctr = model(frames)
    assert len(cls) == 3
    dets = fcos_postprocess(cls, reg, ctr, cfg, cfg.img_size)
    assert len(dets) == 1
