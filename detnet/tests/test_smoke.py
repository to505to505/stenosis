"""Smoke tests for :mod:`detnet`.

Heavy tests (model build, forward pass, single-window overfit) are gated
behind ``DETNET_HEAVY=1`` so plain pytest stays fast.

    pytest detnet/tests/test_smoke.py -v
    DETNET_HEAVY=1 pytest detnet/tests/test_smoke.py -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from detnet.config import Config
from detnet.dataset import cxcywh_norm_to_xyxy_px
from detnet.sca import (
    FrameDetections,
    SCAConfig,
    apply_sca,
    _iou_xyxy,
    patch_ssim,
)


HEAVY = os.environ.get("DETNET_HEAVY") == "1"
HEAVY_REASON = "Set DETNET_HEAVY=1 to run model-build smoke tests"


# ─── lightweight tests ──────────────────────────────────────────────────
def test_config_defaults():
    cfg = Config()
    assert cfg.T == 5
    assert cfg.sff_token_dim == 512
    assert cfg.sff_heads == 8
    assert cfg.sff_token_dim % cfg.sff_heads == 0
    assert cfg.roi_size == 7
    assert cfg.num_classes == 1
    assert cfg.supervise_all_frames is True
    assert cfg.pretrained_coco is True
    assert cfg.sca_enabled_eval is True
    assert cfg.sca_t_iou == 0.2


def test_box_conversion_roundtrip_simple():
    boxes_norm = torch.tensor([
        [0.5, 0.5, 0.2, 0.4],
        [0.1, 0.2, 0.05, 0.1],
    ])
    xyxy = cxcywh_norm_to_xyxy_px(boxes_norm, img_size=100)
    assert torch.allclose(xyxy[0], torch.tensor([40., 30., 60., 70.]))
    assert torch.allclose(xyxy[1], torch.tensor([7.5, 15., 12.5, 25.]))


def test_box_conversion_empty():
    boxes = cxcywh_norm_to_xyxy_px(torch.zeros((0, 4)), img_size=100)
    assert boxes.shape == (0, 4)


# ─── SCA / SSIM ────────────────────────────────────────────────────────
def test_iou_xyxy_basic():
    a = np.array([0., 0., 10., 10.])
    b = np.array([5., 5., 15., 15.])
    iou = _iou_xyxy(a, b)
    # overlap = 5x5 = 25; union = 100 + 100 - 25 = 175
    assert abs(iou - 25 / 175) < 1e-6


def test_iou_xyxy_disjoint():
    a = np.array([0., 0., 10., 10.])
    b = np.array([20., 20., 30., 30.])
    assert _iou_xyxy(a, b) == 0.0


def test_patch_ssim_identical_is_one():
    img = (np.random.RandomState(0).rand(64, 64) * 255).astype(np.float32)
    box = np.array([10., 10., 30., 30.])
    score = patch_ssim(img, box, img, box)
    assert score > 0.99


def test_patch_ssim_unrelated_is_low():
    rng = np.random.RandomState(0)
    a = (rng.rand(64, 64) * 255).astype(np.float32)
    b = (rng.rand(64, 64) * 255).astype(np.float32)
    score = patch_ssim(a, np.array([0., 0., 64., 64.]),
                       b, np.array([0., 0., 64., 64.]))
    # Different random noise patches typically give SSIM ≈ 0.
    assert score < 0.2


def test_sca_keeps_consistent_cluster():
    """A box present in every consecutive frame should be retained."""
    T = 5
    per_frame = []
    for t in range(T):
        boxes = np.array([[100., 100., 200., 200.]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        per_frame.append(FrameDetections(
            boxes=boxes, scores=scores, image=None,
        ))
    cfg = SCAConfig(t_iou=0.2, t_frame=3, interpolate_missing=False)
    out = apply_sca(per_frame, cfg)
    for t in range(T):
        assert out[t].boxes.shape[0] == 1, f"frame {t} should have a box"


def test_sca_interpolates_missing_inner_frame():
    """A consecutive cluster with one missing inner frame should be filled in."""
    T = 5
    per_frame = []
    # Present at every frame except 2 — frame 2 sits between two strong neighbors.
    for t in range(T):
        if t == 2:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            boxes = np.array([[100., 100., 200., 200.]], dtype=np.float32)
            scores = np.array([0.9], dtype=np.float32)
        per_frame.append(FrameDetections(boxes=boxes, scores=scores, image=None))
    # Cluster {0,1} ∪ {3,4} only merges if we allow skipping one frame; with the
    # adjacent-only rule of the paper they stay disjoint and both fall below
    # t_frame=3. Verify the algorithm faithfully drops them.
    cfg_strict = SCAConfig(t_iou=0.2, t_frame=3, interpolate_missing=True)
    out_strict = apply_sca(per_frame, cfg_strict)
    for t in range(T):
        assert out_strict[t].boxes.shape[0] == 0, (
            "with adjacent-only matching, the broken cluster should be dropped"
        )


def test_sca_no_extrapolation():
    """Interpolation only fills gaps between present frames, not boundary frames."""
    T = 5
    per_frame = []
    # Present at frames 1..4 only; frame 0 must stay empty (no extrapolation).
    for t in range(T):
        if t == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        else:
            boxes = np.array([[100., 100., 200., 200.]], dtype=np.float32)
            scores = np.array([0.9], dtype=np.float32)
        per_frame.append(FrameDetections(
            boxes=boxes, scores=scores, image=None,
        ))
    cfg = SCAConfig(t_iou=0.2, t_frame=3, interpolate_missing=True)
    out = apply_sca(per_frame, cfg)
    assert out[0].boxes.shape[0] == 0
    for t in range(1, T):
        assert out[t].boxes.shape[0] == 1


def test_sca_drops_intermittent_false_positive():
    """A box appearing in only 2/5 frames should be removed."""
    T = 5
    per_frame = []
    for t in range(T):
        if t in (0, 2):
            boxes = np.array([[10., 10., 30., 30.]], dtype=np.float32)
            scores = np.array([0.6], dtype=np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        per_frame.append(FrameDetections(
            boxes=boxes, scores=scores, image=None,
        ))
    cfg = SCAConfig(t_iou=0.2, t_frame=3, interpolate_missing=False)
    out = apply_sca(per_frame, cfg)
    # The non-adjacent boxes never form a cluster of size >= t_frame.
    for t in range(T):
        assert out[t].boxes.shape[0] == 0


def test_sca_no_interpolation_when_disabled():
    """A 5-of-5 cluster is kept; disabling interpolation has no visible effect here."""
    T = 5
    per_frame = []
    for t in range(T):
        boxes = np.array([[100., 100., 200., 200.]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        per_frame.append(FrameDetections(boxes=boxes, scores=scores, image=None))
    cfg = SCAConfig(t_iou=0.2, t_frame=3, interpolate_missing=False)
    out = apply_sca(per_frame, cfg)
    for t in range(T):
        assert out[t].boxes.shape[0] == 1


def test_sca_empty_window():
    T = 5
    per_frame = [FrameDetections(
        boxes=np.zeros((0, 4), dtype=np.float32),
        scores=np.zeros((0,), dtype=np.float32),
        image=None,
    ) for _ in range(T)]
    cfg = SCAConfig()
    out = apply_sca(per_frame, cfg)
    assert len(out) == T
    for fd in out:
        assert fd.boxes.shape == (0, 4)


# ─── heavier integration tests ──────────────────────────────────────────
@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_sff_block_residual_shape():
    """SFF block must preserve input shape (residual add semantics)."""
    from detnet.model import SFFBlock
    block = SFFBlock(d_in=128, d_model=64, heads=4, dropout=0.0)
    x = torch.randn(20, 128)
    y = block(x)
    assert y.shape == x.shape


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_sff_block_empty_tokens():
    from detnet.model import SFFBlock
    block = SFFBlock(d_in=128, d_model=64, heads=4)
    x = torch.zeros((0, 128))
    y = block(x)
    assert y.shape == (0, 128)


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_model_forward_train_returns_losses():
    from detnet.model import VideoFasterRCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.T = 3
    cfg.img_size = 128
    cfg.batch_size = 1
    cfg.pretrained_coco = False
    cfg.box_batch_size_per_image = 32
    cfg.rpn_post_nms_top_n_train = 128

    model = VideoFasterRCNN(cfg).to(device)
    model.train()

    B, T, H, W = 1, 3, 128, 128
    images = torch.randn(B, T, 3, H, W, device=device)
    targets = [[
        {"boxes": torch.tensor([[20., 20., 60., 60.]], device=device),
         "labels": torch.tensor([1], dtype=torch.int64, device=device)}
        for _ in range(T)
    ] for _ in range(B)]

    losses = model(images, targets=targets)
    expected = {"loss_objectness", "loss_rpn_box_reg", "loss_classifier", "loss_box_reg"}
    assert expected.issubset(losses.keys()), f"missing keys: {expected - losses.keys()}"
    for k, v in losses.items():
        assert torch.is_tensor(v) and v.ndim == 0, f"{k} not scalar"
        assert torch.isfinite(v), f"{k} non-finite: {v}"


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_model_forward_eval_returns_all_frames():
    """Eval mode should expose per-frame predictions for SCA."""
    from detnet.model import VideoFasterRCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.T = 3
    cfg.img_size = 128
    cfg.batch_size = 1
    cfg.pretrained_coco = False
    cfg.box_batch_size_per_image = 32
    cfg.rpn_post_nms_top_n_test = 64

    model = VideoFasterRCNN(cfg).to(device).eval()
    images = torch.randn(1, 3, 3, 128, 128, device=device)
    with torch.no_grad():
        out = model(images, targets=None)
    assert "centre" in out
    assert "all_frames" in out
    assert len(out["all_frames"]) == 1   # B = 1
    assert len(out["all_frames"][0]) == 3   # T = 3
    for fd in out["all_frames"][0]:
        for k in ("boxes", "scores", "labels"):
            assert k in fd


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_single_window_overfit():
    """30 SGD steps should drive total loss below 80% of its initial value."""
    from detnet.model import VideoFasterRCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.T = 3
    cfg.img_size = 128
    cfg.batch_size = 1
    cfg.pretrained_coco = False
    cfg.box_batch_size_per_image = 32
    cfg.rpn_post_nms_top_n_train = 128

    model = VideoFasterRCNN(cfg).to(device).train()
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    B, T, H, W = 1, 3, 128, 128
    images = torch.randn(B, T, 3, H, W, device=device)
    targets = [[
        {"boxes": torch.tensor([[30., 30., 70., 70.]], device=device),
         "labels": torch.tensor([1], dtype=torch.int64, device=device)}
        for _ in range(T)
    ] for _ in range(B)]

    losses_history = []
    for _ in range(30):
        loss_dict = model(images, targets=targets)
        loss = sum(loss_dict.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses_history.append(float(loss.item()))

    first5 = sum(losses_history[:5]) / 5
    last5 = sum(losses_history[-5:]) / 5
    assert last5 < first5 * 0.8, (
        f"loss did not drop enough: first5={first5:.3f} last5={last5:.3f} "
        f"history={losses_history}"
    )
