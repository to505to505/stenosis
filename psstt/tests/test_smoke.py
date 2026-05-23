"""Smoke tests for :mod:`psstt`.

Heavy tests (model build, forward pass, single-window overfit) are gated
behind ``PSSTT_HEAVY=1`` so plain pytest stays fast.

    pytest psstt/tests/test_smoke.py -v
    PSSTT_HEAVY=1 pytest psstt/tests/test_smoke.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from psstt.config import Config
from psstt.dataset import cxcywh_norm_to_xyxy_px


HEAVY = os.environ.get("PSSTT_HEAVY") == "1"
HEAVY_REASON = "Set PSSTT_HEAVY=1 to run model-build smoke tests"


# ─── lightweight tests ──────────────────────────────────────────────────
def test_config_defaults():
    cfg = Config()
    assert cfg.T == 5
    assert cfg.num_shifts == 4
    assert cfg.token_dim == 512
    assert cfg.tfa_depth == 4
    assert cfg.tfa_heads == 8
    assert cfg.roi_size == 7
    assert cfg.num_classes == 1
    assert cfg.supervise_all_frames is True
    assert cfg.pretrained_coco is True


def test_box_conversion_roundtrip_simple():
    boxes_norm = torch.tensor([
        [0.5, 0.5, 0.2, 0.4],
        [0.1, 0.2, 0.05, 0.1],
    ])
    xyxy = cxcywh_norm_to_xyxy_px(boxes_norm, img_size=100)
    # First box: cx=50, cy=50, w=20, h=40 → x1=40, y1=30, x2=60, y2=70
    assert torch.allclose(xyxy[0], torch.tensor([40., 30., 60., 70.]))
    # Second: cx=10, cy=20, w=5, h=10 → x1=7.5, y1=15, x2=12.5, y2=25
    assert torch.allclose(xyxy[1], torch.tensor([7.5, 15., 12.5, 25.]))


def test_box_conversion_empty():
    boxes = cxcywh_norm_to_xyxy_px(torch.zeros((0, 4)), img_size=100)
    assert boxes.shape == (0, 4)


def test_shift_vectors_paper_layout():
    """The first shift must be the zero-shift; remaining four are
    (up, down, left, right) per paper Eq. (1)."""
    from psstt.model import SHIFT_VECTORS_DEFAULT
    assert SHIFT_VECTORS_DEFAULT[0] == (0, 0)
    assert (0, -1) in SHIFT_VECTORS_DEFAULT     # up
    assert (0, 1) in SHIFT_VECTORS_DEFAULT      # down
    assert (-1, 0) in SHIFT_VECTORS_DEFAULT     # left
    assert (1, 0) in SHIFT_VECTORS_DEFAULT      # right
    assert len(SHIFT_VECTORS_DEFAULT) == 5      # K=4 + zero


# ─── heavier integration tests ──────────────────────────────────────────
@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_build_shifted_boxes_geometry():
    from psstt.model import PSSTTBoxHead
    cfg = Config()
    head = PSSTTBoxHead(cfg)
    proposals = torch.tensor([
        [100., 200., 200., 300.],   # w=100, h=100
        [50., 50., 70., 80.],       # w=20, h=30 (small)
    ])
    shifted = head.build_shifted_boxes(proposals, image_size=(512, 512))
    assert shifted.shape == (2, 5, 4)
    # Index 0 = zero-shift, identical to original.
    assert torch.allclose(shifted[0, 0], proposals[0])
    # Up shift (αy = -1): y1 and y2 both shift by -h.
    # For box 0 (h=100), up shift => y1=200-100=100, y2=300-100=200.
    # Find the "up" index (αx=0, αy=-1) in SHIFT_VECTORS_DEFAULT.
    from psstt.model import SHIFT_VECTORS_DEFAULT
    up_idx = SHIFT_VECTORS_DEFAULT.index((0, -1))
    assert torch.allclose(shifted[0, up_idx], torch.tensor([100., 100., 200., 200.]))


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_build_shifted_boxes_clip_at_image_boundary():
    from psstt.model import PSSTTBoxHead, SHIFT_VECTORS_DEFAULT
    cfg = Config()
    head = PSSTTBoxHead(cfg)
    # Box at top-left corner; "up" and "left" shifts should clip without crashing.
    proposals = torch.tensor([[10., 10., 50., 50.]])
    shifted = head.build_shifted_boxes(proposals, image_size=(512, 512))
    # All shifted boxes must remain inside [0, 511] and have positive area.
    assert (shifted[..., 0] >= 0).all()
    assert (shifted[..., 1] >= 0).all()
    assert (shifted[..., 2] < 512).all()
    assert (shifted[..., 3] < 512).all()
    assert (shifted[..., 2] > shifted[..., 0]).all()
    assert (shifted[..., 3] > shifted[..., 1]).all()


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_model_forward_train_returns_losses():
    """Mini forward (T=3, H=W=128, B=1) on synthetic data — train mode."""
    from psstt.model import VideoFasterRCNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config()
    cfg.T = 3
    cfg.img_size = 128
    cfg.batch_size = 1
    cfg.pretrained_coco = False   # avoid downloading weights in tests
    cfg.box_batch_size_per_image = 32  # keep memory small
    cfg.rpn_post_nms_top_n_train = 128

    model = VideoFasterRCNN(cfg).to(device)
    model.train()

    B, T, H, W = 1, 3, 128, 128
    images = torch.randn(B, T, 3, H, W, device=device)
    # One target box per frame, all the same simple box.
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
def test_model_forward_eval_centre_predictions():
    """Eval mode returns a 'centre' key with B per-image detections."""
    from psstt.model import VideoFasterRCNN
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
    assert len(out["centre"]) == 1
    pred = out["centre"][0]
    for k in ("boxes", "scores", "labels"):
        assert k in pred
        assert torch.is_tensor(pred[k])


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_single_window_overfit():
    """Loss decreases monotonically-ish on one synthetic window.

    We don't aim for a specific final loss — just that 30 SGD steps drive
    the box-cls + box-reg losses lower than at init by a meaningful margin.
    """
    from psstt.model import VideoFasterRCNN
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
    for step in range(30):
        loss_dict = model(images, targets=targets)
        loss = sum(loss_dict.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses_history.append(float(loss.item()))

    # First-5 mean vs last-5 mean: expect at least 20% drop.
    first5 = sum(losses_history[:5]) / 5
    last5 = sum(losses_history[-5:]) / 5
    assert last5 < first5 * 0.8, (
        f"loss did not drop enough: first5={first5:.3f} last5={last5:.3f} "
        f"history={losses_history}"
    )
