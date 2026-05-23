"""Smoke tests for the Post-Network Tuning and Prompt Tuning alternatives.

Heavy model-build tests are gated behind ``RFDETR_VIDEO_HEAVY=1`` so
they do not run in CI by default.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from rfdetr_video.config import Config, apply_adapt_mode
from rfdetr_video.postnet import TemporalPostNet
from rfdetr_video.prompt import TemporalPromptBank


HEAVY = os.environ.get("RFDETR_VIDEO_HEAVY") == "1"
HEAVY_REASON = "Set RFDETR_VIDEO_HEAVY=1 to run model-build smoke tests"
ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────
#  Config / apply_adapt_mode
# ─────────────────────────────────────────────────────────────────────

def test_config_has_adapt_fields():
    cfg = Config()
    assert cfg.adapt_mode == "full"
    assert cfg.postnet_enabled is False
    assert cfg.prompt_enabled is False
    assert cfg.postnet_heads == 8
    assert cfg.postnet_layers == 1
    assert cfg.prompt_num_prompts == 16
    assert cfg.prompt_propagate == "gru"


def test_apply_adapt_mode_full_is_noop():
    cfg = Config(
        adapt_mode="full",
        etf_enabled=True,
        distill_enabled=True,
        consistency_enabled=True,
    )
    cfg = apply_adapt_mode(cfg)
    assert cfg.etf_enabled is True
    assert cfg.distill_enabled is True
    assert cfg.consistency_enabled is True
    assert cfg.postnet_enabled is False
    assert cfg.prompt_enabled is False


def test_apply_adapt_mode_postnet_forces_freezes():
    cfg = Config(
        adapt_mode="postnet",
        etf_enabled=True,
        distill_enabled=True,
        crrcd_enabled=True,
        consistency_enabled=True,
        freeze_backbone=False,
        freeze_decoder=False,
    )
    cfg = apply_adapt_mode(cfg)
    assert cfg.postnet_enabled is True
    assert cfg.prompt_enabled is False
    assert cfg.freeze_backbone is True
    assert cfg.freeze_decoder is True
    assert cfg.etf_enabled is False
    assert cfg.distill_enabled is False
    assert cfg.crrcd_enabled is False
    assert cfg.consistency_enabled is False


def test_apply_adapt_mode_prompt_forces_freezes():
    cfg = Config(adapt_mode="prompt", etf_enabled=True, distill_enabled=True)
    cfg = apply_adapt_mode(cfg)
    assert cfg.prompt_enabled is True
    assert cfg.postnet_enabled is False
    assert cfg.freeze_backbone is True
    assert cfg.freeze_decoder is True
    assert cfg.etf_enabled is False
    assert cfg.distill_enabled is False


def test_apply_adapt_mode_rejects_bogus():
    cfg = Config(adapt_mode="bogus")
    with pytest.raises(ValueError, match="adapt_mode"):
        apply_adapt_mode(cfg)


# ─────────────────────────────────────────────────────────────────────
#  TemporalPostNet
# ─────────────────────────────────────────────────────────────────────

def test_postnet_output_shape_matches_input():
    torch.manual_seed(0)
    B, T, Q, D = 2, 4, 6, 32
    postnet = TemporalPostNet(d_model=D, n_heads=4, n_layers=1)
    hs = torch.randn(B * T, Q, D)
    out = postnet(hs, B, T)
    assert out.shape == hs.shape


def test_postnet_is_identity_at_init():
    """Zero-init residual: at construction, postnet(hs) == hs."""
    torch.manual_seed(1)
    B, T, Q, D = 2, 3, 5, 16
    postnet = TemporalPostNet(d_model=D, n_heads=2, n_layers=2)
    hs = torch.randn(B * T, Q, D)
    with torch.no_grad():
        out = postnet(hs, B, T)
    torch.testing.assert_close(out, hs, rtol=1e-5, atol=1e-5)


def test_postnet_gradient_flow():
    torch.manual_seed(2)
    B, T, Q, D = 2, 3, 4, 16
    postnet = TemporalPostNet(d_model=D, n_heads=2, n_layers=1)
    # Nudge out_proj off zero so the gradient sees a non-trivial path.
    with torch.no_grad():
        postnet.attns[0].out_proj.weight.copy_(torch.eye(D) * 0.01)
    hs = torch.randn(B * T, Q, D, requires_grad=True)
    out = postnet(hs, B, T)
    out.square().mean().backward()
    grads = [p.grad for p in postnet.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in grads)


def test_postnet_rejects_wrong_d_model():
    postnet = TemporalPostNet(d_model=8)
    hs = torch.randn(4, 3, 16)
    with pytest.raises(ValueError, match="d_model"):
        postnet(hs, B=2, T=2)


def test_postnet_rejects_inconsistent_BT():
    postnet = TemporalPostNet(d_model=8)
    hs = torch.randn(5, 3, 8)  # 5 != 2*2
    with pytest.raises(ValueError, match="B\\*T"):
        postnet(hs, B=2, T=2)


# ─────────────────────────────────────────────────────────────────────
#  TemporalPromptBank
# ─────────────────────────────────────────────────────────────────────

def _fake_srcs(B: int, T: int, C: int, h: int, w: int) -> list:
    """Two-scale fake backbone outputs; the bank uses the last one."""
    s1 = torch.randn(B * T, C, h * 2, w * 2)
    s2 = torch.randn(B * T, C, h, w)
    return [s1, s2]


def test_prompt_bank_output_shape():
    torch.manual_seed(3)
    B, T, N, D = 2, 4, 8, 16
    bank = TemporalPromptBank(
        n_prompts=N, d_model=D, feat_channels=D, init_std=0.02,
    )
    srcs = _fake_srcs(B, T, D, 3, 3)
    out = bank(srcs, B, T)
    assert out.shape == (B, T, N, D)


def test_prompt_bank_propagates_across_time():
    """After several GRU steps the prompts should diverge from P0."""
    torch.manual_seed(4)
    B, T, N, D = 1, 5, 4, 8
    bank = TemporalPromptBank(
        n_prompts=N, d_model=D, feat_channels=D, init_std=0.02,
    )
    srcs = _fake_srcs(B, T, D, 2, 2)
    out = bank(srcs, B, T)
    P_last = out[0, -1]
    P0_broadcast = bank.P0
    # Not bitwise-equal because the GRU rewrites the state at every step.
    assert (P_last - P0_broadcast).abs().sum().item() > 1e-3


def test_prompt_bank_gradient_reaches_P0_and_gru():
    torch.manual_seed(5)
    B, T, N, D = 1, 3, 4, 8
    bank = TemporalPromptBank(
        n_prompts=N, d_model=D, feat_channels=D, init_std=0.02,
    )
    srcs = _fake_srcs(B, T, D, 2, 2)
    out = bank(srcs, B, T)
    out.square().mean().backward()
    assert bank.P0.grad is not None
    assert bank.P0.grad.abs().sum().item() > 0
    gru_grads = [p.grad for p in bank.gru.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in gru_grads)


def test_prompt_bank_lazy_projector_init():
    torch.manual_seed(6)
    B, T, N, D = 1, 2, 4, 8
    bank = TemporalPromptBank(n_prompts=N, d_model=D)
    srcs = _fake_srcs(B, T, C=12, h=2, w=2)  # feat dim != d_model on purpose
    _ = bank(srcs, B, T)
    assert bank._proj_initialised
    assert isinstance(bank.feat_proj, torch.nn.Linear)
    assert bank.feat_proj.in_features == 12
    assert bank.feat_proj.out_features == D


# ─────────────────────────────────────────────────────────────────────
#  _param_group_for: ensure the new modules land in the "new" bucket
# ─────────────────────────────────────────────────────────────────────

def test_param_group_for_postnet_and_prompt():
    from rfdetr_video.model import _param_group_for
    assert _param_group_for("postnet.attns.0.in_proj_weight") == "new"
    assert _param_group_for("postnet.norms.0.weight") == "new"
    assert _param_group_for("prompt_bank.P0") == "new"
    assert _param_group_for("prompt_bank.gru.weight_ih") == "new"
    assert _param_group_for("prompt_bank.feat_proj.weight") == "new"


# ─────────────────────────────────────────────────────────────────────
#  Model wiring — heavy tests
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_postnet_model_forward_and_trainables():
    from rfdetr_video.model import VideoRFDETR

    cfg = Config(
        batch_size=1, T=3, img_size=384,
        adapt_mode="postnet", consistency_enabled=False,
    )
    cfg = apply_adapt_mode(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoRFDETR(cfg).to(device).eval()
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size, device=device)
    with torch.no_grad():
        out = model(frames, query_mode="student")
    Q = model.num_queries
    assert out["pred_logits"].shape[:3] == (1, cfg.T, Q)
    assert out["pred_boxes"].shape == (1, cfg.T, Q, 4)
    assert "aux_outputs" not in out  # dropped in postnet mode

    # Only postnet params should require grad.
    trainable_names = {
        n for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable_names, "no trainable params in postnet mode"
    assert all(n.startswith("postnet.") for n in trainable_names), (
        f"unexpected trainable params outside postnet: "
        f"{sorted(trainable_names - {n for n in trainable_names if n.startswith('postnet.')})}"
    )


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_prompt_model_forward_and_trainables():
    from rfdetr_video.model import VideoRFDETR

    cfg = Config(
        batch_size=1, T=3, img_size=384,
        adapt_mode="prompt", consistency_enabled=False,
        prompt_num_prompts=8,
    )
    cfg = apply_adapt_mode(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoRFDETR(cfg).to(device).eval()
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size, device=device)
    with torch.no_grad():
        out = model(frames, query_mode="student")
    Q = model.num_queries
    assert out["pred_logits"].shape[:3] == (1, cfg.T, Q)
    assert out["pred_boxes"].shape == (1, cfg.T, Q, 4)

    trainable_names = {
        n for n, p in model.named_parameters() if p.requires_grad
    }
    assert trainable_names, "no trainable params in prompt mode"
    assert all(n.startswith("prompt_bank.") for n in trainable_names), (
        f"unexpected trainable params outside prompt_bank: "
        f"{sorted(trainable_names - {n for n in trainable_names if n.startswith('prompt_bank.')})}"
    )


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_postnet_loss_gradient_flow_endtoend():
    from rfdetr_video.model import VideoRFDETR, build_criterion
    from rfdetr_video.train import _flatten_predictions, _flatten_targets

    cfg = Config(
        batch_size=1, T=3, img_size=384,
        adapt_mode="postnet", consistency_enabled=False,
    )
    cfg = apply_adapt_mode(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoRFDETR(cfg).to(device).train()
    criterion, _ = build_criterion(cfg)
    criterion = criterion.to(device).train()

    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size, device=device)
    targets_list = [[
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
         "labels": torch.tensor([0], device=device)}
        for _ in range(cfg.T)
    ]]
    targets_flat = _flatten_targets(targets_list, device, cfg.img_size)

    out = model(frames, query_mode="student")
    pred_flat = _flatten_predictions(out, 1, cfg.T)
    loss_dict = criterion(pred_flat, targets_flat)
    loss = sum(
        loss_dict[k] * criterion.weight_dict[k]
        for k in loss_dict if k in criterion.weight_dict
    )
    loss.backward()
    pn_grads = [p.grad for p in model.postnet.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in pn_grads)


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_prompt_loss_gradient_flow_endtoend():
    from rfdetr_video.model import VideoRFDETR, build_criterion
    from rfdetr_video.train import _flatten_predictions, _flatten_targets

    cfg = Config(
        batch_size=1, T=3, img_size=384,
        adapt_mode="prompt", consistency_enabled=False,
        prompt_num_prompts=8,
    )
    cfg = apply_adapt_mode(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoRFDETR(cfg).to(device).train()
    criterion, _ = build_criterion(cfg)
    criterion = criterion.to(device).train()

    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size, device=device)
    targets_list = [[
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
         "labels": torch.tensor([0], device=device)}
        for _ in range(cfg.T)
    ]]
    targets_flat = _flatten_targets(targets_list, device, cfg.img_size)

    out = model(frames, query_mode="student")
    pred_flat = _flatten_predictions(out, 1, cfg.T)
    loss_dict = criterion(pred_flat, targets_flat)
    loss = sum(
        loss_dict[k] * criterion.weight_dict[k]
        for k in loss_dict if k in criterion.weight_dict
    )
    loss.backward()
    pb_grads = [p.grad for p in model.prompt_bank.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in pb_grads)
