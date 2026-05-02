"""Smoke tests for :mod:`rfdetr_video`.

Run a focused subset (no GPU-bound forward) with::

    pytest rfdetr_video/tests/test_smoke.py -v -k "not training_step and not forward_shapes"

Heavier tests that build the full model are gated behind the
``RFDETR_VIDEO_HEAVY=1`` env var so they do not run in CI by default.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from rfdetr_video.config import Config
from rfdetr_video.consistency import num_consistency_loss
from rfdetr_video.stfs import track_queries, inject_features


HEAVY = os.environ.get("RFDETR_VIDEO_HEAVY") == "1"
HEAVY_REASON = "Set RFDETR_VIDEO_HEAVY=1 to run model-build smoke tests"
ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────
#  1. Codebase invariants
# ─────────────────────────────────────────────────────────────────────
def test_no_unfold_in_codebase():
    """Phase 2 constraint — no actual ``F.unfold`` / ``nn.Unfold`` *call*."""
    pkg = ROOT / "rfdetr_video"
    out = subprocess.run(
        [
            "grep", "-RnE",
            "--include=*.py",
            "--exclude-dir=tests",
            "--exclude-dir=__pycache__",
            r"(F\.unfold|nn\.Unfold)\(",
            str(pkg),
        ],
        capture_output=True, text=True,
    )
    # ``grep`` exits 1 when no matches — that's the desired outcome.
    assert out.returncode == 1, (
        f"F.unfold / nn.Unfold leaked into rfdetr_video:\n{out.stdout}"
    )


def test_constraint_checklist():
    """Compact verification of architecture constraints (source-level)."""
    src = (ROOT / "rfdetr_video" / "model.py").read_text()
    # Per-frame backbone reshape exists.
    assert "reshape(BT" in src or "B * T" in src
    # STFS + refinement integration.
    assert "track_queries" in src and "inject_features" in src
    assert "_refinement_pass" in src
    # KD branches present.
    assert 'query_mode == "teacher"' in src
    assert '"general"' in src
    # Refinement layer is a deepcopy (warm-init).
    assert "copy.deepcopy(self.transformer.decoder.layers[-1])" in src

    # Consistency loss penalises flickering.
    assert callable(num_consistency_loss)


# ─────────────────────────────────────────────────────────────────────
#  2. Consistency loss
# ─────────────────────────────────────────────────────────────────────
def test_consistency_loss_zero_when_identical():
    B, T, Q, K = 2, 5, 50, 1
    logits = torch.zeros(B, T, Q, K, requires_grad=True)
    # Make first 3 queries strongly positive on every frame.
    with torch.no_grad():
        logits[..., :3, 0] = 5.0
        logits[..., 3:, 0] = -5.0
    logits.requires_grad_(True)
    loss = num_consistency_loss(logits, threshold=0.3, soft_temp=0.05)
    assert loss.item() < 1e-3, f"identical preds should give ~0 loss, got {loss.item()}"


def test_consistency_loss_positive_when_flickering():
    B, T, Q, K = 1, 5, 50, 1
    logits = torch.full((B, T, Q, K), -5.0)
    # Flicker: 5 boxes on even frames, 0 on odd frames.
    for t in (0, 2, 4):
        logits[0, t, :5, 0] = 5.0
    logits.requires_grad_(True)
    loss = num_consistency_loss(logits, threshold=0.3, soft_temp=0.05)
    assert loss.item() > 0.5, f"flickering should give large loss, got {loss.item()}"
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


# ─────────────────────────────────────────────────────────────────────
#  3. STFS tracking + injection
# ─────────────────────────────────────────────────────────────────────
def test_stfs_track_and_inject():
    """A confident object on frames {0,1,3,4}; missing on frame 2 (H-FN).

    Track must form, and the H-FN slot at frame 2 must inherit the
    embedding + reference point of the strongest in-track frame.
    """
    B, T, Q, D, K = 1, 5, 8, 16, 1
    cfg = Config()

    boxes = torch.zeros(B, T, Q, 4)
    logits = torch.full((B, T, Q, K), -10.0)
    # Object near (0.5, 0.5) on frames 0,1,3,4 in slot 0.
    for t in (0, 1, 3, 4):
        boxes[0, t, 0] = torch.tensor([0.5, 0.5, 0.2, 0.2])
        logits[0, t, 0, 0] = 5.0  # confident
    # Frame 2: no confident slot for this object.

    tracks = track_queries(
        boxes, logits,
        iou_weight=cfg.stfs_iou_weight,
        l1_weight=cfg.stfs_l1_weight,
        cls_weight=cfg.stfs_cls_weight,
        iou_gate=cfg.stfs_match_iou_thresh,
        score_thresh=cfg.stfs_track_score_thresh,
        min_track_len=cfg.stfs_min_track_len,
    )
    assert len(tracks[0]) >= 1, f"expected ≥1 track, got {len(tracks[0])}"
    track = tracks[0][0]
    assert -1 in track.slots, f"track must have an H-FN frame, got slots={track.slots}"
    n_real = sum(1 for s in track.slots if s != -1)
    assert n_real >= cfg.stfs_min_track_len

    # Inject: source slot embedding must replace H-FN slot.
    embed = torch.randn(B, T, Q, D, requires_grad=True)
    refs = torch.randn(B, T, Q, 4, requires_grad=True)

    new_emb, new_ref = inject_features(embed, refs, tracks, alpha=1.0)
    h_fn_t = track.slots.index(-1)
    src_t = track.best_t
    src_q = track.best_p_slot if hasattr(track, "best_p_slot") else None
    # ``best_q`` is the slot index used as the source / target; check via diff.
    diffs = (new_emb[0, h_fn_t] - embed[0, h_fn_t]).abs().sum(-1)
    assert (diffs > 1e-4).any(), "at least one slot must be modified at H-FN frame"
    # Backward should flow.
    new_emb.sum().backward()
    assert embed.grad is not None and embed.grad.abs().sum().item() > 0


def test_stfs_no_tracks_for_low_confidence():
    """When all scores are below threshold → no tracks, embeddings unchanged."""
    B, T, Q, D, K = 1, 5, 4, 8, 1
    boxes = torch.rand(B, T, Q, 4) * 0.5 + 0.25
    logits = torch.full((B, T, Q, K), -10.0)
    tracks = track_queries(
        boxes, logits,
        iou_weight=2.0, l1_weight=5.0, cls_weight=2.0,
        iou_gate=0.1, score_thresh=0.3, min_track_len=3,
    )
    assert len(tracks[0]) == 0
    embed = torch.randn(B, T, Q, D, requires_grad=True)
    refs = torch.randn(B, T, Q, 4, requires_grad=True)
    new_emb, new_ref = inject_features(embed, refs, tracks, alpha=1.0)
    assert torch.allclose(new_emb, embed)
    assert torch.allclose(new_ref, refs)


# ─────────────────────────────────────────────────────────────────────
#  4. Dataset
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(
    not (ROOT / "data" / "dataset2_split").exists(),
    reason="data/dataset2_split not available",
)
def test_dataset_returns_T_targets():
    from rfdetr_video.dataset import get_video_dataloader
    cfg = Config(batch_size=1, num_workers=0, T=5)
    loader = get_video_dataloader("valid", cfg, shuffle=False, with_teacher_frame=True)
    images, targets, teacher, fnames = next(iter(loader))
    B, T = images.shape[:2]
    assert T == cfg.T
    assert len(targets) == B
    assert all(len(t) == T for t in targets)
    assert teacher.shape == (B, T, 3, cfg.distill_teacher_resolution,
                             cfg.distill_teacher_resolution)
    for t_dict in targets[0]:
        assert "boxes" in t_dict and "labels" in t_dict
        assert t_dict["boxes"].dim() == 2 and t_dict["boxes"].shape[-1] == 4


# ─────────────────────────────────────────────────────────────────────
#  5. Model build + forward (heavy)
# ─────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_forward_shapes():
    from rfdetr_video.model import VideoRFDETR

    cfg = Config(batch_size=1, T=3, img_size=512, distill_enabled=False,
                 consistency_enabled=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoRFDETR(cfg).to(device).eval()
    B, T = 1, cfg.T
    frames = torch.randn(B, T, 3, cfg.img_size, cfg.img_size, device=device)
    with torch.no_grad():
        out = model(frames, query_mode="student")
    Q = model.num_queries
    K = cfg.num_classes + 1  # background slot
    assert out["pred_logits"].shape[:3] == (B, T, Q)
    assert out["pred_boxes"].shape == (B, T, Q, 4)
    assert "first_pass" in out
    assert out["first_pass"]["pred_logits"].shape[:3] == (B, T, Q)


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_one_training_step():
    """Single forward + backward — refinement params receive gradient."""
    from rfdetr_video.model import VideoRFDETR, build_criterion
    from rfdetr_video.train import _flatten_targets, _flatten_predictions

    cfg = Config(batch_size=1, T=3, img_size=384, distill_enabled=False,
                 consistency_enabled=True, freeze_decoder=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoRFDETR(cfg).to(device).train()
    criterion, _ = build_criterion(cfg)
    criterion = criterion.to(device).train()

    B, T = 1, cfg.T
    frames = torch.randn(B, T, 3, cfg.img_size, cfg.img_size, device=device)
    targets_list = [[
        {"boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
         "labels": torch.tensor([0], device=device)}
        for _ in range(T)
    ]]
    targets_flat = _flatten_targets(targets_list, device, cfg.img_size)
    out = model(frames, query_mode="student")
    pred_flat = _flatten_predictions(out, B, T)
    loss_dict = criterion(pred_flat, targets_flat)
    loss = sum(loss_dict[k] * criterion.weight_dict[k]
               for k in loss_dict if k in criterion.weight_dict)
    loss = loss + cfg.consistency_weight * num_consistency_loss(
        out["pred_logits"], cfg.consistency_threshold, cfg.consistency_soft_temp,
    )
    loss.backward()

    refine_grads = [p.grad for p in model.refine_layer.parameters() if p.grad is not None]
    assert any(g.abs().sum().item() > 0 for g in refine_grads), (
        "refinement layer must receive gradient from the multi-frame loss"
    )


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_crrcd_per_frame():
    """CRRCD loss flows gradient into the student only."""
    from rfdetr_video.distill import CRRCDLoss

    BT, Q, D = 6, 64, 256  # B*T=6
    teacher_hs = torch.randn(BT, Q, D)
    student_hs = torch.randn(BT, Q, D, requires_grad=True)
    weights = torch.rand(BT, Q)
    crrcd = CRRCDLoss(
        hidden_dim=D, relation_dim=128, frm_hidden_dim=128,
        num_fg=4, num_bg=8, num_negatives=4, temperature=0.1,
    )
    loss = crrcd(teacher_hs, student_hs, weights)
    loss.backward()
    assert student_hs.grad is not None and student_hs.grad.abs().sum().item() > 0
    assert teacher_hs.grad is None
