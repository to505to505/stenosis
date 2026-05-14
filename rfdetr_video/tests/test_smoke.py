"""Smoke tests for :mod:`rfdetr_video`.

Run a focused subset with::

    pytest rfdetr_video/tests/test_smoke.py -v -k "not heavy"

Model-build tests are gated behind ``RFDETR_VIDEO_HEAVY=1`` so they do
not run in CI by default.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import torch

from rfdetr_video.config import Config, resolve_distill_frame_indices
from rfdetr_video.consistency import num_consistency_loss
from rfdetr_video.ema import ModelEMA
from rfdetr_video.schedule import build_scheduler
from rfdetr_video.selection import (
    composite_selection_score,
    SmoothedTracker,
    EarlyStopper,
)


HEAVY = os.environ.get("RFDETR_VIDEO_HEAVY") == "1"
HEAVY_REASON = "Set RFDETR_VIDEO_HEAVY=1 to run model-build smoke tests"
ROOT = Path(__file__).resolve().parents[2]


def test_config_has_regime_fields():
    cfg = Config()
    assert cfg.epochs == 20
    assert cfg.lr == 1e-4
    assert cfg.lr_pretrained == 3e-5
    assert cfg.lr_schedule == "cosine"
    assert cfg.ema_enabled is True
    assert cfg.ema_decay == 0.999
    assert cfg.selection_smooth_k == 3
    assert cfg.selection_weights == (0.5, 0.3, 0.2)
    assert cfg.early_stop_enabled is True
    assert cfg.early_stop_patience == 6
    assert cfg.early_stop_min_delta == 0.0


def test_no_unfold_in_codebase():
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
    assert out.returncode == 1, f"F.unfold / nn.Unfold call found:\n{out.stdout}"


def test_source_invariants():
    src = (ROOT / "rfdetr_video" / "model.py").read_text()
    assert "EarlyTemporalFusion" in src
    assert "etf_spatial_radius" in src
    assert "reshape(BT" in src or "B * T" in src
    assert "_captured_decoder_hs" in src
    assert 'query_mode == "teacher"' in src
    assert '"general"' in src
    assert callable(num_consistency_loss)


def test_etf_spatial_radius_zero_matches_temporal_only_reference():
    from rfdetr_video.model import EarlyTemporalFusion

    torch.manual_seed(7)
    B, T, C, h, w = 2, 3, 4, 2, 3
    etf = EarlyTemporalFusion(
        d_model=C, n_heads=2, dropout=0.0, spatial_radius=0,
    ).eval()
    with torch.no_grad():
        etf.attn.out_proj.weight.copy_(torch.eye(C))
        etf.attn.out_proj.bias.zero_()

    src = torch.randn(B * T, C, h, w)
    with torch.no_grad():
        out = etf([src.clone()], B, T)[0]

        x = src.reshape(B, T, C, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        x = x.reshape(B * h * w, T, C)
        x_n = etf.norm(x)
        attn_out, _ = etf.attn(x_n, x_n, x_n)
        ref = x + attn_out
        ref = ref.reshape(B, h, w, T, C)
        ref = ref.permute(0, 3, 4, 1, 2).contiguous()
        ref = ref.reshape(B * T, C, h, w)

    torch.testing.assert_close(out, ref)


def test_etf_spatial_radius_one_builds_valid_local_window():
    from rfdetr_video.model import EarlyTemporalFusion

    B, T, C, h, w = 1, 2, 1, 3, 3
    etf = EarlyTemporalFusion(d_model=C, n_heads=1, spatial_radius=1)
    x = torch.arange(B * T * C * h * w, dtype=torch.float32)
    x = x.reshape(B, T, C, h, w)

    key_value, key_padding_mask = etf._local_key_value_tokens(x)

    assert key_value.shape == (B * h * w, T * 9, C)
    assert key_padding_mask.shape == (B * h * w, T * 9)

    corner_index = 0
    centre_index = 4
    assert key_padding_mask[centre_index].sum().item() == 0
    assert key_padding_mask[corner_index].sum().item() == 10
    assert (~key_padding_mask[corner_index]).sum().item() == 8

    corner_values = key_value[corner_index][~key_padding_mask[corner_index], 0]
    expected_corner_values = torch.tensor(
        [0.0, 9.0, 1.0, 10.0, 3.0, 12.0, 4.0, 13.0],
    )
    torch.testing.assert_close(corner_values, expected_corner_values)


def test_etf_spatial_radius_one_forward_and_gradient():
    from rfdetr_video.model import EarlyTemporalFusion

    torch.manual_seed(11)
    B, T, C, h, w = 2, 3, 4, 3, 3
    etf = EarlyTemporalFusion(d_model=C, n_heads=2, spatial_radius=1)
    src = torch.randn(B * T, C, h, w, requires_grad=True)

    out = etf([src], B, T)[0]
    assert out.shape == src.shape
    assert torch.isfinite(out).all()

    loss = out.square().mean()
    loss.backward()
    grads = [param.grad for param in etf.parameters() if param.grad is not None]
    assert any(grad.abs().sum().item() > 0 for grad in grads)


def test_distill_frame_index_resolution():
    all_frames_cfg = Config(T=5)
    assert resolve_distill_frame_indices(all_frames_cfg.T, all_frames_cfg) is None

    centre_cfg = Config(T=5, distill_centre_frame_only=True)
    assert resolve_distill_frame_indices(centre_cfg.T, centre_cfg) == [2]

    neighbour_cfg = Config(T=5, distill_frame_offsets=(-1, 1))
    assert resolve_distill_frame_indices(neighbour_cfg.T, neighbour_cfg) == [1, 3]

    with pytest.raises(ValueError, match="outside"):
        invalid_cfg = Config(T=1, distill_frame_offsets=(-1, 1))
        resolve_distill_frame_indices(invalid_cfg.T, invalid_cfg)


def test_consistency_loss_zero_when_identical():
    B, T, Q, K = 2, 5, 50, 1
    logits = torch.zeros(B, T, Q, K, requires_grad=True)
    with torch.no_grad():
        logits[..., :3, 0] = 5.0
        logits[..., 3:, 0] = -5.0
    logits.requires_grad_(True)
    loss = num_consistency_loss(logits, threshold=0.3, soft_temp=0.05)
    assert loss.item() < 1e-3


def test_consistency_loss_positive_when_flickering():
    B, T, Q, K = 1, 5, 50, 1
    logits = torch.full((B, T, Q, K), -5.0)
    for t in (0, 2, 4):
        logits[0, t, :5, 0] = 5.0
    logits.requires_grad_(True)
    loss = num_consistency_loss(logits, threshold=0.3, soft_temp=0.05)
    assert loss.item() > 0.5
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0


@pytest.mark.skipif(
    not (ROOT / "data" / "dataset2_split").exists(),
    reason="data/dataset2_split not available",
)
def test_dataset_returns_T_targets():
    pytest.importorskip("rfdetr_temporal.dataset")
    from rfdetr_video.dataset import get_video_dataloader

    cfg = Config(batch_size=1, num_workers=0, T=5)
    loader = get_video_dataloader("valid", cfg, shuffle=False, with_teacher_frame=True)
    images, targets, teacher, _fnames = next(iter(loader))
    B, T = images.shape[:2]
    assert T == cfg.T
    assert len(targets) == B
    assert all(len(target_list) == T for target_list in targets)
    assert teacher.shape == (
        B, T, 3, cfg.distill_teacher_resolution, cfg.distill_teacher_resolution,
    )
    for target in targets[0]:
        assert "boxes" in target and "labels" in target
        assert target["boxes"].dim() == 2 and target["boxes"].shape[-1] == 4


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_forward_shapes():
    from rfdetr_video.model import VideoRFDETR

    cfg = Config(batch_size=1, T=3, img_size=512, consistency_enabled=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VideoRFDETR(cfg).to(device).eval()
    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size, device=device)
    with torch.no_grad():
        out = model(frames, query_mode="student")
    Q = model.num_queries
    assert out["pred_logits"].shape[:3] == (1, cfg.T, Q)
    assert out["pred_boxes"].shape == (1, cfg.T, Q, 4)
    assert "first_pass" in out
    torch.testing.assert_close(out["pred_logits"], out["first_pass"]["pred_logits"])
    torch.testing.assert_close(out["pred_boxes"], out["first_pass"]["pred_boxes"])


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_etf_receives_gradient():
    from rfdetr_video.model import VideoRFDETR, build_criterion
    from rfdetr_video.train import _flatten_predictions, _flatten_targets

    cfg = Config(
        batch_size=1,
        T=3,
        img_size=384,
        consistency_enabled=True,
        etf_enabled=True,
        etf_spatial_radius=1,
        freeze_decoder=True,
    )
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
        loss_dict[key] * criterion.weight_dict[key]
        for key in loss_dict if key in criterion.weight_dict
    )
    loss.backward()

    assert model.etf is not None
    grads = [param.grad for param in model.etf.parameters() if param.grad is not None]
    assert any(grad.abs().sum().item() > 0 for grad in grads)


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_crrcd_per_frame():
    from rfdetr_video.distill import CRRCDLoss

    BT, Q, D = 6, 64, 256
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


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_distill_one_step():
    teacher_ckpt = ROOT / "rfdetr_runs" / "rfdetr_large_arcade2x_704_reg" \
        / "checkpoint_best_total.pth"
    if not teacher_ckpt.exists():
        pytest.skip(f"Teacher checkpoint not available: {teacher_ckpt}")

    from rfdetr_video.model import VideoRFDETR, build_criterion
    from rfdetr_video.distill import (
        VideoFrozenRFDETRTeacher,
        distillation_loss,
        CRRCDLoss,
    )
    from rfdetr_video.train import _flatten_predictions, _flatten_targets

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = Config(
        batch_size=1,
        T=3,
        img_size=384,
        distill_teacher_resolution=384,
        distill_enabled=True,
        distill_general_enabled=True,
        distill_num_general_queries=16,
        crrcd_enabled=True,
        crrcd_num_fg=4,
        crrcd_num_bg=8,
        crrcd_num_negatives=4,
        consistency_enabled=True,
        etf_enabled=True,
        wandb_enabled=False,
    )

    model = VideoRFDETR(cfg).to(device).train()
    criterion, _ = build_criterion(cfg)
    criterion = criterion.to(device).train()

    teacher = VideoFrozenRFDETRTeacher(cfg).to(device).eval()
    model.register_teacher_queries(
        teacher.refpoint_embed_weight, teacher.query_feat_weight,
    )
    crrcd_module = CRRCDLoss(
        hidden_dim=int(teacher.hidden_dim),
        relation_dim=int(cfg.crrcd_relation_dim),
        frm_hidden_dim=int(cfg.crrcd_hidden_dim),
        num_fg=int(cfg.crrcd_num_fg),
        num_bg=int(cfg.crrcd_num_bg),
        num_negatives=int(cfg.crrcd_num_negatives),
        temperature=float(cfg.crrcd_temperature),
    ).to(device)

    frames = torch.randn(1, cfg.T, 3, cfg.img_size, cfg.img_size, device=device)
    teacher_frames = torch.randn(
        1, cfg.T, 3, cfg.distill_teacher_resolution,
        cfg.distill_teacher_resolution, device=device,
    )
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
        loss_dict[key] * criterion.weight_dict[key]
        for key in loss_dict if key in criterion.weight_dict
    )
    loss = loss + cfg.consistency_weight * num_consistency_loss(
        out["pred_logits"], cfg.consistency_threshold, cfg.consistency_soft_temp,
    )

    with torch.no_grad():
        teacher_out = teacher.forward_video(teacher_frames)
    student_specific = model(
        frames,
        query_mode="teacher",
        decoder_inputs={
            "tgt": teacher_out["decoder_tgt"],
            "refpoints": teacher_out["decoder_refpoints"],
        },
    )
    assert model._captured_decoder_hs is not None
    distill_specific = distillation_loss(student_specific, teacher_out, cfg)
    loss = loss + cfg.distill_loss_weight * distill_specific["loss_distill"]
    loss_rcd = crrcd_module(
        teacher_hs=teacher_out["decoder_hs"],
        student_hs=model._captured_decoder_hs,
        weights=teacher_out["foreground_weight"],
    )
    loss = loss + cfg.crrcd_loss_weight * loss_rcd

    general_queries = model.sample_general_queries(
        cfg.distill_num_general_queries, device=device, dtype=frames.dtype,
    )
    with torch.no_grad():
        teacher_general = teacher.forward_video_general(
            teacher_frames,
            general_queries["refpoint"],
            general_queries["query_feat"],
            min_weight=cfg.distill_general_min_weight,
        )
    student_general = model(
        frames,
        query_mode="general",
        general_queries=general_queries,
        decoder_inputs={
            "tgt": teacher_general["decoder_tgt"],
            "refpoints": teacher_general["decoder_refpoints"],
        },
    )
    distill_general = distillation_loss(student_general, teacher_general, cfg)
    loss = loss + (
        cfg.distill_loss_weight * cfg.distill_general_loss_weight
        * distill_general["loss_distill"]
    )

    assert torch.isfinite(loss)
    loss.backward()

    assert model.etf is not None
    etf_grads = [param.grad for param in model.etf.parameters() if param.grad is not None]
    assert any(grad.abs().sum().item() > 0 for grad in etf_grads)

    crrcd_grads = [param.grad for param in crrcd_module.parameters() if param.grad is not None]
    assert any(grad.abs().sum().item() > 0 for grad in crrcd_grads)

    for name, param in teacher.named_parameters():
        assert param.grad is None, f"teacher param '{name}' received gradient"


def _tiny_model_with_frozen():
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    # freeze the second layer
    for p in model[1].parameters():
        p.requires_grad_(False)
    return model


def test_model_ema_tracks_only_trainable_params():
    import torch.nn as nn
    model = _tiny_model_with_frozen()
    ema = ModelEMA(model, decay=0.9)
    # shadow holds exactly the trainable param names
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(ema.shadow.keys()) == trainable


def test_model_ema_update_math():
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    model = nn.Linear(3, 3)
    ema = ModelEMA(model, decay=0.8)
    before = {n: v.clone() for n, v in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # move the live model
    ema.update(model)
    for n, p in model.named_parameters():
        expected = before[n] * 0.8 + p.detach() * 0.2
        torch.testing.assert_close(ema.shadow[n], expected)


def test_model_ema_constant_param_stays_constant():
    import torch
    import torch.nn as nn
    model = nn.Linear(3, 3)
    ema = ModelEMA(model, decay=0.5)
    snapshot = {n: v.clone() for n, v in ema.shadow.items()}
    ema.update(model)  # model unchanged -> EMA unchanged
    for n in snapshot:
        torch.testing.assert_close(ema.shadow[n], snapshot[n])


def test_model_ema_applied_to_round_trips():
    import torch
    import torch.nn as nn
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.5)
    # force a known EMA value
    for n in ema.shadow:
        ema.shadow[n].fill_(7.0)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    with ema.applied_to(model):
        for n, p in model.named_parameters():
            assert torch.all(p == 7.0)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.detach(), original[n])


def test_model_ema_applied_to_leaves_frozen_params_untouched():
    import torch
    model = _tiny_model_with_frozen()
    ema = ModelEMA(model, decay=0.5)
    for n in ema.shadow:
        ema.shadow[n].fill_(7.0)
    frozen_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters() if not p.requires_grad
    }
    assert frozen_before  # the helper does have a frozen layer
    with ema.applied_to(model):
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert not torch.all(p == 7.0)  # frozen params NOT swapped
    for n, p in model.named_parameters():
        if not p.requires_grad:
            torch.testing.assert_close(p.detach(), frozen_before[n])


def test_model_ema_applied_to_restores_on_exception():
    import torch
    import torch.nn as nn
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.5)
    for n in ema.shadow:
        ema.shadow[n].fill_(99.0)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    with pytest.raises(RuntimeError):
        with ema.applied_to(model):
            raise RuntimeError("boom")
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.detach(), original[n])


def _two_group_optimizer():
    import torch
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.SGD([
        {"params": [p1], "lr": 1e-4},
        {"params": [p2], "lr": 3e-5},
    ], lr=1e-4)


def test_build_scheduler_cosine_decays_all_groups():
    cfg = Config(epochs=10, lr_schedule="cosine")
    opt = _two_group_optimizer()
    sched = build_scheduler(opt, cfg)
    start = [pg["lr"] for pg in opt.param_groups]
    for _ in range(cfg.epochs):
        sched.step()
    end = [pg["lr"] for pg in opt.param_groups]
    assert end[0] < start[0] and end[1] < start[1]
    assert end[0] < 1e-5 and end[1] < 1e-5  # near eta_min


def test_build_scheduler_multistep_passthrough():
    cfg = Config(lr_schedule="multistep", lr_step_milestones=(2,), lr_gamma=0.1)
    opt = _two_group_optimizer()
    sched = build_scheduler(opt, cfg)
    sched.step(); sched.step()  # hit milestone 2
    lrs = [pg["lr"] for pg in opt.param_groups]
    torch.testing.assert_close(lrs[0], 1e-5)   # 1e-4 * 0.1
    torch.testing.assert_close(lrs[1], 3e-6)   # 3e-5 * 0.1


def test_build_scheduler_rejects_unknown():
    cfg = Config(lr_schedule="bogus")
    opt = _two_group_optimizer()
    with pytest.raises(ValueError, match="lr_schedule"):
        build_scheduler(opt, cfg)


def test_composite_selection_score_weighted_sum():
    metrics = {"AP@0.3": 1.0, "AP@0.5": 1.0, "F1": 1.0, "other": 9.0}
    assert composite_selection_score(metrics, (0.5, 0.3, 0.2)) == pytest.approx(1.0)
    metrics2 = {"AP@0.3": 1.0, "AP@0.5": 0.0, "F1": 0.0}
    assert composite_selection_score(metrics2, (0.5, 0.3, 0.2)) == pytest.approx(0.5)
    # missing keys default to 0.0
    assert composite_selection_score({}, (0.5, 0.3, 0.2)) == pytest.approx(0.0)


def test_smoothed_tracker_rolling_mean():
    tracker = SmoothedTracker(k=3)
    assert tracker.add(1.0) == pytest.approx(1.0)
    assert tracker.add(2.0) == pytest.approx(1.5)
    assert tracker.add(3.0) == pytest.approx(2.0)
    assert tracker.add(4.0) == pytest.approx(3.0)  # window = [2,3,4]


def test_early_stopper_triggers_after_patience():
    stopper = EarlyStopper(patience=2, min_delta=0.0)
    assert stopper.update(0.5) is False   # new best
    assert stopper.update(0.4) is False   # bad 1
    assert stopper.update(0.4) is True    # bad 2 -> stop


def test_early_stopper_resets_on_improvement():
    stopper = EarlyStopper(patience=2, min_delta=0.0)
    assert stopper.update(0.5) is False
    assert stopper.update(0.4) is False   # bad 1
    assert stopper.update(0.6) is False   # new best -> reset
    assert stopper.update(0.5) is False   # bad 1
    assert stopper.update(0.5) is True    # bad 2 -> stop


def test_param_group_for_classifies_by_name():
    from rfdetr_video.model import _param_group_for
    assert _param_group_for("backbone.blocks.0.attn.qkv.weight") == "backbone"
    assert _param_group_for("etf.attn.in_proj_weight") == "new"
    assert _param_group_for("etf.norm.weight") == "new"
    assert _param_group_for("crrcd.relation_t.0.weight") == "new"
    assert _param_group_for("transformer.decoder.layers.0.self_attn.weight") == "pretrained"
    assert _param_group_for("class_embed.weight") == "pretrained"
    assert _param_group_for("bbox_embed.layers.0.weight") == "pretrained"
    assert _param_group_for("refpoint_embed.weight") == "pretrained"
    assert _param_group_for("query_feat.weight") == "pretrained"
    assert _param_group_for("transformer.backbone_proj.weight") == "pretrained"


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_get_param_groups_three_way_split():
    from rfdetr_video.model import VideoRFDETR
    cfg = Config(etf_enabled=True, lr=1e-4, lr_pretrained=3e-5, freeze_backbone=True)
    model = VideoRFDETR(cfg)
    groups = model.get_param_groups()
    # backbone frozen -> only pretrained + new groups remain
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == [3e-5, 1e-4]
    by_lr = {g["lr"]: g["params"] for g in groups}
    # ETF params land in the 1e-4 (new) group
    etf_param_ids = {id(p) for p in model.etf.parameters()}
    new_group_ids = {id(p) for p in by_lr[1e-4]}
    assert etf_param_ids and etf_param_ids.issubset(new_group_ids)
    # transformer params land in the 3e-5 (pretrained) group
    tf_param_ids = {id(p) for p in model.transformer.parameters() if p.requires_grad}
    pre_group_ids = {id(p) for p in by_lr[3e-5]}
    assert tf_param_ids and tf_param_ids.issubset(pre_group_ids)
