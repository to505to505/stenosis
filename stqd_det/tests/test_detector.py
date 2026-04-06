"""Integration tests for the full STQDDet detector pipeline."""

import torch
import pytest
from stqd_det.config import Config
from stqd_det.model.detector import STQDDet


@pytest.fixture
def small_cfg():
    """Minimal config for fast integration tests."""
    return Config(
        img_h=256, img_w=256,
        T=4,
        num_proposals=16,
        decoder_layers=2,
        decoder_dim=64,
        decoder_heads=4,
        decoder_ffn_dim=128,
        C=256,
        gfe_num_heads=4,
        stfs_num_heads=4,
        stfs_ffn_dim=128,
        num_classes=2,
        diffusion_steps=100,
        roi_output_size=7,
        score_thresh=0.3,
        nms_thresh=0.5,
    )


def _make_targets(T, device, n_boxes_per_frame=2):
    """Create synthetic targets for T frames."""
    targets = []
    for t in range(T):
        boxes = torch.tensor(
            [[50 + t, 60 + t, 100 + t, 110 + t]] * n_boxes_per_frame,
            dtype=torch.float32, device=device,
        )
        labels = torch.zeros(n_boxes_per_frame, dtype=torch.long, device=device)
        targets.append({"boxes": boxes, "labels": labels})
    return targets


class TestDetectorTraining:
    def test_training_returns_loss_dict(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device).train()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        targets = [_make_targets(T, device)]
        losses = model(images, targets)
        assert isinstance(losses, dict)
        assert "total_loss" in losses
        assert "loss_cls" in losses
        assert "loss_l1" in losses
        assert "loss_giou" in losses
        assert "loss_consistency" in losses

    def test_training_loss_finite(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device).train()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        targets = [_make_targets(T, device)]
        losses = model(images, targets)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_training_backward(self, small_cfg, device):
        """Full backward pass should not crash and produce gradients."""
        model = STQDDet(small_cfg).to(device).train()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        targets = [_make_targets(T, device)]
        losses = model(images, targets)
        losses["total_loss"].backward()
        # Check at least some parameters got gradients
        n_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        assert n_grads > 0, "No parameters received gradients"

    def test_training_empty_gt(self, small_cfg, device):
        """Training with no GT boxes should not crash."""
        model = STQDDet(small_cfg).to(device).train()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        targets = [[{
            "boxes": torch.zeros(0, 4, device=device),
            "labels": torch.zeros(0, dtype=torch.long, device=device),
        } for _ in range(T)]]
        losses = model(images, targets)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_training_single_gt(self, small_cfg, device):
        """Single GT box per frame."""
        model = STQDDet(small_cfg).to(device).train()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        targets = [_make_targets(T, device, n_boxes_per_frame=1)]
        losses = model(images, targets)
        assert torch.isfinite(losses["total_loss"])

    def test_training_many_gt(self, small_cfg, device):
        """Many GT boxes per frame."""
        model = STQDDet(small_cfg).to(device).train()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        targets = [_make_targets(T, device, n_boxes_per_frame=10)]
        losses = model(images, targets)
        assert torch.isfinite(losses["total_loss"])


class TestDetectorInference:
    def test_inference_returns_list(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device).eval()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        with torch.no_grad():
            results = model(images)
        assert isinstance(results, list)
        assert len(results) == 1  # batch=1
        assert len(results[0]) == T

    def test_inference_output_structure(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device).eval()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        with torch.no_grad():
            results = model(images)
        for t in range(T):
            frame_result = results[0][t]
            assert "boxes" in frame_result
            assert "scores" in frame_result
            assert "labels" in frame_result
            # boxes should be (n, 4), scores (n,), labels (n,)
            n = frame_result["boxes"].shape[0]
            assert frame_result["scores"].shape == (n,)
            assert frame_result["labels"].shape == (n,)
            if n > 0:
                assert frame_result["boxes"].shape[1] == 4

    def test_inference_boxes_within_bounds(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device).eval()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        with torch.no_grad():
            results = model(images)
        for t in range(T):
            boxes = results[0][t]["boxes"]
            if len(boxes) > 0:
                assert boxes[:, 0::2].min() >= 0
                assert boxes[:, 1::2].min() >= 0
                assert boxes[:, 0::2].max() <= small_cfg.img_w
                assert boxes[:, 1::2].max() <= small_cfg.img_h

    def test_inference_scores_valid(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device).eval()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        with torch.no_grad():
            results = model(images)
        for t in range(T):
            scores = results[0][t]["scores"]
            if len(scores) > 0:
                assert (scores >= 0).all()
                assert (scores <= 1).all()
                # All scores above threshold
                assert (scores >= small_cfg.score_thresh).all()

    def test_inference_no_grad_context(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device).eval()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        with torch.no_grad():
            results = model(images)
        for t in range(T):
            boxes = results[0][t]["boxes"]
            assert not boxes.requires_grad

    def test_inference_deterministic(self, small_cfg, device):
        """Same input + same seed → same output in eval mode (Poisson noise is random)."""
        model = STQDDet(small_cfg).to(device).eval()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        torch.manual_seed(42)
        with torch.no_grad():
            r1 = model(images)
        torch.manual_seed(42)
        with torch.no_grad():
            r2 = model(images)
        for t in range(T):
            assert torch.allclose(r1[0][t]["boxes"], r2[0][t]["boxes"])
            assert torch.allclose(r1[0][t]["scores"], r2[0][t]["scores"])


class TestDetectorMisc:
    def test_init_weights(self, small_cfg, device):
        """init_weights should not crash."""
        model = STQDDet(small_cfg).to(device)
        model.init_weights()  # should not raise

    def test_parameter_count_positive(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_all_submodules_present(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device)
        assert hasattr(model, "backbone")
        assert hasattr(model, "gfe")
        assert hasattr(model, "sqnb")
        assert hasattr(model, "decoder")
        assert hasattr(model, "stfs")
        assert hasattr(model, "criterion")

    def test_train_eval_mode_toggle(self, small_cfg, device):
        model = STQDDet(small_cfg).to(device)
        model.train()
        assert model.training
        model.eval()
        assert not model.training

    def test_mixed_gt_per_frame(self, small_cfg, device):
        """Frames with different numbers of GT boxes."""
        model = STQDDet(small_cfg).to(device).train()
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        targets = [[]]
        for t in range(T):
            n_boxes = t + 1  # 1, 2, 3, 4 boxes
            boxes = torch.rand(n_boxes, 4, device=device) * 100
            boxes[:, 2:] = boxes[:, :2] + 20
            labels = torch.zeros(n_boxes, dtype=torch.long, device=device)
            targets[0].append({"boxes": boxes, "labels": labels})
        losses = model(images, targets)
        assert torch.isfinite(losses["total_loss"])

    def test_optimizer_step(self, small_cfg, device):
        """Full optimizer step should work."""
        model = STQDDet(small_cfg).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        T = small_cfg.T
        images = torch.randn(1, T, 1, small_cfg.img_h, small_cfg.img_w, device=device)
        targets = [_make_targets(T, device)]

        # Store initial params
        initial_params = {n: p.clone() for n, p in model.named_parameters() if p.requires_grad}

        losses = model(images, targets)
        losses["total_loss"].backward()
        optimizer.step()

        # At least one parameter should have changed
        changed = False
        for n, p in model.named_parameters():
            if n in initial_params and not torch.equal(p, initial_params[n]):
                changed = True
                break
        assert changed, "No parameters changed after optimizer step"
