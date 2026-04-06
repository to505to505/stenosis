"""Extensive tests for loss functions: HungarianMatcher and STQDDetCriterion."""

import torch
import pytest
from stqd_det.config import Config
from stqd_det.model.losses import HungarianMatcher, STQDDetCriterion


# ─── HungarianMatcher ────────────────────────────────────────────────────────

class TestHungarianMatcher:
    @pytest.fixture
    def matcher(self):
        return HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)

    def test_perfect_match(self, matcher, device):
        """When pred == GT, should give correct matching."""
        cls_logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0]], device=device)  # P=2
        gt_boxes = torch.tensor([[10, 10, 50, 50], [100, 100, 200, 200]], dtype=torch.float32, device=device)
        gt_labels = torch.tensor([0, 1], dtype=torch.long, device=device)
        pred_boxes = gt_boxes.clone()

        pred_idx, gt_idx = matcher(cls_logits, pred_boxes, gt_boxes, gt_labels)
        assert len(pred_idx) == 2
        assert len(gt_idx) == 2
        # Each pred should match to its corresponding GT
        for pi, gi in zip(pred_idx, gt_idx):
            assert torch.allclose(pred_boxes[pi], gt_boxes[gi])

    def test_empty_gt(self, matcher, device):
        """No GT → empty matching."""
        cls_logits = torch.randn(10, 2, device=device)
        pred_boxes = torch.rand(10, 4, device=device) * 200
        gt_boxes = torch.zeros(0, 4, device=device)
        gt_labels = torch.zeros(0, dtype=torch.long, device=device)

        pred_idx, gt_idx = matcher(cls_logits, pred_boxes, gt_boxes, gt_labels)
        assert len(pred_idx) == 0
        assert len(gt_idx) == 0

    def test_more_preds_than_gt(self, matcher, device):
        """P > M → only M matches."""
        P, M = 10, 2
        cls_logits = torch.randn(P, 2, device=device)
        pred_boxes = torch.rand(P, 4, device=device) * 200
        pred_boxes[:, 2:] = pred_boxes[:, :2] + 20
        gt_boxes = torch.tensor([[50, 50, 100, 100], [150, 150, 200, 200]], dtype=torch.float32, device=device)
        gt_labels = torch.tensor([0, 1], dtype=torch.long, device=device)

        pred_idx, gt_idx = matcher(cls_logits, pred_boxes, gt_boxes, gt_labels)
        assert len(pred_idx) == M
        assert len(gt_idx) == M
        # All GT indices present
        assert set(gt_idx.tolist()) == {0, 1}

    def test_single_gt(self, matcher, device):
        cls_logits = torch.randn(5, 2, device=device)
        pred_boxes = torch.rand(5, 4, device=device) * 200
        pred_boxes[:, 2:] = pred_boxes[:, :2] + 20
        gt_boxes = torch.tensor([[80, 80, 120, 120]], dtype=torch.float32, device=device)
        gt_labels = torch.tensor([0], dtype=torch.long, device=device)

        pred_idx, gt_idx = matcher(cls_logits, pred_boxes, gt_boxes, gt_labels)
        assert len(pred_idx) == 1
        assert gt_idx[0] == 0

    def test_no_grad(self, matcher, device):
        """Matcher should not create computation graph."""
        cls_logits = torch.randn(5, 2, device=device, requires_grad=True)
        pred_boxes = torch.rand(5, 4, device=device, requires_grad=True)
        pred_boxes_comp = pred_boxes.detach()
        pred_boxes_comp[:, 2:] = pred_boxes_comp[:, :2] + 20
        gt_boxes = torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)
        gt_labels = torch.tensor([0], dtype=torch.long, device=device)

        pred_idx, gt_idx = matcher(cls_logits, pred_boxes_comp, gt_boxes, gt_labels)
        assert not pred_idx.requires_grad
        assert not gt_idx.requires_grad

    def test_prefers_close_boxes(self, matcher, device):
        """Matcher should prefer close boxes over far ones."""
        cls_logits = torch.zeros(2, 2, device=device)  # no class preference
        pred_boxes = torch.tensor(
            [[50, 50, 100, 100], [300, 300, 350, 350]], dtype=torch.float32, device=device
        )
        gt_boxes = torch.tensor([[55, 55, 105, 105]], dtype=torch.float32, device=device)
        gt_labels = torch.tensor([0], dtype=torch.long, device=device)

        pred_idx, gt_idx = matcher(cls_logits, pred_boxes, gt_boxes, gt_labels)
        assert pred_idx[0] == 0  # first pred is closer


# ─── STQDDetCriterion ────────────────────────────────────────────────────────

class TestSTQDDetCriterion:
    def test_all_loss_keys_present(self, cfg, device):
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals
        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]
        gt_boxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)] * T
        gt_labels = [torch.tensor([0], dtype=torch.long, device=device)] * T

        losses = criterion(layer_outputs, gt_boxes, gt_labels)
        assert "loss_cls" in losses
        assert "loss_l1" in losses
        assert "loss_giou" in losses
        assert "loss_consistency" in losses
        assert "total_loss" in losses

    def test_all_losses_finite(self, cfg, device):
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals
        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]
        gt_boxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)] * T
        gt_labels = [torch.tensor([0], dtype=torch.long, device=device)] * T

        losses = criterion(layer_outputs, gt_boxes, gt_labels, voted_count=1.0)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_empty_gt_all_frames(self, cfg, device):
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals
        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]
        gt_boxes = [torch.zeros(0, 4, device=device)] * T
        gt_labels = [torch.zeros(0, dtype=torch.long, device=device)] * T

        losses = criterion(layer_outputs, gt_boxes, gt_labels)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_consistency_loss_present(self, cfg, device):
        """Consistency loss should be > 0 when voted_count is given."""
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals
        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]
        gt_boxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)] * T
        gt_labels = [torch.tensor([0], dtype=torch.long, device=device)] * T

        losses = criterion(layer_outputs, gt_boxes, gt_labels, voted_count=1.0)
        assert losses["loss_consistency"] > 0

    def test_consistency_loss_absent(self, cfg, device):
        """Consistency loss should be 0 when voted_count is None."""
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals
        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]
        gt_boxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)] * T
        gt_labels = [torch.tensor([0], dtype=torch.long, device=device)] * T

        losses = criterion(layer_outputs, gt_boxes, gt_labels, voted_count=None)
        assert losses["loss_consistency"].item() == 0.0

    def test_multi_layer_averaging(self, cfg, device):
        """Loss should be averaged across decoder layers."""
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals

        single_layer = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]
        # Duplicate the same layer twice
        double_layer = [single_layer[0], single_layer[0]]

        gt_boxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)] * T
        gt_labels = [torch.tensor([0], dtype=torch.long, device=device)] * T

        loss1 = criterion(single_layer, gt_boxes, gt_labels)
        loss2 = criterion(double_layer, gt_boxes, gt_labels)
        # With 2 identical layers, averaged cls loss should equal single layer's
        assert torch.isclose(loss1["loss_cls"], loss2["loss_cls"], atol=1e-4)

    def test_total_loss_positive(self, cfg, device):
        """Total loss should generally be > 0 (unless perfect pred)."""
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals
        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]
        gt_boxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)] * T
        gt_labels = [torch.tensor([0], dtype=torch.long, device=device)] * T

        losses = criterion(layer_outputs, gt_boxes, gt_labels)
        assert losses["total_loss"] > 0

    def test_gradient_flow_through_loss(self, cfg, device):
        """Gradients should flow from total_loss to cls_logits."""
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals
        cls_logits = torch.randn(T, P, cfg.num_classes, device=device, requires_grad=True)
        # box_pred must be leaf tensor for .grad to be populated
        box_pred = torch.rand(T, P, 4, device=device, requires_grad=True)

        layer_outputs = [{"cls_logits": cls_logits, "box_pred": box_pred}]
        gt_boxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)] * T
        gt_labels = [torch.tensor([0], dtype=torch.long, device=device)] * T

        losses = criterion(layer_outputs, gt_boxes, gt_labels)
        losses["total_loss"].backward()
        assert cls_logits.grad is not None
        assert cls_logits.grad.abs().sum() > 0

    def test_many_gt_boxes(self, cfg, device):
        """Multiple GT boxes per frame should work."""
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals
        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]
        gt_boxes = [
            torch.tensor([
                [10, 10, 50, 50], [60, 60, 100, 100], [120, 120, 180, 180]
            ], dtype=torch.float32, device=device)
            for _ in range(T)
        ]
        gt_labels = [torch.tensor([0, 1, 0], dtype=torch.long, device=device)] * T

        losses = criterion(layer_outputs, gt_boxes, gt_labels)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_cls_loss_decreases_for_correct_pred(self, cfg, device):
        """Classification loss should be lower when predictions match GT class."""
        criterion = STQDDetCriterion(cfg).to(device)
        T, P = cfg.T, cfg.num_proposals

        gt_boxes = [torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device)] * T
        gt_labels = [torch.tensor([0], dtype=torch.long, device=device)] * T

        # Wrong prediction: high confidence for class 1 everywhere
        wrong_logits = torch.full((T, P, cfg.num_classes), -5.0, device=device)
        wrong_logits[..., 1] = 5.0
        output_wrong = [{"cls_logits": wrong_logits, "box_pred": torch.rand(T, P, 4, device=device) * 200}]

        # Right prediction: high confidence for class 0 everywhere
        right_logits = torch.full((T, P, cfg.num_classes), -5.0, device=device)
        right_logits[..., 0] = 5.0
        output_right = [{"cls_logits": right_logits, "box_pred": output_wrong[0]["box_pred"]}]

        loss_wrong = criterion(output_wrong, gt_boxes, gt_labels)
        loss_right = criterion(output_right, gt_boxes, gt_labels)
        assert loss_right["loss_cls"] < loss_wrong["loss_cls"]
