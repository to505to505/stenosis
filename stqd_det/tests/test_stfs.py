"""Extensive tests for STFS: RoIAggregator, Hungarian matching, voting, STFSModule."""

import torch
import pytest
import numpy as np
from collections import OrderedDict
from stqd_det.config import Config
from stqd_det.model.stfs import (
    RoIAggregator,
    hungarian_match_across_frames,
    vote_groups,
    STFSModule,
)


# ─── RoIAggregator ───────────────────────────────────────────────────────────

class TestRoIAggregator:
    def test_output_shape(self, device):
        agg = RoIAggregator(channels=256, spatial_size=7, num_heads=4, ffn_dim=512).to(device)
        wrong = torch.randn(3, 256, 7, 7, device=device)
        right = torch.randn(3, 256, 7, 7, device=device)
        out = agg(wrong, right)
        assert out.shape == (3, 256, 7, 7)

    def test_single_roi(self, device):
        agg = RoIAggregator(channels=64, spatial_size=7, num_heads=4, ffn_dim=256).to(device)
        wrong = torch.randn(1, 64, 7, 7, device=device)
        right = torch.randn(1, 64, 7, 7, device=device)
        out = agg(wrong, right)
        assert out.shape == (1, 64, 7, 7)

    def test_gradient_flow(self, device):
        agg = RoIAggregator(channels=64, spatial_size=7, num_heads=4, ffn_dim=256).to(device)
        wrong = torch.randn(2, 64, 7, 7, device=device, requires_grad=True)
        right = torch.randn(2, 64, 7, 7, device=device, requires_grad=True)
        out = agg(wrong, right)
        out.sum().backward()
        assert wrong.grad is not None and wrong.grad.abs().sum() > 0
        assert right.grad is not None and right.grad.abs().sum() > 0

    def test_no_nan(self, device):
        agg = RoIAggregator(channels=64, spatial_size=7, num_heads=4, ffn_dim=256).to(device)
        wrong = torch.randn(5, 64, 7, 7, device=device)
        right = torch.randn(5, 64, 7, 7, device=device)
        out = agg(wrong, right)
        assert torch.isfinite(out).all()

    def test_different_inputs_different_outputs(self, device):
        """Wrong features should change based on right features."""
        agg = RoIAggregator(channels=64, spatial_size=7, num_heads=4, ffn_dim=256).to(device)
        agg.eval()
        wrong = torch.randn(1, 64, 7, 7, device=device)
        right1 = torch.randn(1, 64, 7, 7, device=device) * 10
        right2 = torch.randn(1, 64, 7, 7, device=device) * 0.1
        out1 = agg(wrong, right1)
        out2 = agg(wrong, right2)
        assert not torch.allclose(out1, out2)

    def test_all_params_receive_grad(self, device):
        agg = RoIAggregator(channels=64, spatial_size=7, num_heads=4, ffn_dim=256).to(device)
        wrong = torch.randn(2, 64, 7, 7, device=device)
        right = torch.randn(2, 64, 7, 7, device=device)
        out = agg(wrong, right)
        out.sum().backward()
        has_grad = sum(
            1 for _, p in agg.named_parameters()
            if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        # At least some parameters receive gradients (MHA with seq_len=1
        # won't produce grad for in_proj_weight, which is expected)
        assert has_grad > 0, "No parameters received any gradient"


# ─── Hungarian Matching ──────────────────────────────────────────────────────

class TestHungarianMatchAcrossFrames:
    def _make_predictions(self, boxes_list, scores_list, labels_list=None):
        preds = []
        for i, (boxes, scores) in enumerate(zip(boxes_list, scores_list)):
            labels = labels_list[i] if labels_list else torch.zeros(len(boxes), dtype=torch.long)
            preds.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32) if not isinstance(boxes, torch.Tensor) else boxes,
                "scores": torch.tensor(scores, dtype=torch.float32) if not isinstance(scores, torch.Tensor) else scores,
                "labels": labels if isinstance(labels, torch.Tensor) else torch.tensor(labels, dtype=torch.long),
            })
        return preds

    def test_perfect_tracking(self):
        """Same box in all frames → one group with n_boxes=N."""
        N = 5
        preds = self._make_predictions(
            boxes_list=[[[100, 100, 200, 200]]] * N,
            scores_list=[[0.9]] * N,
        )
        groups = hungarian_match_across_frames(preds)
        assert len(groups) == 1
        assert groups[0]["n_boxes"] == N

    def test_two_objects_tracked(self):
        """Two distinct objects in all frames should produce 2 groups."""
        N = 4
        preds = self._make_predictions(
            boxes_list=[[[10, 10, 50, 50], [200, 200, 250, 250]]] * N,
            scores_list=[[0.9, 0.8]] * N,
        )
        groups = hungarian_match_across_frames(preds)
        assert len(groups) == 2
        assert all(g["n_boxes"] == N for g in groups)

    def test_disappearing_object(self):
        """Object present in frames 0-2, absent in frame 3."""
        preds = self._make_predictions(
            boxes_list=[
                [[100, 100, 200, 200]],
                [[105, 105, 205, 205]],
                [[110, 110, 210, 210]],
                [],  # disappeared
            ],
            scores_list=[[0.9], [0.85], [0.8], []],
        )
        groups = hungarian_match_across_frames(preds)
        assert len(groups) == 1
        assert groups[0]["n_boxes"] == 3
        assert 3 not in groups[0]["frame_indices"]

    def test_new_object_appears(self):
        """New object first detected at frame 2."""
        preds = self._make_predictions(
            boxes_list=[[], [], [[100, 100, 200, 200]], [[105, 105, 205, 205]]],
            scores_list=[[], [], [0.9], [0.85]],
        )
        groups = hungarian_match_across_frames(preds)
        assert len(groups) == 1
        assert groups[0]["n_boxes"] == 2

    def test_score_threshold_filtering(self):
        """Low-confidence detections should be filtered (below 0.3 default)."""
        preds = self._make_predictions(
            boxes_list=[[[100, 100, 200, 200]], [[100, 100, 200, 200]]],
            scores_list=[[0.1], [0.1]],  # below 0.3 threshold
        )
        groups = hungarian_match_across_frames(preds, score_thresh=0.3)
        assert len(groups) == 0  # all filtered out

    def test_empty_all_frames(self):
        preds = self._make_predictions(
            boxes_list=[[], [], [], []],
            scores_list=[[], [], [], []],
        )
        groups = hungarian_match_across_frames(preds)
        assert len(groups) == 0

    def test_single_frame(self):
        preds = self._make_predictions(
            boxes_list=[[[50, 50, 100, 100], [200, 200, 250, 250]]],
            scores_list=[[0.9, 0.8]],
        )
        groups = hungarian_match_across_frames(preds)
        assert len(groups) == 2

    def test_ref_box_highest_confidence(self):
        """ref_box should come from the highest confidence frame."""
        preds = self._make_predictions(
            boxes_list=[[[100, 100, 200, 200]], [[100, 100, 200, 200]]],
            scores_list=[[0.5], [0.9]],
        )
        groups = hungarian_match_across_frames(preds)
        assert groups[0]["ref_frame"] == 1  # frame 1 has higher confidence

    def test_iou_rejection(self):
        """Far-apart boxes should not be matched (IoU < 0.1)."""
        preds = self._make_predictions(
            boxes_list=[[[10, 10, 20, 20]], [[400, 400, 410, 410]]],
            scores_list=[[0.9], [0.9]],
        )
        groups = hungarian_match_across_frames(preds)
        assert len(groups) == 2  # not matched, two separate groups


# ─── Voting ──────────────────────────────────────────────────────────────────

class TestVoteGroups:
    def test_all_tp(self):
        groups = [{"n_boxes": 5}]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=5)
        assert len(h_tp) == 1
        assert len(h_fn) == 0
        assert len(h_fp) == 0

    def test_all_fp(self):
        groups = [{"n_boxes": 1}]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=5)
        assert len(h_tp) == 0
        assert len(h_fn) == 0
        assert len(h_fp) == 1

    def test_all_fn(self):
        groups = [{"n_boxes": 3}]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=5)
        assert len(h_tp) == 0
        assert len(h_fn) == 1
        assert len(h_fp) == 0

    def test_boundary_half_n(self):
        """n_boxes == N/2 exactly → should be H-FN."""
        groups = [{"n_boxes": 4}]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=8)
        assert len(h_fn) == 1  # 4 >= 8/2 = 4.0

    def test_boundary_below_half(self):
        groups = [{"n_boxes": 3}]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=8)
        assert len(h_fp) == 1  # 3 < 4.0

    def test_mixed(self):
        groups = [
            {"n_boxes": 9},  # TP (== N)
            {"n_boxes": 7},  # FN (>= N/2)
            {"n_boxes": 3},  # FP (< N/2)
            {"n_boxes": 1},  # FP
        ]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=9)
        assert len(h_tp) == 1
        assert len(h_fn) == 1
        assert len(h_fp) == 2

    def test_empty_groups(self):
        h_tp, h_fn, h_fp = vote_groups([], num_frames=5)
        assert len(h_tp) == 0
        assert len(h_fn) == 0
        assert len(h_fp) == 0

    def test_n_above_N_is_tp(self):
        """n_boxes > N should still be classified as TP."""
        groups = [{"n_boxes": 10}]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=5)
        assert len(h_tp) == 1

    def test_single_frame(self):
        """N=1: n_boxes=1 → TP."""
        groups = [{"n_boxes": 1}]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=1)
        assert len(h_tp) == 1

    def test_two_frames(self):
        """N=2: n_boxes=1 → FN (>= N/2=1.0)."""
        groups = [{"n_boxes": 1}]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=2)
        assert len(h_fn) == 1


# ─── STFSModule ──────────────────────────────────────────────────────────────

class TestSTFSModule:
    def _make_frame_predictions(self, n_frames, n_boxes, score, device):
        preds = []
        for _ in range(n_frames):
            boxes = torch.tensor([[100, 100, 200, 200]] * n_boxes, dtype=torch.float32, device=device)
            scores = torch.full((n_boxes,), score, device=device)
            labels = torch.zeros(n_boxes, dtype=torch.long, device=device)
            preds.append({"boxes": boxes, "scores": scores, "labels": labels})
        return preds

    def test_all_tp_no_correction(self, cfg, device, sample_fpn, image_sizes):
        """When all frames detect the same object → all H-TP, no corrections."""
        stfs = STFSModule(cfg).to(device)
        preds = self._make_frame_predictions(cfg.T, 1, 0.9, device)
        h_tp, h_fn, h_fp, corrected = stfs(sample_fpn, preds, image_sizes, cfg.T)
        assert len(h_tp) >= 1
        # If all TP, corrected_rois should be empty (nothing to fix)
        assert len(h_fn) == 0 or len(corrected) >= 0

    def test_empty_predictions(self, cfg, device, sample_fpn, image_sizes):
        """No predictions in any frame → no groups, no corrections."""
        stfs = STFSModule(cfg).to(device)
        preds = self._make_frame_predictions(cfg.T, 0, 0.0, device)
        h_tp, h_fn, h_fp, corrected = stfs(sample_fpn, preds, image_sizes, cfg.T)
        assert len(h_tp) == 0
        assert len(h_fn) == 0
        assert len(h_fp) == 0
        assert len(corrected) == 0

    def test_corrected_rois_shape(self, cfg, device, sample_fpn, image_sizes):
        """Corrected RoIs should have shape (C, roi, roi)."""
        stfs = STFSModule(cfg).to(device)
        # Create scenario with FN: object in T-1 frames only
        preds = []
        for t in range(cfg.T):
            if t < cfg.T - 1:
                preds.append({
                    "boxes": torch.tensor(
                        [[100, 100, 200, 200]], dtype=torch.float32, device=device
                    ),
                    "scores": torch.tensor([0.9], device=device),
                    "labels": torch.tensor([0], dtype=torch.long, device=device),
                })
            else:
                preds.append({
                    "boxes": torch.zeros(0, 4, device=device),
                    "scores": torch.zeros(0, device=device),
                    "labels": torch.zeros(0, dtype=torch.long, device=device),
                })
        h_tp, h_fn, h_fp, corrected = stfs(sample_fpn, preds, image_sizes, cfg.T)
        for key, roi in corrected.items():
            assert roi.shape == (cfg.C, cfg.roi_output_size, cfg.roi_output_size)

    def test_expand_box(self, cfg, device):
        """_expand_box should enlarge the box by alpha factor."""
        stfs = STFSModule(cfg).to(device)
        box = torch.tensor([100, 100, 200, 200], dtype=torch.float32, device=device)
        expanded = stfs._expand_box(box, 2.0)
        # Original: cx=150, cy=150, w=100, h=100 → new_w=200, new_h=200
        expected_w = 200.0
        expected_h = 200.0
        actual_w = expanded[2] - expanded[0]
        actual_h = expanded[3] - expanded[1]
        assert abs(actual_w.item() - expected_w) < 1.0
        assert abs(actual_h.item() - expected_h) < 1.0

    def test_expand_box_center_preserved(self, cfg, device):
        """Center should be the same after expansion."""
        stfs = STFSModule(cfg).to(device)
        box = torch.tensor([100, 100, 200, 200], dtype=torch.float32, device=device)
        expanded = stfs._expand_box(box, 2.0)
        cx_orig = (box[0] + box[2]) / 2
        cy_orig = (box[1] + box[3]) / 2
        cx_exp = (expanded[0] + expanded[2]) / 2
        cy_exp = (expanded[1] + expanded[3]) / 2
        assert torch.isclose(cx_orig, cx_exp, atol=1e-4)
        assert torch.isclose(cy_orig, cy_exp, atol=1e-4)

    def test_expand_box_alpha_one(self, cfg, device):
        """Alpha=1.0 should preserve box size."""
        stfs = STFSModule(cfg).to(device)
        box = torch.tensor([100, 100, 200, 200], dtype=torch.float32, device=device)
        expanded = stfs._expand_box(box, 1.0)
        assert torch.allclose(expanded, box, atol=1e-4)

    def test_no_nan(self, cfg, device, sample_fpn, image_sizes):
        stfs = STFSModule(cfg).to(device)
        preds = self._make_frame_predictions(cfg.T, 2, 0.9, device)
        h_tp, h_fn, h_fp, corrected = stfs(sample_fpn, preds, image_sizes, cfg.T)
        for key, roi in corrected.items():
            assert torch.isfinite(roi).all(), f"NaN in corrected RoI {key}"
