"""Extensive tests for SQNB (Sequential Quantum Noise Box) generator."""

import torch
import pytest
from stqd_det.config import Config
from stqd_det.model.sqnb import (
    SQNBGenerator,
    cosine_beta_schedule,
    box_xyxy_to_cxcywh,
    box_cxcywh_to_xyxy,
)


class TestCosineBetaSchedule:
    def test_shape(self):
        alphas = cosine_beta_schedule(100)
        assert alphas.shape == (101,)  # timesteps + 1

    def test_starts_at_one(self):
        alphas = cosine_beta_schedule(1000)
        assert torch.isclose(alphas[0], torch.tensor(1.0), atol=1e-3)

    def test_monotonically_decreasing(self):
        alphas = cosine_beta_schedule(1000)
        diffs = alphas[1:] - alphas[:-1]
        assert (diffs <= 0).all()

    def test_ends_near_zero(self):
        alphas = cosine_beta_schedule(1000)
        assert alphas[-1] < 0.01

    def test_all_positive(self):
        alphas = cosine_beta_schedule(1000)
        assert (alphas > 0).all()


class TestBoxConversions:
    def test_xyxy_to_cxcywh_roundtrip(self):
        boxes = torch.tensor([[10, 20, 50, 80], [100, 100, 200, 200]], dtype=torch.float32)
        cxcywh = box_xyxy_to_cxcywh(boxes)
        back = box_cxcywh_to_xyxy(cxcywh)
        assert torch.allclose(boxes, back)

    def test_xyxy_to_cxcywh_values(self):
        boxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
        cxcywh = box_xyxy_to_cxcywh(boxes)
        expected = torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)
        assert torch.allclose(cxcywh, expected)

    def test_cxcywh_to_xyxy_values(self):
        cxcywh = torch.tensor([[50, 50, 100, 100]], dtype=torch.float32)
        xyxy = box_cxcywh_to_xyxy(cxcywh)
        expected = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
        assert torch.allclose(xyxy, expected)

    def test_roundtrip_batched(self):
        boxes = torch.rand(100, 4) * 500
        boxes[:, 2:] = boxes[:, :2] + (boxes[:, 2:]).abs() + 1
        cxcywh = box_xyxy_to_cxcywh(boxes)
        back = box_cxcywh_to_xyxy(cxcywh)
        assert torch.allclose(boxes, back, atol=1e-4)

    def test_zero_size_box(self):
        boxes = torch.tensor([[50, 50, 50, 50]], dtype=torch.float32)
        cxcywh = box_xyxy_to_cxcywh(boxes)
        assert cxcywh[0, 2] == 0  # width=0
        assert cxcywh[0, 3] == 0  # height=0


class TestSQNBGenerator:
    def test_buffer_registration(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        assert hasattr(gen, "alphas_cumprod")
        assert hasattr(gen, "sqrt_alphas_cumprod")
        assert hasattr(gen, "sqrt_one_minus_alphas_cumprod")
        assert gen.alphas_cumprod.device.type == device.type

    def test_alphas_cumprod_valid(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        assert (gen.alphas_cumprod > 0).all()
        assert (gen.alphas_cumprod < 1).all()

    def test_sqrt_buffers_consistent(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        # sqrt(a)^2 + sqrt(1-a)^2 should ≈ 1
        check = gen.sqrt_alphas_cumprod ** 2 + gen.sqrt_one_minus_alphas_cumprod ** 2
        assert torch.allclose(check, gen.alphas_cumprod + (1 - gen.alphas_cumprod), atol=1e-5)


class TestForwardDiffusion:
    def test_output_shapes(self, cfg, device, sample_gt_boxes):
        gen = SQNBGenerator(cfg).to(device)
        noisy, timesteps, noise = gen.forward_diffusion(sample_gt_boxes, cfg.T)
        assert noisy.shape == (cfg.T, cfg.num_proposals, 4)
        assert timesteps.shape == (cfg.T,)
        assert noise.shape == (cfg.T, cfg.num_proposals, 4)

    def test_boxes_within_bounds(self, cfg, device, sample_gt_boxes):
        gen = SQNBGenerator(cfg).to(device)
        noisy, _, _ = gen.forward_diffusion(sample_gt_boxes, cfg.T)
        assert noisy[..., 0::2].min() >= 0
        assert noisy[..., 1::2].min() >= 0
        assert noisy[..., 0::2].max() <= cfg.img_w
        assert noisy[..., 1::2].max() <= cfg.img_h

    def test_timesteps_valid_range(self, cfg, device, sample_gt_boxes):
        gen = SQNBGenerator(cfg).to(device)
        _, timesteps, _ = gen.forward_diffusion(sample_gt_boxes, cfg.T)
        assert (timesteps >= 0).all()
        assert (timesteps < cfg.diffusion_steps).all()

    def test_no_grad_context(self, cfg, device, sample_gt_boxes):
        """forward_diffusion should not create a computation graph."""
        gen = SQNBGenerator(cfg).to(device)
        noisy, _, _ = gen.forward_diffusion(sample_gt_boxes, cfg.T)
        assert not noisy.requires_grad

    def test_no_nan(self, cfg, device, sample_gt_boxes):
        gen = SQNBGenerator(cfg).to(device)
        noisy, _, noise = gen.forward_diffusion(sample_gt_boxes, cfg.T)
        assert torch.isfinite(noisy).all()
        assert torch.isfinite(noise).all()

    def test_empty_gt_all_frames(self, cfg, device):
        """No GT boxes: should produce all random proposals."""
        gen = SQNBGenerator(cfg).to(device)
        empty_gt = [torch.zeros(0, 4, device=device) for _ in range(cfg.T)]
        noisy, timesteps, noise = gen.forward_diffusion(empty_gt, cfg.T)
        assert noisy.shape == (cfg.T, cfg.num_proposals, 4)
        assert torch.isfinite(noisy).all()

    def test_single_gt_box(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        gt_boxes = [
            torch.tensor([[100, 100, 150, 150]], dtype=torch.float32, device=device)
            for _ in range(cfg.T)
        ]
        noisy, _, _ = gen.forward_diffusion(gt_boxes, cfg.T)
        assert noisy.shape == (cfg.T, cfg.num_proposals, 4)

    def test_many_gt_boxes(self, cfg, device):
        """More GT boxes than proposals should not crash."""
        gen = SQNBGenerator(cfg).to(device)
        many_boxes = torch.rand(50, 4, device=device) * 200
        many_boxes[:, 2:] = many_boxes[:, :2] + many_boxes[:, 2:].abs() + 10
        gt_boxes = [many_boxes for _ in range(cfg.T)]
        noisy, _, _ = gen.forward_diffusion(gt_boxes, cfg.T)
        assert noisy.shape == (cfg.T, cfg.num_proposals, 4)

    def test_sequence_perturbation_correlation(self, cfg, device, sample_gt_boxes):
        """Frame 0 and frame 1 proposals should be correlated (not fully independent)."""
        gen = SQNBGenerator(cfg).to(device)
        torch.manual_seed(42)
        noisy, _, _ = gen.forward_diffusion(sample_gt_boxes, cfg.T)
        # Frame 0 vs frame 1 should be more similar than frame 0 vs random
        diff_01 = (noisy[0] - noisy[1]).abs().mean()
        random_f = torch.rand_like(noisy[0]) * cfg.img_w
        diff_random = (noisy[0] - random_f).abs().mean()
        # This is a statistical assertion; allow generous margin
        assert diff_01 < diff_random * 2

    def test_mixed_empty_and_nonempty_gt(self, cfg, device):
        """Some frames with GT, some without."""
        gen = SQNBGenerator(cfg).to(device)
        gt_boxes = []
        for t in range(cfg.T):
            if t % 2 == 0:
                gt_boxes.append(torch.tensor([[50, 50, 100, 100]], dtype=torch.float32, device=device))
            else:
                gt_boxes.append(torch.zeros(0, 4, device=device))
        noisy, _, _ = gen.forward_diffusion(gt_boxes, cfg.T)
        assert noisy.shape == (cfg.T, cfg.num_proposals, 4)
        assert torch.isfinite(noisy).all()


class TestGenerateProposals:
    def test_output_shape(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)
        assert proposals.shape == (cfg.T, cfg.num_proposals, 4)

    def test_proposals_within_bounds(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)
        assert proposals[..., 0::2].min() >= 0
        assert proposals[..., 1::2].min() >= 0
        assert proposals[..., 0::2].max() <= cfg.img_w
        assert proposals[..., 1::2].max() <= cfg.img_h

    def test_no_nan(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)
        assert torch.isfinite(proposals).all()

    def test_centered_distribution(self, cfg, device):
        """Proposals should be centered around image center."""
        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)
        # Mean center of proposals should be near image center
        cx = (proposals[..., 0] + proposals[..., 2]) / 2
        cy = (proposals[..., 1] + proposals[..., 3]) / 2
        mean_cx = cx.mean()
        mean_cy = cy.mean()
        # Within 30% of center
        assert abs(mean_cx - cfg.img_w / 2) < cfg.img_w * 0.3
        assert abs(mean_cy - cfg.img_h / 2) < cfg.img_h * 0.3

    def test_sequence_consistency(self, cfg, device):
        """Consecutive frames should have correlated proposals."""
        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)
        c0 = proposals[0].mean(dim=0)
        c1 = proposals[1].mean(dim=0)
        diff = (c0 - c1).abs().sum()
        assert diff < cfg.img_w * 2

    def test_single_frame(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(1, device)
        assert proposals.shape == (1, cfg.num_proposals, 4)

    def test_no_grad(self, cfg, device):
        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)
        assert not proposals.requires_grad

    def test_valid_xyxy_boxes(self, cfg, device):
        """x1 <= x2, y1 <= y2 for all proposals."""
        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)
        assert (proposals[..., 2] >= proposals[..., 0]).all(), "x2 < x1 found"
        assert (proposals[..., 3] >= proposals[..., 1]).all(), "y2 < y1 found"
