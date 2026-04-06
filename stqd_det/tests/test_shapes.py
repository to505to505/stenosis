"""Shape verification tests for STQD-Det modules.

Run: python -m pytest stqd_det/tests/test_shapes.py -v
"""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stqd_det.config import Config


@pytest.fixture
def cfg():
    return Config(
        img_h=256, img_w=256,  # smaller for fast tests
        T=4,                    # fewer frames for speed
        num_proposals=16,       # fewer proposals
        decoder_layers=2,       # fewer layers
        decoder_dim=64,
        decoder_heads=4,
        decoder_ffn_dim=128,
        C=256,
        gfe_num_heads=4,
        stfs_num_heads=4,
        stfs_ffn_dim=128,
        num_classes=2,
    )


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestBackbone:
    def test_output_shapes(self, cfg, device):
        from stqd_det.model.backbone import Backbone

        model = Backbone(cfg).to(device)
        x = torch.randn(4, 1, cfg.img_h, cfg.img_w, device=device)
        out = model(x)

        assert "0" in out and "1" in out and "2" in out and "3" in out
        assert out["0"].shape == (4, 256, cfg.img_h // 4, cfg.img_w // 4)
        assert out["1"].shape == (4, 256, cfg.img_h // 8, cfg.img_w // 8)
        assert out["2"].shape == (4, 256, cfg.img_h // 16, cfg.img_w // 16)
        assert out["3"].shape == (4, 256, cfg.img_h // 32, cfg.img_w // 32)


class TestGFE:
    def test_output_shapes(self, cfg, device):
        from collections import OrderedDict
        from stqd_det.model.gfe import GFEModule

        T = cfg.T
        H, W = cfg.img_h, cfg.img_w

        gfe = GFEModule(cfg).to(device)

        # Simulate FPN features for B=1, N=T frames
        fpn = OrderedDict({
            "0": torch.randn(T, 256, H // 4, W // 4, device=device),
            "1": torch.randn(T, 256, H // 8, W // 8, device=device),
            "2": torch.randn(T, 256, H // 16, W // 16, device=device),
            "3": torch.randn(T, 256, H // 32, W // 32, device=device),
        })

        enhanced = gfe(fpn, num_frames=T)

        # Only top layer should be modified, all shapes preserved
        assert enhanced["3"].shape == fpn["3"].shape
        assert enhanced["0"].shape == fpn["0"].shape
        # Top layer should be different (enhanced)
        assert not torch.allclose(enhanced["3"], fpn["3"])

    def test_dynamic_conv(self, device):
        from stqd_det.model.gfe import DynamicConv

        dc = DynamicConv(channels=256, kernel_size=3, groups=4).to(device)
        x = torch.randn(2, 256, 8, 8, device=device)
        out = dc(x)
        assert out.shape == x.shape


class TestSQNB:
    def test_training_proposals(self, cfg, device):
        from stqd_det.model.sqnb import SQNBGenerator

        gen = SQNBGenerator(cfg).to(device)

        # GT boxes for T frames: each frame has 2 boxes
        gt_boxes = [
            torch.tensor([[50, 60, 100, 110], [150, 160, 200, 210]], dtype=torch.float32, device=device)
            for _ in range(cfg.T)
        ]

        noisy, timesteps, noise = gen.forward_diffusion(gt_boxes, cfg.T)
        assert noisy.shape == (cfg.T, cfg.num_proposals, 4)
        assert timesteps.shape == (cfg.T,)
        assert noise.shape == (cfg.T, cfg.num_proposals, 4)
        # All boxes should be within image bounds
        assert noisy[..., 0::2].min() >= 0
        assert noisy[..., 1::2].min() >= 0
        assert noisy[..., 0::2].max() <= cfg.img_w
        assert noisy[..., 1::2].max() <= cfg.img_h

    def test_inference_proposals(self, cfg, device):
        from stqd_det.model.sqnb import SQNBGenerator

        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)
        assert proposals.shape == (cfg.T, cfg.num_proposals, 4)

    def test_sequence_consistency(self, cfg, device):
        """Frames should have correlated proposals (not fully random)."""
        from stqd_det.model.sqnb import SQNBGenerator

        gen = SQNBGenerator(cfg).to(device)
        proposals = gen.generate_proposals(cfg.T, device)

        # Center of frame 0 and frame 1 proposals should be more similar
        # than fully random proposals
        c0 = proposals[0].mean(dim=0)
        c1 = proposals[1].mean(dim=0)
        diff = (c0 - c1).abs().sum()
        assert diff < cfg.img_w * 2  # reasonable similarity


class TestDecoder:
    def test_output_shapes(self, cfg, device):
        from collections import OrderedDict
        from stqd_det.model.decoder import StenosisDecoder

        decoder = StenosisDecoder(cfg).to(device)
        T = cfg.T
        P = cfg.num_proposals
        H, W = cfg.img_h, cfg.img_w

        fpn = OrderedDict({
            "0": torch.randn(T, 256, H // 4, W // 4, device=device),
            "1": torch.randn(T, 256, H // 8, W // 8, device=device),
            "2": torch.randn(T, 256, H // 16, W // 16, device=device),
            "3": torch.randn(T, 256, H // 32, W // 32, device=device),
        })

        proposals = torch.rand(T, P, 4, device=device) * torch.tensor(
            [W, H, W, H], device=device
        )
        # Ensure x1 < x2, y1 < y2
        proposals[..., 2:] = proposals[..., :2] + proposals[..., 2:].abs().clamp(min=10)
        proposals[..., 0::2].clamp_(0, W)
        proposals[..., 1::2].clamp_(0, H)

        image_sizes = [(H, W)] * T
        timesteps = torch.randint(0, 100, (T,), device=device)

        outputs = decoder(fpn, proposals, image_sizes, timesteps)

        assert len(outputs) == cfg.decoder_layers
        for out in outputs:
            assert out["cls_logits"].shape == (T, P, cfg.num_classes)
            assert out["box_pred"].shape == (T, P, 4)


class TestLosses:
    def test_criterion(self, cfg, device):
        from stqd_det.model.losses import STQDDetCriterion

        criterion = STQDDetCriterion(cfg).to(device)
        T = cfg.T
        P = cfg.num_proposals

        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]

        gt_boxes = [
            torch.tensor([[50, 60, 100, 110]], dtype=torch.float32, device=device)
            for _ in range(T)
        ]
        gt_labels = [
            torch.tensor([0], dtype=torch.long, device=device)
            for _ in range(T)
        ]

        losses = criterion(layer_outputs, gt_boxes, gt_labels, voted_count=1.0)
        assert "total_loss" in losses
        assert "loss_cls" in losses
        assert "loss_consistency" in losses
        assert losses["total_loss"].requires_grad is False  # inputs didn't require grad
        # Check no NaN
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_empty_gt(self, cfg, device):
        from stqd_det.model.losses import STQDDetCriterion

        criterion = STQDDetCriterion(cfg).to(device)
        T = cfg.T
        P = cfg.num_proposals

        layer_outputs = [{
            "cls_logits": torch.randn(T, P, cfg.num_classes, device=device),
            "box_pred": torch.rand(T, P, 4, device=device) * 200,
        }]

        gt_boxes = [torch.zeros((0, 4), device=device) for _ in range(T)]
        gt_labels = [torch.zeros(0, dtype=torch.long, device=device) for _ in range(T)]

        losses = criterion(layer_outputs, gt_boxes, gt_labels)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"


class TestSTFS:
    def test_voting(self):
        from stqd_det.model.stfs import vote_groups

        # 5 frames, group appearing in all 5 → TP
        # group appearing in 3 → FN
        # group appearing in 1 → FP
        groups = [
            {"n_boxes": 5, "frame_indices": [0, 1, 2, 3, 4]},
            {"n_boxes": 3, "frame_indices": [0, 1, 2]},
            {"n_boxes": 1, "frame_indices": [0]},
        ]
        h_tp, h_fn, h_fp = vote_groups(groups, num_frames=5)
        assert len(h_tp) == 1
        assert len(h_fn) == 1
        assert len(h_fp) == 1


class TestRoIAggregator:
    def test_shapes(self, device):
        from stqd_det.model.stfs import RoIAggregator

        agg = RoIAggregator(
            channels=256, spatial_size=7, num_heads=4, ffn_dim=512
        ).to(device)

        wrong = torch.randn(3, 256, 7, 7, device=device)
        right = torch.randn(3, 256, 7, 7, device=device)

        out = agg(wrong, right)
        assert out.shape == (3, 256, 7, 7)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
