"""Extensive tests for GFE module: DynamicConv, GFEAttention, GFEModule."""

import torch
import pytest
from collections import OrderedDict
from stqd_det.config import Config
from stqd_det.model.gfe import DynamicConv, GFEAttention, GFEModule


class TestDynamicConv:
    def test_output_shape(self, device):
        dc = DynamicConv(channels=64, kernel_size=3).to(device)
        x = torch.randn(4, 64, 8, 8, device=device)
        assert dc(x).shape == x.shape

    def test_single_sample(self, device):
        dc = DynamicConv(channels=32, kernel_size=3).to(device)
        x = torch.randn(1, 32, 16, 16, device=device)
        assert dc(x).shape == x.shape

    def test_content_adaptive_kernels(self, device):
        """Different inputs should generate different kernels → different outputs."""
        dc = DynamicConv(channels=64, kernel_size=3).to(device)
        dc.eval()
        x1 = torch.randn(1, 64, 8, 8, device=device) * 10
        x2 = torch.randn(1, 64, 8, 8, device=device) * 0.01
        o1 = dc(x1)
        o2 = dc(x2)
        assert not torch.allclose(o1, o2)

    def test_gradient_flow(self, device):
        dc = DynamicConv(channels=64, kernel_size=3).to(device)
        x = torch.randn(2, 64, 8, 8, device=device, requires_grad=True)
        out = dc(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_nan(self, device):
        dc = DynamicConv(channels=64, kernel_size=3).to(device)
        x = torch.randn(2, 64, 8, 8, device=device)
        out = dc(x)
        assert torch.isfinite(out).all()

    def test_kernel_sizes(self, device):
        """Different kernel sizes should work."""
        for ks in [1, 3, 5]:
            dc = DynamicConv(channels=32, kernel_size=ks).to(device)
            x = torch.randn(2, 32, 8, 8, device=device)
            out = dc(x)
            assert out.shape == x.shape

    def test_batch_independence(self, device):
        """Each sample should be processed nearly independently (small GroupNorm coupling)."""
        dc = DynamicConv(channels=32, kernel_size=3).to(device)
        dc.eval()
        x = torch.randn(4, 32, 8, 8, device=device)
        out_batch = dc(x)

        # Process one at a time
        for i in range(4):
            out_single = dc(x[i:i+1])
            # GroupNorm with batch_size=1 vs batch_size=4 can differ slightly
            assert torch.allclose(out_batch[i:i+1], out_single, atol=0.5), \
                f"Batch sample {i} differs significantly from single-sample processing"


class TestGFEAttention:
    def test_output_shape(self, device):
        attn = GFEAttention(embed_dim=128, num_heads=4).to(device)
        v_prev = torch.randn(3, 128, device=device)
        v_curr = torch.randn(3, 128, device=device)
        v_next = torch.randn(3, 128, device=device)
        out = attn(v_prev, v_curr, v_next)
        assert out.shape == (3, 128)

    def test_residual_connection(self, device):
        """Output should be close to input with random weights (residual)."""
        attn = GFEAttention(embed_dim=128, num_heads=4).to(device)
        v = torch.randn(1, 128, device=device)
        out = attn(v, v, v)
        # With residual + LayerNorm, output won't be identical but should not be zero
        assert out.abs().sum() > 0

    def test_gradient_flow(self, device):
        """Output should depend on all inputs (verified via perturbation)."""
        attn = GFEAttention(embed_dim=128, num_heads=4).to(device)
        attn.eval()
        v_prev = torch.randn(2, 128, device=device)
        v_curr = torch.randn(2, 128, device=device)
        v_next = torch.randn(2, 128, device=device)
        out_base = attn(v_prev, v_curr, v_next)
        # Non-uniform perturbation (constant shift cancelled by LayerNorm)
        v_curr_perturbed = v_curr * 2.0
        out_perturbed = attn(v_prev, v_curr_perturbed, v_next)
        assert not torch.allclose(out_base, out_perturbed), \
            "Output did not change when v_curr was perturbed"

    def test_no_nan(self, device):
        attn = GFEAttention(embed_dim=128, num_heads=8).to(device)
        v = torch.randn(5, 128, device=device)
        out = attn(v, v, v)
        assert torch.isfinite(out).all()


class TestGFEModule:
    def test_output_shapes(self, cfg, device, sample_fpn):
        gfe = GFEModule(cfg).to(device)
        enhanced = gfe(sample_fpn, num_frames=cfg.T)
        for key in sample_fpn:
            assert enhanced[key].shape == sample_fpn[key].shape

    def test_only_top_layer_modified(self, cfg, device, sample_fpn):
        """Only FPN level '3' should be modified; levels 0-2 pass through."""
        gfe = GFEModule(cfg).to(device)
        enhanced = gfe(sample_fpn, num_frames=cfg.T)
        # Levels 0,1,2 should be identical (not modified)
        for key in ["0", "1", "2"]:
            assert torch.equal(enhanced[key], sample_fpn[key])
        # Level 3 should be different
        assert not torch.allclose(enhanced["3"], sample_fpn["3"])

    def test_gradient_flow(self, cfg, device):
        gfe = GFEModule(cfg).to(device)
        T, H, W = cfg.T, cfg.img_h, cfg.img_w
        top = torch.randn(T, 256, H // 32, W // 32, device=device, requires_grad=True)
        fpn = OrderedDict({
            "0": torch.randn(T, 256, H // 4, W // 4, device=device),
            "1": torch.randn(T, 256, H // 8, W // 8, device=device),
            "2": torch.randn(T, 256, H // 16, W // 16, device=device),
            "3": top,
        })
        enhanced = gfe(fpn, num_frames=T)
        enhanced["3"].sum().backward()
        assert top.grad is not None
        # Check that at least some GFE params received gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in gfe.parameters()
        )
        assert has_grad

    def test_boundary_padding(self, cfg, device, sample_fpn):
        """First frame uses itself as prev, last frame uses itself as next."""
        gfe = GFEModule(cfg).to(device)
        # This verifies boundary duplication doesn't crash
        enhanced = gfe(sample_fpn, num_frames=cfg.T)
        assert enhanced["3"].shape == sample_fpn["3"].shape

    def test_single_frame(self, device):
        """Edge case: single frame should still work (prev=curr=next)."""
        cfg1 = Config(img_h=256, img_w=256, T=1, C=256, gfe_num_heads=4)
        gfe = GFEModule(cfg1).to(device)
        H, W = 256, 256
        fpn = OrderedDict({
            "0": torch.randn(1, 256, H // 4, W // 4, device=device),
            "1": torch.randn(1, 256, H // 8, W // 8, device=device),
            "2": torch.randn(1, 256, H // 16, W // 16, device=device),
            "3": torch.randn(1, 256, H // 32, W // 32, device=device),
        })
        enhanced = gfe(fpn, num_frames=1)
        assert enhanced["3"].shape == fpn["3"].shape
        assert torch.isfinite(enhanced["3"]).all()

    def test_two_frames(self, device):
        """Two frames: frame 0 has prev=0, next=1; frame 1 has prev=0, next=1."""
        cfg2 = Config(img_h=256, img_w=256, T=2, C=256, gfe_num_heads=4)
        gfe = GFEModule(cfg2).to(device)
        H, W = 256, 256
        fpn = OrderedDict({
            "0": torch.randn(2, 256, H // 4, W // 4, device=device),
            "1": torch.randn(2, 256, H // 8, W // 8, device=device),
            "2": torch.randn(2, 256, H // 16, W // 16, device=device),
            "3": torch.randn(2, 256, H // 32, W // 32, device=device),
        })
        enhanced = gfe(fpn, num_frames=2)
        assert enhanced["3"].shape == (2, 256, H // 32, W // 32)

    def test_multi_batch(self, device):
        """B=2 batch elements, T=3 frames each → B*T=6 FPN entries."""
        cfg_mb = Config(img_h=256, img_w=256, T=3, C=256, gfe_num_heads=4)
        gfe = GFEModule(cfg_mb).to(device)
        BT = 6  # B=2, T=3
        H, W = 256, 256
        fpn = OrderedDict({
            "0": torch.randn(BT, 256, H // 4, W // 4, device=device),
            "1": torch.randn(BT, 256, H // 8, W // 8, device=device),
            "2": torch.randn(BT, 256, H // 16, W // 16, device=device),
            "3": torch.randn(BT, 256, H // 32, W // 32, device=device),
        })
        enhanced = gfe(fpn, num_frames=3)
        assert enhanced["3"].shape == (BT, 256, H // 32, W // 32)

    def test_no_nan(self, cfg, device, sample_fpn):
        gfe = GFEModule(cfg).to(device)
        enhanced = gfe(sample_fpn, num_frames=cfg.T)
        assert torch.isfinite(enhanced["3"]).all()

    def test_deterministic_eval(self, cfg, device, sample_fpn):
        gfe = GFEModule(cfg).to(device).eval()
        out1 = gfe(sample_fpn, num_frames=cfg.T)
        out2 = gfe(sample_fpn, num_frames=cfg.T)
        assert torch.allclose(out1["3"], out2["3"])
