"""Extensive tests for Backbone (ResNet50 + FPN + grayscale adapter)."""

import torch
import pytest
from collections import OrderedDict
from stqd_det.config import Config
from stqd_det.model.backbone import Backbone, ResNet50FPN


class TestChannelAdapter:
    def test_adapter_weight_init(self, cfg, device):
        """Channel adapter should be initialized to 1/3."""
        model = Backbone(cfg).to(device)
        w = model.channel_adapter.weight.data
        assert torch.allclose(w, torch.full_like(w, 1.0 / 3.0))

    def test_adapter_output_channels(self, cfg, device):
        """1ch → 3ch adapter output."""
        model = Backbone(cfg).to(device)
        x = torch.randn(2, 1, cfg.img_h, cfg.img_w, device=device)
        out3 = model.channel_adapter(x)
        assert out3.shape == (2, 3, cfg.img_h, cfg.img_w)

    def test_adapter_no_bias(self, cfg):
        model = Backbone(cfg)
        assert model.channel_adapter.bias is None


class TestResNet50FPN:
    def test_fpn_keys(self, device):
        model = ResNet50FPN().to(device)
        x = torch.randn(1, 3, 256, 256, device=device)
        out = model(x)
        assert set(out.keys()) == {"0", "1", "2", "3"}

    def test_fpn_strides(self, device):
        model = ResNet50FPN().to(device)
        H, W = 256, 256
        x = torch.randn(1, 3, H, W, device=device)
        out = model(x)
        assert out["0"].shape[2:] == (H // 4, W // 4)
        assert out["1"].shape[2:] == (H // 8, W // 8)
        assert out["2"].shape[2:] == (H // 16, W // 16)
        assert out["3"].shape[2:] == (H // 32, W // 32)

    def test_fpn_channels_all_256(self, device):
        model = ResNet50FPN().to(device)
        x = torch.randn(1, 3, 256, 256, device=device)
        out = model(x)
        for key in out:
            assert out[key].shape[1] == 256

    def test_fpn_batch_dim_preserved(self, device):
        model = ResNet50FPN().to(device)
        B = 3
        x = torch.randn(B, 3, 256, 256, device=device)
        out = model(x)
        for key in out:
            assert out[key].shape[0] == B


class TestBackboneEndToEnd:
    def test_grayscale_input(self, cfg, device):
        """Backbone accepts 1-channel input."""
        model = Backbone(cfg).to(device)
        x = torch.randn(2, 1, cfg.img_h, cfg.img_w, device=device)
        out = model(x)
        assert "0" in out
        assert out["0"].shape[0] == 2

    def test_output_shapes_exact(self, cfg, device):
        model = Backbone(cfg).to(device)
        N = cfg.T
        x = torch.randn(N, 1, cfg.img_h, cfg.img_w, device=device)
        out = model(x)
        H, W = cfg.img_h, cfg.img_w
        assert out["0"].shape == (N, 256, H // 4, W // 4)
        assert out["1"].shape == (N, 256, H // 8, W // 8)
        assert out["2"].shape == (N, 256, H // 16, W // 16)
        assert out["3"].shape == (N, 256, H // 32, W // 32)

    def test_gradient_flow(self, cfg, device):
        """Gradients should flow back through backbone."""
        model = Backbone(cfg).to(device)
        x = torch.randn(1, 1, cfg.img_h, cfg.img_w, device=device, requires_grad=True)
        out = model(x)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_nan_output(self, cfg, device):
        model = Backbone(cfg).to(device)
        x = torch.randn(2, 1, cfg.img_h, cfg.img_w, device=device)
        out = model(x)
        for key, feat in out.items():
            assert torch.isfinite(feat).all(), f"NaN/Inf in FPN level {key}"

    def test_different_inputs_different_outputs(self, cfg, device):
        """Different inputs should produce different outputs."""
        model = Backbone(cfg).to(device)
        model.eval()
        x1 = torch.randn(1, 1, cfg.img_h, cfg.img_w, device=device)
        x2 = torch.randn(1, 1, cfg.img_h, cfg.img_w, device=device)
        out1 = model(x1)
        out2 = model(x2)
        assert not torch.allclose(out1["3"], out2["3"])

    def test_deterministic_eval(self, cfg, device):
        """Same input → same output in eval mode."""
        model = Backbone(cfg).to(device).eval()
        x = torch.randn(1, 1, cfg.img_h, cfg.img_w, device=device)
        out1 = model(x)
        out2 = model(x)
        for key in out1:
            assert torch.allclose(out1[key], out2[key])

    def test_gradient_checkpointing(self, device):
        """Gradient checkpointing should produce same output."""
        cfg_no_ckpt = Config(img_h=256, img_w=256, gradient_checkpointing=False)
        cfg_ckpt = Config(img_h=256, img_w=256, gradient_checkpointing=True)

        model_no = Backbone(cfg_no_ckpt).to(device).train()
        model_ck = Backbone(cfg_ckpt).to(device).train()
        model_ck.load_state_dict(model_no.state_dict())

        x = torch.randn(1, 1, 256, 256, device=device)
        out_no = model_no(x)
        out_ck = model_ck(x)
        for key in out_no:
            assert torch.allclose(out_no[key], out_ck[key], atol=1e-5)
