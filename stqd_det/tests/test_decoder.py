"""Extensive tests for StenosisDecoder and DecoderLayer."""

import torch
import pytest
from collections import OrderedDict
from stqd_det.config import Config
from stqd_det.model.decoder import DecoderLayer, StenosisDecoder


class TestDecoderLayer:
    def test_output_shapes(self, device):
        layer = DecoderLayer(d_model=64, num_heads=4, ffn_dim=128, dropout=0.0, num_classes=2).to(device)
        x = torch.randn(2, 16, 64, device=device)
        features, cls_logits, box_deltas = layer(x)
        assert features.shape == (2, 16, 64)
        assert cls_logits.shape == (2, 16, 2)
        assert box_deltas.shape == (2, 16, 4)

    def test_gradient_flow(self, device):
        layer = DecoderLayer(d_model=64, num_heads=4, ffn_dim=128, dropout=0.0, num_classes=2).to(device)
        x = torch.randn(1, 8, 64, device=device, requires_grad=True)
        features, cls_logits, box_deltas = layer(x)
        loss = cls_logits.sum() + box_deltas.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_all_params_have_grad(self, device):
        layer = DecoderLayer(d_model=64, num_heads=4, ffn_dim=128, dropout=0.0, num_classes=2).to(device)
        x = torch.randn(1, 8, 64, device=device)
        features, cls_logits, box_deltas = layer(x)
        (cls_logits.sum() + box_deltas.sum()).backward()
        for name, p in layer.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan(self, device):
        layer = DecoderLayer(d_model=64, num_heads=4, ffn_dim=128, dropout=0.0, num_classes=2).to(device)
        x = torch.randn(3, 32, 64, device=device)
        features, cls_logits, box_deltas = layer(x)
        assert torch.isfinite(features).all()
        assert torch.isfinite(cls_logits).all()
        assert torch.isfinite(box_deltas).all()


class TestStenosisDecoder:
    def test_output_structure(self, cfg, device, sample_fpn, sample_proposals, image_sizes):
        decoder = StenosisDecoder(cfg).to(device)
        timesteps = torch.randint(0, 100, (cfg.T,), device=device)
        outputs = decoder(sample_fpn, sample_proposals, image_sizes, timesteps)

        assert len(outputs) == cfg.decoder_layers
        for out in outputs:
            assert "cls_logits" in out
            assert "box_pred" in out
            assert out["cls_logits"].shape == (cfg.T, cfg.num_proposals, cfg.num_classes)
            assert out["box_pred"].shape == (cfg.T, cfg.num_proposals, 4)

    def test_box_predictions_within_bounds(self, cfg, device, sample_fpn, sample_proposals, image_sizes):
        decoder = StenosisDecoder(cfg).to(device)
        outputs = decoder(sample_fpn, sample_proposals, image_sizes)
        last_boxes = outputs[-1]["box_pred"]
        assert last_boxes[..., 0::2].min() >= 0
        assert last_boxes[..., 1::2].min() >= 0
        assert last_boxes[..., 0::2].max() <= cfg.img_w
        assert last_boxes[..., 1::2].max() <= cfg.img_h

    def test_without_timesteps(self, cfg, device, sample_fpn, sample_proposals, image_sizes):
        """Decoder should work without time conditioning."""
        decoder = StenosisDecoder(cfg).to(device)
        outputs = decoder(sample_fpn, sample_proposals, image_sizes, timesteps=None)
        assert len(outputs) == cfg.decoder_layers

    def test_gradient_flow(self, cfg, device, sample_fpn, image_sizes):
        decoder = StenosisDecoder(cfg).to(device)
        proposals = torch.rand(cfg.T, cfg.num_proposals, 4, device=device, requires_grad=False)
        proposals = proposals * torch.tensor([cfg.img_w, cfg.img_h, cfg.img_w, cfg.img_h], device=device)
        proposals[..., 2:] = proposals[..., :2] + 20
        proposals = proposals.clamp(0, min(cfg.img_w, cfg.img_h))

        # Make FPN features require grad
        fpn_grad = OrderedDict()
        for key, feat in sample_fpn.items():
            fpn_grad[key] = feat.detach().requires_grad_(True)

        outputs = decoder(fpn_grad, proposals, image_sizes)
        loss = outputs[-1]["cls_logits"].sum() + outputs[-1]["box_pred"].sum()
        loss.backward()
        # At least the top FPN level should have gradients
        has_grad = any(fpn_grad[k].grad is not None and fpn_grad[k].grad.abs().sum() > 0 for k in fpn_grad)
        assert has_grad, "No gradient flowed to FPN features"

    def test_iterative_refinement(self, cfg, device, sample_fpn, sample_proposals, image_sizes):
        """Each layer should produce different predictions (refinement)."""
        decoder = StenosisDecoder(cfg).to(device)
        outputs = decoder(sample_fpn, sample_proposals, image_sizes)
        assert len(outputs) >= 2
        # Layer 0 and layer 1 predictions should differ
        box0 = outputs[0]["box_pred"]
        box1 = outputs[1]["box_pred"]
        assert not torch.allclose(box0, box1)

    def test_time_embedding_effect(self, cfg, device, sample_fpn, sample_proposals, image_sizes):
        """Different timesteps should produce different outputs."""
        decoder = StenosisDecoder(cfg).to(device)
        decoder.eval()
        t_low = torch.zeros(cfg.T, device=device, dtype=torch.long)
        t_high = torch.full((cfg.T,), 99, device=device, dtype=torch.long)
        out_low = decoder(sample_fpn, sample_proposals, image_sizes, t_low)
        out_high = decoder(sample_fpn, sample_proposals, image_sizes, t_high)
        # Different timesteps → different outputs
        assert not torch.allclose(out_low[-1]["cls_logits"], out_high[-1]["cls_logits"])

    def test_no_nan(self, cfg, device, sample_fpn, sample_proposals, image_sizes):
        decoder = StenosisDecoder(cfg).to(device)
        outputs = decoder(sample_fpn, sample_proposals, image_sizes)
        for i, out in enumerate(outputs):
            assert torch.isfinite(out["cls_logits"]).all(), f"NaN in layer {i} cls_logits"
            assert torch.isfinite(out["box_pred"]).all(), f"NaN in layer {i} box_pred"

    def test_apply_deltas_identity(self, cfg, device):
        """Zero deltas should preserve boxes (approximately)."""
        decoder = StenosisDecoder(cfg).to(device)
        boxes = torch.tensor([[[50, 50, 100, 100], [200, 200, 250, 250]]], dtype=torch.float32, device=device)
        deltas = torch.zeros_like(boxes)
        refined = decoder._apply_deltas(boxes, deltas)
        assert torch.allclose(refined, boxes, atol=1e-3)

    def test_apply_deltas_clamped(self, cfg, device):
        """Large deltas should be clamped and boxes stay within bounds."""
        decoder = StenosisDecoder(cfg).to(device)
        boxes = torch.tensor([[[128, 128, 130, 130]]], dtype=torch.float32, device=device)
        deltas = torch.tensor([[[100.0, 100.0, 100.0, 100.0]]], dtype=torch.float32, device=device)
        refined = decoder._apply_deltas(boxes, deltas)
        assert refined[..., 0::2].max() <= cfg.img_w
        assert refined[..., 1::2].max() <= cfg.img_h
        assert refined[..., 0::2].min() >= 0
        assert refined[..., 1::2].min() >= 0

    def test_single_proposal(self, cfg, device, sample_fpn, image_sizes):
        """Works with P=1."""
        cfg1 = Config(
            img_h=cfg.img_h, img_w=cfg.img_w, T=cfg.T,
            num_proposals=1, decoder_layers=1,
            decoder_dim=64, decoder_heads=4, decoder_ffn_dim=128,
            C=256, num_classes=2, roi_output_size=7,
        )
        decoder = StenosisDecoder(cfg1).to(device)
        proposals = torch.tensor([[128, 128, 200, 200]], dtype=torch.float32, device=device)
        proposals = proposals.unsqueeze(0).expand(cfg.T, -1, -1)
        outputs = decoder(sample_fpn, proposals, image_sizes)
        assert outputs[-1]["cls_logits"].shape == (cfg.T, 1, 2)
