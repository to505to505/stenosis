"""Tests for Config dataclass."""

import pytest
from stqd_det.config import Config


class TestConfigDefaults:
    def test_default_values(self):
        cfg = Config()
        assert cfg.T == 9
        assert cfg.img_h == 512
        assert cfg.img_w == 512
        assert cfg.C == 256
        assert cfg.num_proposals == 100
        assert cfg.decoder_layers == 6
        assert cfg.num_classes == 2
        assert cfg.lr == 2.5e-5
        assert cfg.weight_decay == 1e-4

    def test_custom_values(self):
        cfg = Config(img_h=1024, img_w=1024, T=5, num_proposals=100)
        assert cfg.img_h == 1024
        assert cfg.T == 5
        assert cfg.num_proposals == 100

    def test_top_fpn_spatial_default(self):
        cfg = Config()
        assert cfg.top_fpn_spatial == 512 // 32  # 16

    def test_top_fpn_spatial_custom(self):
        cfg = Config(img_h=256)
        assert cfg.top_fpn_spatial == 256 // 32  # 8

    def test_gfe_token_dim(self):
        cfg = Config(img_h=512, img_w=512, C=256)
        s = cfg.top_fpn_spatial  # 16
        assert cfg.gfe_token_dim == 256 * s * s

    def test_gfe_token_dim_small(self):
        cfg = Config(img_h=256, img_w=256, C=128)
        s = cfg.top_fpn_spatial  # 8
        assert cfg.gfe_token_dim == 128 * s * s

    def test_loss_weights_positive(self):
        cfg = Config()
        assert cfg.lambda_l1 > 0
        assert cfg.lambda_giou > 0
        assert cfg.lambda_num > 0
        assert cfg.focal_alpha > 0
        assert cfg.focal_gamma > 0

    def test_dataclass_equality(self):
        c1 = Config(T=5)
        c2 = Config(T=5)
        assert c1 == c2

    def test_dataclass_inequality(self):
        c1 = Config(T=5)
        c2 = Config(T=9)
        assert c1 != c2
