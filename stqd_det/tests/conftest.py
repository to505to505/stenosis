"""Shared fixtures for STQD-Det test suite."""

import sys
from pathlib import Path
from collections import OrderedDict

import torch
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stqd_det.config import Config


@pytest.fixture
def cfg():
    """Small config for fast unit tests."""
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
        stfs_alpha=2.0,
    )


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_fpn(cfg, device):
    """FPN features for T frames, batch size 1."""
    T = cfg.T
    H, W = cfg.img_h, cfg.img_w
    return OrderedDict({
        "0": torch.randn(T, 256, H // 4, W // 4, device=device),
        "1": torch.randn(T, 256, H // 8, W // 8, device=device),
        "2": torch.randn(T, 256, H // 16, W // 16, device=device),
        "3": torch.randn(T, 256, H // 32, W // 32, device=device),
    })


@pytest.fixture
def sample_gt_boxes(cfg, device):
    """GT boxes per frame: 2 boxes each in xyxy."""
    return [
        torch.tensor(
            [[50, 60, 100, 110], [150, 160, 200, 210]],
            dtype=torch.float32, device=device,
        )
        for _ in range(cfg.T)
    ]


@pytest.fixture
def sample_gt_labels(cfg, device):
    """GT labels per frame: 2 labels each."""
    return [
        torch.tensor([0, 1], dtype=torch.long, device=device)
        for _ in range(cfg.T)
    ]


@pytest.fixture
def sample_proposals(cfg, device):
    """Valid random proposals in xyxy format."""
    T, P = cfg.T, cfg.num_proposals
    H, W = cfg.img_h, cfg.img_w
    x1y1 = torch.rand(T, P, 2, device=device) * torch.tensor([W * 0.6, H * 0.6], device=device)
    wh = torch.rand(T, P, 2, device=device) * 50 + 10
    x2y2 = x1y1 + wh
    x2y2[..., 0].clamp_(max=W)
    x2y2[..., 1].clamp_(max=H)
    return torch.cat([x1y1, x2y2], dim=-1)


@pytest.fixture
def image_sizes(cfg):
    return [(cfg.img_h, cfg.img_w)] * cfg.T
