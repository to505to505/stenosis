"""Comprehensive tests for the Spatio-Temporal Stenosis Detector.

Run: cd /home/dsa/stenosis && python -m pytest stenosis_temporal/tests/ -v
  or: cd /home/dsa/stenosis && python stenosis_temporal/tests/test_all.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stenosis_temporal.config import Config
from stenosis_temporal.dataset import (
    parse_filename,
    build_sequence_index,
    build_windows,
    load_yolo_labels,
    StenosisTemporalDataset,
    collate_fn,
)
from stenosis_temporal.model.fpe import FPE, ResNet50FPN
from stenosis_temporal.model.pstfa import PSSTT, TFA, PSTFA
from stenosis_temporal.model.mto import MTO
from stenosis_temporal.model.detector import (
    StenosisTemporalDetector,
    encode_boxes,
    decode_boxes,
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Use smaller S for faster unit tests
TEST_S = 16


def _test_cfg(**overrides) -> Config:
    """Config with reduced S for faster tests."""
    defaults = dict(S=TEST_S, rpn_post_nms_top_n_train=TEST_S, rpn_post_nms_top_n_test=TEST_S)
    defaults.update(overrides)
    return Config(**defaults)


# ═══════════════════════════════════════════════════════════════════════
#  1. Config
# ═══════════════════════════════════════════════════════════════════════

def test_config_defaults():
    cfg = Config()
    assert cfg.T == 5
    assert cfg.K == 4
    assert cfg.S == 400
    assert cfg.C == 256
    assert cfg.D == 512
    assert cfg.num_transformer_layers == 4
    assert cfg.roi_output_size == 7
    assert cfg.num_classes == 2
    assert cfg.img_h == 512
    assert cfg.img_w == 512
    assert cfg.in_channels == 1
    assert cfg.batch_size == 4
    assert cfg.grad_accum_steps == 1
    assert cfg.lr == 0.02
    assert cfg.momentum == 0.9
    assert cfg.weight_decay == 1e-4
    assert cfg.warmup_iters == 500
    assert cfg.epochs == 100
    print("  [PASS] test_config_defaults")


def test_config_num_tokens():
    cfg = Config()
    assert cfg.num_tokens == 5 * (4 + 1)  # T * (K + 1) = 25
    cfg2 = Config(T=3, K=2)
    assert cfg2.num_tokens == 3 * 3  # 9
    print("  [PASS] test_config_num_tokens")


# ═══════════════════════════════════════════════════════════════════════
#  2. Dataset: parse_filename
# ═══════════════════════════════════════════════════════════════════════

def test_parse_filename_valid():
    r = parse_filename("14_021_1_0046_bmp_jpg.rf.21ed26b42925d46f0add3a4dd6868cf8.jpg")
    assert r is not None
    pid, seq, frame = r
    assert pid == "14_021"
    assert seq == 1
    assert frame == 46
    print("  [PASS] test_parse_filename_valid")


def test_parse_filename_different_ids():
    r = parse_filename("14_095_8_0001_bmp_jpg.rf.abcdef0123456789abcdef0123456789.jpg")
    assert r == ("14_095", 8, 1)
    print("  [PASS] test_parse_filename_different_ids")


def test_parse_filename_invalid():
    assert parse_filename("random_file.jpg") is None
    assert parse_filename("14_021_1_0046.jpg") is None
    assert parse_filename("") is None
    assert parse_filename("something_else.txt") is None
    print("  [PASS] test_parse_filename_invalid")


# ═══════════════════════════════════════════════════════════════════════
#  3. Dataset: build_windows
# ═══════════════════════════════════════════════════════════════════════

def test_build_windows_exact_T():
    """Sequence with exactly T frames → 1 window."""
    paths = [Path(f"frame_{i}.jpg") for i in range(5)]
    seqs = [("p1", 1, paths)]
    windows = build_windows(seqs, T=5)
    assert len(windows) == 1
    assert windows[0] == paths
    print("  [PASS] test_build_windows_exact_T")


def test_build_windows_longer():
    """Sequence with 7 frames, T=5 → 3 sliding windows."""
    paths = [Path(f"frame_{i}.jpg") for i in range(7)]
    seqs = [("p1", 1, paths)]
    windows = build_windows(seqs, T=5)
    assert len(windows) == 3  # 0-4, 1-5, 2-6
    assert windows[0] == paths[0:5]
    assert windows[1] == paths[1:6]
    assert windows[2] == paths[2:7]
    print("  [PASS] test_build_windows_longer")


def test_build_windows_shorter_than_T():
    """Sequence with 3 frames, T=5 → padded to 5 by repeating last."""
    paths = [Path(f"frame_{i}.jpg") for i in range(3)]
    seqs = [("p1", 1, paths)]
    windows = build_windows(seqs, T=5)
    assert len(windows) == 1
    assert len(windows[0]) == 5
    assert windows[0][0] == paths[0]
    assert windows[0][2] == paths[2]
    assert windows[0][3] == paths[2]  # padded
    assert windows[0][4] == paths[2]  # padded
    print("  [PASS] test_build_windows_shorter_than_T")


def test_build_windows_single_frame():
    paths = [Path("frame_0.jpg")]
    seqs = [("p1", 1, paths)]
    windows = build_windows(seqs, T=5)
    assert len(windows) == 1
    assert len(windows[0]) == 5
    assert all(w == paths[0] for w in windows[0])
    print("  [PASS] test_build_windows_single_frame")


def test_build_windows_empty_sequence():
    seqs = [("p1", 1, [])]
    windows = build_windows(seqs, T=5)
    assert len(windows) == 0
    print("  [PASS] test_build_windows_empty_sequence")


def test_build_windows_multiple_sequences():
    paths_a = [Path(f"a_{i}.jpg") for i in range(6)]
    paths_b = [Path(f"b_{i}.jpg") for i in range(5)]
    seqs = [("p1", 1, paths_a), ("p2", 2, paths_b)]
    windows = build_windows(seqs, T=5)
    assert len(windows) == 2 + 1  # 6-5+1=2 from a, 5-5+1=1 from b
    print("  [PASS] test_build_windows_multiple_sequences")


# ═══════════════════════════════════════════════════════════════════════
#  4. Dataset: load_yolo_labels
# ═══════════════════════════════════════════════════════════════════════

def test_load_yolo_labels_single_box(tmp_path):
    lbl = tmp_path / "test.txt"
    # class cx cy w h (normalized)
    lbl.write_text("0 0.5 0.5 0.1 0.2\n")
    result = load_yolo_labels(lbl, img_w=512, img_h=512)
    assert result.shape == (1, 5)
    assert result[0, 0] == 0  # class
    np.testing.assert_allclose(result[0, 1], 512 * 0.5 - 512 * 0.1 / 2, atol=1e-3)  # x1
    np.testing.assert_allclose(result[0, 2], 512 * 0.5 - 512 * 0.2 / 2, atol=1e-3)  # y1
    np.testing.assert_allclose(result[0, 3], 512 * 0.5 + 512 * 0.1 / 2, atol=1e-3)  # x2
    np.testing.assert_allclose(result[0, 4], 512 * 0.5 + 512 * 0.2 / 2, atol=1e-3)  # y2
    print("  [PASS] test_load_yolo_labels_single_box")


def test_load_yolo_labels_multiple_boxes(tmp_path):
    lbl = tmp_path / "test.txt"
    lbl.write_text("0 0.5 0.5 0.1 0.2\n0 0.8 0.3 0.05 0.1\n")
    result = load_yolo_labels(lbl, img_w=512, img_h=512)
    assert result.shape == (2, 5)
    print("  [PASS] test_load_yolo_labels_multiple_boxes")


def test_load_yolo_labels_empty_file(tmp_path):
    lbl = tmp_path / "empty.txt"
    lbl.write_text("")
    result = load_yolo_labels(lbl, img_w=512, img_h=512)
    assert result.shape == (0, 5)
    print("  [PASS] test_load_yolo_labels_empty_file")


def test_load_yolo_labels_missing_file(tmp_path):
    result = load_yolo_labels(tmp_path / "nonexistent.txt", img_w=512, img_h=512)
    assert result.shape == (0, 5)
    print("  [PASS] test_load_yolo_labels_missing_file")


def test_load_yolo_labels_box_coords_correct(tmp_path):
    """Verify exact coordinate conversion: center → xyxy."""
    lbl = tmp_path / "test.txt"
    lbl.write_text("0 0.25 0.75 0.1 0.2\n")
    result = load_yolo_labels(lbl, img_w=100, img_h=200)
    # cx=25, cy=150, w=10, h=40
    np.testing.assert_allclose(result[0, 1:], [20.0, 130.0, 30.0, 170.0], atol=1e-4)
    print("  [PASS] test_load_yolo_labels_box_coords_correct")


# ═══════════════════════════════════════════════════════════════════════
#  5. Dataset: real data loading
# ═══════════════════════════════════════════════════════════════════════

def test_dataset_loads_train():
    """Load train split and verify shapes, types, and value ranges."""
    cfg = Config()
    ds = StenosisTemporalDataset("train", cfg)
    assert len(ds) > 0, f"Dataset empty, expected >0 windows"

    images, targets = ds[0]
    # Shape
    assert images.shape == (cfg.T, 1, cfg.img_h, cfg.img_w), \
        f"Expected ({cfg.T}, 1, {cfg.img_h}, {cfg.img_w}), got {images.shape}"
    # Type
    assert images.dtype == torch.float32
    # Number of target frames
    assert len(targets) == cfg.T
    # Each target has correct keys
    for t_dict in targets:
        assert "boxes" in t_dict and "labels" in t_dict
        assert t_dict["boxes"].ndim == 2 and t_dict["boxes"].shape[1] == 4
        assert t_dict["labels"].dtype == torch.int64
    print("  [PASS] test_dataset_loads_train")


def test_dataset_collate_fn():
    """Verify collate produces correct batch dimensions."""
    cfg = Config()
    ds = StenosisTemporalDataset("train", cfg)
    batch = [ds[0], ds[1]]
    images, targets = collate_fn(batch)
    assert images.shape == (2, cfg.T, 1, cfg.img_h, cfg.img_w)
    assert len(targets) == 2
    assert len(targets[0]) == cfg.T
    print("  [PASS] test_dataset_collate_fn")


def test_dataset_boxes_within_image():
    """All bounding boxes should be within [0, img_w] × [0, img_h]."""
    cfg = Config()
    ds = StenosisTemporalDataset("train", cfg)
    for i in range(min(50, len(ds))):
        _, targets = ds[i]
        for t_dict in targets:
            boxes = t_dict["boxes"]
            if boxes.numel() == 0:
                continue
            assert (boxes[:, 0] >= 0).all(), f"x1 < 0 at window {i}"
            assert (boxes[:, 1] >= 0).all(), f"y1 < 0 at window {i}"
            assert (boxes[:, 2] <= cfg.img_w + 1).all(), f"x2 > img_w at window {i}"
            assert (boxes[:, 3] <= cfg.img_h + 1).all(), f"y2 > img_h at window {i}"
            assert (boxes[:, 2] > boxes[:, 0]).all(), f"x2 <= x1 at window {i}"
            assert (boxes[:, 3] > boxes[:, 1]).all(), f"y2 <= y1 at window {i}"
    print("  [PASS] test_dataset_boxes_within_image")


def test_dataset_normalization():
    """Verify pixel normalization is applied (not raw 0-255)."""
    cfg = Config()
    ds = StenosisTemporalDataset("train", cfg)
    images, _ = ds[0]
    assert images.min() < 0 or images.max() < 10, \
        f"Images seem unnormalized: range [{images.min():.1f}, {images.max():.1f}]"
    print("  [PASS] test_dataset_normalization")


# ═══════════════════════════════════════════════════════════════════════
#  6. Encode/Decode boxes (roundtrip)
# ═══════════════════════════════════════════════════════════════════════

def test_encode_decode_roundtrip():
    """encode then decode should recover original GT boxes."""
    proposals = torch.tensor([
        [100, 100, 200, 200],
        [50, 50, 80, 90],
        [300, 300, 400, 450],
    ], dtype=torch.float32)
    gt_boxes = torch.tensor([
        [110, 105, 210, 195],
        [55, 48, 82, 92],
        [290, 310, 410, 440],
    ], dtype=torch.float32)

    deltas = encode_boxes(gt_boxes, proposals)
    recovered = decode_boxes(deltas, proposals)
    torch.testing.assert_close(recovered, gt_boxes, atol=1e-4, rtol=1e-4)
    print("  [PASS] test_encode_decode_roundtrip")


def test_encode_decode_identity():
    """When GT == proposal, deltas should be zero."""
    proposals = torch.tensor([[100, 100, 200, 200]], dtype=torch.float32)
    deltas = encode_boxes(proposals, proposals)
    assert torch.allclose(deltas, torch.zeros(1, 4), atol=1e-6)
    print("  [PASS] test_encode_decode_identity")


# ═══════════════════════════════════════════════════════════════════════
#  7. ResNet50 + FPN backbone shapes
# ═══════════════════════════════════════════════════════════════════════

def test_resnet50_fpn_shapes():
    backbone = ResNet50FPN().to(DEVICE)
    x = torch.randn(2, 3, 512, 512, device=DEVICE)
    with torch.no_grad():
        out = backbone(x)

    # FPN levels: "0" (stride 4), "1" (stride 8), "2" (stride 16), "3" (stride 32), "pool"
    expected_sizes = {
        "0": (2, 256, 128, 128),
        "1": (2, 256, 64, 64),
        "2": (2, 256, 32, 32),
        "3": (2, 256, 16, 16),
        "pool": (2, 256, 8, 8),
    }
    for key, expected in expected_sizes.items():
        assert key in out, f"Missing FPN level {key}"
        assert out[key].shape == expected, \
            f"FPN level {key}: expected {expected}, got {out[key].shape}"

    print("  [PASS] test_resnet50_fpn_shapes")


# ═══════════════════════════════════════════════════════════════════════
#  8. FPE module (full: adapter + backbone + RPN + RoI Align)
# ═══════════════════════════════════════════════════════════════════════

def test_fpe_output_shapes():
    cfg = _test_cfg()
    fpe = FPE(cfg).to(DEVICE)
    N = 4
    images = torch.randn(N, 1, cfg.img_h, cfg.img_w, device=DEVICE)
    targets = [
        {"boxes": torch.tensor([[100, 100, 130, 130]], dtype=torch.float32, device=DEVICE),
         "labels": torch.tensor([0], dtype=torch.int64, device=DEVICE)}
        for _ in range(N)
    ]
    fpe.train()
    features, proposals, rpn_losses, roi_features = fpe(images, targets)

    # Features: OrderedDict with FPN levels
    assert "0" in features
    assert features["0"].shape[0] == N  # batch dim

    # Proposals: list of N tensors, each (S, 4)
    assert len(proposals) == N
    for p in proposals:
        assert p.shape == (cfg.S, 4), f"Expected ({cfg.S}, 4), got {p.shape}"

    # RPN losses present during training
    assert "loss_objectness" in rpn_losses
    assert "loss_rpn_box_reg" in rpn_losses

    # RoI features
    assert roi_features.shape == (N * cfg.S, cfg.C, cfg.roi_output_size, cfg.roi_output_size)

    print("  [PASS] test_fpe_output_shapes")


def test_fpe_eval_no_rpn_loss():
    cfg = _test_cfg()
    fpe = FPE(cfg).to(DEVICE)
    fpe.eval()
    images = torch.randn(2, 1, cfg.img_h, cfg.img_w, device=DEVICE)
    with torch.no_grad():
        features, proposals, rpn_losses, roi_features = fpe(images, None)
    # In eval mode RPN should not compute losses
    assert len(rpn_losses) == 0 or all(v == 0 for v in rpn_losses.values()), \
        f"Expected no losses in eval, got {rpn_losses}"
    print("  [PASS] test_fpe_eval_no_rpn_loss")


def test_fpe_channel_adapter():
    """1→3 channel adapter should produce 3-channel output."""
    cfg = _test_cfg()
    fpe = FPE(cfg).to(DEVICE)
    x = torch.randn(1, 1, 64, 64, device=DEVICE)
    out = fpe.channel_adapter(x)
    assert out.shape == (1, 3, 64, 64)
    print("  [PASS] test_fpe_channel_adapter")


# ═══════════════════════════════════════════════════════════════════════
#  9. PSSTT (Tokenization)
# ═══════════════════════════════════════════════════════════════════════

def test_psstt_shifted_boxes_shape():
    cfg = _test_cfg()
    psstt = PSSTT(cfg).to(DEVICE)
    S = 10
    boxes = torch.tensor(
        [[100, 100, 150, 150]] * S, dtype=torch.float32, device=DEVICE
    )
    shifted = psstt._generate_shifted_boxes(boxes)
    assert shifted.shape == (S, cfg.K + 1, 4)
    print("  [PASS] test_psstt_shifted_boxes_shape")


def test_psstt_shifted_boxes_first_is_original():
    cfg = _test_cfg()
    psstt = PSSTT(cfg).to(DEVICE)
    box = torch.tensor([[100, 100, 150, 150]], dtype=torch.float32, device=DEVICE)
    shifted = psstt._generate_shifted_boxes(box)
    # First shift should be the original box
    torch.testing.assert_close(shifted[0, 0], box[0])
    print("  [PASS] test_psstt_shifted_boxes_first_is_original")


def test_psstt_shifted_boxes_directions():
    """Verify 4 shifts are in correct directions."""
    cfg = Config(shift_fraction=0.5)
    psstt = PSSTT(cfg)
    # box: x1=100, y1=100, x2=200, y2=200 → w=100, h=100
    box = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    shifted = psstt._generate_shifted_boxes(box)  # (1, 5, 4)
    s = shifted[0]  # (5, 4)

    orig = s[0]  # original
    up = s[1]    # y decreased
    down = s[2]  # y increased
    left = s[3]  # x decreased
    right = s[4] # x increased

    # Up: y shifted down by 50 (0.5 * h=100)
    assert up[1] < orig[1], "Up shift should decrease y1"
    assert up[3] < orig[3], "Up shift should decrease y2"
    # Down
    assert down[1] > orig[1], "Down shift should increase y1"
    # Left
    assert left[0] < orig[0], "Left shift should decrease x1"
    # Right
    assert right[0] > orig[0], "Right shift should increase x1"
    print("  [PASS] test_psstt_shifted_boxes_directions")


def test_psstt_clamping():
    """Boxes near image edge should be clamped."""
    cfg = Config(img_w=512, img_h=512, shift_fraction=0.5)
    psstt = PSSTT(cfg)
    # Box at top-left corner
    box = torch.tensor([[0.0, 0.0, 50.0, 50.0]])
    shifted = psstt._generate_shifted_boxes(box)
    assert (shifted >= 0).all(), "Shifted boxes should not go below 0"
    assert (shifted[..., 0::2] <= 512).all(), "x coords should be ≤ img_w"
    assert (shifted[..., 1::2] <= 512).all(), "y coords should be ≤ img_h"
    print("  [PASS] test_psstt_clamping")


def test_psstt_output_shapes():
    cfg = _test_cfg()
    psstt = PSSTT(cfg).to(DEVICE)
    S = 8
    T = cfg.T
    C = cfg.C
    # Fake features for T frames (one FPN level): each (C, 128, 128)
    features = [torch.randn(C, 128, 128, device=DEVICE) for _ in range(T)]
    proposals = torch.rand(S, 4, device=DEVICE) * 400 + 10
    # Ensure x2 > x1, y2 > y1
    proposals[:, 2] = proposals[:, 0] + 30
    proposals[:, 3] = proposals[:, 1] + 30

    tokens = psstt(features, proposals, ref_idx=2)
    expected = (S, T * (cfg.K + 1), cfg.D)
    assert tokens.shape == expected, f"Expected {expected}, got {tokens.shape}"
    print("  [PASS] test_psstt_output_shapes")


# ═══════════════════════════════════════════════════════════════════════
#  10. TFA (Transformer aggregation)
# ═══════════════════════════════════════════════════════════════════════

def test_tfa_output_shapes():
    cfg = _test_cfg()
    tfa = TFA(cfg).to(DEVICE)
    S = 8
    tokens = torch.randn(S, cfg.num_tokens, cfg.D, device=DEVICE)
    out = tfa(tokens)
    expected = (S, cfg.C, cfg.roi_output_size, cfg.roi_output_size)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("  [PASS] test_tfa_output_shapes")


def test_tfa_pos_embed_shape():
    cfg = _test_cfg()
    tfa = TFA(cfg).to(DEVICE)
    assert tfa.pos_embed.shape == (1, cfg.num_tokens, cfg.D)
    print("  [PASS] test_tfa_pos_embed_shape")


# ═══════════════════════════════════════════════════════════════════════
#  11. PSTFA (combined PSSTT + TFA)
# ═══════════════════════════════════════════════════════════════════════

def test_pstfa_output_shapes():
    cfg = _test_cfg()
    pstfa = PSTFA(cfg).to(DEVICE)
    S = 8
    features = [torch.randn(cfg.C, 128, 128, device=DEVICE) for _ in range(cfg.T)]
    proposals = torch.rand(S, 4, device=DEVICE) * 400 + 10
    proposals[:, 2] = proposals[:, 0] + 30
    proposals[:, 3] = proposals[:, 1] + 30

    out = pstfa(features, proposals, ref_idx=0)
    expected = (S, cfg.C, cfg.roi_output_size, cfg.roi_output_size)
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("  [PASS] test_pstfa_output_shapes")


def test_pstfa_chunking():
    """Verify chunked processing produces same shape."""
    cfg = _test_cfg(proposal_chunk_size=4)
    pstfa = PSTFA(cfg).to(DEVICE)
    S = 12  # > chunk_size=4, will process in 3 chunks
    features = [torch.randn(cfg.C, 128, 128, device=DEVICE) for _ in range(cfg.T)]
    proposals = torch.rand(S, 4, device=DEVICE) * 400 + 10
    proposals[:, 2] = proposals[:, 0] + 30
    proposals[:, 3] = proposals[:, 1] + 30

    out = pstfa(features, proposals, ref_idx=0)
    assert out.shape == (S, cfg.C, cfg.roi_output_size, cfg.roi_output_size)
    print("  [PASS] test_pstfa_chunking")


# ═══════════════════════════════════════════════════════════════════════
#  12. MTO (classification + regression heads)
# ═══════════════════════════════════════════════════════════════════════

def test_mto_output_shapes():
    cfg = _test_cfg()
    mto = MTO(cfg).to(DEVICE)
    N = 10
    roi_features = torch.randn(N, cfg.C, cfg.roi_output_size, cfg.roi_output_size, device=DEVICE)
    cls_logits, box_deltas = mto(roi_features)
    assert cls_logits.shape == (N, cfg.num_classes), \
        f"Expected ({N}, {cfg.num_classes}), got {cls_logits.shape}"
    assert box_deltas.shape == (N, 4), f"Expected ({N}, 4), got {box_deltas.shape}"
    print("  [PASS] test_mto_output_shapes")


def test_mto_single_input():
    cfg = _test_cfg()
    mto = MTO(cfg).to(DEVICE)
    roi_features = torch.randn(1, cfg.C, cfg.roi_output_size, cfg.roi_output_size, device=DEVICE)
    cls_logits, box_deltas = mto(roi_features)
    assert cls_logits.shape == (1, 2)
    assert box_deltas.shape == (1, 4)
    print("  [PASS] test_mto_single_input")


# ═══════════════════════════════════════════════════════════════════════
#  13. Full detector — training mode
# ═══════════════════════════════════════════════════════════════════════

def _make_dummy_data(cfg, B=2, device=DEVICE):
    T = cfg.T
    images = torch.randn(B, T, 1, cfg.img_h, cfg.img_w, device=device)
    targets = []
    for b in range(B):
        frame_targets = []
        for t in range(T):
            frame_targets.append({
                "boxes": torch.tensor([[100.0, 100.0, 130.0, 130.0]], device=device),
                "labels": torch.tensor([0], dtype=torch.int64, device=device),
            })
        targets.append(frame_targets)
    return images, targets


def test_detector_train_returns_losses():
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.init_weights()
    model.train()

    images, targets = _make_dummy_data(cfg, B=1)
    losses = model(images, targets)

    required_keys = {"rpn_objectness_loss", "rpn_box_loss", "det_cls_loss", "det_reg_loss", "total_loss"}
    assert required_keys.issubset(losses.keys()), \
        f"Missing keys: {required_keys - losses.keys()}"

    for k, v in losses.items():
        assert v.ndim == 0, f"Loss {k} should be scalar, got shape {v.shape}"
        assert torch.isfinite(v), f"Loss {k} is not finite: {v.item()}"

    print("  [PASS] test_detector_train_returns_losses")


def test_detector_train_total_loss_is_sum():
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.init_weights()
    model.train()

    images, targets = _make_dummy_data(cfg, B=1)
    losses = model(images, targets)

    components_sum = (
        losses["rpn_objectness_loss"]
        + losses["rpn_box_loss"]
        + losses["det_cls_loss"]
        + losses["det_reg_loss"]
    )
    torch.testing.assert_close(losses["total_loss"], components_sum, atol=1e-5, rtol=1e-5)
    print("  [PASS] test_detector_train_total_loss_is_sum")


def test_detector_train_backward():
    """Verify gradients flow through the full model."""
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.init_weights()
    model.train()

    images, targets = _make_dummy_data(cfg, B=1)
    losses = model(images, targets)
    losses["total_loss"].backward()

    # Check that at least some parameters got gradients
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients found — backward pass broken"
    print("  [PASS] test_detector_train_backward")


def test_detector_train_empty_gt():
    """Training with empty GT boxes should not crash."""
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.init_weights()
    model.train()

    B, T = 1, cfg.T
    images = torch.randn(B, T, 1, cfg.img_h, cfg.img_w, device=DEVICE)
    targets = [[
        {"boxes": torch.zeros(0, 4, device=DEVICE),
         "labels": torch.zeros(0, dtype=torch.int64, device=DEVICE)}
        for _ in range(T)
    ]]

    losses = model(images, targets)
    assert torch.isfinite(losses["total_loss"]), "Loss should be finite with empty GT"
    print("  [PASS] test_detector_train_empty_gt")


# ═══════════════════════════════════════════════════════════════════════
#  14. Full detector — inference mode
# ═══════════════════════════════════════════════════════════════════════

def test_detector_inference_returns_detections():
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.init_weights()
    model.eval()

    B, T = 2, cfg.T
    images = torch.randn(B, T, 1, cfg.img_h, cfg.img_w, device=DEVICE)

    with torch.no_grad():
        results = model(images, None)

    # Should get B * T results (one per reference frame)
    assert len(results) == B * T, f"Expected {B*T} results, got {len(results)}"

    for r in results:
        assert "boxes" in r and "scores" in r and "labels" in r
        assert "batch_idx" in r and "frame_idx" in r
        assert r["boxes"].ndim == 2 and r["boxes"].shape[1] == 4
        assert r["scores"].ndim == 1
        assert r["boxes"].shape[0] == r["scores"].shape[0]
        assert r["labels"].shape[0] == r["scores"].shape[0]
        # Scores should be in [0, 1]
        if r["scores"].numel() > 0:
            assert r["scores"].min() >= 0 and r["scores"].max() <= 1
        # Labels should all be 1 (stenosis)
        assert (r["labels"] == 1).all()
    print("  [PASS] test_detector_inference_returns_detections")


def test_detector_inference_boxes_clamped():
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.eval()

    images = torch.randn(1, cfg.T, 1, cfg.img_h, cfg.img_w, device=DEVICE)
    with torch.no_grad():
        results = model(images, None)

    for r in results:
        if r["boxes"].numel() > 0:
            assert (r["boxes"][:, 0] >= 0).all(), "x1 < 0"
            assert (r["boxes"][:, 1] >= 0).all(), "y1 < 0"
            assert (r["boxes"][:, 2] <= cfg.img_w).all(), "x2 > img_w"
            assert (r["boxes"][:, 3] <= cfg.img_h).all(), "y2 > img_h"
    print("  [PASS] test_detector_inference_boxes_clamped")


def test_detector_inference_score_filtering():
    """Only detections above score_thresh should be returned."""
    cfg = _test_cfg(score_thresh=0.9)  # very high threshold
    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.eval()

    images = torch.randn(1, cfg.T, 1, cfg.img_h, cfg.img_w, device=DEVICE)
    with torch.no_grad():
        results = model(images, None)

    for r in results:
        if r["scores"].numel() > 0:
            assert (r["scores"] > 0.9).all(), \
                f"Scores below threshold found: min={r['scores'].min():.3f}"
    print("  [PASS] test_detector_inference_score_filtering")


def test_detector_inference_max_detections():
    """Should not return more than detections_per_img."""
    cfg = _test_cfg(detections_per_img=5, score_thresh=0.0)
    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.eval()

    images = torch.randn(1, cfg.T, 1, cfg.img_h, cfg.img_w, device=DEVICE)
    with torch.no_grad():
        results = model(images, None)

    for r in results:
        assert r["boxes"].shape[0] <= 5, \
            f"Got {r['boxes'].shape[0]} detections, expected ≤ 5"
    print("  [PASS] test_detector_inference_max_detections")


# ═══════════════════════════════════════════════════════════════════════
#  15. Xavier initialization
# ═══════════════════════════════════════════════════════════════════════

def test_init_weights_preserves_backbone():
    """init_weights should not modify the pretrained backbone weights."""
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)

    # Snapshot backbone weights before init
    before = {n: p.clone() for n, p in model.named_parameters() if "fpe.backbone" in n}

    model.init_weights()

    for name, param in model.named_parameters():
        if "fpe.backbone" in name:
            torch.testing.assert_close(
                param.data, before[name],
                msg=f"Backbone param {name} was modified by init_weights"
            )
    print("  [PASS] test_init_weights_preserves_backbone")


def test_init_weights_modifies_heads():
    """init_weights should modify non-backbone parameters."""
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)

    # Record initial weights of MTO cls head
    before = model.mto.cls_head[0].weight.clone()

    model.init_weights()

    # After xavier init, weights should have changed (very high probability)
    changed = not torch.equal(model.mto.cls_head[0].weight, before)
    assert changed, "MTO weights should change after init_weights"
    print("  [PASS] test_init_weights_modifies_heads")


# ═══════════════════════════════════════════════════════════════════════
#  16. Target assignment
# ═══════════════════════════════════════════════════════════════════════

def test_assign_targets_perfect_overlap():
    """Proposal exactly matching GT should be assigned foreground."""
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)

    proposals = torch.tensor([
        [100.0, 100.0, 200.0, 200.0],  # exact match
        [400.0, 400.0, 450.0, 450.0],  # no overlap
    ], device=DEVICE)
    gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]], device=DEVICE)
    gt_labels = torch.tensor([0], dtype=torch.int64, device=DEVICE)

    labels, matched = model._assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
    assert labels[0] == 1, "Exact overlap should be foreground"
    assert labels[1] == 0, "No overlap should be background"
    print("  [PASS] test_assign_targets_perfect_overlap")


def test_assign_targets_empty_gt():
    cfg = _test_cfg()
    model = StenosisTemporalDetector(cfg).to(DEVICE)

    proposals = torch.tensor([[100.0, 100.0, 200.0, 200.0]], device=DEVICE)
    gt_boxes = torch.zeros(0, 4, device=DEVICE)
    gt_labels = torch.zeros(0, dtype=torch.int64, device=DEVICE)

    labels, matched = model._assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
    assert (labels == 0).all(), "All should be background with no GT"
    assert matched.shape == (1, 4)
    print("  [PASS] test_assign_targets_empty_gt")


# ═══════════════════════════════════════════════════════════════════════
#  17. Real data end-to-end
# ═══════════════════════════════════════════════════════════════════════

def test_real_data_forward():
    """Run model on a real batch from the dataset."""
    cfg = _test_cfg()
    ds = StenosisTemporalDataset("train", cfg)
    batch = collate_fn([ds[0]])
    images, targets = batch
    images = images.to(DEVICE)

    model = StenosisTemporalDetector(cfg).to(DEVICE)
    model.init_weights()

    # Training
    model.train()
    losses = model(images, targets)
    assert torch.isfinite(losses["total_loss"]), f"Loss not finite: {losses['total_loss']}"

    # Inference
    model.eval()
    with torch.no_grad():
        results = model(images, None)
    assert len(results) == cfg.T
    print("  [PASS] test_real_data_forward")


# ═══════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════

def main():
    import tempfile

    print("=" * 60)
    print(" Stenosis Temporal Detector — Comprehensive Tests")
    print("=" * 60)

    # Use a temp dir for file-based tests
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        all_tests = [
            # Config
            ("Config defaults", test_config_defaults),
            ("Config num_tokens", test_config_num_tokens),
            # Filename parsing
            ("parse_filename valid", test_parse_filename_valid),
            ("parse_filename different IDs", test_parse_filename_different_ids),
            ("parse_filename invalid", test_parse_filename_invalid),
            # Windows
            ("build_windows exact T", test_build_windows_exact_T),
            ("build_windows longer", test_build_windows_longer),
            ("build_windows shorter", test_build_windows_shorter_than_T),
            ("build_windows single frame", test_build_windows_single_frame),
            ("build_windows empty", test_build_windows_empty_sequence),
            ("build_windows multiple seq", test_build_windows_multiple_sequences),
            # YOLO labels
            ("load_yolo single box", lambda: test_load_yolo_labels_single_box(tmp_path)),
            ("load_yolo multiple boxes", lambda: test_load_yolo_labels_multiple_boxes(tmp_path)),
            ("load_yolo empty file", lambda: test_load_yolo_labels_empty_file(tmp_path)),
            ("load_yolo missing file", lambda: test_load_yolo_labels_missing_file(tmp_path)),
            ("load_yolo coords correct", lambda: test_load_yolo_labels_box_coords_correct(tmp_path)),
            # Encode/decode
            ("encode/decode roundtrip", test_encode_decode_roundtrip),
            ("encode/decode identity", test_encode_decode_identity),
            # Real dataset
            ("dataset loads train", test_dataset_loads_train),
            ("dataset collate_fn", test_dataset_collate_fn),
            ("dataset boxes within image", test_dataset_boxes_within_image),
            ("dataset normalization", test_dataset_normalization),
            # Backbone
            ("ResNet50+FPN shapes", test_resnet50_fpn_shapes),
            # FPE
            ("FPE output shapes", test_fpe_output_shapes),
            ("FPE eval no rpn loss", test_fpe_eval_no_rpn_loss),
            ("FPE channel adapter", test_fpe_channel_adapter),
            # PSSTT
            ("PSSTT shifted boxes shape", test_psstt_shifted_boxes_shape),
            ("PSSTT first is original", test_psstt_shifted_boxes_first_is_original),
            ("PSSTT shift directions", test_psstt_shifted_boxes_directions),
            ("PSSTT clamping", test_psstt_clamping),
            ("PSSTT output shapes", test_psstt_output_shapes),
            # TFA
            ("TFA output shapes", test_tfa_output_shapes),
            ("TFA pos_embed shape", test_tfa_pos_embed_shape),
            # PSTFA
            ("PSTFA output shapes", test_pstfa_output_shapes),
            ("PSTFA chunking", test_pstfa_chunking),
            # MTO
            ("MTO output shapes", test_mto_output_shapes),
            ("MTO single input", test_mto_single_input),
            # Detector train
            ("Detector train losses", test_detector_train_returns_losses),
            ("Detector total = sum", test_detector_train_total_loss_is_sum),
            ("Detector backward", test_detector_train_backward),
            ("Detector empty GT", test_detector_train_empty_gt),
            # Detector inference
            ("Detector inference dets", test_detector_inference_returns_detections),
            ("Detector boxes clamped", test_detector_inference_boxes_clamped),
            ("Detector score filtering", test_detector_inference_score_filtering),
            ("Detector max detections", test_detector_inference_max_detections),
            # Init
            ("init preserves backbone", test_init_weights_preserves_backbone),
            ("init modifies heads", test_init_weights_modifies_heads),
            # Target assignment
            ("assign perfect overlap", test_assign_targets_perfect_overlap),
            ("assign empty GT", test_assign_targets_empty_gt),
            # Real data e2e
            ("real data forward", test_real_data_forward),
        ]

        passed = 0
        failed = 0
        errors = []

        for name, fn in all_tests:
            try:
                fn()
                passed += 1
            except Exception as e:
                failed += 1
                errors.append((name, str(e)))
                print(f"  [FAIL] {name}: {e}")

    print("\n" + "=" * 60)
    print(f" Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    if errors:
        print("\nFailed tests:")
        for name, err in errors:
            print(f"  ✗ {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")


if __name__ == "__main__":
    main()
