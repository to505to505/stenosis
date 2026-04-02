"""Extensive tests for the ViT-Small + YOLO11 hybrid detector.

Run:
    cd /home/dsa/stenosis
    python -m pytest vit_yolo11/tests.py -v --tb=short
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO / "ultralytics"))

from ultralytics import YOLO
from ultralytics.nn.modules.vit_encoder import ViTEncoder
from ultralytics.nn.modules import ViTEncoder as ViTEncoderImported

YAML_PATH = str(ROOT / "vit-yolo11.yaml")
WEIGHTS_PATH = REPO / "vasomim" / "weights" / "vit_small_encoder_512.pth"
DATA_YAML = REPO / "data" / "dataset2_split_90_10" / "data.yaml"


# =====================================================================
# 1. ViTEncoder unit tests
# =====================================================================

class TestViTEncoder:
    """Unit tests for the ViTEncoder module."""

    def test_default_init(self):
        """Default constructor: c1=3, c2=384, img_size=512."""
        enc = ViTEncoder()
        assert enc.patch_size == 16
        assert enc.embed_dim == 384
        assert isinstance(enc.vit, nn.Module)

    def test_custom_init(self):
        """Custom parameters forwarded correctly."""
        enc = ViTEncoder(c1=1, c2=192, img_size=256)
        assert enc.embed_dim == 192
        assert enc.patch_size == 16

    def test_output_shape_512(self):
        """512×512 input → (B, 384, 32, 32)."""
        enc = ViTEncoder(c1=3, c2=384, img_size=512)
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (2, 384, 32, 32)

    def test_output_shape_256(self):
        """256×256 input works (dynamic_img_size) → (B, 384, 16, 16)."""
        enc = ViTEncoder(c1=3, c2=384, img_size=512)
        x = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, 384, 16, 16)

    def test_output_shape_640(self):
        """640×640 input works (dynamic_img_size) → (B, 384, 40, 40)."""
        enc = ViTEncoder(c1=3, c2=384, img_size=512)
        x = torch.randn(1, 3, 640, 640)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, 384, 40, 40)

    def test_output_shape_non_square(self):
        """Non-square input works → (B, 384, H//16, W//16)."""
        enc = ViTEncoder(c1=3, c2=384, img_size=512)
        x = torch.randn(1, 3, 256, 512)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, 384, 16, 32)

    def test_single_channel_input(self):
        """c1=1 single-channel input."""
        enc = ViTEncoder(c1=1, c2=384, img_size=256)
        x = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = enc(x)
        assert out.shape == (1, 384, 16, 16)

    def test_output_dtype_float32(self):
        """Output dtype matches input dtype."""
        enc = ViTEncoder()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = enc(x)
        assert out.dtype == torch.float32

    def test_output_finite(self):
        """Output contains no NaN or Inf values."""
        enc = ViTEncoder()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out = enc(x)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self):
        """Gradients flow through the encoder."""
        enc = ViTEncoder(c1=3, c2=384, img_size=256)
        x = torch.randn(1, 3, 256, 256, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_parameter_count(self):
        """ViT-Small with 384d, 12 blocks, 6 heads ≈ 22M params."""
        enc = ViTEncoder()
        total = sum(p.numel() for p in enc.parameters())
        assert 20_000_000 < total < 25_000_000, f"Expected ~22M params, got {total:,}"

    def test_batch_size_1(self):
        enc = ViTEncoder(c1=3, c2=384, img_size=256)
        with torch.no_grad():
            out = enc(torch.randn(1, 3, 256, 256))
        assert out.shape[0] == 1

    def test_batch_size_8(self):
        enc = ViTEncoder(c1=3, c2=384, img_size=256)
        with torch.no_grad():
            out = enc(torch.randn(8, 3, 256, 256))
        assert out.shape[0] == 8


# =====================================================================
# 2. Module registration tests
# =====================================================================

class TestModuleRegistration:
    """Verify ViTEncoder is correctly registered in the Ultralytics framework."""

    def test_import_from_modules(self):
        """ViTEncoder importable from ultralytics.nn.modules."""
        assert ViTEncoderImported is ViTEncoder

    def test_in_modules_all(self):
        """ViTEncoder listed in __all__."""
        from ultralytics.nn import modules
        assert "ViTEncoder" in modules.__all__

    def test_importable_from_tasks(self):
        """ViTEncoder importable from tasks.py (needed for globals())."""
        from ultralytics.nn import tasks
        assert hasattr(tasks, "ViTEncoder") or "ViTEncoder" in dir(tasks)

    def test_in_base_modules(self):
        """ViTEncoder recognized by parse_model as a base module."""
        # Build model — if ViTEncoder is NOT in base_modules, YAML parsing would fail
        model = YOLO(YAML_PATH)
        layer0 = model.model.model[0]
        assert isinstance(layer0, ViTEncoder)


# =====================================================================
# 3. YAML config / model build tests
# =====================================================================

class TestModelBuild:
    """Test that the YAML config produces a valid detection model."""

    @pytest.fixture(scope="class")
    def model(self):
        return YOLO(YAML_PATH)

    def test_model_builds(self, model):
        """Model builds from YAML without errors."""
        assert model is not None
        assert model.model is not None

    def test_num_layers(self, model):
        """YAML defines 19 layers (0..18)."""
        assert len(model.model.model) == 19

    def test_layer0_is_vit(self, model):
        """Layer 0 is ViTEncoder."""
        assert isinstance(model.model.model[0], ViTEncoder)

    def test_last_layer_is_detect(self, model):
        """Last layer is Detect head."""
        from ultralytics.nn.modules.head import Detect
        assert isinstance(model.model.model[-1], Detect)

    def test_strides(self, model):
        """Detect strides are [8, 16, 32]."""
        strides = model.model.stride.tolist()
        assert strides == [8.0, 16.0, 32.0]

    def test_nc(self, model):
        """Number of classes is 1 (stenosis)."""
        detect = model.model.model[-1]
        assert detect.nc == 1

    def test_detect_nl(self, model):
        """Detect head has 3 output levels."""
        detect = model.model.model[-1]
        assert detect.nl == 3

    def test_detect_channels(self, model):
        """Detect head input channels: [256, 256, 512]."""
        detect = model.model.model[-1]
        # cv2 (box branch) first conv input channels reflect the feature map channels
        chs = [detect.cv2[i][0].conv.in_channels for i in range(detect.nl)]
        assert chs == [256, 256, 512]


# =====================================================================
# 4. Forward pass / inference tests
# =====================================================================

class TestForwardPass:
    """Test forward pass through the complete model."""

    @pytest.fixture(scope="class")
    def model(self):
        m = YOLO(YAML_PATH)
        m.model.eval()
        return m

    def test_inference_512(self, model):
        """512×512 inference produces valid output."""
        x = torch.zeros(1, 3, 512, 512)
        with torch.no_grad():
            out = model.model(x)
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_inference_output_tensor(self, model):
        """First output is the decoded predictions tensor."""
        x = torch.zeros(1, 3, 512, 512)
        with torch.no_grad():
            preds, extra = model.model(x)
        assert isinstance(preds, torch.Tensor)
        # (B, num_outputs, num_anchors)
        assert preds.ndim == 3
        assert preds.shape[0] == 1
        assert preds.shape[1] == 5  # 4 box + 1 class

    def test_inference_num_anchors(self, model):
        """Number of anchors for 512×512: 64² + 32² + 16² = 4096+1024+256 = 5376."""
        x = torch.zeros(1, 3, 512, 512)
        with torch.no_grad():
            preds, _ = model.model(x)
        assert preds.shape[2] == 5376

    def test_inference_feats_dict(self, model):
        """Second output is a dict with 'feats' key (feature maps for loss)."""
        x = torch.zeros(1, 3, 512, 512)
        with torch.no_grad():
            _, extra = model.model(x)
        assert isinstance(extra, dict)
        assert "feats" in extra

    def test_feats_shapes(self, model):
        """Feature maps have expected spatial sizes for 512×512 input."""
        x = torch.zeros(1, 3, 512, 512)
        with torch.no_grad():
            _, extra = model.model(x)
        feats = extra["feats"]
        assert len(feats) == 3
        assert feats[0].shape == (1, 256, 64, 64)   # P3/8
        assert feats[1].shape == (1, 256, 32, 32)   # P4/16
        assert feats[2].shape == (1, 512, 16, 16)   # P5/32

    def test_inference_output_finite(self, model):
        """Output contains no NaN/Inf."""
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            preds, _ = model.model(x)
        assert torch.isfinite(preds).all()

    def test_inference_batch_2(self, model):
        """Batch size 2 works correctly."""
        x = torch.zeros(2, 3, 512, 512)
        with torch.no_grad():
            preds, _ = model.model(x)
        assert preds.shape[0] == 2

    def test_different_input_resolution(self, model):
        """Model handles 640×640 input (different from training size)."""
        x = torch.zeros(1, 3, 640, 640)
        with torch.no_grad():
            preds, extra = model.model(x)
        feats = extra["feats"]
        assert feats[0].shape == (1, 256, 80, 80)   # P3/8
        assert feats[1].shape == (1, 256, 40, 40)   # P4/16
        assert feats[2].shape == (1, 512, 20, 20)   # P5/32

    def test_grayscale_repeated(self, model):
        """Simulated grayscale (same value across 3 channels) works."""
        gray = torch.randn(1, 1, 512, 512)
        x = gray.expand(1, 3, 512, 512).contiguous()
        with torch.no_grad():
            preds, _ = model.model(x)
        assert torch.isfinite(preds).all()


# =====================================================================
# 5. Training mode tests
# =====================================================================

class TestTrainingMode:
    """Test model behavior in training mode."""

    def test_train_mode_returns_feats(self):
        """In training mode, forward returns feature list for loss computation."""
        model = YOLO(YAML_PATH)
        model.model.train()
        x = torch.randn(2, 3, 512, 512)
        out = model.model(x)
        # In training mode, Detect returns the raw feature maps
        assert isinstance(out, (list, dict))

    def test_gradient_flows_through_full_model(self):
        """Gradients flow from loss back through entire model."""
        model = YOLO(YAML_PATH)
        model.model.train()
        x = torch.randn(1, 3, 512, 512, requires_grad=True)
        out = model.model(x)
        # Sum all feature maps
        if isinstance(out, dict):
            feats = out.get("feats", out.get("one2many", []))
        elif isinstance(out, list):
            feats = out
        else:
            feats = [out]
        loss = sum(f.sum() for f in feats if isinstance(f, torch.Tensor))
        loss.backward()
        assert x.grad is not None


# =====================================================================
# 6. Pretrained weight loading tests
# =====================================================================

class TestPretrainedWeights:
    """Test loading VasoMIM pretrained encoder weights."""

    @pytest.fixture(scope="class")
    def model_with_weights(self):
        model = YOLO(YAML_PATH)
        if WEIGHTS_PATH.exists():
            vit = model.model.model[0]
            state = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
            vit.vit.load_state_dict(state, strict=False)
        return model

    @pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="Pretrained weights not found")
    def test_weights_load_no_missing(self):
        """All ViT state dict keys match — zero missing keys."""
        model = YOLO(YAML_PATH)
        vit = model.model.model[0]
        state = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
        msg = vit.vit.load_state_dict(state, strict=False)
        assert len(msg.missing_keys) == 0, f"Missing keys: {msg.missing_keys}"

    @pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="Pretrained weights not found")
    def test_weights_load_no_unexpected(self):
        """Zero unexpected keys."""
        model = YOLO(YAML_PATH)
        vit = model.model.model[0]
        state = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
        msg = vit.vit.load_state_dict(state, strict=False)
        assert len(msg.unexpected_keys) == 0, f"Unexpected keys: {msg.unexpected_keys}"

    @pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="Pretrained weights not found")
    def test_weights_change_output(self):
        """Pretrained weights produce different output than random init."""
        x = torch.randn(1, 3, 256, 256)

        enc_random = ViTEncoder(c1=3, c2=384, img_size=512)
        with torch.no_grad():
            out_random = enc_random(x).clone()

        state = torch.load(str(WEIGHTS_PATH), map_location="cpu", weights_only=True)
        enc_random.vit.load_state_dict(state, strict=False)
        with torch.no_grad():
            out_loaded = enc_random(x)

        assert not torch.allclose(out_random, out_loaded, atol=1e-3)

    @pytest.mark.skipif(not WEIGHTS_PATH.exists(), reason="Pretrained weights not found")
    def test_pretrained_forward_finite(self, model_with_weights):
        """Pretrained model output is finite."""
        model_with_weights.model.eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            preds, _ = model_with_weights.model(x)
        assert torch.isfinite(preds).all()


# =====================================================================
# 7. Layer-by-layer feature map shape tests
# =====================================================================

class TestLayerShapes:
    """Verify intermediate feature map shapes match YAML comments."""

    @pytest.fixture(scope="class")
    def layer_outputs(self):
        """Run forward pass and capture all intermediate outputs."""
        model = YOLO(YAML_PATH)
        model.model.eval()
        x = torch.zeros(1, 3, 512, 512)

        outputs = {}
        hooks = []

        def make_hook(idx):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    outputs[idx] = output.shape
            return hook_fn

        for i, layer in enumerate(model.model.model):
            hooks.append(layer.register_forward_hook(make_hook(i)))

        with torch.no_grad():
            model.model(x)

        for h in hooks:
            h.remove()

        return outputs

    def test_layer0_vit(self, layer_outputs):
        """Layer 0 (ViTEncoder): (1, 384, 32, 32)."""
        assert layer_outputs[0] == (1, 384, 32, 32)

    def test_layer1_convtranspose(self, layer_outputs):
        """Layer 1 (ConvTranspose): (1, 256, 64, 64)."""
        assert layer_outputs[1] == (1, 256, 64, 64)

    def test_layer2_conv_p4(self, layer_outputs):
        """Layer 2 (Conv 1×1): (1, 256, 32, 32)."""
        assert layer_outputs[2] == (1, 256, 32, 32)

    def test_layer3_conv_p5(self, layer_outputs):
        """Layer 3 (Conv 3×3 s2): (1, 512, 16, 16)."""
        assert layer_outputs[3] == (1, 512, 16, 16)

    def test_layer4_sppf(self, layer_outputs):
        """Layer 4 (SPPF): (1, 512, 16, 16)."""
        assert layer_outputs[4] == (1, 512, 16, 16)

    def test_layer5_c2psa(self, layer_outputs):
        """Layer 5 (C2PSA): (1, 512, 16, 16)."""
        assert layer_outputs[5] == (1, 512, 16, 16)

    def test_layer8_neck_mid(self, layer_outputs):
        """Layer 8 (C3k2 after first concat): (1, 256, 32, 32)."""
        assert layer_outputs[8] == (1, 256, 32, 32)

    def test_layer11_p3_out(self, layer_outputs):
        """Layer 11 (P3/8 output): (1, 256, 64, 64)."""
        assert layer_outputs[11] == (1, 256, 64, 64)

    def test_layer14_p4_out(self, layer_outputs):
        """Layer 14 (P4/16 output): (1, 256, 32, 32)."""
        assert layer_outputs[14] == (1, 256, 32, 32)

    def test_layer17_p5_out(self, layer_outputs):
        """Layer 17 (P5/32 output): (1, 512, 16, 16)."""
        assert layer_outputs[17] == (1, 512, 16, 16)


# =====================================================================
# 8. Model properties and consistency tests
# =====================================================================

class TestModelProperties:
    """Miscellaneous model property checks."""

    def test_total_param_count(self):
        """Total params ≈ 33M (22M ViT + 11M head)."""
        model = YOLO(YAML_PATH)
        total = sum(p.numel() for p in model.model.parameters())
        assert 30_000_000 < total < 40_000_000, f"Expected ~33M, got {total:,}"

    def test_all_params_trainable(self):
        """All parameters are trainable except DFL conv (fixed arange kernel)."""
        model = YOLO(YAML_PATH)
        for name, p in model.model.named_parameters():
            if "dfl.conv" in name:
                assert not p.requires_grad, f"DFL conv {name} should be frozen"
            else:
                assert p.requires_grad, f"Parameter {name} is not trainable"

    def test_model_to_eval_and_back(self):
        """Model switches between train/eval modes cleanly."""
        model = YOLO(YAML_PATH)
        model.model.train()
        assert model.model.training
        model.model.eval()
        assert not model.model.training
        model.model.train()
        assert model.model.training

    def test_model_state_dict_saveable(self):
        """Model state_dict can be serialized."""
        model = YOLO(YAML_PATH)
        sd = model.model.state_dict()
        assert len(sd) > 0
        # Verify all values are tensors
        for k, v in sd.items():
            assert isinstance(v, torch.Tensor), f"Non-tensor in state_dict: {k}"

    def test_two_builds_identical_structure(self):
        """Two model builds produce same layer count and param count."""
        m1 = YOLO(YAML_PATH)
        m2 = YOLO(YAML_PATH)
        assert len(m1.model.model) == len(m2.model.model)
        p1 = sum(p.numel() for p in m1.model.parameters())
        p2 = sum(p.numel() for p in m2.model.parameters())
        assert p1 == p2


# =====================================================================
# 9. CUDA tests (skip if no GPU)
# =====================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    """GPU-specific tests."""

    def test_model_to_cuda(self):
        """Model moves to GPU without errors."""
        model = YOLO(YAML_PATH)
        model.model.cuda()
        # Check a parameter is on GPU
        p = next(model.model.parameters())
        assert p.is_cuda

    def test_cuda_forward_pass(self):
        """Forward pass on GPU produces valid output."""
        model = YOLO(YAML_PATH)
        model.model.cuda().eval()
        x = torch.zeros(1, 3, 512, 512, device="cuda")
        with torch.no_grad():
            preds, extra = model.model(x)
        assert preds.is_cuda
        assert torch.isfinite(preds).all()

    def test_cuda_training_backward(self):
        """Training backward pass on GPU works."""
        model = YOLO(YAML_PATH)
        model.model.cuda().train()
        x = torch.randn(1, 3, 512, 512, device="cuda")
        out = model.model(x)
        if isinstance(out, dict):
            feats = out.get("feats", out.get("one2many", []))
        elif isinstance(out, list):
            feats = out
        else:
            feats = [out]
        loss = sum(f.sum() for f in feats if isinstance(f, torch.Tensor))
        loss.backward()
        # Check gradients exist on ViT params
        vit = model.model.model[0]
        assert any(p.grad is not None for p in vit.parameters())

    def test_cuda_amp_forward(self):
        """Forward pass works with automatic mixed precision."""
        model = YOLO(YAML_PATH)
        model.model.cuda().eval()
        x = torch.zeros(1, 3, 512, 512, device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            preds, _ = model.model(x)
        assert torch.isfinite(preds).all()


# =====================================================================
# 10. Data pipeline integration test
# =====================================================================

@pytest.mark.skipif(not DATA_YAML.exists(), reason="Dataset not found")
class TestDataIntegration:
    """Test that the model works with the actual dataset config."""

    def test_data_yaml_exists(self):
        assert DATA_YAML.exists()

    def test_data_yaml_valid(self):
        """data.yaml has required fields."""
        import yaml
        with open(DATA_YAML) as f:
            d = yaml.safe_load(f)
        assert "nc" in d
        assert "names" in d
        assert d["nc"] == 1

    def test_val_images_exist(self):
        """Validation images directory is non-empty."""
        val_dir = REPO / "data" / "dataset2_split_90_10" / "valid" / "images"
        assert val_dir.exists()
        images = list(val_dir.glob("*"))
        assert len(images) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
