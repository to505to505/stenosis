"""Comprehensive unit tests for Stenosis-DetNet."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from stenosis_detnet.config import Config


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = Config()
        self.assertEqual(cfg.T, 9)
        self.assertEqual(cfg.img_h, 512)
        self.assertEqual(cfg.img_w, 512)
        self.assertAlmostEqual(cfg.pixel_mean, 0.394)
        self.assertAlmostEqual(cfg.pixel_std, 0.181)
        self.assertAlmostEqual(cfg.lr, 0.002)
        self.assertEqual(cfg.epochs, 50)
        self.assertEqual(cfg.early_stopping_patience, 5)
        self.assertEqual(cfg.batch_size, 2)
        self.assertAlmostEqual(cfg.momentum, 0.9)

    def test_sff_input_dim(self):
        cfg = Config()
        self.assertEqual(cfg.sff_input_dim, 256 * 7 * 7)

    def test_sff_head_dim(self):
        cfg = Config()
        self.assertEqual(cfg.sff_head_dim, 512 // 8)

    def test_sca_thresholds(self):
        cfg = Config()
        self.assertAlmostEqual(cfg.sca_t_iou, 0.2)
        self.assertEqual(cfg.sca_t_frame, 5)


class TestDataset(unittest.TestCase):
    def test_parse_filename_valid(self):
        from stenosis_detnet.dataset import parse_filename
        result = parse_filename("14_021_1_0046_bmp_jpg.rf.2d780c9aeee935f6ed8997efbaa84b1d.jpg")
        self.assertIsNotNone(result)
        patient, seq, frame = result
        self.assertEqual(patient, "14_021")
        self.assertEqual(seq, 1)
        self.assertEqual(frame, 46)

    def test_parse_filename_invalid(self):
        from stenosis_detnet.dataset import parse_filename
        self.assertIsNone(parse_filename("random_image.jpg"))
        self.assertIsNone(parse_filename(""))

    def test_build_windows_short_seq(self):
        from stenosis_detnet.dataset import build_windows
        paths = [Path(f"frame_{i}.jpg") for i in range(5)]
        sequences = [("p1", 1, paths)]
        windows = build_windows(sequences, T=9)
        self.assertEqual(len(windows), 1)
        self.assertEqual(len(windows[0]), 9)
        # Last 4 should be padded with the last frame
        self.assertEqual(windows[0][-1], paths[-1])

    def test_build_windows_exact_length(self):
        from stenosis_detnet.dataset import build_windows
        paths = [Path(f"frame_{i}.jpg") for i in range(9)]
        sequences = [("p1", 1, paths)]
        windows = build_windows(sequences, T=9)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0], paths)

    def test_build_windows_sliding(self):
        from stenosis_detnet.dataset import build_windows
        paths = [Path(f"frame_{i}.jpg") for i in range(12)]
        sequences = [("p1", 1, paths)]
        windows = build_windows(sequences, T=9)
        # 12 - 9 + 1 = 4 windows
        self.assertEqual(len(windows), 4)

    def test_load_yolo_labels_empty(self):
        from stenosis_detnet.dataset import load_yolo_labels
        result = load_yolo_labels(Path("/nonexistent.txt"), 512, 512)
        self.assertEqual(result.shape, (0, 5))

    def test_collate_fn(self):
        from stenosis_detnet.dataset import collate_fn
        B, T = 2, 3
        batch = []
        for _ in range(B):
            images = torch.randn(T, 1, 64, 64)
            targets = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)} for _ in range(T)]
            batch.append((images, targets))
        imgs, tgts = collate_fn(batch)
        self.assertEqual(imgs.shape, (B, T, 1, 64, 64))
        self.assertEqual(len(tgts), B)


class TestBackbone(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.device = torch.device("cpu")

    def test_channel_adapter(self):
        from stenosis_detnet.model.backbone import Backbone
        backbone = Backbone(self.cfg)
        # Weight should be initialized to 1/3
        w = backbone.channel_adapter.weight.data
        self.assertTrue(torch.allclose(w, torch.full_like(w, 1.0 / 3.0)))

    def test_output_shapes(self):
        from stenosis_detnet.model.backbone import Backbone
        backbone = Backbone(self.cfg)
        backbone.eval()
        x = torch.randn(1, 1, 512, 512)
        with torch.no_grad():
            features = backbone(x)
        self.assertIn("0", features)
        self.assertIn("3", features)
        self.assertEqual(features["0"].shape, (1, 256, 128, 128))
        self.assertEqual(features["1"].shape, (1, 256, 64, 64))
        self.assertEqual(features["2"].shape, (1, 256, 32, 32))
        self.assertEqual(features["3"].shape, (1, 256, 16, 16))


class TestFeatureAdaption(unittest.TestCase):
    def test_output_shape(self):
        from stenosis_detnet.model.feature_adaption import FeatureAdaption
        fa = FeatureAdaption(256, 256, kernel_size=3, deform_groups=4)
        x = torch.randn(2, 256, 16, 16)
        shape = torch.randn(2, 2, 16, 16)
        out = fa(x, shape)
        self.assertEqual(out.shape, (2, 256, 16, 16))


class TestGARPN(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.device = torch.device("cpu")

    def test_square_anchors(self):
        from stenosis_detnet.model.ga_rpn import generate_square_anchors
        anchors = generate_square_anchors(4, 4, stride=8, scale=4, device=self.device)
        self.assertEqual(anchors.shape, (16, 4))

    def test_guided_anchor_transform(self):
        from stenosis_detnet.model.ga_rpn import generate_square_anchors, guided_anchor_transform
        anchors = generate_square_anchors(2, 2, stride=8, scale=4, device=self.device)
        shape = torch.zeros(4, 2)  # No deformation
        transformed = guided_anchor_transform(anchors, shape, stride=8)
        self.assertTrue(torch.allclose(anchors, transformed, atol=1e-5))

    def test_bounded_iou_loss(self):
        from stenosis_detnet.model.ga_rpn import bounded_iou_loss
        pred = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        target = torch.tensor([[10, 10, 30, 30]], dtype=torch.float32)
        loss = bounded_iou_loss(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_focal_loss(self):
        from stenosis_detnet.model.ga_rpn import focal_loss
        pred = torch.tensor([10.0, -10.0])
        target = torch.tensor([1.0, 0.0])
        loss = focal_loss(pred, target)
        self.assertGreater(loss.item(), 0)

    def test_encode_decode_roundtrip(self):
        from stenosis_detnet.model.ga_rpn import GuidedAnchoringRPN
        gt = torch.tensor([[50., 50., 100., 100.]])
        proposals = torch.tensor([[45., 45., 95., 95.]])
        deltas = GuidedAnchoringRPN._encode_deltas(gt, proposals)
        decoded = GuidedAnchoringRPN._decode_deltas(deltas, proposals)
        self.assertTrue(torch.allclose(decoded, gt, atol=1e-3))


class TestSFF(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_output_shape(self):
        from stenosis_detnet.model.sff import SequenceFeatureFusion
        sff = SequenceFeatureFusion(self.cfg)
        S, T = 10, self.cfg.T
        roi_features = torch.randn(S, T, 256, 7, 7)
        output = sff(roi_features)
        self.assertEqual(output.shape, (S, T, 256, 7, 7))

    def test_residual_connection(self):
        from stenosis_detnet.model.sff import SequenceFeatureFusion
        sff = SequenceFeatureFusion(self.cfg)
        sff.eval()
        S, T = 5, self.cfg.T
        roi_features = torch.randn(S, T, 256, 7, 7)
        output = sff(roi_features)
        # Output should not be identical to input due to attention
        self.assertFalse(torch.allclose(output, roi_features, atol=1e-3))

    def test_attention_dimensions(self):
        from stenosis_detnet.model.sff import SequenceFeatureFusion
        sff = SequenceFeatureFusion(self.cfg)
        # Check d_model divides evenly by num_heads
        self.assertEqual(self.cfg.sff_d_model % self.cfg.sff_num_heads, 0)


class TestHeads(unittest.TestCase):
    def test_output_shapes(self):
        cfg = Config()
        from stenosis_detnet.model.heads import DetectionHeads
        heads = DetectionHeads(cfg)
        x = torch.randn(20, 256, 7, 7)
        cls_logits, box_deltas = heads(x)
        self.assertEqual(cls_logits.shape, (20, 2))
        self.assertEqual(box_deltas.shape, (20, 4))

    def test_dropout_present(self):
        cfg = Config()
        from stenosis_detnet.model.heads import DetectionHeads
        heads = DetectionHeads(cfg)
        has_dropout = any(isinstance(m, torch.nn.Dropout) for m in heads.cls_head.modules())
        self.assertTrue(has_dropout)


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()
        self.cfg.S = 50  # Fewer proposals for faster tests
        self.device = torch.device("cpu")

    def _build_dummy_batch(self, B=1, T=None):
        T = T or self.cfg.T
        images = torch.randn(B, T, 1, self.cfg.img_h, self.cfg.img_w)
        targets = []
        for b in range(B):
            frame_targets = []
            for t in range(T):
                frame_targets.append({
                    "boxes": torch.tensor([[100., 100., 130., 130.]]),
                    "labels": torch.tensor([0], dtype=torch.int64),
                })
            targets.append(frame_targets)
        return images, targets

    def test_training_loss(self):
        from stenosis_detnet.model.detector import StenosisDetNet
        model = StenosisDetNet(self.cfg)
        model.train()
        images, targets = self._build_dummy_batch(B=1)
        losses = model(images, targets)
        self.assertIn("total_loss", losses)
        self.assertIn("det_cls_loss", losses)
        self.assertIn("det_reg_loss", losses)
        self.assertGreater(losses["total_loss"].item(), 0)

    def test_inference_output(self):
        from stenosis_detnet.model.detector import StenosisDetNet
        model = StenosisDetNet(self.cfg)
        model.eval()
        images, _ = self._build_dummy_batch(B=1)
        with torch.no_grad():
            results = model(images, None)
        self.assertEqual(len(results), self.cfg.T)
        for r in results:
            self.assertIn("boxes", r)
            self.assertIn("scores", r)
            self.assertIn("labels", r)
            self.assertEqual(r["boxes"].shape[1], 4)

    def test_empty_gt(self):
        from stenosis_detnet.model.detector import StenosisDetNet
        model = StenosisDetNet(self.cfg)
        model.train()
        images = torch.randn(1, self.cfg.T, 1, self.cfg.img_h, self.cfg.img_w)
        targets = [[{
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.int64),
        } for _ in range(self.cfg.T)]]
        losses = model(images, targets)
        self.assertIn("total_loss", losses)


class TestBoxEncoding(unittest.TestCase):
    def test_roundtrip(self):
        from stenosis_detnet.model.detector import encode_boxes, decode_boxes
        gt = torch.tensor([[50., 50., 100., 100.]])
        proposals = torch.tensor([[45., 45., 95., 95.]])
        deltas = encode_boxes(gt, proposals)
        decoded = decode_boxes(deltas, proposals)
        self.assertTrue(torch.allclose(decoded, gt, atol=1e-3))

    def test_identity(self):
        from stenosis_detnet.model.detector import encode_boxes
        boxes = torch.tensor([[50., 50., 100., 100.]])
        deltas = encode_boxes(boxes, boxes)
        # dx, dy should be 0; dw, dh should be 0
        self.assertTrue(torch.allclose(deltas, torch.zeros_like(deltas), atol=1e-5))


class TestCenterline(unittest.TestCase):
    def test_snap_to_empty_skeleton(self):
        from stenosis_detnet.centerline import snap_to_centerline
        skeleton = np.zeros((64, 64), dtype=bool)
        result = snap_to_centerline((32, 32), skeleton)
        self.assertEqual(result, (32, 32))

    def test_snap_to_skeleton(self):
        from stenosis_detnet.centerline import snap_to_centerline
        skeleton = np.zeros((64, 64), dtype=bool)
        skeleton[30, 35] = True
        result = snap_to_centerline((32, 32), skeleton)
        self.assertEqual(result, (30, 35))

    def test_snap_bbox_center(self):
        from stenosis_detnet.centerline import snap_bbox_center_to_centerline
        skeleton = np.zeros((64, 64), dtype=bool)
        skeleton[20, 20] = True
        bbox = np.array([10., 10., 30., 30.], dtype=np.float32)
        snapped = snap_bbox_center_to_centerline(bbox, skeleton)
        cx = (snapped[0] + snapped[2]) / 2
        cy = (snapped[1] + snapped[3]) / 2
        self.assertAlmostEqual(cx, 20.0, places=1)
        self.assertAlmostEqual(cy, 20.0, places=1)


class TestSCA(unittest.TestCase):
    def test_iou_clustering(self):
        from stenosis_detnet.postprocess import compute_iou
        box_a = np.array([10, 10, 30, 30])
        box_b = np.array([15, 15, 35, 35])
        iou = compute_iou(box_a, box_b)
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)

    def test_iou_identical(self):
        from stenosis_detnet.postprocess import compute_iou
        box = np.array([10, 10, 30, 30])
        self.assertAlmostEqual(compute_iou(box, box), 1.0)

    def test_iou_no_overlap(self):
        from stenosis_detnet.postprocess import compute_iou
        box_a = np.array([10, 10, 20, 20])
        box_b = np.array([30, 30, 40, 40])
        self.assertAlmostEqual(compute_iou(box_a, box_b), 0.0)

    def test_center_distance(self):
        from stenosis_detnet.postprocess import box_center_distance
        box_a = np.array([0, 0, 10, 10])
        box_b = np.array([10, 0, 20, 10])
        dist = box_center_distance(box_a, box_b)
        self.assertAlmostEqual(dist, 10.0)

    def test_temporal_filtering(self):
        from stenosis_detnet.postprocess import sca_postprocess
        cfg = Config()
        cfg.sca_t_frame = 3
        # Sequence with detection in 4 out of 9 frames
        detections = []
        for t in range(9):
            if t < 4:
                detections.append({
                    "boxes": np.array([[10., 10., 30., 30.]]),
                    "scores": np.array([0.9]),
                })
            else:
                detections.append({
                    "boxes": np.zeros((0, 4), dtype=np.float32),
                    "scores": np.zeros(0, dtype=np.float32),
                })
        result = sca_postprocess(detections, cfg=cfg, skip_centerline=True)
        self.assertEqual(len(result), 9)

    def test_union_find(self):
        from stenosis_detnet.postprocess import UnionFind
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        self.assertEqual(uf.find(0), uf.find(2))
        self.assertNotEqual(uf.find(0), uf.find(3))

    def test_empty_detections(self):
        from stenosis_detnet.postprocess import sca_postprocess
        detections = [{"boxes": np.zeros((0, 4), dtype=np.float32),
                       "scores": np.zeros(0, dtype=np.float32)} for _ in range(9)]
        result = sca_postprocess(detections, skip_centerline=True)
        self.assertEqual(len(result), 9)
        for r in result:
            self.assertEqual(r["boxes"].shape[0], 0)


class TestInitWeights(unittest.TestCase):
    def test_preserves_backbone(self):
        cfg = Config()
        cfg.S = 50
        from stenosis_detnet.model.detector import StenosisDetNet
        model = StenosisDetNet(cfg)
        # Store backbone weights before init
        before = model.backbone.resnet_fpn.layer1[0].conv1.weight.data.clone()
        model.init_weights()
        after = model.backbone.resnet_fpn.layer1[0].conv1.weight.data
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main(verbosity=2)
