"""Lightweight temporal student detector.

ConvNeXt-V2-Tiny + Temporal Shift Module (TSM) backbone, Detail-Aware
Cross-Attention FPN neck, anchor-free FCOS head.

Takes 9 consecutive frames per sample, predicts boxes for the centre frame.
"""
