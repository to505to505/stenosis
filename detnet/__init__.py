"""Stenosis-DetNet — sequence-aware coronary stenosis detection.

Implementation of:
    Pang et al., "Stenosis-DetNet: Sequence consistency-based stenosis
    detection for X-ray coronary angiography",
    Computerized Medical Imaging and Graphics 89 (2021) 101900.

Key modules:
  • :class:`detnet.model.SFFBoxHead` — Sequence Feature Fusion: global
    multi-head self-attention over all candidate-box features from every
    frame in the T-frame window, with residual concat (Eqs. 1–3 / Fig. 4).
  • :class:`detnet.model.VideoFasterRCNN` — end-to-end detector wrapping a
    torchvision Faster-R-CNN backbone + RPN, with per-frame SFF + Fast-R-CNN
    cls/box-reg heads.
  • :mod:`detnet.sca` — Sequence Consistency Alignment: cross-frame
    clustering (IoU + SSIM-on-displaced-patches fallback), T_frame
    filtering, and linear corner interpolation for missing frames.

The training scaffold (config, dataset, train, evaluate) mirrors
:mod:`psstt` so the two trainers can be compared apples-to-apples.
"""
