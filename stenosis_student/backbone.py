"""ConvNeXt-V2-Tiny backbone with optional TSM hooks.

Loads ``facebook/dinov3-convnext-tiny-pretrain-lvd1689m`` from HuggingFace
(``transformers.AutoModel``) and exposes the per-stage feature maps.

The backbone collects activations from the four ConvNeXt stages via forward
hooks so we can pull the multi-scale features (P2/P3/P4/P5 at strides
4/8/16/32) without depending on a specific HF output schema.

When TSM is enabled, a forward-pre-hook is attached to every ConvNeXt block
(or to each stage's first block, depending on config) that performs a
parameter-free temporal channel shift over the ``T`` frames packed in the
batch dimension.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .config import Config
from .tsm import TSMState, install_tsm_hooks


def _is_convnext_block(module: nn.Module) -> bool:
    """Heuristic: a ConvNeXt block is a leaf-ish module whose class name
    contains 'Layer' or 'Block' and which lives inside a stage."""
    name = type(module).__name__.lower()
    return ("convnextv2layer" in name
            or "convnextlayer" in name
            or "convnextv2block" in name
            or "convnextblock" in name)


def _find_stages(model: nn.Module) -> List[nn.Module]:
    """Locate the four ConvNeXt stages inside an HF model object.

    HF ``ConvNextModel`` / ``ConvNextV2Model`` exposes ``encoder.stages``;
    when wrapped (e.g. by an ``AutoBackbone``) the path may differ.  We try
    a few well-known locations and fall back to a generic search.
    """
    candidates = [
        ("convnextv2", "encoder", "stages"),
        ("convnext", "encoder", "stages"),
        ("encoder", "stages"),
        ("model", "encoder", "stages"),
    ]
    for path in candidates:
        cur = model
        ok = True
        for attr in path:
            if not hasattr(cur, attr):
                ok = False
                break
            cur = getattr(cur, attr)
        if ok and isinstance(cur, (nn.ModuleList, nn.Sequential)) and len(cur) == 4:
            return list(cur)
    # Fallback: any ModuleList of length 4 of nn.Modules whose children look
    # like blocks.
    for module in model.modules():
        if isinstance(module, (nn.ModuleList, nn.Sequential)) and len(module) == 4:
            children_ok = all(any(_is_convnext_block(c) for c in stg.modules())
                              for stg in module)
            if children_ok:
                return list(module)
    raise RuntimeError("Could not locate the 4 ConvNeXt stages in the HF model")


class ConvNeXtV2TinyBackbone(nn.Module):
    """ConvNeXt-V2-Tiny backbone returning multi-scale features.

    Forward signature: ``(B*T, 3, H, W) → list[Tensor]`` of length
    ``len(cfg.fpn_stage_indices)`` with channel counts taken from
    ``cfg.stage_channels`` and spatial strides from ``cfg.stage_strides``.

    The backbone keeps activations from all four stages internally; only the
    requested ones are returned.  This makes it cheap to re-configure which
    pyramid levels feed the neck.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        from transformers import AutoModel  # local import; heavy dep

        src = cfg.backbone_local_path or cfg.hf_model_id
        self.model = AutoModel.from_pretrained(src)
        self.model.eval()  # default; .train() flips it back

        # Locate stages and (optionally) install TSM hooks
        self.stages = _find_stages(self.model)
        self._stage_outputs: List[Optional[torch.Tensor]] = [None] * 4
        for i, stg in enumerate(self.stages):
            stg.register_forward_hook(self._make_stage_hook(i))

        self.tsm_state: Optional[TSMState] = None
        if cfg.tsm_enabled:
            tsm_targets: List[nn.Module] = []
            for stg in self.stages:
                if cfg.tsm_per_block:
                    seen = set()
                    for m in stg.modules():
                        if _is_convnext_block(m) and id(m) not in seen:
                            seen.add(id(m))
                            tsm_targets.append(m)
                else:
                    tsm_targets.append(stg)
            if tsm_targets:
                self.tsm_state = install_tsm_hooks(
                    tsm_targets, T=cfg.T, fold_div=cfg.tsm_fold_div, enabled=True,
                )

        # Apply freezing
        if cfg.freeze_stem:
            self._freeze_stem()
        if cfg.freeze_stage0:
            for p in self.stages[0].parameters():
                p.requires_grad = False

        # Detect channels from a dummy forward
        self._auto_detect_channels()

    # ─── Stage activation capture ────────────────────────────────────
    def _make_stage_hook(self, idx: int):
        def _hook(_module, _inputs, output):
            self._stage_outputs[idx] = output
        return _hook

    def _freeze_stem(self) -> None:
        # Freeze the convnext embeddings/stem if it exists
        for attr_chain in [("convnextv2", "embeddings"), ("convnext", "embeddings"),
                            ("embeddings",), ("model", "embeddings")]:
            cur = self.model
            ok = True
            for a in attr_chain:
                if not hasattr(cur, a):
                    ok = False
                    break
                cur = getattr(cur, a)
            if ok:
                for p in cur.parameters():
                    p.requires_grad = False
                return

    @torch.no_grad()
    def _auto_detect_channels(self) -> None:
        was_training = self.training
        self.eval()
        # Disable TSM during the probe (T=1) to avoid divisibility issues
        prev_enabled = self.tsm_state.enabled if self.tsm_state else None
        if self.tsm_state is not None:
            self.tsm_state.enabled = False
        try:
            dummy = torch.zeros(1, 3, 64, 64)
            self._stage_outputs = [None] * 4
            self.model(dummy)
            chans = []
            for i in range(4):
                out = self._stage_outputs[i]
                if out is None:
                    raise RuntimeError(f"Stage {i} produced no output during probe")
                chans.append(int(out.shape[1]))
            self.cfg.stage_channels = tuple(chans)  # type: ignore[assignment]
        finally:
            if self.tsm_state is not None and prev_enabled is not None:
                self.tsm_state.enabled = prev_enabled
            if was_training:
                self.train()
        self._stage_outputs = [None] * 4

    @property
    def out_channels(self) -> Tuple[int, ...]:
        return tuple(self.cfg.stage_channels[i] for i in self.cfg.fpn_stage_indices)

    @property
    def out_strides(self) -> Tuple[int, ...]:
        return tuple(self.cfg.stage_strides[i] for i in self.cfg.fpn_stage_indices)

    # ─── Forward ─────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Args:
            x: ``(B*T, 3, H, W)`` (frames already folded into the batch axis).
        Returns:
            list of feature maps for the requested FPN stages.
        """
        self._stage_outputs = [None] * 4
        self.model(x)
        feats = []
        for i in self.cfg.fpn_stage_indices:
            out = self._stage_outputs[i]
            if out is None:
                raise RuntimeError(f"Stage {i} did not produce an output")
            feats.append(out)
        # Free reference list to avoid retaining tensors
        self._stage_outputs = [None] * 4
        return feats
