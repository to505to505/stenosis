"""Video RF-DETR model.

Pipeline (Phases 2–5 of the spec):

  1. Backbone + transformer encoder process every frame **independently**
     by reshaping ``(B, T, 3, H, W) → (B*T, 3, H, W)``. No
     ``F.unfold`` / pixel-level temporal mixing is performed before the
     decoder.

  2. The standard RF-DETR decoder produces a *first-pass* set of
     predictions for every frame: ``hs[L, B*T, Q, D]``,
     ``ref_unsigmoid[B*T, Q, 4]``.

  3. STFS (``stfs.track_queries`` + ``stfs.inject_features``) chains
     confident slots into tracks across the T frames and overwrites
     Hypothesis-False-Negative slot embeddings + reference points with
     the strongest in-track frame's counterparts.

  4. A *refinement* deformable decoder layer (warm-init deepcopy of the
     last RF-DETR decoder layer) re-runs cross-attention over the
     per-frame memory using the enriched queries, producing the final
     ``pred_logits[B,T,Q,K]`` / ``pred_boxes[B,T,Q,4]``.

The KD-DETR ``query_mode="teacher"`` / ``"general"`` branches reuse the
existing slot-injection / decoder-output capture hooks for per-frame
CRRCD distillation; STFS + refinement are skipped in those branches.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.models.weights import load_pretrain_weights
from rfdetr.models.lwdetr import build_model_from_config, build_criterion_from_config
from rfdetr.models.transformer import gen_sineembed_for_position
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

from .config import Config
from .stfs import track_queries, inject_features, FeatureAggregator, RefPointShift


# ─────────────────────────────────────────────────────────────────────
#  Early Temporal Fusion (ETF)
# ─────────────────────────────────────────────────────────────────────
class EarlyTemporalFusion(nn.Module):
    """Lightweight temporal self-attention applied to backbone feature maps.

    Placed between the DINOv2 backbone output and the RF-DETR transformer
    encoder so that every spatial location accumulates cross-frame
    information *before* query slots are formed.

    For each feature level ``src`` of shape ``(B*T, C, h, w)``:

      1. Reshape → ``(B*h*w, T, C)``  (treat every spatial cell as a
         "sequence" of T frame tokens).
      2. Pre-norm → temporal multi-head self-attention → residual add.
      3. Reshape back → ``(B*T, C, h, w)``.

    The ``out_proj`` weight and bias of the attention layer are
    zero-initialised so the module starts as an exact identity mapping
    (warm-start safe, no geometric distortion on epoch 0).

    Parameters
    ----------
    d_model : int
        Channel dimension of the projected backbone features (256 for
        RFDETRSmallConfig with P4 projector).
    n_heads : int
        Number of temporal attention heads (default 8).
    dropout : float
        Attention dropout probability (default 0.0).
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Zero-init out_proj → identity at start (residual = 0)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(
        self,
        srcs: List[torch.Tensor],
        B: int,
        T: int,
    ) -> List[torch.Tensor]:
        """Apply temporal self-attention to each feature level.

        Args:
            srcs:  List of ``(B*T, C, h, w)`` feature tensors (one per
                   feature scale; typically just one level P4 for Small).
            B:     Batch size.
            T:     Temporal window size.

        Returns:
            List of ``(B*T, C, h, w)`` tensors with temporal context
            mixed in.
        """
        enriched = []
        for src in srcs:
            BT, C, h, w = src.shape
            # (B*T, C, h, w) → (B, T, C, h, w) → (B, h, w, T, C) → (B*h*w, T, C)
            x = src.reshape(B, T, C, h, w)
            x = x.permute(0, 3, 4, 1, 2).contiguous()   # (B, h, w, T, C)
            x = x.reshape(B * h * w, T, C)               # (B*h*w, T, C)

            # Pre-norm temporal self-attention with residual
            x_n = self.norm(x)
            attn_out, _ = self.attn(x_n, x_n, x_n)
            x = x + attn_out                              # (B*h*w, T, C)

            # Reshape back to (B*T, C, h, w)
            x = x.reshape(B, h, w, T, C)
            x = x.permute(0, 3, 4, 1, 2).contiguous()   # (B, T, C, h, w)
            x = x.reshape(B * T, C, h, w)
            enriched.append(x)
        return enriched


# ─────────────────────────────────────────────────────────────────────
#  Pretrained loader (mirrors rfdetr_temporal.model._build_rfdetr…)
# ─────────────────────────────────────────────────────────────────────
def _build_rfdetr_from_checkpoint(cfg: Config) -> nn.Module:
    model_cfg = RFDETRSmallConfig(num_classes=cfg.num_classes)
    lwdetr = build_model_from_config(model_cfg)

    ckpt_path = Path(cfg.rfdetr_checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        model_sd = lwdetr.state_dict()
        filtered, skipped = {}, []
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        msg = lwdetr.load_state_dict(filtered, strict=False)
        print(
            f"[VideoRFDETR] Loaded fine-tuned checkpoint '{ckpt_path.name}'  "
            f"loaded={len(filtered)}  missing={len(msg.missing_keys)}  "
            f"skipped(shape)={len(skipped)}"
        )
    else:
        print(
            f"[VideoRFDETR] Checkpoint '{cfg.rfdetr_checkpoint}' not found – "
            f"downloading Small pretrained weights …"
        )
        load_pretrain_weights(lwdetr, model_cfg)
        print("[VideoRFDETR] Small pretrained weights loaded.")
    return lwdetr


def build_criterion(cfg: Config):
    """Build SetCriterion + PostProcess using RF-DETR defaults."""
    model_cfg = RFDETRSmallConfig(num_classes=cfg.num_classes)
    train_cfg = TrainConfig(dataset_dir=".", output_dir=".")
    criterion, postprocessors = build_criterion_from_config(model_cfg, train_cfg)
    return criterion, postprocessors


# ─────────────────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────────────────
class VideoRFDETR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T
        self.centre = cfg.T // 2

        lwdetr = _build_rfdetr_from_checkpoint(cfg)
        self.backbone = lwdetr.backbone
        self.transformer = lwdetr.transformer
        self.class_embed = lwdetr.class_embed
        self.bbox_embed = lwdetr.bbox_embed
        self.refpoint_embed = lwdetr.refpoint_embed
        self.query_feat = lwdetr.query_feat
        self.num_queries = lwdetr.num_queries
        self.group_detr = lwdetr.group_detr
        self.aux_loss = lwdetr.aux_loss
        self.two_stage = lwdetr.two_stage
        self.bbox_reparam = lwdetr.bbox_reparam
        self.lite_refpoint_refine = lwdetr.lite_refpoint_refine

        if self.two_stage:
            self.transformer.enc_out_bbox_embed = lwdetr.transformer.enc_out_bbox_embed
            self.transformer.enc_out_class_embed = lwdetr.transformer.enc_out_class_embed

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if cfg.freeze_decoder:
            for p in self.transformer.parameters():
                p.requires_grad = False
            for p in self.class_embed.parameters():
                p.requires_grad = False
            for p in self.bbox_embed.parameters():
                p.requires_grad = False

        # ── Refinement decoder layer (warm-init from last layer) ────
        # Deepcopy so its parameters are independent and trainable
        # regardless of ``freeze_decoder``.
        self.refine_layer = copy.deepcopy(self.transformer.decoder.layers[-1])
        # Refinement layer always runs ungrouped (group_detr=1) so it
        # cross-attends to the full Q query set without the SA group split.
        self.refine_layer.group_detr = 1
        for p in self.refine_layer.parameters():
            p.requires_grad = True
        self.refine_norm = copy.deepcopy(self.transformer.decoder.norm)
        for p in self.refine_norm.parameters():
            p.requires_grad = True
        self.refine_ref_point_head = copy.deepcopy(
            self.transformer.decoder.ref_point_head
        )
        for p in self.refine_ref_point_head.parameters():
            p.requires_grad = True

        # ── Early Temporal Fusion (ETF) ────────────────────────────────
        # Lightweight temporal self-attention applied to backbone feature
        # maps before the RF-DETR encoder / decoder see them.  Disabled by
        # default (etf_enabled=False); when enabled the module is trainable
        # and is kept separate from the frozen backbone.
        if cfg.etf_enabled:
            self.etf: Optional[EarlyTemporalFusion] = EarlyTemporalFusion(
                d_model=cfg.hidden_dim,
                n_heads=cfg.etf_heads,
                dropout=cfg.etf_dropout,
            )
        else:
            self.etf = None

        # ── STFS soft aggregator + proposal-shift refpoint grid ────────
        # Replaces the original hard torch.where injection. FeatureAggregator
        # is zero-init on its attention out_proj. RefPointShift is now a
        # deterministic 5-point grid (centre/up/down/left/right), not a
        # blind MLP offset regressor.
        if cfg.stfs_aggregator_enabled:
            self.stfs_aggregator: Optional[FeatureAggregator] = FeatureAggregator(
                d_model=cfg.hidden_dim,
                n_heads=cfg.stfs_aggregator_heads,
                dropout=cfg.stfs_aggregator_dropout,
            )
        else:
            self.stfs_aggregator = None
        if cfg.stfs_shifter_enabled:
            self.stfs_shifter: Optional[RefPointShift] = RefPointShift(
                d_model=cfg.hidden_dim,
                hidden_dim=cfg.stfs_shifter_hidden_dim,
                padding_alpha=cfg.stfs_shifter_padding_alpha,
            )
        else:
            self.stfs_shifter = None

        # ── KD-DETR teacher-query buffers ───────────────────────────
        self._has_teacher_queries: bool = False
        self.register_buffer("teacher_refpoint", torch.zeros(1, 4), persistent=False)
        self.register_buffer(
            "teacher_query_feat", torch.zeros(1, cfg.hidden_dim), persistent=False,
        )

        # ── Decoder-input injection hook (KD slot alignment) ────────
        self._inject_decoder_inputs: Optional[Dict[str, torch.Tensor]] = None

        def _inject_pre_hook(_m, args, kwargs):
            inj = self._inject_decoder_inputs
            if inj is None:
                return None
            new_args = (inj["tgt"], *args[1:])
            new_kwargs = dict(kwargs)
            new_kwargs["refpoints_unsigmoid"] = inj["refpoints"]
            return new_args, new_kwargs

        self.transformer.decoder.register_forward_pre_hook(
            _inject_pre_hook, with_kwargs=True,
        )

        # ── Decoder-output capture (CRRCD) ──────────────────────────
        self._captured_decoder_hs: Optional[torch.Tensor] = None
        # Post-refinement hidden state capture (E1 — CRRCD on the
        # tensor that actually feeds the inference heads).
        self._captured_refined_hs: Optional[torch.Tensor] = None
        # STFS feature-alignment capture: enriched embeddings produced by
        # inject_features plus a mask of slots modified by STFS.
        self._captured_stfs_hs: Optional[torch.Tensor] = None
        self._captured_stfs_mask: Optional[torch.Tensor] = None

        def _capture_hs_post_hook(_m, _a, _kw, output):
            if isinstance(output, (list, tuple)) and len(output) >= 1:
                hs = output[0]
            else:
                hs = output
            if hs is not None:
                self._captured_decoder_hs = hs[-1]
            return None

        self.transformer.decoder.register_forward_hook(
            _capture_hs_post_hook, with_kwargs=True,
        )

        # ── Last-decoder-layer cross-attention input capture ────────
        # Needed so the refinement layer can re-run deformable cross-
        # attention over the same per-frame ``memory`` / ``spatial_shapes``
        # / ``level_start_index`` / ``valid_ratios`` the first pass used.
        self._captured_cross_inputs: Dict[str, Any] = {}

        def _capture_cross_inputs_pre_hook(_m, args, kwargs):
            # Decoder layer signature (see TransformerDecoderLayer.forward):
            #   forward(tgt, memory, *, ..., reference_points, spatial_shapes,
            #            level_start_index)
            # ``memory`` is positional[1] from TransformerDecoder.forward.
            self._captured_cross_inputs["memory"] = args[1] if len(args) > 1 \
                else kwargs.get("memory")
            self._captured_cross_inputs["memory_key_padding_mask"] = (
                kwargs.get("memory_key_padding_mask")
            )
            self._captured_cross_inputs["spatial_shapes"] = (
                kwargs.get("spatial_shapes")
            )
            self._captured_cross_inputs["level_start_index"] = (
                kwargs.get("level_start_index")
            )
            return None

        self.transformer.decoder.layers[-1].register_forward_pre_hook(
            _capture_cross_inputs_pre_hook, with_kwargs=True,
        )

        # Note: ``valid_ratios`` are reconstructed in :meth:`_refinement_pass`
        # from the captured spatial shapes (our pipeline never pads memory,
        # so every ratio is 1.0). Re-derivation avoids plumbing masks
        # through an extra hook.

    # ─────────────────────────────────────────────────────────────────
    #  KD helpers (unchanged from the temporal package)
    # ─────────────────────────────────────────────────────────────────
    def register_teacher_queries(
        self, refpoint_w: torch.Tensor, query_feat_w: torch.Tensor,
    ) -> None:
        device = self.refpoint_embed.weight.device
        self.teacher_refpoint = refpoint_w.detach().to(device).clone()
        self.teacher_query_feat = query_feat_w.detach().to(device).clone()
        self._has_teacher_queries = True

    def _swap_group_detr(self, new_group: int, new_num_queries: int):
        tr = self.transformer
        orig_nq_t = getattr(tr, "num_queries", None)
        patched: List[Tuple[nn.Module, int]] = []
        for m in tr.modules():
            if hasattr(m, "group_detr"):
                patched.append((m, int(m.group_detr)))
                m.group_detr = int(new_group)
        if orig_nq_t is not None:
            tr.num_queries = int(new_num_queries)

        def restore() -> None:
            for m, g in patched:
                m.group_detr = g
            if orig_nq_t is not None:
                tr.num_queries = orig_nq_t

        return restore

    def sample_general_queries(
        self, num_queries: int, device, dtype=torch.float32,
    ) -> Dict[str, torch.Tensor]:
        std = float(self.cfg.distill_general_query_std)
        refpoint = torch.zeros(num_queries, 4, device=device, dtype=dtype)
        query_feat = torch.randn(
            num_queries, int(self.cfg.hidden_dim), device=device, dtype=dtype,
        ) * std
        return {"refpoint": refpoint, "query_feat": query_feat}

    # ─────────────────────────────────────────────────────────────────
    #  Forward
    # ─────────────────────────────────────────────────────────────────
    def _run_backbone(self, frames_bt: torch.Tensor):
        """frames_bt: (BT, 3, H, W) → (srcs, masks, poss)."""
        nested = nested_tensor_from_tensor_list(frames_bt)
        with torch.set_grad_enabled(not self.cfg.freeze_backbone):
            features, poss = self.backbone(nested)
        srcs, masks = [], []
        for feat in features:
            src, mask = feat.decompose()
            srcs.append(src)
            masks.append(mask)
        return srcs, masks, poss

    def _heads(
        self, hs_last: torch.Tensor, ref_unsigmoid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """hs_last: (BT, Q, D), ref_unsigmoid: (BT, Q, 4)."""
        if self.bbox_reparam:
            delta = self.bbox_embed(hs_last)
            cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
            outputs_coord = torch.cat([cxcy, wh], dim=-1)
        else:
            outputs_coord = (self.bbox_embed(hs_last) + ref_unsigmoid).sigmoid()
        outputs_class = self.class_embed(hs_last)
        return outputs_class, outputs_coord

    def _aux_outputs(self, outputs_class_stack, outputs_coord_stack):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class_stack[:-1], outputs_coord_stack[:-1])
        ]

    def _refinement_pass(
        self,
        query_embed: torch.Tensor,        # (B, T, Q, D)
        refpoints_unsigmoid: torch.Tensor,  # (B, T, Q, 4)
        candidate_refpoints_unsigmoid: Optional[torch.Tensor] = None,  # (B,T,Q,5,4)
        candidate_mask: Optional[torch.Tensor] = None,  # (B,T,Q)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run a single deformable-decoder layer over per-frame memory."""
        cap = self._captured_cross_inputs
        memory = cap.get("memory")
        spatial_shapes = cap.get("spatial_shapes")
        level_start_index = cap.get("level_start_index")
        memory_key_padding_mask = cap.get("memory_key_padding_mask")
        if memory is None or spatial_shapes is None:
            raise RuntimeError(
                "Refinement pass requires the first decoder pass to have "
                "captured memory + spatial_shapes via the layer hook."
            )
        BT = memory.shape[0]
        B, T, Q, D = query_embed.shape
        assert BT == B * T, f"memory BT={BT} vs B*T={B*T}"

        # Build geometric inputs analogous to TransformerDecoder.forward.
        if self.bbox_reparam:
            obj_center = refpoints_unsigmoid.reshape(BT, Q, 4)
        else:
            obj_center = refpoints_unsigmoid.sigmoid().reshape(BT, Q, 4)

        # valid_ratios: shape (BT, num_levels, 2). Memory has no padding in
        # our pipeline (dense feature maps), so every ratio is 1.0; we
        # construct it directly to avoid plumbing masks through.
        n_levels = spatial_shapes.shape[0]
        valid_ratios = torch.ones(
            BT, n_levels, 2, device=memory.device, dtype=memory.dtype,
        )

        refpoints_input = (
            obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        )
        query_sine_embed = gen_sineembed_for_position(
            refpoints_input[:, :, 0, :], self.transformer.d_model / 2,
        )
        query_pos = self.refine_ref_point_head(query_sine_embed)

        tgt = query_embed.reshape(BT, Q, D)
        # Refinement layer: SA + Cross-attn + FFN. We force group_detr=1
        # by setting it on the layer in __init__.
        was_training = self.refine_layer.training
        # Layer SA splits the queries by group_detr only when in train mode.
        # group_detr=1 disables the split in either case.
        out = self.refine_layer(
            tgt,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
            query_pos=query_pos,
            query_sine_embed=query_sine_embed,
            is_first=False,
            reference_points=refpoints_input,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        out = self.refine_norm(out)              # (BT, Q, D)
        outputs_class, outputs_coord = self._heads(out, obj_center)

        # Proposal-shifted sparse refinement: for STFS-injected slots only,
        # run deformable cross-attention over five explicit spatial hypotheses
        # (centre/up/down/left/right), then softly collapse by foreground
        # confidence. This replaces blind Δcxcywh regression with visual
        # evidence from the refinement layer itself.
        if candidate_refpoints_unsigmoid is not None and candidate_mask is not None:
            mask_flat = candidate_mask.reshape(BT, Q).to(device=memory.device, dtype=torch.bool)
            if mask_flat.any():
                candidate_obj = candidate_refpoints_unsigmoid.reshape(BT, Q, 5, 4)
                candidate_obj = candidate_obj.to(device=memory.device, dtype=memory.dtype)
                if not self.bbox_reparam:
                    candidate_obj = candidate_obj.sigmoid()

                bt_idx = mask_flat.nonzero(as_tuple=False)[:, 0]
                selected_tgt = tgt[mask_flat].unsqueeze(1).expand(-1, 5, -1).contiguous()
                selected_refs = candidate_obj[mask_flat]
                selected_memory = memory.index_select(0, bt_idx)
                if memory_key_padding_mask is not None:
                    selected_padding = memory_key_padding_mask.index_select(0, bt_idx)
                else:
                    selected_padding = None
                selected_ratios = valid_ratios.index_select(0, bt_idx)
                selected_refpoints_input = (
                    selected_refs[:, :, None]
                    * torch.cat([selected_ratios, selected_ratios], -1)[:, None]
                )
                selected_sine = gen_sineembed_for_position(
                    selected_refpoints_input[:, :, 0, :], self.transformer.d_model / 2,
                )
                selected_query_pos = self.refine_ref_point_head(selected_sine)
                selected_out = self.refine_layer(
                    selected_tgt,
                    selected_memory,
                    memory_key_padding_mask=selected_padding,
                    query_pos=selected_query_pos,
                    query_sine_embed=selected_sine,
                    is_first=False,
                    reference_points=selected_refpoints_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                )
                selected_out = self.refine_norm(selected_out)  # (N, 5, D)
                cand_class, cand_coord = self._heads(selected_out, selected_refs)
                cand_score = cand_class.sigmoid().amax(dim=-1)
                cand_weight = torch.softmax(cand_score / 0.1, dim=1).unsqueeze(-1)

                out = out.clone()
                outputs_class = outputs_class.clone()
                outputs_coord = outputs_coord.clone()
                out[mask_flat] = (selected_out * cand_weight).sum(dim=1)
                outputs_class[mask_flat] = (cand_class * cand_weight).sum(dim=1)
                outputs_coord[mask_flat] = (cand_coord * cand_weight).sum(dim=1)

        # Capture post-refinement hidden state so CRRCD can hook the exact
        # tensor consumed by the inference heads (E1), including candidate
        # replacements for STFS-injected slots.
        self._captured_refined_hs = out
        # Reshape back to (B, T, Q, *).
        outputs_class = outputs_class.reshape(B, T, Q, -1)
        outputs_coord = outputs_coord.reshape(B, T, Q, 4)
        _ = was_training
        return outputs_class, outputs_coord

    def forward(
        self,
        frames: torch.Tensor,
        targets=None,
        query_mode: str = "student",
        general_queries: Optional[Dict[str, torch.Tensor]] = None,
        decoder_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Forward.

        Args:
            frames: (B, T, 3, H, W) student frames.
            query_mode: "student" → full STFS + refinement. "teacher" /
                "general" → per-frame slot-aligned forward used by KD/CRRCD
                branches; STFS + refinement are skipped and the output
                is ``(B*T, Q, *)``.
            general_queries / decoder_inputs: as in
                :class:`rfdetr_temporal.model.TemporalRFDETR`.

        Returns (student mode):
            ``{"pred_logits": (B,T,Q,K), "pred_boxes": (B,T,Q,4),
                "first_pass": {"pred_logits", "pred_boxes"},
                "aux_outputs": ..., "enc_outputs": ...}``.
        """
        assert query_mode in ("student", "teacher", "general"), query_mode
        B, T, C, H, W = frames.shape
        BT = B * T

        # Reset captures.
        self._captured_decoder_hs = None
        self._captured_refined_hs = None
        self._captured_stfs_hs = None
        self._captured_stfs_mask = None
        self._captured_cross_inputs = {}

        # ── 1. Backbone on B*T frames ────────────────────────────────
        srcs, masks, poss = self._run_backbone(
            frames.reshape(BT, C, H, W),
        )

        # ── 1b. Early Temporal Fusion (optional) ─────────────────────
        # Enrich per-level feature maps with cross-frame information via
        # temporal self-attention before the decoder forms query slots.
        if self.etf is not None:
            srcs = self.etf(srcs, B, T)

        # ── 2. Resolve queries ───────────────────────────────────────
        restore_fn = None
        if query_mode == "student":
            if self.training:
                refpoint_w = self.refpoint_embed.weight
                query_feat_w = self.query_feat.weight
            else:
                refpoint_w = self.refpoint_embed.weight[: self.num_queries]
                query_feat_w = self.query_feat.weight[: self.num_queries]
        elif query_mode == "teacher":
            assert self._has_teacher_queries, (
                "query_mode='teacher' requires register_teacher_queries() first"
            )
            refpoint_w = self.teacher_refpoint
            query_feat_w = self.teacher_query_feat
            restore_fn = self._swap_group_detr(1, refpoint_w.shape[0])
        else:  # general
            assert (
                general_queries is not None
                and "refpoint" in general_queries
                and "query_feat" in general_queries
            ), "query_mode='general' requires general_queries={'refpoint','query_feat'}"
            refpoint_w = general_queries["refpoint"]
            query_feat_w = general_queries["query_feat"]
            restore_fn = self._swap_group_detr(1, refpoint_w.shape[0])

        if decoder_inputs is not None:
            assert query_mode in ("teacher", "general")
            self._inject_decoder_inputs = {
                "tgt": decoder_inputs["tgt"].detach(),
                "refpoints": decoder_inputs["refpoints"].detach(),
            }

        # ── 3. First decoder pass on B*T frames ──────────────────────
        try:
            hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
                srcs, masks, poss, refpoint_w, query_feat_w,
            )
        finally:
            self._inject_decoder_inputs = None
            if restore_fn is not None:
                restore_fn()

        # ── 4. KD branches: return per-frame outputs (no STFS) ──────
        if query_mode != "student":
            out: Optional[Dict[str, torch.Tensor]] = None
            if hs is not None:
                outputs_class, outputs_coord = self._heads(hs[-1], ref_unsigmoid)
                # hs is (L, BT, Q, D); class_embed produces (L, BT, Q, K)
                # only on hs[-1] above; for aux we need full stack.
                stacked_class = self.class_embed(hs)
                if self.bbox_reparam:
                    delta_full = self.bbox_embed(hs)
                    cxcy = delta_full[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                    wh = delta_full[..., 2:].exp() * ref_unsigmoid[..., 2:]
                    stacked_coord = torch.cat([cxcy, wh], dim=-1)
                else:
                    stacked_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
                out = {
                    "pred_logits": stacked_class[-1],     # (BT, Q, K)
                    "pred_boxes": stacked_coord[-1],
                }
                if self.aux_loss:
                    out["aux_outputs"] = self._aux_outputs(stacked_class, stacked_coord)

                # E1: route the teacher-aligned queries through the
                # refinement layer so KD/CRRCD supervise the exact
                # tensors used at inference. STFS stays bypassed
                # (1:1 teacher↔student slot alignment is preserved).
                if getattr(self.cfg, "distill_through_refine", False):
                    BT_kd, Q_kd, D_kd = hs[-1].shape
                    assert BT_kd == B * T, (
                        f"KD-branch BT mismatch: hs[-1] has BT={BT_kd}, "
                        f"expected B*T={B*T}"
                    )
                    query_embed_btq = hs[-1].reshape(B, T, Q_kd, D_kd)
                    refpoint_btq = ref_unsigmoid.reshape(B, T, Q_kd, 4)
                    refined_class_btq, refined_coord_btq = self._refinement_pass(
                        query_embed_btq, refpoint_btq,
                    )
                    K_kd = refined_class_btq.shape[-1]
                    # KD branches expect ``(BT, Q, *)`` — flatten back.
                    out["pred_logits"] = refined_class_btq.reshape(BT_kd, Q_kd, K_kd)
                    out["pred_boxes"] = refined_coord_btq.reshape(BT_kd, Q_kd, 4)
                    # Refinement is single-layer; ``aux_outputs`` from the
                    # first decoder pass would mix unrefined predictions
                    # into the KD aux loss. Drop them on this path —
                    # ``distillation_loss`` only needs the main fields.
                    out.pop("aux_outputs", None)

            if self.two_stage and hs_enc is not None:
                hs_enc_list = hs_enc.chunk(1, dim=1)
                cls_enc = self.transformer.enc_out_class_embed[0](hs_enc_list[0])
                if out is not None:
                    out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
                else:
                    out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
            return out

        # ── 5. Student mode: build first-pass predictions ────────────
        assert hs is not None, "RF-DETR student mode must run the decoder"
        stacked_class = self.class_embed(hs)               # (L, BT, Q, K)
        if self.bbox_reparam:
            delta_full = self.bbox_embed(hs)
            cxcy = delta_full[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            wh = delta_full[..., 2:].exp() * ref_unsigmoid[..., 2:]
            stacked_coord = torch.cat([cxcy, wh], dim=-1)
        else:
            stacked_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

        first_pred_logits = stacked_class[-1]              # (BT, Q, K)
        first_pred_boxes = stacked_coord[-1]               # (BT, Q, 4)
        K = first_pred_logits.shape[-1]
        Q = first_pred_logits.shape[-2]
        D = hs.shape[-1]

        # ── 6. STFS: track + inject ─────────────────────────────────
        pred_boxes_btq = first_pred_boxes.reshape(B, T, Q, 4)
        pred_logits_btq = first_pred_logits.reshape(B, T, Q, K)
        tracks = track_queries(
            pred_boxes_btq, pred_logits_btq,
            iou_weight=self.cfg.stfs_iou_weight,
            l1_weight=self.cfg.stfs_l1_weight,
            cls_weight=self.cfg.stfs_cls_weight,
            iou_gate=self.cfg.stfs_match_iou_thresh,
            score_thresh=self.cfg.stfs_track_score_thresh,
            min_track_len=self.cfg.stfs_min_track_len,
        )

        query_embed_btq = hs[-1].reshape(B, T, Q, D)        # gradient through decoder
        refpoint_btq = ref_unsigmoid.reshape(B, T, Q, 4)
        enriched_emb, enriched_ref, shift_candidates, inject_mask = inject_features(
            query_embed_btq, refpoint_btq, tracks,
            alpha=self.cfg.stfs_inject_alpha,
            aggregator=self.stfs_aggregator,
            shifter=self.stfs_shifter,
            return_shift_candidates=True,
        )
        self._captured_stfs_hs = enriched_emb
        self._captured_stfs_mask = inject_mask

        # ── 7. Refinement pass on enriched queries ───────────────────
        outputs_class_btq, outputs_coord_btq = self._refinement_pass(
            enriched_emb, enriched_ref,
            candidate_refpoints_unsigmoid=shift_candidates,
            candidate_mask=self._captured_stfs_mask,
        )

        out = {
            "pred_logits": outputs_class_btq,    # (B, T, Q, K)
            "pred_boxes": outputs_coord_btq,
            "first_pass": {
                "pred_logits": first_pred_logits.reshape(B, T, Q, K),
                "pred_boxes": first_pred_boxes.reshape(B, T, Q, 4),
            },
        }

        if self.aux_loss:
            # Provide the first-pass intermediate decoder layers as DETR
            # aux supervision (already per-frame, shape (BT, Q, *)).
            out["aux_outputs"] = self._aux_outputs(stacked_class, stacked_coord)

        if self.two_stage and hs_enc is not None:
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g in range(group_detr):
                cls_enc.append(
                    self.transformer.enc_out_class_embed[g](hs_enc_list[g])
                )
            cls_enc = torch.cat(cls_enc, dim=1)
            out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        return out

    # ─────────────────────────────────────────────────────────────────
    #  Optimiser groups
    # ─────────────────────────────────────────────────────────────────
    def get_param_groups(self):
        backbone_params = []
        decoder_params = []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(p)
            else:
                decoder_params.append(p)
        groups = []
        if decoder_params:
            groups.append({"params": decoder_params, "lr": self.cfg.lr})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.cfg.lr_backbone})
        return groups
