"""Temporal RF-DETR: pre-decoder temporal feature fusion.

Architecture
────────────
1. Shared DINOv2 backbone (frozen): extracts per-frame multi-scale features
2. TemporalFusion (new, trainable): lightweight temporal self-attention that
   enriches the centre frame features with context from neighbouring frames
3. Standard RF-DETR decoder + heads (fine-tuned): produces detections
"""

import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.models.weights import load_pretrain_weights
from rfdetr.models.lwdetr import build_model_from_config, build_criterion_from_config
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list

from .config import Config


# ─────────────────────────────────────────────────────────────────────
#  Temporal fusion module (Neighbourhood Temporal Attention)
# ─────────────────────────────────────────────────────────────────────
class TemporalFusion(nn.Module):
    """Neighbourhood temporal cross-attention.

    For every centre-frame position (h, w) the module attends to a small
    spatial window (h ± k, w ± k) in *all* T frames. This compensates for
    object motion between frames (e.g. cardiac displacement in angiography),
    so the centre token is enriched even when the object is not in the
    exact same pixel location across frames.

      • Query : centre frame, single token at (h, w)         shape (1,   C)
      • Key/V : T · (2k+1)² tokens from local windows of all shape (T·K², C)
                T frames

    Implementation uses a TransformerDecoder (cross-attention from Q to KV)
    applied independently for each spatial position.
    """

    def __init__(
        self,
        hidden_dim: int,
        T: int,
        k: int = 1,
        num_layers: int = 2,
        nhead: int = 8,
    ):
        super().__init__()
        self.T = T
        self.k = k
        self.K = 2 * k + 1                      # window side length
        self.centre = T // 2

        # Positional embeddings for KV tokens: temporal index + spatial offset
        self.temporal_pos = nn.Embedding(T, hidden_dim)
        self.spatial_pos = nn.Embedding(self.K * self.K, hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, T, C, H, W) – features from backbone for one level
        Returns:
            fused: (B, C, H, W) – temporally-enriched centre-frame features
        """
        B, T, C, H, W = feats.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"
        K = self.K
        K2 = K * K
        HW = H * W

        # ── Build KV: local (K×K) neighbourhood for every (h, w) of every frame
        x = feats.reshape(B * T, C, H, W)
        # F.unfold → (B*T, C*K*K, H*W)
        unfolded = F.unfold(x, kernel_size=K, padding=self.k)
        # → (B, T, C, K*K, H*W) → (B, H*W, T, K*K, C)
        unfolded = unfolded.view(B, T, C, K2, HW).permute(0, 4, 1, 3, 2)
        kv = unfolded.reshape(B * HW, T * K2, C)

        # Add temporal + spatial positional embeddings to KV tokens
        tp = self.temporal_pos.weight                 # (T, C)
        sp = self.spatial_pos.weight                  # (K*K, C)
        pos_kv = (tp[:, None, :] + sp[None, :, :]).reshape(T * K2, C)
        kv = kv + pos_kv[None]                        # broadcast over (B*HW)

        # ── Build Q: centre frame's centre-position feature (single token)
        centre_feat = feats[:, self.centre]           # (B, C, H, W)
        q = centre_feat.permute(0, 2, 3, 1).reshape(B * HW, 1, C)

        # ── Cross-attention: Q (len 1) attends to KV (len T*K²)
        out = self.decoder(q, kv)                     # (B*HW, 1, C)

        return out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────
#  Build helpers
# ─────────────────────────────────────────────────────────────────────
def _build_rfdetr_from_checkpoint(cfg: Config) -> nn.Module:
    """Instantiate an RFDETRNano and load weights.

    Loading strategy (in order):
    1. If cfg.rfdetr_checkpoint points to an existing file → treat it as a
       fine-tuned stenosis checkpoint and load it (shape-filtered, to handle
       any class/resolution mismatch).
    2. Otherwise → use rfdetr's built-in load_pretrain_weights(), which
       auto-downloads 'rf-detr-nano.pth' from the official source exactly
       the same way the standard rf-detr training script does.
    """
    model_cfg = RFDETRSmallConfig(num_classes=cfg.num_classes)
    lwdetr = build_model_from_config(model_cfg)

    ckpt_path = Path(cfg.rfdetr_checkpoint)
    if ckpt_path.exists():
        # ── Fine-tuned stenosis checkpoint ──────────────────────────
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        # Skip keys whose shapes don't match (class head, pos embeddings …)
        model_sd = lwdetr.state_dict()
        filtered, skipped = {}, []
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)

        msg = lwdetr.load_state_dict(filtered, strict=False)
        print(f"[RF-DETR] Loaded fine-tuned checkpoint '{ckpt_path.name}'  "
              f"loaded={len(filtered)}  missing={len(msg.missing_keys)}  "
              f"skipped(shape)={len(skipped)}")
    else:
        # Auto-download official Small pretrained weights
        print(f"[RF-DETR] Checkpoint '{cfg.rfdetr_checkpoint}' not found – "
              f"downloading Small pretrained weights (rf-detr-small.pth) …")
        load_pretrain_weights(lwdetr, model_cfg)
        print("[RF-DETR] Small pretrained weights loaded.")

    return lwdetr


def _build_criterion(cfg: Config):
    """Build SetCriterion + PostProcess using RF-DETR defaults."""
    model_cfg = RFDETRSmallConfig(num_classes=cfg.num_classes)
    train_cfg = TrainConfig(dataset_dir=".", output_dir=".")
    criterion, postprocessors = build_criterion_from_config(model_cfg, train_cfg)
    return criterion, postprocessors


# ─────────────────────────────────────────────────────────────────────
#  Main model
# ─────────────────────────────────────────────────────────────────────
class TemporalRFDETR(nn.Module):
    """Temporal RF-DETR with pre-decoder feature fusion.

    The model wraps a pretrained LWDETR and injects a lightweight
    TemporalFusion module between the backbone and transformer decoder.
    Only the temporal module and (optionally) the decoder are trained;
    the DINOv2 backbone + projector stay frozen.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T
        self.centre = cfg.T // 2

        # ── pretrained RF-DETR components ───────────────────────────
        lwdetr = _build_rfdetr_from_checkpoint(cfg)
        self.backbone = lwdetr.backbone          # Joiner(Backbone, PosEmbed)
        self.transformer = lwdetr.transformer    # Transformer(decoder)
        self.class_embed = lwdetr.class_embed    # nn.Linear
        self.bbox_embed = lwdetr.bbox_embed      # MLP
        self.refpoint_embed = lwdetr.refpoint_embed  # nn.Embedding
        self.query_feat = lwdetr.query_feat          # nn.Embedding
        self.num_queries = lwdetr.num_queries
        self.group_detr = lwdetr.group_detr
        self.aux_loss = lwdetr.aux_loss
        self.two_stage = lwdetr.two_stage
        self.bbox_reparam = lwdetr.bbox_reparam
        self.lite_refpoint_refine = lwdetr.lite_refpoint_refine

        # Copy two-stage modules if present
        if self.two_stage:
            self.transformer.enc_out_bbox_embed = lwdetr.transformer.enc_out_bbox_embed
            self.transformer.enc_out_class_embed = lwdetr.transformer.enc_out_class_embed

        # ── freeze controls ─────────────────────────────────────────
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

        # ── temporal fusion (one per backbone output level) ─────────
        # RFDETRSmall with projector_scale=["P4"] produces 1 level
        n_levels = len(lwdetr.backbone[0].projector_scale)  # Backbone is backbone[0]
        self.temporal_fusions = nn.ModuleList([
            TemporalFusion(
                hidden_dim=cfg.hidden_dim,
                T=cfg.T,
                k=cfg.neighborhood_k,
                num_layers=cfg.temporal_attn_layers,
                nhead=cfg.temporal_nhead,
            )
            for _ in range(n_levels)
        ])
        # ── KD-DETR specific-sampling state ─────────────────────────
        # Filled in by ``register_teacher_queries`` once a teacher is built.
        self._has_teacher_queries: bool = False
        self.register_buffer(
            "teacher_refpoint",
            torch.zeros(1, 4),
            persistent=False,
        )
        self.register_buffer(
            "teacher_query_feat",
            torch.zeros(1, cfg.hidden_dim),
            persistent=False,
        )

        # ── Decoder-input injection hook (KD-DETR slot alignment) ───
        # When ``self._inject_decoder_inputs`` is set to a dict with keys
        # ``tgt`` (B, Q, D) and ``refpoints`` (B, Q, 4), the next forward
        # through ``self.transformer.decoder`` will use *those* tensors as
        # the per-slot decoder inputs instead of whatever the student's own
        # encoder + two-stage block computed.  This is the mechanism that
        # makes the student's deformable attention sample the same spatial
        # locations the teacher does, which is the core requirement for
        # KD-DETR slot-aligned distillation on a two-stage detector.
        self._inject_decoder_inputs: dict | None = None

        def _decoder_inject_pre_hook(_module, args, kwargs):
            inj = self._inject_decoder_inputs
            if inj is None:
                return None
            new_args = (inj["tgt"], *args[1:])
            new_kwargs = dict(kwargs)
            new_kwargs["refpoints_unsigmoid"] = inj["refpoints"]
            return new_args, new_kwargs

        self.transformer.decoder.register_forward_pre_hook(
            _decoder_inject_pre_hook, with_kwargs=True,
        )

        # ── Decoder output capture (CRRCD relational distillation) ──
        # Stores the last decoder layer's hidden state (B, Q, D) — i.e. the
        # embedding fed into the class / box heads.  Reset before every
        # forward; CRRCD reads it after the Branch-2 forward.
        self._captured_decoder_hs: torch.Tensor | None = None

        def _decoder_capture_post_hook(_module, _args, _kwargs, output):
            # rfdetr's TransformerDecoder returns [stacked_hs, stacked_refs]
            # when ``return_intermediate=True`` — see
            # rfdetr/models/transformer.py::TransformerDecoder.forward.
            if isinstance(output, (list, tuple)) and len(output) >= 1:
                hs = output[0]
            else:
                hs = output
            if hs is not None:
                # Keep the graph: student embeddings carry gradient back into
                # the decoder + temporal fusion through the CRRCD loss.
                self._captured_decoder_hs = hs[-1]
            return None

        self.transformer.decoder.register_forward_hook(
            _decoder_capture_post_hook, with_kwargs=True,
        )

    # ─────────────────────────────────────────────────────────────────
    #  KD-DETR helpers
    # ─────────────────────────────────────────────────────────────────
    def register_teacher_queries(
        self, refpoint_w: torch.Tensor, query_feat_w: torch.Tensor,
    ) -> None:
        """Cache the frozen teacher's object queries on the student.

        The student keeps its own learnable ``refpoint_embed`` / ``query_feat``
        for the detection forward pass.  The teacher queries are stored as
        non-persistent buffers and used only by ``forward(query_mode='teacher')``
        for the KD-DETR specific-sampling distillation branch (no Hungarian
        matching required — predictions are slot-aligned).
        """
        device = self.refpoint_embed.weight.device
        self.teacher_refpoint = refpoint_w.detach().to(device).clone()
        self.teacher_query_feat = query_feat_w.detach().to(device).clone()
        self._has_teacher_queries = True

    def _swap_group_detr(self, new_group: int, new_num_queries: int):
        """Temporarily set ``group_detr`` and ``num_queries`` on the
        transformer + every decoder submodule that carries those fields.

        Returns a no-arg ``restore`` closure to be called in a ``finally``
        block.  The student's own ``self.group_detr`` / ``self.num_queries``
        are left untouched.
        """
        tr = self.transformer
        orig_g_tr = getattr(tr, "group_detr", None)
        orig_nq_tr = getattr(tr, "num_queries", None)
        # Walk every transformer module — decoder layers carry their own
        # ``group_detr`` (used by SA-Q grouping at line 568 in transformer.py).
        patched: list[tuple[nn.Module, int]] = []
        for m in tr.modules():
            if hasattr(m, "group_detr"):
                patched.append((m, int(m.group_detr)))
                m.group_detr = int(new_group)
        if orig_nq_tr is not None:
            tr.num_queries = int(new_num_queries)

        def restore() -> None:
            for m, g in patched:
                m.group_detr = g
            if orig_nq_tr is not None:
                tr.num_queries = orig_nq_tr

        return restore
    # ─────────────────────────────────────────────────────────────────
    #  Forward
    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        frames: torch.Tensor,
        targets=None,
        query_mode: str = "student",
        general_queries: dict | None = None,
        decoder_inputs: dict | None = None,
    ):
        """
        Args:
            frames: (B, T, 3, H, W) – temporally ordered RGB frames
            targets: list[list[dict]] – per-batch, per-frame targets
                     (only centre-frame targets used for loss)
            query_mode:
                * ``"student"``  – the default detection forward.  Uses the
                  student's own learnable ``refpoint_embed`` / ``query_feat``
                  with the configured ``group_detr``.
                * ``"teacher"``  – KD-DETR *specific sampling* branch.  Uses
                  the teacher queries stored by :meth:`register_teacher_queries`,
                  with ``group_detr=1`` and the matching slot count.  The
                  backbone + temporal fusion + decoder weights are shared
                  with the student forward, only the queries differ.
                * ``"general"``  – KD-DETR *general sampling* branch.  Uses
                  the random queries supplied in ``general_queries``
                  (a dict with ``refpoint`` (Q,4) and ``query_feat`` (Q,D)),
                  again with ``group_detr=1``.
            decoder_inputs: optional dict with ``tgt`` (B, Q, D) and
                ``refpoints`` (B, Q, 4) — the *final* per-slot decoder
                inputs captured from the teacher's decoder.  When supplied,
                these tensors override whatever the student's own encoder
                + two-stage block produced, so the student's deformable
                attention samples at the teacher's spatial locations.
                Only meaningful in ``"teacher"`` / ``"general"`` modes.

        Returns:
            dict with "pred_logits", "pred_boxes", optionally "aux_outputs"
        """
        assert query_mode in ("student", "teacher", "general"), query_mode
        B, T, C, H, W = frames.shape

        # Reset the decoder hidden-state capture (used by CRRCD).
        self._captured_decoder_hs = None

        # ── 1. Run backbone on all frames simultaneously ────────────
        all_frames = frames.reshape(B * T, C, H, W)
        nested = nested_tensor_from_tensor_list(all_frames)
        with torch.set_grad_enabled(not self.cfg.freeze_backbone):
            features, poss = self.backbone(nested)

        # ── 2. Temporal fusion per level ────────────────────────────
        fused_srcs = []
        fused_masks = []
        fused_poss = []

        for lvl_idx, feat in enumerate(features):
            src, mask = feat.decompose()  # (B*T, C, h, w), (B*T, h, w)
            _, Cl, h, w = src.shape

            # Reshape to (B, T, C, h, w)
            src_bt = src.reshape(B, T, Cl, h, w)

            # Apply temporal fusion → (B, C, h, w) enriched centre frame
            fused = self.temporal_fusions[lvl_idx](src_bt)
            fused_srcs.append(fused)

            # Use centre frame's mask
            mask_bt = mask.reshape(B, T, h, w)
            fused_masks.append(mask_bt[:, self.centre])

            # Use centre frame's position embeddings
            pos = poss[lvl_idx]  # (B*T, C, h, w)
            pos_bt = pos.reshape(B, T, Cl, h, w)
            fused_poss.append(pos_bt[:, self.centre])

        # ── 3. Resolve queries + run decoder ────────────────────────
        restore_fn = None
        if query_mode == "student":
            if self.training:
                refpoint_embed_weight = self.refpoint_embed.weight
                query_feat_weight = self.query_feat.weight
            else:
                refpoint_embed_weight = self.refpoint_embed.weight[: self.num_queries]
                query_feat_weight = self.query_feat.weight[: self.num_queries]
        elif query_mode == "teacher":
            assert self._has_teacher_queries, (
                "query_mode='teacher' requires register_teacher_queries() first"
            )
            refpoint_embed_weight = self.teacher_refpoint
            query_feat_weight = self.teacher_query_feat
            restore_fn = self._swap_group_detr(1, refpoint_embed_weight.shape[0])
        else:  # "general"
            assert (
                general_queries is not None
                and "refpoint" in general_queries
                and "query_feat" in general_queries
            ), "query_mode='general' requires general_queries={'refpoint','query_feat'}"
            refpoint_embed_weight = general_queries["refpoint"]
            query_feat_weight = general_queries["query_feat"]
            restore_fn = self._swap_group_detr(1, refpoint_embed_weight.shape[0])

        # KD-DETR slot-alignment injection: when decoder_inputs is supplied,
        # the pre-hook on ``self.transformer.decoder`` will swap in the
        # teacher-derived (tgt, refpoints) right before the decoder runs.
        if decoder_inputs is not None:
            assert query_mode in ("teacher", "general"), (
                "decoder_inputs only valid in teacher/general modes"
            )
            assert "tgt" in decoder_inputs and "refpoints" in decoder_inputs
            self._inject_decoder_inputs = {
                "tgt": decoder_inputs["tgt"].detach(),
                "refpoints": decoder_inputs["refpoints"].detach(),
            }

        try:
            hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
                fused_srcs, fused_masks, fused_poss,
                refpoint_embed_weight, query_feat_weight,
            )
        finally:
            self._inject_decoder_inputs = None
            if restore_fn is not None:
                restore_fn()

        # ── 4. Detection heads ──────────────────────────────────────
        out: dict | None = None
        if hs is not None:
            if self.bbox_reparam:
                delta = self.bbox_embed(hs)
                cxcy = delta[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
                wh = delta[..., 2:].exp() * ref_unsigmoid[..., 2:]
                outputs_coord = torch.cat([cxcy, wh], dim=-1)
            else:
                outputs_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()

            outputs_class = self.class_embed(hs)

            out = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if self.two_stage:
            # Teacher / general branches always run single-group; student
            # follows the train/eval convention.
            if query_mode == "student":
                group_detr = self.group_detr if self.training else 1
            else:
                group_detr = 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc.append(
                    self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                )
            cls_enc = torch.cat(cls_enc, dim=1)
            if out is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        return out

    # ─────────────────────────────────────────────────────────────────
    #  Random general-sampling queries (re-drawn per training step)
    # ─────────────────────────────────────────────────────────────────
    def sample_general_queries(self, num_queries: int, device, dtype=torch.float32):
        """Draw a fresh set of random queries for KD-DETR general sampling.

        ``refpoint`` is initialised at zero (matching LWDETR's
        ``refpoint_embed`` init) so two-stage encoder proposals govern the
        spatial coverage; ``query_feat`` is N(0, std) noise that probes the
        decoder with content embeddings unrelated to any trained slot.
        """
        std = float(self.cfg.distill_general_query_std)
        refpoint = torch.zeros(num_queries, 4, device=device, dtype=dtype)
        query_feat = torch.randn(
            num_queries, int(self.cfg.hidden_dim), device=device, dtype=dtype
        ) * std
        return {"refpoint": refpoint, "query_feat": query_feat}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def get_param_groups(self):
        """Return parameter groups with different learning rates."""
        temporal_params = []
        decoder_params = []
        backbone_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "temporal_fusion" in name:
                temporal_params.append(p)
            elif "backbone" in name:
                backbone_params.append(p)
            else:
                decoder_params.append(p)

        groups = [
            {"params": temporal_params, "lr": self.cfg.lr},
            {"params": decoder_params, "lr": self.cfg.lr},
        ]
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.cfg.lr_backbone})
        return groups
