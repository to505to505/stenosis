"""Frozen RF-DETR-Large @ 704 teacher for HR→LR query-aligned distillation.

Loads the fine-tuned 2D RF-DETR-Large checkpoint from
``rfdetr_runs/rfdetr_large_arcade2x_704_reg`` and exposes:

  • ``refpoint_embed_weight``  — (Q, 4) shared object queries (anchor side)
  • ``query_feat_weight``      — (Q, hidden_dim) shared object queries (content)
  • ``forward(centre_clean)``  — runs the full HR detector and returns
        ``pred_logits`` (B, Q, K_t), ``pred_boxes`` (B, Q, 4) and the
        per-query foreground weight ``w_i = max_k sigmoid(pred_logits)``.

All parameters are frozen and the module is forced into eval mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from ..config import Config


class FrozenRFDETRTeacher(nn.Module):
    """Frozen 2D RF-DETR-Large teacher exposing shared decoder queries."""

    def __init__(self, cfg: Config):
        super().__init__()
        # Local imports to avoid pulling rfdetr unless distillation is on.
        from rfdetr.config import RFDETRLargeConfig
        from rfdetr.models.lwdetr import build_model_from_config
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        self._nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
        self.cfg = cfg

        # Build with the checkpoint's class count and group_detr=1 (inference
        # slice). num_queries is matched to cfg.distill_num_queries.
        Q = int(cfg.distill_num_queries)
        model_cfg = RFDETRLargeConfig(
            num_classes=cfg.distill_teacher_num_classes,
            num_queries=Q,
            num_select=Q,
        )
        # group_detr is not on the model config — it lives on the LWDETR
        # constructor; the default builder uses TrainConfig.group_detr=1, but
        # the checkpoint has group_detr=13. We trim queries below.
        lwdetr = build_model_from_config(model_cfg)

        ckpt_path = Path(cfg.distill_teacher_ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Distillation teacher checkpoint not found: {ckpt_path}"
            )
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)

        # The checkpoint was trained with group_detr>1, so the query
        # embeddings are sized (num_queries * group_detr, ·). Our model is
        # built with the same default group_detr (no trimming needed): the
        # LWDETR.forward eval-mode path slices to the first ``num_queries``
        # rows automatically — exactly the inference convention.

        # Shape-filter the rest so any incidental head/projector mismatches
        # are tolerated.
        model_sd = lwdetr.state_dict()
        filtered, skipped = {}, []
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        msg = lwdetr.load_state_dict(filtered, strict=False)
        print(
            f"[Teacher] Loaded '{ckpt_path.name}'  "
            f"loaded={len(filtered)}  missing={len(msg.missing_keys)}  "
            f"skipped(shape)={len(skipped)}"
        )

        self.lwdetr = lwdetr
        # Force inference-mode behaviour: no aux loss returned, exactly
        # ``num_queries`` predictions.
        self.lwdetr.aux_loss = False
        self.lwdetr.num_queries = Q

        # Cache the *first Q* rows of the loaded query embeddings — these are
        # the canonical inference queries (group 0). They become the shared
        # q_teacher passed to the student.
        with torch.no_grad():
            shared_refpoint = self.lwdetr.refpoint_embed.weight[:Q].detach().clone()
            shared_query_feat = self.lwdetr.query_feat.weight[:Q].detach().clone()
        self.register_buffer(
            "refpoint_embed_weight", shared_refpoint, persistent=False,
        )
        self.register_buffer(
            "query_feat_weight", shared_query_feat, persistent=False,
        )

        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        # ── Hook to capture the *final* decoder inputs ──────────────
        # ``tgt`` (= query_feat tiled to B) and ``refpoints_unsigmoid``
        # (= encoder-topk anchors mixed with the learned per-slot offset)
        # are exactly the per-slot inputs the teacher's decoder layers see
        # at every layer.  KD-DETR slot alignment requires the student's
        # decoder to receive these exact tensors so deformable attention
        # samples the *same* spatial locations and slot identities.
        self._captured_decoder_inputs: Dict[str, torch.Tensor] = {}

        def _decoder_pre_hook(_module, args, kwargs):
            # args = (tgt, memory, ...);  refpoints_unsigmoid is kwarg.
            if len(args) >= 1:
                self._captured_decoder_inputs["tgt"] = args[0].detach()
            if "refpoints_unsigmoid" in kwargs:
                self._captured_decoder_inputs["refpoints"] = (
                    kwargs["refpoints_unsigmoid"].detach()
                )
            return None  # don't modify

        self.lwdetr.transformer.decoder.register_forward_pre_hook(
            _decoder_pre_hook, with_kwargs=True,
        )

        # Forward hook: capture the decoder's *output* — ``hs`` is
        # ``(num_layers, B, Q, D)``; we keep the last layer ``hs[-1]``
        # (B, Q, D), which is the embedding fed into the classification
        # and bounding-box heads.  Required by CRRCD relational distillation.
        def _decoder_post_hook(_module, _args, _kwargs, output):
            # rfdetr's TransformerDecoder.forward returns either:
            #   • [stacked_intermediate, stacked_refpoints]  (return_intermediate=True)
            #   • a single tensor (export path)
            # In the intermediate case ``stacked_intermediate`` has shape
            # (num_layers, B, Q, D); we keep the last layer (B, Q, D).
            if isinstance(output, (list, tuple)) and len(output) >= 1:
                hs = output[0]
            else:
                hs = output
            if hs is not None:
                self._captured_decoder_inputs["hs"] = hs[-1].detach()
            return None

        self.lwdetr.transformer.decoder.register_forward_hook(
            _decoder_post_hook, with_kwargs=True,
        )

    @property
    def hidden_dim(self) -> int:
        return int(self.lwdetr.transformer.d_model)

    def train(self, mode: bool = True):  # type: ignore[override]
        # Always stay in eval mode regardless of parent .train(True).
        return super().train(False)

    @torch.no_grad()
    def forward(self, centre_clean: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Args:
            centre_clean: (B, 3, H, W) ImageNet-normalised HR frames.
        Returns:
            dict with ``pred_logits`` (B, Q, K_t), ``pred_boxes`` (B, Q, 4),
            ``foreground_weight`` (B, Q).
        """
        return self._forward_impl(centre_clean, refpoint_w=None, query_feat_w=None)

    @torch.no_grad()
    def forward_general(
        self,
        centre_clean: torch.Tensor,
        refpoint_w: torch.Tensor,
        query_feat_w: torch.Tensor,
        min_weight: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        """Run the teacher with an externally-supplied set of queries.

        Used by the KD-DETR *general sampling* branch: ``refpoint_w`` and
        ``query_feat_w`` are random tensors freshly drawn on each step and
        shared with the student so distillation losses are slot-aligned.
        """
        return self._forward_impl(
            centre_clean,
            refpoint_w=refpoint_w,
            query_feat_w=query_feat_w,
            min_weight_override=min_weight,
        )

    @torch.no_grad()
    def _forward_impl(
        self,
        centre_clean: torch.Tensor,
        refpoint_w,
        query_feat_w,
        min_weight_override: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        assert centre_clean.dim() == 4, (
            f"teacher expects (B, 3, H, W), got {tuple(centre_clean.shape)}"
        )
        nested = self._nested_tensor_from_tensor_list(centre_clean)

        if refpoint_w is None:
            out = self.lwdetr(nested)
        else:
            # Inject custom queries by temporarily swapping the embeddings.
            # Embedding objects are replaced (not their .weight Parameter) so
            # downstream LWDETR.forward picks them up via attribute access.
            Q = int(refpoint_w.shape[0])
            orig_rp = self.lwdetr.refpoint_embed
            orig_qf = self.lwdetr.query_feat
            orig_nq_l = self.lwdetr.num_queries
            orig_nq_t = self.lwdetr.transformer.num_queries

            new_rp = nn.Embedding(Q, refpoint_w.shape[1]).to(refpoint_w.device)
            new_rp.weight = nn.Parameter(
                refpoint_w.detach().to(refpoint_w.device).clone(),
                requires_grad=False,
            )
            new_qf = nn.Embedding(Q, query_feat_w.shape[1]).to(query_feat_w.device)
            new_qf.weight = nn.Parameter(
                query_feat_w.detach().to(query_feat_w.device).clone(),
                requires_grad=False,
            )
            self.lwdetr.refpoint_embed = new_rp
            self.lwdetr.query_feat = new_qf
            self.lwdetr.num_queries = Q
            self.lwdetr.transformer.num_queries = Q
            try:
                out = self.lwdetr(nested)
            finally:
                self.lwdetr.refpoint_embed = orig_rp
                self.lwdetr.query_feat = orig_qf
                self.lwdetr.num_queries = orig_nq_l
                self.lwdetr.transformer.num_queries = orig_nq_t

        logits = out["pred_logits"]                       # (B, Q, K_real + 1)
        boxes = out["pred_boxes"]                         # (B, Q, 4)
        # Drop the trailing no-object slot (rfdetr's build_model does
        # ``num_classes = args.num_classes + 1`` and treats the last index
        # as background; see SetCriterion.loss_labels).
        K_real = int(self.cfg.distill_teacher_num_classes)
        if logits.shape[-1] > K_real:
            logits = logits[..., :K_real]
        # Foreground rebalance weight: per-query max foreground probability.
        # RF-DETR uses sigmoid focal (no explicit bg slot in the loss), so the
        # requested "max prob over non-bg classes" reduces to max-over-classes
        # of the sigmoid logits.
        w = logits.sigmoid().amax(dim=-1)                 # (B, Q)
        floor = (
            float(min_weight_override)
            if min_weight_override is not None
            else float(self.cfg.distill_min_weight)
        )
        if floor > 0.0:
            w = w.clamp(min=floor)

        result = {
            "pred_logits": logits.detach(),
            "pred_boxes": boxes.detach(),
            "foreground_weight": w.detach(),
        }
        # Always expose the captured per-slot decoder inputs — these are
        # what the student's decoder must receive for proper KD-DETR slot
        # alignment (geometry + content).
        if "tgt" in self._captured_decoder_inputs:
            result["decoder_tgt"] = self._captured_decoder_inputs["tgt"]
        if "refpoints" in self._captured_decoder_inputs:
            result["decoder_refpoints"] = self._captured_decoder_inputs["refpoints"]
        if "hs" in self._captured_decoder_inputs:
            result["decoder_hs"] = self._captured_decoder_inputs["hs"]
        # Reset for the next call so we don't leak stale tensors.
        self._captured_decoder_inputs = {}
        return result
