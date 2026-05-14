"""Video RF-DETR model.

The model treats a video window as ``B*T`` independent frames for the
backbone and RF-DETR transformer, then reshapes the predictions back to
``(B, T, Q, *)`` for multi-frame detection losses and evaluation.

Optional Early Temporal Fusion (ETF) can mix backbone feature maps across
the temporal axis before the transformer. Distillation modes keep the
teacher/student query-slot alignment hooks used by KD-DETR and CRRCD.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.models.weights import load_pretrain_weights
from rfdetr.models.lwdetr import build_model_from_config, build_criterion_from_config
from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

from .config import Config


class EarlyTemporalFusion(nn.Module):
    """Lightweight temporal self-attention applied to backbone feature maps."""

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.0,
        spatial_radius: int = 0,
    ):
        super().__init__()
        self.spatial_radius = int(spatial_radius)
        if self.spatial_radius < 0:
            raise ValueError(
                f"spatial_radius must be non-negative, got {spatial_radius}",
            )
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def _local_key_value_tokens(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C, h, w = x.shape
        radius = self.spatial_radius
        padded = x.new_zeros(B, T, C, h + 2 * radius, w + 2 * radius)
        padded[:, :, :, radius: radius + h, radius: radius + w] = x

        valid = torch.zeros(
            1, 1, h + 2 * radius, w + 2 * radius,
            device=x.device,
            dtype=torch.bool,
        )
        valid[:, :, radius: radius + h, radius: radius + w] = True

        token_windows = []
        mask_windows = []
        for y_offset in range(2 * radius + 1):
            for x_offset in range(2 * radius + 1):
                window = padded[
                    :, :, :,
                    y_offset: y_offset + h,
                    x_offset: x_offset + w,
                ]
                tokens = window.permute(0, 3, 4, 1, 2).contiguous()
                token_windows.append(tokens.reshape(B * h * w, T, C))

                valid_window = valid[
                    :, :,
                    y_offset: y_offset + h,
                    x_offset: x_offset + w,
                ]
                valid_window = valid_window.expand(B, 1, h, w)
                valid_window = valid_window.permute(0, 2, 3, 1).contiguous()
                valid_window = valid_window.reshape(B * h * w, 1)
                mask_windows.append(~valid_window.expand(B * h * w, T))

        return torch.cat(token_windows, dim=1), torch.cat(mask_windows, dim=1)

    def forward(
        self,
        srcs: List[torch.Tensor],
        B: int,
        T: int,
    ) -> List[torch.Tensor]:
        enriched = []
        for src in srcs:
            _BT, C, h, w = src.shape
            x = src.reshape(B, T, C, h, w)
            x = x.permute(0, 3, 4, 1, 2).contiguous()
            x = x.reshape(B * h * w, T, C)

            x_n = self.norm(x)
            if self.spatial_radius == 0:
                attn_out, _ = self.attn(x_n, x_n, x_n)
            else:
                x_n_grid = x_n.reshape(B, h, w, T, C)
                x_n_grid = x_n_grid.permute(0, 3, 4, 1, 2).contiguous()
                key_value, key_padding_mask = self._local_key_value_tokens(x_n_grid)
                attn_out, _ = self.attn(
                    x_n,
                    key_value,
                    key_value,
                    key_padding_mask=key_padding_mask,
                )
            x = x + attn_out

            x = x.reshape(B, h, w, T, C)
            x = x.permute(0, 3, 4, 1, 2).contiguous()
            enriched.append(x.reshape(B * T, C, h, w))
        return enriched


def _build_rfdetr_from_checkpoint(cfg: Config) -> nn.Module:
    model_cfg = RFDETRSmallConfig(num_classes=cfg.num_classes)
    lwdetr = build_model_from_config(model_cfg)

    ckpt_path = Path(cfg.rfdetr_checkpoint)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model", ckpt)
        model_sd = lwdetr.state_dict()
        filtered, skipped = {}, []
        for key, value in state_dict.items():
            if key in model_sd and model_sd[key].shape == value.shape:
                filtered[key] = value
            else:
                skipped.append(key)
        msg = lwdetr.load_state_dict(filtered, strict=False)
        print(
            f"[VideoRFDETR] Loaded fine-tuned checkpoint '{ckpt_path.name}'  "
            f"loaded={len(filtered)}  missing={len(msg.missing_keys)}  "
            f"skipped(shape)={len(skipped)}"
        )
    else:
        print(
            f"[VideoRFDETR] Checkpoint '{cfg.rfdetr_checkpoint}' not found - "
            f"downloading Small pretrained weights ..."
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


def _param_group_for(name: str) -> str:
    """Classify a parameter name into an LR group: backbone / new / pretrained."""
    if name.startswith("backbone"):
        return "backbone"
    if name.startswith("etf") or name.startswith("crrcd"):
        return "new"
    return "pretrained"


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

        if self.two_stage:
            self.transformer.enc_out_bbox_embed = lwdetr.transformer.enc_out_bbox_embed
            self.transformer.enc_out_class_embed = lwdetr.transformer.enc_out_class_embed

        if cfg.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        if cfg.freeze_decoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
            for param in self.class_embed.parameters():
                param.requires_grad = False
            for param in self.bbox_embed.parameters():
                param.requires_grad = False

        if cfg.etf_enabled:
            self.etf: Optional[EarlyTemporalFusion] = EarlyTemporalFusion(
                d_model=cfg.hidden_dim,
                n_heads=cfg.etf_heads,
                dropout=cfg.etf_dropout,
                spatial_radius=cfg.etf_spatial_radius,
            )
        else:
            self.etf = None

        self._has_teacher_queries: bool = False
        self.register_buffer("teacher_refpoint", torch.zeros(1, 4), persistent=False)
        self.register_buffer(
            "teacher_query_feat", torch.zeros(1, cfg.hidden_dim), persistent=False,
        )

        self._inject_decoder_inputs: Optional[Dict[str, torch.Tensor]] = None

        def _inject_pre_hook(_module, args, kwargs):
            injected = self._inject_decoder_inputs
            if injected is None:
                return None
            new_args = (injected["tgt"], *args[1:])
            new_kwargs = dict(kwargs)
            new_kwargs["refpoints_unsigmoid"] = injected["refpoints"]
            return new_args, new_kwargs

        self.transformer.decoder.register_forward_pre_hook(
            _inject_pre_hook, with_kwargs=True,
        )

        self._captured_decoder_hs: Optional[torch.Tensor] = None

        def _capture_hs_post_hook(_module, _args, _kwargs, output):
            hs = output[0] if isinstance(output, (list, tuple)) and output else output
            if hs is not None:
                self._captured_decoder_hs = hs[-1]
            return None

        self.transformer.decoder.register_forward_hook(
            _capture_hs_post_hook, with_kwargs=True,
        )

    def register_teacher_queries(
        self, refpoint_w: torch.Tensor, query_feat_w: torch.Tensor,
    ) -> None:
        device = self.refpoint_embed.weight.device
        self.teacher_refpoint = refpoint_w.detach().to(device).clone()
        self.teacher_query_feat = query_feat_w.detach().to(device).clone()
        self._has_teacher_queries = True

    def _swap_group_detr(self, new_group: int, new_num_queries: int):
        transformer = self.transformer
        original_num_queries = getattr(transformer, "num_queries", None)
        patched: List[Tuple[nn.Module, int]] = []
        for module in transformer.modules():
            if hasattr(module, "group_detr"):
                patched.append((module, int(module.group_detr)))
                module.group_detr = int(new_group)
        if original_num_queries is not None:
            transformer.num_queries = int(new_num_queries)

        def restore() -> None:
            for module, group in patched:
                module.group_detr = group
            if original_num_queries is not None:
                transformer.num_queries = original_num_queries

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

    def _run_backbone(self, frames_bt: torch.Tensor):
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
            {"pred_logits": logits, "pred_boxes": boxes}
            for logits, boxes in zip(outputs_class_stack[:-1], outputs_coord_stack[:-1])
        ]

    def _stacked_heads(
        self, hs: torch.Tensor, ref_unsigmoid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        stacked_class = self.class_embed(hs)
        if self.bbox_reparam:
            delta_full = self.bbox_embed(hs)
            cxcy = delta_full[..., :2] * ref_unsigmoid[..., 2:] + ref_unsigmoid[..., :2]
            wh = delta_full[..., 2:].exp() * ref_unsigmoid[..., 2:]
            stacked_coord = torch.cat([cxcy, wh], dim=-1)
        else:
            stacked_coord = (self.bbox_embed(hs) + ref_unsigmoid).sigmoid()
        return stacked_class, stacked_coord

    def forward(
        self,
        frames: torch.Tensor,
        targets=None,
        query_mode: str = "student",
        general_queries: Optional[Dict[str, torch.Tensor]] = None,
        decoder_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Run the video model.

        Student mode returns ``(B,T,Q,*)`` predictions. Teacher and general
        query modes return ``(B*T,Q,*)`` predictions for KD/CRRCD.
        """
        del targets
        assert query_mode in ("student", "teacher", "general"), query_mode
        B, T, C, H, W = frames.shape
        BT = B * T

        self._captured_decoder_hs = None

        srcs, masks, poss = self._run_backbone(frames.reshape(BT, C, H, W))

        if self.etf is not None:
            srcs = self.etf(srcs, B, T)

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
        else:
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

        try:
            hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
                srcs, masks, poss, refpoint_w, query_feat_w,
            )
        finally:
            self._inject_decoder_inputs = None
            if restore_fn is not None:
                restore_fn()

        out: Optional[Dict[str, torch.Tensor]] = None
        if hs is not None:
            stacked_class, stacked_coord = self._stacked_heads(hs, ref_unsigmoid)
            out = {
                "pred_logits": stacked_class[-1],
                "pred_boxes": stacked_coord[-1],
            }
            if self.aux_loss:
                out["aux_outputs"] = self._aux_outputs(stacked_class, stacked_coord)

        if self.two_stage and hs_enc is not None:
            if query_mode == "student":
                group_detr = self.group_detr if self.training else 1
                hs_enc_list = hs_enc.chunk(group_detr, dim=1)
                cls_enc = []
                for group_idx in range(group_detr):
                    cls_enc.append(
                        self.transformer.enc_out_class_embed[group_idx](
                            hs_enc_list[group_idx]
                        )
                    )
                cls_enc = torch.cat(cls_enc, dim=1)
            else:
                hs_enc_list = hs_enc.chunk(1, dim=1)
                cls_enc = self.transformer.enc_out_class_embed[0](hs_enc_list[0])

            if out is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        assert out is not None, "RF-DETR forward produced no decoder or encoder output"

        if query_mode != "student":
            return out

        logits_btq = out["pred_logits"]
        boxes_btq = out["pred_boxes"]
        Q = logits_btq.shape[-2]
        K = logits_btq.shape[-1]
        out["pred_logits"] = logits_btq.reshape(B, T, Q, K)
        out["pred_boxes"] = boxes_btq.reshape(B, T, Q, 4)
        out["first_pass"] = {
            "pred_logits": out["pred_logits"],
            "pred_boxes": out["pred_boxes"],
        }
        return out

    def get_param_groups(self):
        buckets = {"backbone": [], "pretrained": [], "new": []}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            buckets[_param_group_for(name)].append(param)
        print(
            f"[param groups] pretrained={len(buckets['pretrained'])} "
            f"new={len(buckets['new'])} backbone={len(buckets['backbone'])}"
        )
        assert buckets["pretrained"], (
            "no pretrained-detector params found — check _param_group_for prefixes"
        )
        groups = []
        if buckets["pretrained"]:
            groups.append({"params": buckets["pretrained"], "lr": self.cfg.lr_pretrained})
        if buckets["new"]:
            groups.append({"params": buckets["new"], "lr": self.cfg.lr})
        if buckets["backbone"]:
            groups.append({"params": buckets["backbone"], "lr": self.cfg.lr_backbone})
        return groups