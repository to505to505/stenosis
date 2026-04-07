"""Temporal RF-DETR: pre-decoder temporal feature fusion.

Architecture
────────────
1. Shared DINOv2 backbone (frozen): extracts per-frame multi-scale features
2. TemporalFusion (new, trainable): lightweight temporal self-attention that
   enriches the centre frame features with context from neighbouring frames
3. Standard RF-DETR decoder + heads (fine-tuned): produces detections
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── make rf-detr importable ──────────────────────────────────────────
_RFDETR_ROOT = str(Path(__file__).resolve().parent.parent / "rf-detr" / "src")
if _RFDETR_ROOT not in sys.path:
    sys.path.insert(0, _RFDETR_ROOT)

from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.models.lwdetr import build_model_from_config, build_criterion_from_config
from rfdetr.utilities.tensors import NestedTensor, nested_tensor_from_tensor_list

from .config import Config


# ─────────────────────────────────────────────────────────────────────
#  Temporal fusion module
# ─────────────────────────────────────────────────────────────────────
class TemporalFusion(nn.Module):
    """Per-spatial-position temporal self-attention across T frames.

    For each feature level the module:
      1. adds learnable temporal position embeddings
      2. runs a small TransformerEncoder across the T dimension
      3. selects the centre-frame output
    """

    def __init__(self, hidden_dim: int, T: int, num_layers: int = 2, nhead: int = 8):
        super().__init__()
        self.T = T
        self.temporal_pos = nn.Embedding(T, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feats: (B, T, C, H, W) – features from backbone for one level
        Returns:
            fused: (B, C, H, W) – temporally-enriched centre-frame features
        """
        B, T, C, H, W = feats.shape
        assert T == self.T

        # add temporal positional embeddings  (broadcast over B, H, W)
        tp = self.temporal_pos.weight  # (T, C)
        feats = feats + tp[None, :, :, None, None]

        # reshape to (B*H*W, T, C) for temporal self-attention
        feats = feats.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        feats = self.encoder(feats)  # (B*H*W, T, C)

        # select centre frame
        centre = T // 2
        out = feats[:, centre]  # (B*H*W, C)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)


# ─────────────────────────────────────────────────────────────────────
#  Build helpers
# ─────────────────────────────────────────────────────────────────────
def _build_rfdetr_from_checkpoint(cfg: Config) -> nn.Module:
    """Instantiate an RFDETRSmall and load fine-tuned checkpoint weights."""
    model_cfg = RFDETRSmallConfig(num_classes=cfg.num_classes)
    lwdetr = build_model_from_config(model_cfg)

    ckpt_path = Path(cfg.rfdetr_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"RF-DETR checkpoint not found: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # The .pth from PTL training stores raw LWDETR weights under 'model'
    state_dict = ckpt.get("model", ckpt)
    msg = lwdetr.load_state_dict(state_dict, strict=False)
    print(f"[RF-DETR] Loaded checkpoint  missing={len(msg.missing_keys)}  "
          f"unexpected={len(msg.unexpected_keys)}")
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
                num_layers=cfg.temporal_attn_layers,
                nhead=cfg.temporal_nhead,
            )
            for _ in range(n_levels)
        ])

    # ─────────────────────────────────────────────────────────────────
    #  Forward
    # ─────────────────────────────────────────────────────────────────
    def forward(self, frames: torch.Tensor, targets=None):
        """
        Args:
            frames: (B, T, 3, H, W) – temporally ordered RGB frames
            targets: list[list[dict]] – per-batch, per-frame targets
                     (only centre-frame targets used for loss)

        Returns:
            dict with "pred_logits", "pred_boxes", optionally "aux_outputs"
        """
        B, T, C, H, W = frames.shape

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

        # ── 3. Run RF-DETR decoder ──────────────────────────────────
        if self.training:
            refpoint_embed_weight = self.refpoint_embed.weight
            query_feat_weight = self.query_feat.weight
        else:
            refpoint_embed_weight = self.refpoint_embed.weight[:self.num_queries]
            query_feat_weight = self.query_feat.weight[:self.num_queries]

        hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(
            fused_srcs, fused_masks, fused_poss,
            refpoint_embed_weight, query_feat_weight,
        )

        # ── 4. Detection heads ──────────────────────────────────────
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
            group_detr = self.group_detr if self.training else 1
            hs_enc_list = hs_enc.chunk(group_detr, dim=1)
            cls_enc = []
            for g_idx in range(group_detr):
                cls_enc.append(
                    self.transformer.enc_out_class_embed[g_idx](hs_enc_list[g_idx])
                )
            cls_enc = torch.cat(cls_enc, dim=1)
            if hs is not None:
                out["enc_outputs"] = {"pred_logits": cls_enc, "pred_boxes": ref_enc}
            else:
                out = {"pred_logits": cls_enc, "pred_boxes": ref_enc}

        return out

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
