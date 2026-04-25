"""Stage-4 auxiliary losses: feature distillation + InfoNCE.

These helpers operate on already-pooled / aligned tensors; the model and
trainer are responsible for producing the right inputs (e.g. running the
neighbour-frame feature pass and RoI-pooling box embeddings).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    """1×1 conv aligning student feature channels to the teacher's hidden dim.

    Per the user spec we keep this as a single 1×1 conv (no normalisation,
    no activation) so the student is forced to do the heavy lifting in its
    own backbone/neck rather than in the adapter.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="linear")
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def cosine_distill_loss(
    student_feat: torch.Tensor, teacher_feat: torch.Tensor,
) -> torch.Tensor:
    """Mean cosine-distance loss between two feature maps.

    Args:
        student_feat: ``(B, C, H, W)`` — already aligned to teacher channels.
        teacher_feat: ``(B, C, H', W')`` — un-grad teacher output.

    The teacher map is bilinearly resized to the student's spatial resolution
    if the two differ (cheap; happens at most once per batch).
    """
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        teacher_feat = F.interpolate(
            teacher_feat, size=student_feat.shape[-2:],
            mode="bilinear", align_corners=False,
        )
    s = F.normalize(student_feat.float(), dim=1, eps=1e-6)
    t = F.normalize(teacher_feat.float(), dim=1, eps=1e-6)
    cos = (s * t).sum(dim=1)         # (B, H, W)
    return (1.0 - cos).mean()


def info_nce_loss(
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric-style InfoNCE on a paired (anchor, positive) batch.

    Args:
        anchors:    ``(P, D)``
        positives:  ``(P, D)`` — paired one-to-one with ``anchors``
        negatives:  ``(N, D)`` — shared negative pool (e.g. all OTHER GT-box
                    embeddings in the batch). May be empty; if so the loss
                    degenerates to a trivial 0 (cross-entropy of a 1-class
                    problem) and is returned as ``anchors.sum() * 0`` to
                    keep gradients connected.
    """
    assert anchors.shape == positives.shape, \
        f"anchor/positive shape mismatch: {anchors.shape} vs {positives.shape}"
    P, D = anchors.shape
    if P == 0:
        return anchors.sum() * 0.0

    a = F.normalize(anchors.float(), dim=1, eps=1e-6)
    p = F.normalize(positives.float(), dim=1, eps=1e-6)

    pos_logits = (a * p).sum(dim=1, keepdim=True) / temperature  # (P, 1)

    if negatives.numel() == 0:
        # Only one logit → cross-entropy is log(1) = 0; keep grad alive.
        return (pos_logits * 0.0).sum()

    n = F.normalize(negatives.float(), dim=1, eps=1e-6)
    neg_logits = a @ n.t() / temperature                         # (P, N)
    logits = torch.cat([pos_logits, neg_logits], dim=1)          # (P, 1+N)
    targets = torch.zeros(P, dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)
