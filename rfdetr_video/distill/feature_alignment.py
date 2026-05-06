"""Feature-space alignment losses for video distillation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def stfs_feature_alignment_loss(
    student_hs: torch.Tensor,
    teacher_hs: torch.Tensor,
    *,
    stfs_mask: Optional[torch.Tensor] = None,
    teacher_weights: Optional[torch.Tensor] = None,
    teacher_topk: int = 16,
) -> torch.Tensor:
    """Align STFS-injected student features to teacher decoder geometry.

    This loss deliberately avoids query-index matching. For every selected
    student STFS slot, it finds the nearest teacher decoder slot by cosine
    similarity and minimises ``1 - max_cosine``. If teacher foreground weights
    are supplied, the nearest-neighbor pool is restricted to the top-k teacher
    foreground slots per frame so background queries do not dominate.

    Args:
        student_hs: ``(B,T,Q,D)`` or ``(BT,Q,D)`` student STFS embeddings.
        teacher_hs: ``(B,T,Qt,D)`` or ``(BT,Qt,D)`` teacher decoder states.
        stfs_mask: optional ``(B,T,Q)`` or ``(BT,Q)`` bool mask. When present,
            only injected/modified STFS slots contribute.
        teacher_weights: optional ``(B,T,Qt)`` or ``(BT,Qt)`` foreground
            weights used to select teacher top-k slots.
        teacher_topk: number of teacher slots kept per frame when weights are
            available. Values <=0 keep all teacher slots.
    """
    if student_hs.dim() == 4:
        student = student_hs.reshape(-1, student_hs.shape[-2], student_hs.shape[-1])
    elif student_hs.dim() == 3:
        student = student_hs
    else:
        raise ValueError(
            "student_hs must be (B,T,Q,D) or (BT,Q,D), "
            f"got {tuple(student_hs.shape)}"
        )

    if teacher_hs.dim() == 4:
        teacher = teacher_hs.reshape(-1, teacher_hs.shape[-2], teacher_hs.shape[-1])
    elif teacher_hs.dim() == 3:
        teacher = teacher_hs
    else:
        raise ValueError(
            "teacher_hs must be (B,T,Q,D) or (BT,Q,D), "
            f"got {tuple(teacher_hs.shape)}"
        )

    if student.shape[0] != teacher.shape[0]:
        raise ValueError(
            f"student/teacher BT mismatch: {student.shape[0]} vs {teacher.shape[0]}"
        )
    if student.shape[-1] != teacher.shape[-1]:
        raise ValueError(
            f"student/teacher hidden dim mismatch: {student.shape[-1]} vs {teacher.shape[-1]}"
        )

    if stfs_mask is None:
        mask = torch.ones(student.shape[:2], dtype=torch.bool, device=student.device)
    else:
        mask = stfs_mask.reshape(student.shape[0], student.shape[1]).to(
            device=student.device, dtype=torch.bool,
        )

    if not mask.any():
        return student.sum() * 0.0

    teacher = teacher.detach()
    if teacher_weights is not None and int(teacher_topk) > 0:
        weights = teacher_weights.reshape(teacher.shape[0], teacher.shape[1]).detach()
        weights = weights.to(device=teacher.device)
        topk = min(int(teacher_topk), teacher.shape[1])
        idx = weights.topk(topk, dim=1).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, teacher.shape[-1])
        teacher = teacher.gather(1, gather_idx)

    student_norm = F.normalize(student, dim=-1)
    teacher_norm = F.normalize(teacher, dim=-1)
    cosine = torch.einsum("bqd,bkd->bqk", student_norm, teacher_norm)
    nearest = cosine.max(dim=-1).values
    return (1.0 - nearest[mask]).mean()
