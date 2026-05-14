"""Checkpoint-selection and early-stopping helpers for Video RF-DETR.

The per-epoch validation metric is noisy (~0.04 swings on a small val split), so
checkpoint selection rides a *smoothed composite* score rather than a single
greedy ``argmax(AP@0.3)``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def composite_selection_score(
    metrics: Dict[str, float],
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2),
) -> float:
    """Weighted sum of AP@0.3 / AP@0.5 / F1 (missing keys count as 0.0)."""
    w3, w5, wf = weights
    return (
        w3 * float(metrics.get("AP@0.3", 0.0))
        + w5 * float(metrics.get("AP@0.5", 0.0))
        + wf * float(metrics.get("F1", 0.0))
    )


class SmoothedTracker:
    """Rolling mean of the last ``k`` values."""

    def __init__(self, k: int = 3):
        self.k = max(int(k), 1)
        self.values: List[float] = []

    def add(self, value: float) -> float:
        self.values.append(float(value))
        window = self.values[-self.k:]
        return sum(window) / len(window)


class EarlyStopper:
    """Stop after ``patience`` consecutive evals without ``min_delta`` improvement."""

    def __init__(self, patience: int = 6, min_delta: float = 0.0):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best = float("-inf")
        self.num_bad = 0

    def update(self, score: float) -> bool:
        """Feed the latest score; return True if training should stop."""
        if score > self.best + self.min_delta:
            self.best = score
            self.num_bad = 0
        else:
            self.num_bad += 1
        return self.num_bad >= self.patience
