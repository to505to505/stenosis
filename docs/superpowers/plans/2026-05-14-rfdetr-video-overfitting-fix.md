# Video RF-DETR Overfitting Fix (Stage 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the from-scratch training regime in `rfdetr_video/` with a fine-tuning regime — differential LR, cosine schedule, short budget + early stopping, weight EMA + smoothed selection — and add a diagnostic sweep.

**Architecture:** Three new small, light-to-import helper modules (`ema.py`, `schedule.py`, `selection.py`), a reworked `VideoRFDETR.get_param_groups()` (3 LR groups), new `Config` fields, and `train.py` wiring that evaluates an EMA model, selects `best.pth` on a smoothed composite score, and early-stops. A `_eval_stfs_ablations.py` CLI tweak + a sweep shell script drive the 5-6 run experiment.

**Tech Stack:** Python, PyTorch, pytest. Tests live in `rfdetr_video/tests/test_smoke.py` (existing convention; heavy model-build tests gated by `RFDETR_VIDEO_HEAVY=1`).

**Spec:** `docs/superpowers/specs/2026-05-14-rfdetr-video-overfitting-fix-design.md`

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `rfdetr_video/config.py` | modify | New `Config` fields; `epochs` default 50→20 |
| `rfdetr_video/ema.py` | create | `ModelEMA` — EMA of trainable params, swap-in context manager |
| `rfdetr_video/schedule.py` | create | `build_scheduler(optimizer, cfg)` — cosine or multistep |
| `rfdetr_video/selection.py` | create | `composite_selection_score`, `SmoothedTracker`, `EarlyStopper` |
| `rfdetr_video/model.py` | modify | `_param_group_for` + reworked `get_param_groups` (3 groups) |
| `rfdetr_video/train.py` | modify | Wire EMA / scheduler / selection / early-stop; new argparse flags |
| `_eval_stfs_ablations.py` | modify | `main()` accepts a run name (CLI arg) |
| `rfdetr_video/tools/overfitting_sweep.sh` | create | Per-run sweep launcher (`R0`..`R5`) |
| `rfdetr_video/tests/test_smoke.py` | modify | Unit tests for all of the above |

---

## Task 1: Config fields

**Files:**
- Modify: `rfdetr_video/config.py:34-44` (the `# Training` block)
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `rfdetr_video/tests/test_smoke.py`:

```python
def test_config_has_regime_fields():
    cfg = Config()
    assert cfg.epochs == 20
    assert cfg.lr == 1e-4
    assert cfg.lr_pretrained == 3e-5
    assert cfg.lr_schedule == "cosine"
    assert cfg.ema_enabled is True
    assert cfg.ema_decay == 0.999
    assert cfg.selection_smooth_k == 3
    assert cfg.selection_weights == (0.5, 0.3, 0.2)
    assert cfg.early_stop_enabled is True
    assert cfg.early_stop_patience == 6
    assert cfg.early_stop_min_delta == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_config_has_regime_fields -v`
Expected: FAIL — `AttributeError: 'Config' object has no attribute 'lr_pretrained'` (or `assert 50 == 20`).

- [ ] **Step 3: Write minimal implementation**

In `rfdetr_video/config.py`, replace the `# Training` block (currently lines 34-44):

```python
    # Training
    epochs: int = 50
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    lr_step_milestones: tuple = (30, 40)
    lr_gamma: float = 0.1
    grad_accum_steps: int = 2
```

with:

```python
    # Training
    epochs: int = 20
    batch_size: int = 2
    num_workers: int = 4
    lr: float = 1e-4                 # LR for new modules (ETF / CRRCD)
    lr_pretrained: float = 3e-5      # LR for pretrained detector (transformer + heads)
    lr_backbone: float = 1e-5        # LR for backbone (unused while frozen)
    lr_schedule: str = "cosine"      # "cosine" | "multistep"
    weight_decay: float = 1e-4
    warmup_iters: int = 500
    lr_step_milestones: tuple = (30, 40)   # only used when lr_schedule == "multistep"
    lr_gamma: float = 0.1
    grad_accum_steps: int = 2

    # EMA + checkpoint selection + early stopping
    ema_enabled: bool = True
    ema_decay: float = 0.999
    selection_smooth_k: int = 3
    selection_weights: tuple = (0.5, 0.3, 0.2)   # weights for (AP@0.3, AP@0.5, F1)
    early_stop_enabled: bool = True
    early_stop_patience: int = 6
    early_stop_min_delta: float = 0.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_config_has_regime_fields -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rfdetr_video/config.py rfdetr_video/tests/test_smoke.py
git commit -m "feat(rfdetr_video): add fine-tuning regime config fields"
```

---

## Task 2: ModelEMA

**Files:**
- Create: `rfdetr_video/ema.py`
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `rfdetr_video/tests/test_smoke.py` (top-level import + tests):

```python
from rfdetr_video.ema import ModelEMA
```

```python
def _tiny_model_with_frozen():
    import torch.nn as nn
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    # freeze the second layer
    for p in model[1].parameters():
        p.requires_grad_(False)
    return model


def test_model_ema_tracks_only_trainable_params():
    import torch.nn as nn
    model = _tiny_model_with_frozen()
    ema = ModelEMA(model, decay=0.9)
    # shadow holds exactly the trainable param names
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert set(ema.shadow.keys()) == trainable


def test_model_ema_update_math():
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    model = nn.Linear(3, 3)
    ema = ModelEMA(model, decay=0.8)
    before = {n: v.clone() for n, v in ema.shadow.items()}
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # move the live model
    ema.update(model)
    for n, p in model.named_parameters():
        expected = before[n] * 0.8 + p.detach() * 0.2
        torch.testing.assert_close(ema.shadow[n], expected)


def test_model_ema_constant_param_stays_constant():
    import torch
    import torch.nn as nn
    model = nn.Linear(3, 3)
    ema = ModelEMA(model, decay=0.5)
    snapshot = {n: v.clone() for n, v in ema.shadow.items()}
    ema.update(model)  # model unchanged → EMA unchanged
    for n in snapshot:
        torch.testing.assert_close(ema.shadow[n], snapshot[n])


def test_model_ema_applied_to_round_trips():
    import torch
    import torch.nn as nn
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.5)
    # force a known EMA value
    for n in ema.shadow:
        ema.shadow[n].fill_(7.0)
    original = {n: p.detach().clone() for n, p in model.named_parameters()}
    with ema.applied_to(model):
        for n, p in model.named_parameters():
            assert torch.all(p == 7.0)
    for n, p in model.named_parameters():
        torch.testing.assert_close(p.detach(), original[n])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py -v -k model_ema`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'rfdetr_video.ema'`.

- [ ] **Step 3: Write minimal implementation**

Create `rfdetr_video/ema.py`:

```python
"""Exponential moving average of a model's trainable parameters.

Frozen parameters (``requires_grad=False``) are never copied — the EMA is a
plain dict of the trainable parameter tensors. ``applied_to`` temporarily swaps
the EMA tensors into a live model for evaluation / checkpoint saving and then
restores the live parameters, so no second full model is held in memory.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.detach(), alpha=1.0 - self.decay,
            )

    @contextmanager
    def applied_to(self, model: nn.Module):
        backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.detach().clone()
                    param.copy_(self.shadow[name])
        try:
            yield model
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.copy_(backup[name])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rfdetr_video/tests/test_smoke.py -v -k model_ema`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rfdetr_video/ema.py rfdetr_video/tests/test_smoke.py
git commit -m "feat(rfdetr_video): add ModelEMA for trainable-param weight averaging"
```

---

## Task 3: build_scheduler

**Files:**
- Create: `rfdetr_video/schedule.py`
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `rfdetr_video/tests/test_smoke.py` (top-level import + tests):

```python
from rfdetr_video.schedule import build_scheduler
```

```python
def _two_group_optimizer():
    import torch
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.SGD([
        {"params": [p1], "lr": 1e-4},
        {"params": [p2], "lr": 3e-5},
    ], lr=1e-4)


def test_build_scheduler_cosine_decays_all_groups():
    cfg = Config(epochs=10, lr_schedule="cosine")
    opt = _two_group_optimizer()
    sched = build_scheduler(opt, cfg)
    start = [pg["lr"] for pg in opt.param_groups]
    for _ in range(cfg.epochs):
        sched.step()
    end = [pg["lr"] for pg in opt.param_groups]
    assert end[0] < start[0] and end[1] < start[1]
    assert end[0] < 1e-5 and end[1] < 1e-5  # near eta_min


def test_build_scheduler_multistep_passthrough():
    cfg = Config(lr_schedule="multistep", lr_step_milestones=(2,), lr_gamma=0.1)
    opt = _two_group_optimizer()
    sched = build_scheduler(opt, cfg)
    sched.step(); sched.step()  # hit milestone 2
    lrs = [pg["lr"] for pg in opt.param_groups]
    torch.testing.assert_close(lrs[0], 1e-5)   # 1e-4 * 0.1
    torch.testing.assert_close(lrs[1], 3e-6)   # 3e-5 * 0.1


def test_build_scheduler_rejects_unknown():
    cfg = Config(lr_schedule="bogus")
    opt = _two_group_optimizer()
    with pytest.raises(ValueError, match="lr_schedule"):
        build_scheduler(opt, cfg)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py -v -k build_scheduler`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'rfdetr_video.schedule'`.

- [ ] **Step 3: Write minimal implementation**

Create `rfdetr_video/schedule.py`:

```python
"""LR scheduler factory for Video RF-DETR.

``cosine`` decays every param group from its base LR toward ``eta_min`` over the
epoch budget; ``multistep`` preserves the legacy step schedule. The per-iteration
linear warmup in ``train.py`` is unchanged and hands off to whichever scheduler
this returns.
"""

from __future__ import annotations

from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


def build_scheduler(optimizer, cfg):
    if cfg.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    if cfg.lr_schedule == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=list(cfg.lr_step_milestones),
            gamma=cfg.lr_gamma,
        )
    raise ValueError(
        f"unknown lr_schedule={cfg.lr_schedule!r}; expected 'cosine' or 'multistep'"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rfdetr_video/tests/test_smoke.py -v -k build_scheduler`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rfdetr_video/schedule.py rfdetr_video/tests/test_smoke.py
git commit -m "feat(rfdetr_video): add cosine/multistep scheduler factory"
```

---

## Task 4: Selection helpers

**Files:**
- Create: `rfdetr_video/selection.py`
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `rfdetr_video/tests/test_smoke.py` (top-level import + tests):

```python
from rfdetr_video.selection import (
    composite_selection_score,
    SmoothedTracker,
    EarlyStopper,
)
```

```python
def test_composite_selection_score_weighted_sum():
    metrics = {"AP@0.3": 1.0, "AP@0.5": 1.0, "F1": 1.0, "other": 9.0}
    assert composite_selection_score(metrics, (0.5, 0.3, 0.2)) == pytest.approx(1.0)
    metrics2 = {"AP@0.3": 1.0, "AP@0.5": 0.0, "F1": 0.0}
    assert composite_selection_score(metrics2, (0.5, 0.3, 0.2)) == pytest.approx(0.5)
    # missing keys default to 0.0
    assert composite_selection_score({}, (0.5, 0.3, 0.2)) == pytest.approx(0.0)


def test_smoothed_tracker_rolling_mean():
    tracker = SmoothedTracker(k=3)
    assert tracker.add(1.0) == pytest.approx(1.0)
    assert tracker.add(2.0) == pytest.approx(1.5)
    assert tracker.add(3.0) == pytest.approx(2.0)
    assert tracker.add(4.0) == pytest.approx(3.0)  # window = [2,3,4]


def test_early_stopper_triggers_after_patience():
    stopper = EarlyStopper(patience=2, min_delta=0.0)
    assert stopper.update(0.5) is False   # new best
    assert stopper.update(0.4) is False   # bad 1
    assert stopper.update(0.4) is True    # bad 2 -> stop


def test_early_stopper_resets_on_improvement():
    stopper = EarlyStopper(patience=2, min_delta=0.0)
    assert stopper.update(0.5) is False
    assert stopper.update(0.4) is False   # bad 1
    assert stopper.update(0.6) is False   # new best -> reset
    assert stopper.update(0.5) is False   # bad 1
    assert stopper.update(0.5) is True    # bad 2 -> stop
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py -v -k "composite_selection or smoothed_tracker or early_stopper"`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'rfdetr_video.selection'`.

- [ ] **Step 3: Write minimal implementation**

Create `rfdetr_video/selection.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rfdetr_video/tests/test_smoke.py -v -k "composite_selection or smoothed_tracker or early_stopper"`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add rfdetr_video/selection.py rfdetr_video/tests/test_smoke.py
git commit -m "feat(rfdetr_video): add smoothed-composite selection + early stopper"
```

---

## Task 5: Differential-LR param groups

**Files:**
- Modify: `rfdetr_video/model.py` — add `_param_group_for` (module level) + rewrite `get_param_groups` (currently lines 439-453)
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `rfdetr_video/tests/test_smoke.py`:

```python
def test_param_group_for_classifies_by_name():
    from rfdetr_video.model import _param_group_for
    assert _param_group_for("backbone.blocks.0.attn.qkv.weight") == "backbone"
    assert _param_group_for("etf.attn.in_proj_weight") == "new"
    assert _param_group_for("etf.norm.weight") == "new"
    assert _param_group_for("crrcd.relation_t.0.weight") == "new"
    assert _param_group_for("transformer.decoder.layers.0.self_attn.weight") == "pretrained"
    assert _param_group_for("class_embed.weight") == "pretrained"
    assert _param_group_for("bbox_embed.layers.0.weight") == "pretrained"
    assert _param_group_for("refpoint_embed.weight") == "pretrained"
    assert _param_group_for("query_feat.weight") == "pretrained"


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
def test_get_param_groups_three_way_split():
    from rfdetr_video.model import VideoRFDETR
    cfg = Config(etf_enabled=True, lr=1e-4, lr_pretrained=3e-5, freeze_backbone=True)
    model = VideoRFDETR(cfg)
    groups = model.get_param_groups()
    # backbone frozen -> only pretrained + new groups remain
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == [3e-5, 1e-4]
    by_lr = {g["lr"]: g["params"] for g in groups}
    # ETF params land in the 1e-4 (new) group
    etf_param_ids = {id(p) for p in model.etf.parameters()}
    new_group_ids = {id(p) for p in by_lr[1e-4]}
    assert etf_param_ids and etf_param_ids.issubset(new_group_ids)
    # transformer params land in the 3e-5 (pretrained) group
    tf_param_ids = {id(p) for p in model.transformer.parameters() if p.requires_grad}
    pre_group_ids = {id(p) for p in by_lr[3e-5]}
    assert tf_param_ids and tf_param_ids.issubset(pre_group_ids)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_param_group_for_classifies_by_name -v`
Expected: FAIL — `ImportError: cannot import name '_param_group_for'`.

- [ ] **Step 3: Write minimal implementation**

In `rfdetr_video/model.py`, add this module-level function (place it just above the `VideoRFDETR` class definition):

```python
def _param_group_for(name: str) -> str:
    """Classify a parameter name into an LR group: backbone / new / pretrained."""
    if name.startswith("backbone"):
        return "backbone"
    if name.startswith("etf") or name.startswith("crrcd"):
        return "new"
    return "pretrained"
```

Then replace `get_param_groups` (currently lines 439-453):

```python
    def get_param_groups(self):
        backbone_params = []
        decoder_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                decoder_params.append(param)
        groups = []
        if decoder_params:
            groups.append({"params": decoder_params, "lr": self.cfg.lr})
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.cfg.lr_backbone})
        return groups
```

with:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_param_group_for_classifies_by_name -v`
Expected: PASS.
Run (if a GPU + checkpoint are available): `RFDETR_VIDEO_HEAVY=1 pytest rfdetr_video/tests/test_smoke.py::test_get_param_groups_three_way_split -v`
Expected: PASS — and the captured stdout shows non-zero `pretrained=` and `new=` counts.

- [ ] **Step 5: Commit**

```bash
git add rfdetr_video/model.py rfdetr_video/tests/test_smoke.py
git commit -m "feat(rfdetr_video): differential LR — 3-way param-group split"
```

---

## Task 6: Wire EMA / scheduler / selection / early-stop into train.py

**Files:**
- Modify: `rfdetr_video/train.py` — imports, scheduler construction, EMA init, the optimizer-step block, the eval/selection block, end-of-epoch save, final summary
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing tests**

Add to `rfdetr_video/tests/test_smoke.py`:

```python
def test_train_source_wires_regime_fix():
    src = (ROOT / "rfdetr_video" / "train.py").read_text()
    assert "from .ema import ModelEMA" in src
    assert "from .schedule import build_scheduler" in src
    assert "from .selection import" in src
    assert "build_scheduler(optimizer, cfg)" in src
    assert "ema.update(model)" in src
    assert "ema.applied_to(model)" in src
    assert "composite_selection_score" in src
    assert "EarlyStopper" in src
    # the old greedy MultiStepLR construction is gone
    assert "MultiStepLR(" not in src


@pytest.mark.skipif(not HEAVY, reason=HEAVY_REASON)
@pytest.mark.skipif(
    not (ROOT / "data" / "dataset2_split").exists(),
    reason="data/dataset2_split not available",
)
def test_train_one_epoch_smoke(tmp_path):
    from rfdetr_video.train import train
    cfg = Config(
        data_root=ROOT / "data" / "dataset2_split",
        output_dir=tmp_path,
        run_name="smoke",
        epochs=1,
        batch_size=1,
        num_workers=0,
        T=3,
        img_size=384,
        distill_enabled=False,
        consistency_enabled=False,
        etf_enabled=True,
        ema_enabled=True,
        early_stop_enabled=False,
        wandb_enabled=False,
    )
    run_dir = train(cfg)
    assert (run_dir / "best.pth").exists()
    assert (run_dir / "last_ema.pth").exists()
    history = json.loads((run_dir / "history.json").read_text())
    assert history and "ema/AP@0.3" in history[0] and "sel_smoothed" in history[0]
```

(Add `import json` to the test file's imports if not already present — it is not, the existing file only imports `os`, `subprocess`, `Path`. Add `import json` near them.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_train_source_wires_regime_fix -v`
Expected: FAIL — `assert "from .ema import ModelEMA" in src` is False.

- [ ] **Step 3a: Add imports**

In `rfdetr_video/train.py`, after the distill import block (currently lines 39-43, ending `)`), add:

```python
from .ema import ModelEMA
from .schedule import build_scheduler
from .selection import composite_selection_score, SmoothedTracker, EarlyStopper
```

- [ ] **Step 3b: Replace the scheduler construction**

Replace (currently lines 228-230):

```python
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(cfg.lr_step_milestones), gamma=cfg.lr_gamma,
    )
```

with:

```python
    scheduler = build_scheduler(optimizer, cfg)
```

- [ ] **Step 3c: Instantiate the EMA**

Immediately after `scaler = GradScaler(enabled=cfg.amp)` (currently line 231), add:

```python
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_enabled else None
```

- [ ] **Step 3d: Initialise selection + early-stop state**

Replace (currently lines 252-256):

```python
    best_map30 = 0.0
    best_metrics: dict = {}
    best_epoch = 0
    history: list = []
    global_step = 0
```

with:

```python
    best_sel = float("-inf")
    best_metrics: dict = {}
    best_epoch = 0
    history: list = []
    global_step = 0
    sel_tracker = SmoothedTracker(cfg.selection_smooth_k)
    early_stopper = (
        EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)
        if cfg.early_stop_enabled else None
    )
```

- [ ] **Step 3e: Update the EMA inside the optimizer-step block**

Replace (currently lines 382-387):

```python
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
```

with:

```python
            if (batch_idx + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if ema is not None:
                    ema.update(model)
```

- [ ] **Step 3f: Rewrite the eval / selection / early-stop block**

Replace the entire `if (epoch + 1) % cfg.eval_interval == 0:` block (currently lines 426-456):

```python
        if (epoch + 1) % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, criterion, postprocess, cfg, device)
            record = {"epoch": epoch + 1, "train_loss": train_loss, **metrics}
            history.append(record)
            print(
                f"  val — mAP30={metrics['AP@0.3']:.4f}  "
                f"mAP50={metrics['AP@0.5']:.4f}  "

                f"F1={metrics['F1']:.4f}  "
                f"all/mAP30={metrics.get('all/AP@0.3', 0):.4f}  "
                f"val_loss={metrics.get('val_loss', 0):.4f}"
            )
            if cfg.wandb_enabled:
                import wandb
                log_dict = {"epoch": epoch + 1, "train_loss": train_loss}
                for k, v in metrics.items():
                    log_dict[f"val/{k}"] = v
                wandb.log(log_dict, step=global_step)
            if metrics["AP@0.3"] > best_map30:
                best_map30 = metrics["AP@0.3"]
                best_metrics = metrics.copy()
                best_epoch = epoch + 1
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1, **metrics},
                    run_dir / "best.pth",
                )
                write_best_txt(run_dir, best_metrics, best_epoch, cfg)
                print(f"  ★ New best micro mAP@0.3={best_map30:.4f}")
            with open(run_dir / "history.json", "w") as _f:
                json.dump(history, _f, indent=2)
            save_train_csv(run_dir, history)
```

with:

```python
        if (epoch + 1) % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, criterion, postprocess, cfg, device)
            if ema is not None:
                with ema.applied_to(model):
                    ema_metrics = evaluate(
                        model, val_loader, criterion, postprocess, cfg, device,
                    )
            else:
                ema_metrics = metrics
            sel = composite_selection_score(ema_metrics, cfg.selection_weights)
            sel_smoothed = sel_tracker.add(sel)
            record = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                **metrics,
                **{f"ema/{k}": v for k, v in ema_metrics.items()},
                "sel": sel,
                "sel_smoothed": sel_smoothed,
            }
            history.append(record)
            print(
                f"  val — mAP30={metrics['AP@0.3']:.4f} "
                f"| ema mAP30={ema_metrics['AP@0.3']:.4f} "
                f"ema mAP50={ema_metrics['AP@0.5']:.4f} "
                f"ema F1={ema_metrics['F1']:.4f} "
                f"| sel={sel:.4f} sel_smoothed={sel_smoothed:.4f} "
                f"val_loss={metrics.get('val_loss', 0):.4f}"
            )
            if cfg.wandb_enabled:
                import wandb
                log_dict = {"epoch": epoch + 1, "train_loss": train_loss,
                            "sel": sel, "sel_smoothed": sel_smoothed}
                for k, v in metrics.items():
                    log_dict[f"val/{k}"] = v
                for k, v in ema_metrics.items():
                    log_dict[f"val_ema/{k}"] = v
                wandb.log(log_dict, step=global_step)
            if sel_smoothed > best_sel:
                best_sel = sel_smoothed
                best_metrics = {**ema_metrics, "sel_smoothed": sel_smoothed}
                best_epoch = epoch + 1
                ema_ctx = ema.applied_to(model) if ema is not None else _nullcontext()
                with ema_ctx:
                    torch.save(
                        {"model": model.state_dict(), "epoch": epoch + 1,
                         **ema_metrics},
                        run_dir / "best.pth",
                    )
                write_best_txt(run_dir, best_metrics, best_epoch, cfg)
                print(f"  ★ New best smoothed sel={best_sel:.4f} (epoch {epoch + 1})")
            with open(run_dir / "history.json", "w") as _f:
                json.dump(history, _f, indent=2)
            save_train_csv(run_dir, history)
            if early_stopper is not None and early_stopper.update(sel_smoothed):
                print(
                    f"  ⨯ Early stop — no smoothed-sel improvement for "
                    f"{cfg.early_stop_patience} evals"
                )
                break
```

- [ ] **Step 3g: Add the `_nullcontext` import**

`_nullcontext` is used in Step 3f so the `with` block works whether or not EMA is enabled. In `rfdetr_video/train.py`, immediately after the line `from typing import Dict, List` (currently line 20), add:

```python
from contextlib import nullcontext as _nullcontext
```

- [ ] **Step 3h: Save `last_ema.pth` each epoch**

Replace (currently lines 458-461):

```python
        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1},
            run_dir / "last.pth",
        )
```

with:

```python
        torch.save(
            {"model": model.state_dict(), "epoch": epoch + 1},
            run_dir / "last.pth",
        )
        if ema is not None:
            with ema.applied_to(model):
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1},
                    run_dir / "last_ema.pth",
                )
```

- [ ] **Step 3i: Fix the final-summary references to `best_map30`**

Replace (currently lines 476-477):

```python
    print(f"\nTraining complete. Best micro mAP@0.3={best_map30:.4f}")
    print(f"Outputs saved to {run_dir}")
```

with:

```python
    print(
        f"\nTraining complete. Best EMA mAP@0.3="
        f"{best_metrics.get('AP@0.3', 0):.4f} "
        f"(smoothed sel={best_sel:.4f}, epoch {best_epoch})"
    )
    print(f"Outputs saved to {run_dir}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_train_source_wires_regime_fix -v`
Expected: PASS.
Run: `pytest rfdetr_video/tests/test_smoke.py -v -k "not heavy"`
Expected: PASS — no import errors, all light tests green.
Run (if a GPU + checkpoint + data are available): `RFDETR_VIDEO_HEAVY=1 pytest rfdetr_video/tests/test_smoke.py::test_train_one_epoch_smoke -v`
Expected: PASS — `best.pth`, `last_ema.pth` written; `history.json[0]` has `ema/AP@0.3` and `sel_smoothed`.

- [ ] **Step 5: Commit**

```bash
git add rfdetr_video/train.py rfdetr_video/tests/test_smoke.py
git commit -m "feat(rfdetr_video): wire EMA, cosine schedule, smoothed selection, early stop"
```

---

## Task 7: Argparse flags for the new regime knobs

**Files:**
- Modify: `rfdetr_video/train.py` — `parse_args()` and the `cfg_kwargs` builder in `__main__`
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `rfdetr_video/tests/test_smoke.py`:

```python
def test_train_cli_exposes_regime_flags():
    out = subprocess.run(
        ["python", "-m", "rfdetr_video.train", "--help"],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    assert out.returncode == 0, out.stderr
    for flag in (
        "--lr-pretrained", "--lr-schedule", "--weight-decay",
        "--no-ema", "--ema-decay", "--no-early-stop", "--early-stop-patience",
        "--eval-interval",
    ):
        assert flag in out.stdout, f"missing CLI flag: {flag}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_train_cli_exposes_regime_flags -v`
Expected: FAIL — `missing CLI flag: --lr-pretrained`.

- [ ] **Step 3a: Add the argparse arguments**

In `rfdetr_video/train.py` `parse_args()`, immediately after the `--img-size` line (currently line 516, `p.add_argument("--img-size", type=int, default=None)`), add:

```python
    # Fine-tuning regime
    p.add_argument("--lr-pretrained", type=float, default=None,
                   help="LR for the pretrained detector (transformer + heads).")
    p.add_argument("--lr-schedule", type=str, default=None,
                   choices=["cosine", "multistep"])
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--no-ema", action="store_true",
                   help="Disable weight EMA (selection then uses the raw model).")
    p.add_argument("--ema-decay", type=float, default=None)
    p.add_argument("--no-early-stop", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None,
                   help="Run validation every N epochs (default 1).")
```

- [ ] **Step 3b: Wire the arguments into `cfg_kwargs`**

In `rfdetr_video/train.py` `__main__`, immediately after the `_maybe(cfg_kwargs, "img_size", args.img_size, int)` line (currently line 605), add:

```python
    _maybe(cfg_kwargs, "lr_pretrained", args.lr_pretrained, float)
    _maybe(cfg_kwargs, "lr_schedule", args.lr_schedule, str)
    _maybe(cfg_kwargs, "weight_decay", args.weight_decay, float)
    if args.no_ema:
        cfg_kwargs["ema_enabled"] = False
    _maybe(cfg_kwargs, "ema_decay", args.ema_decay, float)
    if args.no_early_stop:
        cfg_kwargs["early_stop_enabled"] = False
    _maybe(cfg_kwargs, "early_stop_patience", args.early_stop_patience, int)
    _maybe(cfg_kwargs, "eval_interval", args.eval_interval, int)
```

- [ ] **Step 3c: Update the `--epochs` default**

Replace (currently line 503):

```python
    p.add_argument("--epochs", type=int, default=50)
```

with:

```python
    p.add_argument("--epochs", type=int, default=20)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_train_cli_exposes_regime_flags -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rfdetr_video/train.py rfdetr_video/tests/test_smoke.py
git commit -m "feat(rfdetr_video): CLI flags for differential LR, schedule, EMA, early stop"
```

---

## Task 8: `_eval_stfs_ablations.py` — accept a run name argument

**Files:**
- Modify: `_eval_stfs_ablations.py` — `main()` signature + the 4 internal `RUN_NAME` references + the `__main__` block
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `rfdetr_video/tests/test_smoke.py` (text-based check, consistent with `test_source_invariants` — avoids executing the heavy ablation module):

```python
def test_eval_ablations_main_accepts_run_name():
    src = (ROOT / "_eval_stfs_ablations.py").read_text()
    assert "def main(run_name" in src
    assert "main(sys.argv[1]" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_eval_ablations_main_accepts_run_name -v`
Expected: FAIL — `assert "def main(run_name" in src` is False (`main()` currently takes no args).

- [ ] **Step 3a: Make `main()` take a `run_name` argument**

In `_eval_stfs_ablations.py`, replace the `main()` signature and its first line (currently lines 376-377):

```python
def main():
    run_dir = ROOT / "rfdetr_video" / "runs" / RUN_NAME
```

with:

```python
def main(run_name: str = RUN_NAME):
    run_dir = ROOT / "rfdetr_video" / "runs" / run_name
```

- [ ] **Step 3b: Replace the remaining `RUN_NAME` uses inside `main()`**

Inside `main()`, three more references use the module global. Replace each `RUN_NAME` with `run_name`:

- Line 379: `print(f"# STFS Ablation Tests – {RUN_NAME}")` → `print(f"# STFS Ablation Tests – {run_name}")`
- Line 435: `f"STFS Ablation Tests – {RUN_NAME}\n"` → `f"STFS Ablation Tests – {run_name}\n"`
- Line 445: `"run": RUN_NAME,` → `"run": run_name,`

- [ ] **Step 3c: Pass `sys.argv[1]` from the entrypoint**

Add `import sys` to the imports at the top of the file (it currently imports `json`, `time`, `from collections import defaultdict`, `from pathlib import Path` — add `import sys` alongside them).

Replace the entrypoint (currently lines 465-466):

```python
if __name__ == "__main__":
    main()
```

with:

```python
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RUN_NAME)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_eval_ablations_main_accepts_run_name -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add _eval_stfs_ablations.py rfdetr_video/tests/test_smoke.py
git commit -m "feat: _eval_stfs_ablations accepts run name as CLI argument"
```

---

## Task 9: Overfitting sweep launcher script

**Files:**
- Create: `rfdetr_video/tools/overfitting_sweep.sh`
- Test: `rfdetr_video/tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

Add to `rfdetr_video/tests/test_smoke.py`:

```python
def test_overfitting_sweep_script_valid_and_complete():
    script = ROOT / "rfdetr_video" / "tools" / "overfitting_sweep.sh"
    assert script.exists(), "overfitting_sweep.sh not created"
    # bash syntax check
    syntax = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert syntax.returncode == 0, syntax.stderr
    # all six run names are handled
    text = script.read_text()
    for run in ("R0", "R1", "R2", "R3", "R4", "R5"):
        assert run in text, f"sweep run {run} not handled"
    # unknown run name exits non-zero
    bad = subprocess.run(
        ["bash", str(script), "RX"], capture_output=True, text=True, cwd=str(ROOT),
    )
    assert bad.returncode != 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_overfitting_sweep_script_valid_and_complete -v`
Expected: FAIL — `overfitting_sweep.sh not created`.

- [ ] **Step 3: Write the script**

Create `rfdetr_video/tools/overfitting_sweep.sh`:

```bash
#!/usr/bin/env bash
# Stage 1 overfitting-fix diagnostic sweep — one lever at a time.
#
# Usage:  bash rfdetr_video/tools/overfitting_sweep.sh <R0|R1|R2|R3|R4|R5>
#
# Base model config = the current "main" (etf + distill + crrcd + consistency,
# centre-frame KD). Only training-regime knobs vary between runs.
# All runs: batch-size 16, validation every 2 epochs (--eval-interval 2).
#
#   R0  anchor: diff-LR (pre 3e-5 / new 1e-4), cosine, 20 ep, etf_dropout 0.1, wd 1e-3
#   R1  no differential LR (pretrained = new = 1e-4)
#   R2  aggressive differential LR (pretrained 1e-5)
#   R3  no extra regularization (etf_dropout 0.0, wd 1e-4)
#   R4  long budget (35 epochs, still cosine + early-stop)
#   R5  EMA off
set -euo pipefail

RUN="${1:?usage: overfitting_sweep.sh <R0|R1|R2|R3|R4|R5>}"
CKPT="rfdetr_runs/rfdetr_small_arcade2x_512_reg/checkpoint_best_regular.pth"

# Anchor (R0) values.
LR_NEW=1e-4
LR_PRE=3e-5
EPOCHS=20
ETF_DROPOUT=0.1
WD=1e-3
EMA_FLAG="--ema-decay 0.999"

case "$RUN" in
  R0) ;;
  R1) LR_PRE=1e-4 ;;
  R2) LR_PRE=1e-5 ;;
  R3) ETF_DROPOUT=0.0; WD=1e-4 ;;
  R4) EPOCHS=35 ;;
  R5) EMA_FLAG="--no-ema" ;;
  *) echo "unknown sweep run: $RUN (expected R0..R5)" >&2; exit 1 ;;
esac

echo "=== overfitting sweep: $RUN ==="
echo "    lr(new)=$LR_NEW lr(pretrained)=$LR_PRE epochs=$EPOCHS etf_dropout=$ETF_DROPOUT wd=$WD ema=$EMA_FLAG"

python -m rfdetr_video.train \
  --dataset data/dataset2_split \
  --checkpoint "$CKPT" \
  --run-name "video_overfit_${RUN}" \
  --img-size 512 --T 5 --batch-size 16 --grad-accum 1 --num-workers 4 \
  --etf --distill --crrcd --distill-centre-frame-only \
  --lr-schedule cosine --eval-interval 2 \
  --lr "$LR_NEW" --lr-pretrained "$LR_PRE" \
  --epochs "$EPOCHS" --etf-dropout "$ETF_DROPOUT" --weight-decay "$WD" \
  $EMA_FLAG
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest rfdetr_video/tests/test_smoke.py::test_overfitting_sweep_script_valid_and_complete -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rfdetr_video/tools/overfitting_sweep.sh rfdetr_video/tests/test_smoke.py
git commit -m "feat(rfdetr_video): overfitting-fix diagnostic sweep launcher"
```

---

## Task 10: Full test-suite green + plan-complete commit

**Files:** none (verification only)

- [ ] **Step 1: Run the full light test suite**

Run: `pytest rfdetr_video/tests/test_smoke.py -v -k "not heavy"`
Expected: PASS — every light test green, no import errors, no collection errors.

- [ ] **Step 2: Run the heavy suite if a GPU + checkpoints + data are available**

Run: `RFDETR_VIDEO_HEAVY=1 pytest rfdetr_video/tests/test_smoke.py -v`
Expected: PASS — including `test_get_param_groups_three_way_split` and `test_train_one_epoch_smoke`. If the environment lacks a GPU/checkpoint, those tests `skip` (not fail) — that is acceptable; note it in the execution report.

- [ ] **Step 3: Commit any remaining changes**

```bash
git add -A rfdetr_video/ _eval_stfs_ablations.py
git status   # confirm nothing unexpected is staged
git commit -m "test(rfdetr_video): Stage 1 overfitting-fix suite green" || echo "nothing to commit"
```

---

## Execution: running the sweep (after the plan is implemented)

The sweep is run **after** all tasks above are merged. Per the user's instruction
("after the first run finishes you need to start second one"), the runs are
**chained** — each launched in the background, the next started on the previous
one's completion notification (no polling). For each run `Rx`:

1. `bash rfdetr_video/tools/overfitting_sweep.sh Rx` (background).
2. On completion, evaluate on test:
   `python _eval_stfs_ablations.py video_overfit_Rx`
   → writes `rfdetr_video/runs/video_overfit_Rx/ablation_results.{txt,json}`.
3. Record test MICRO AP30/AP50/F1 + cadica MICRO, note where the `sel_smoothed`
   peak landed and whether `val_loss` stopped rising, then launch `R(x+1)`.

Run order: R0, R1, R3, R4, R5 (core), then R2 only if R0 > R1 confirms
differential LR matters. Final comparison is against current main
(`consistency_distill_crrcd`: test MICRO AP30 0.593 / F1 0.369; cadica AP30 0.402).

---

## Self-review notes

- **Spec coverage:** §4.A differential LR → Task 5; §4.B cosine schedule → Tasks 3, 6;
  §4.C short budget + early stop → Tasks 1, 4, 6; §4.D EMA + smoothed selection →
  Tasks 2, 4, 6; §4.E regularization knobs → Task 7 (sweep-controlled — `--etf-dropout`
  flag already exists, `--weight-decay` flag added; defaults unchanged); §4.F sweep →
  Tasks 7, 8, 9 + Execution section; §7 testing → tests in every task + Task 10.
  Stage 2 / Approach 3 are explicitly out of scope per the spec.
- **`cfg.lr` meaning change** (now new-modules-only) is realized in Task 5 and
  documented in the Task 1 config comment — consistent with spec §4.A and §8.
- **Type consistency:** `ModelEMA(model, decay)` / `.update(model)` / `.applied_to(model)`;
  `build_scheduler(optimizer, cfg)`; `composite_selection_score(metrics, weights)`;
  `SmoothedTracker(k).add(value)`; `EarlyStopper(patience, min_delta).update(score)`;
  `_param_group_for(name)` — all used consistently across Tasks 2-9.
