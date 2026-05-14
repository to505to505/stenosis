# Video RF-DETR — Severe Overfitting Fix (Stage 1: regime retune)

**Date:** 2026-05-14
**Status:** Design — awaiting user review
**Scope:** `rfdetr_video/` training regime (LR grouping, schedule, budget,
early stopping, EMA, regularization knobs) + a structured diagnostic sweep.
No changes to model architecture or the loss functions.

---

## 1. Context & problem

Across all 11 runs in `rfdetr_video/runs/`, the model overfits hard: best epochs
are 2,3,3,4,4,5,8,10,10,12,14 out of 50; `val_loss` bottoms out early then climbs
monotonically (main run 8.5 → 9.2) while `train_loss` keeps falling (6.4 → 5.1).

The init checkpoint (`rfdetr_small_arcade2x_512_reg`) is **already an ARCADE
coronary detector**, so this is a *fine-tuning* task being run with a
*from-scratch* regime. Five causes, all confirmed in code:

1. **No differential LR.** `VideoRFDETR.get_param_groups()` puts every non-backbone
   param in one group at `cfg.lr=1e-4` — the pretrained `transformer`,
   `class_embed`, `bbox_embed`, `refpoint_embed`, `query_feat` **and** the
   randomly-initialized `etf` + `crrcd` modules all train at the same rate. The
   ARCADE prior is washed out as fast as ETF learns from scratch.
2. **LR never decays.** `MultiStepLR(milestones=(30,40))` is unreachable — runs
   peak by epoch ≤14 and are killed before 30. LR is a flat 1e-4 throughout.
3. **50-epoch budget, no early stopping** — runs grind ~30 epochs past peak.
4. **No regularization headroom** — `etf_dropout=0.0` (knob works, just off),
   `weight_decay=1e-4` (light).
5. **Small effective dataset** — 151 train videos; intra-video frames are highly
   correlated, so effective N ≈ "151", not "5817 frames".

## 2. Goals

1. The model's performance **peak is higher and/or later**, and `val_loss` stops
   rising monotonically — overfitting is visibly reduced.
2. Best test MICRO metrics on `dataset2_split_test` ≥ current main
   (`consistency_distill_crrcd`: AP30 0.593 / F1 0.369), ideally higher; cadica
   MICRO AP30 ≥ 0.402.
3. The structured sweep **identifies which lever** (differential LR / schedule &
   budget / regularization / EMA) actually drives the improvement.

## 3. Non-goals / deferred

- **Stage 2** (parameter-efficient fine-tuning: freeze decoder / LoRA / adapters)
  and **Approach 3** (L2-SP anti-forgetting, SWA) are *escalation paths*,
  documented in §6 but not implemented in Stage 1. Stage 2 is triggered only if
  the Stage-1 sweep's best peak is not meaningfully above current main.
- The **evaluation-methodology overhaul** (test-based results registry, git-SHA
  metadata, `--config` files, noise-floor seed-repeats) stays deferred — see the
  approved spec `2026-05-14-rfdetr-video-eval-methodology-design.md`. EMA +
  smoothed selection (§4.D) is the one piece shared by both; whichever spec lands
  first implements `rfdetr_video/ema.py`.
- No changes to losses (the consistency-loss-hurts-recall finding, ETF spatial
  radius, T, augmentation pipeline are all separate, deferred).

## 4. Design — Stage 1 (regime retune)

### A. Differential learning rate — rework `get_param_groups()`

Replace the backbone/everything-else split with **three** groups, matched by
parameter-name prefix:

| group | matches | LR |
|---|---|---|
| backbone | `backbone.*` | `cfg.lr_backbone` (empty while `freeze_backbone=True`) |
| pretrained detector | `transformer.*`, `class_embed.*`, `bbox_embed.*`, `refpoint_embed.*`, `query_feat.*` | `cfg.lr_pretrained` (low) |
| new modules | `etf.*`, `crrcd.*` | `cfg.lr` (base) |

`cfg.lr` keeps its name but its **meaning changes**: it is now the LR for the
newly-initialized modules only. The implementation must log the three group sizes
at startup and assert none is unexpectedly empty (guards against a prefix typo
silently mis-grouping params).

### B. LR schedule — cosine with warmup

Add `cfg.lr_schedule: str` (`"cosine"` | `"multistep"`, default `"cosine"`).
For `"cosine"`: `CosineAnnealingLR` over the epoch budget, decaying each group
from its base LR toward ~1e-6, stepped per epoch. The existing per-iteration
linear `warmup_lr` for the first `cfg.warmup_iters` is kept and hands off to
cosine after warmup. `"multistep"` preserves the old `MultiStepLR` path for
back-compat. Schedule applies to all param groups proportionally.

### C. Short budget + early stopping

- `cfg.epochs` default 50 → **20**.
- Add `cfg.early_stop_enabled: bool = True`, `cfg.early_stop_patience: int = 6`
  (evals without improvement), `cfg.early_stop_min_delta: float = 0.0`.
- Early stopping watches the **smoothed composite selection score** (§4.D). When
  it has not improved by `min_delta` for `patience` consecutive evals, training
  stops; the loop still writes `last.pth`, `history.json`, `train.csv`, `best.txt`.

### D. EMA + smoothed composite selection (shared with eval spec)

New `rfdetr_video/ema.py`: `ModelEMA` tracking an exponential moving average of
the **trainable** params (frozen params shared by reference; CPU storage allowed
if VRAM is tight), `decay = cfg.ema_decay` (default 0.999), updated every
optimizer step (i.e. inside the `grad_accum_steps` boundary).

- Each eval epoch, run `evaluate()` on **both** the raw and EMA models.
- Selection score: `sel = w₃·AP@0.3 + w₅·AP@0.5 + w_f·F1` on the EMA model
  (`cfg.selection_weights`, default `(0.5, 0.3, 0.2)`); the tracked metric is the
  rolling mean of `sel` over the last `cfg.selection_smooth_k` evals (default 3).
- `best.pth` stores the **EMA** weights at the best smoothed score. `last.pth`
  (raw) still saved; add `last_ema.pth`. `history.json` / `train.csv` gain
  `ema/AP@0.3`, `ema/AP@0.5`, `ema/F1`, `ema/val_loss`, `sel`, `sel_smoothed`.

### E. Regularization knobs

`etf_dropout` and `weight_decay` stay at their current defaults (0.0, 1e-4) — the
sweep (§4.F, run R3) is what measures their effect, so they are not baked into
the new defaults until proven. The sweep's anchor run sets `etf_dropout=0.1`,
`weight_decay=1e-3`. Decoder dropout: RF-DETR's decoder exposes no obvious dropout
arg — implementation checks feasibility; if none exists, it is skipped (not a
blocker).

### F. Structured diagnostic sweep — one lever at a time

Base model config held fixed at the current main (`consistency_distill_crrcd`);
only training-regime knobs vary. Anchor + five single-lever variants:

| run | change vs anchor | isolates |
|---|---|---|
| **R0** anchor | diff-LR (pretrained 3e-5 / new 1e-4), cosine, epochs 20, early-stop, EMA, etf_dropout 0.1, wd 1e-3 | — |
| **R1** | no differential LR (pretrained = new = 1e-4) | differential LR |
| **R2** | pretrained LR 1e-5 (more aggressive) | diff-LR ratio (optional) |
| **R3** | etf_dropout 0.0, wd 1e-4 | dropout + weight decay |
| **R4** | epochs 35 (still cosine + early-stop) | short budget vs early-stop |
| **R5** | EMA off | EMA |

5 core runs (R0, R1, R3, R4, R5); R2 optional, run only if R0 > R1 confirms
differential LR matters. Driven by a committed `rfdetr_video/tools/
overfitting_sweep.sh` invoking `python -m rfdetr_video.train` with explicit
flags (new argparse flags added for the new config fields). This avoids pulling
in the deferred `--config` system.

### Config additions (`config.py`) + argparse flags (`train.py`)

`lr_pretrained: float = 3e-5`, `lr_schedule: str = "cosine"`,
`early_stop_enabled: bool = True`, `early_stop_patience: int = 6`,
`early_stop_min_delta: float = 0.0`, `ema_enabled: bool = True`,
`ema_decay: float = 0.999`, `selection_smooth_k: int = 3`,
`selection_weights: tuple = (0.5, 0.3, 0.2)`. New defaults: `epochs = 20`.
Corresponding `--lr-pretrained`, `--lr-schedule`, `--no-early-stop`,
`--early-stop-patience`, `--no-ema`, `--ema-decay` flags.

## 5. Evaluation & success criteria

Each sweep run → `_eval_stfs_ablations.py` on `dataset2_split_test` +
`cadica_50plus_new` (the existing tooling; full registry deferred).

- **Primary:** best sweep run's test MICRO AP30/F1 ≥ current main (0.593 / 0.369),
  cadica AP30 ≥ 0.402.
- **Overfitting signal (secondary):** `val_loss` curve flattens instead of rising
  monotonically; the `sel_smoothed` peak occurs later and/or higher than main's
  epoch-12 peak; the train↔val loss gap shrinks.
- **Diagnostic:** R0-vs-Rx deltas attribute the improvement to specific levers.
- **Stage 2 trigger:** if the best sweep peak is not meaningfully above current
  main, escalate to freezing the decoder (`freeze_decoder=True` already freezes
  `transformer` + `class_embed` + `bbox_embed`; ETF + `refpoint_embed` +
  `query_feat` stay trainable) and/or LoRA.

## 6. Stage 2 — escalation (documented, not implemented in Stage 1)

If Stage 1 plateaus: (a) `freeze_decoder=True` and train only ETF + heads-adjacent
params; (b) LoRA/adapters on the decoder attention/FFN; (c) L2-SP — add
`λ·‖θ_pretrained_now − θ_pretrained_init‖²` to keep the decoder near its ARCADE
basin (requires stashing the init weights). Each is its own follow-up spec.

## 7. Testing

Extend `rfdetr_video/tests/test_smoke.py` (unit-level; existing
`RFDETR_VIDEO_HEAVY=1` gating unchanged):

- `get_param_groups()` returns three groups with the correct LRs, every
  `etf.*`/`crrcd.*` param in the new-modules group and every `transformer.*` /
  `*_embed.*` param in the pretrained group; no group unexpectedly empty.
- Cosine+warmup schedule produces expected LR at step 0, end of warmup, mid, end.
- Early-stopping triggers after `patience` non-improving evals and not before.
- `ModelEMA.update` math: EMA of a constant param is constant; one update moves
  the EMA toward the model by exactly `(1-decay)`; only trainable params tracked.
- Smoothed composite `sel` score: weighted sum + rolling-mean-K computed as specified.

## 8. Risks

- **Prefix mis-grouping** in `get_param_groups()` → wrong LRs silently. Mitigation:
  startup logging + non-empty assertions + the unit test.
- **`cfg.lr` meaning change** (was decoder+everything, now new-modules-only) →
  confusing vs old runs. Mitigation: documented here and in the flag help; old
  runs are already non-comparable.
- **Cosine/warmup handoff** — the per-iter warmup and per-epoch cosine must
  compose cleanly. Mitigation: LR-trajectory unit test.
- **Early stopping on a noisy val signal** stopping too early. Mitigation:
  smoothed signal + EMA + patience 6.
- **Sweep cost** — 5–6 trainings. Accepted (user chose the structured sweep).
- **EMA VRAM** — distill runs already hold a teacher. Mitigation: EMA trainable
  params only, CPU storage allowed.
