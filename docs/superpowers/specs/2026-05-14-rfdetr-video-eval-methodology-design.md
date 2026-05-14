# Video RF-DETR — Reliable Evaluation & Comparison Protocol

**Date:** 2026-05-14
**Status:** Design — awaiting user review
**Scope:** `rfdetr_video/` training & evaluation tooling. No changes to model
architecture or the training objective.

---

## 1. Context & problem

A review of all 11 runs in `rfdetr_video/runs/` (configs, `best.txt`,
`ablation_results.txt`, `train.csv`) plus the training/eval code surfaced one
dominant problem: **experiment conclusions are not trustworthy**, because model
selection and cross-run comparison both ride a noisy, unrepresentative signal.

Evidence:

- **Val is unrepresentative.** The split is clean by video (train 151 / valid 32
  / test 31 videos, zero overlap), but the `valid` draw is ~15 AP30 points
  *harder* than `test` (val mAP30 ~0.40–0.42 vs test MICRO AP30 ~0.55–0.59),
  using identical metric code (`evaluate_map`, `f1_confidence_sweep`). It is one
  unlucky 32-video draw.
- **Selection reads noise.** `train.py` selects `best.pth` by greedy
  `argmax(val AP@0.3)`. Per-epoch val AP@0.3 swings ±0.04 (e.g. the main run goes
  0.36 → 0.42 → 0.36 on consecutive epochs). All 11 runs land in val mAP30
  ∈ [0.395, 0.423] — a 0.028 spread, inside the noise band.
- **Val→test ranking is uncorrelated.** The highest-val run (`rad_1`, 0.423) is
  mid-pack on test; the two lowest-val runs are 3rd/4th *best* on test. Ranking
  experiments by val mAP30 is actively misleading.
- **Comparisons are confounded.** Runs span augmentation-code changes, were
  stopped at different epochs (17–50), and several change >1 variable at once.
  No git SHA is recorded in any run.

This spec fixes the **evaluation methodology**. The separate overfitting problem
(every run peaks by epoch 2–14; constant LR; no early stopping; no dropout;
consistency loss hurting recall) is explicitly deferred — it only becomes
*measurable* once this protocol exists.

## 2. Goals

1. **Reliable within-run checkpoint selection** — `best.pth` reflects a real
   performance peak, not a per-epoch noise spike.
2. **Trustworthy cross-run comparison** — one canonical, apples-to-apples
   comparison artifact, computed identically for every run.
3. **Reproducible runs** — every run records the exact code version and config
   that produced it; experiments are diffable and one-variable-at-a-time.
4. **A known noise floor** — a measured "Δ smaller than X is noise" threshold so
   every future ablation is interpretable.

## 3. Non-goals / accepted limitations

- **Approach A was chosen** over re-splitting the data (Approach B) and a full
  re-split + cross-validation (Approach C). The `valid` and `test` directories
  are **left byte-identical**. Consequence, explicitly accepted: `valid` remains
  ~15 AP harder than `test`. This is tolerated because (a) val is *demoted* to
  within-run epoch selection only — cross-run comparison moves entirely to
  `test`, and (b) EMA + smoothed selection makes within-run selection robust to a
  biased/noisy val. If the residual bias later proves harmful, re-splitting
  (Approach B) is the follow-up.
- No changes to the training objective, losses, model architecture, LR schedule,
  epoch budget, or augmentation pipeline. Those are the deferred "overfitting
  retune" phase.

## 4. Design

Five components: 1–4 are code, implemented in the order 1 → 2 → 3 → 4;
component 5 is a run plan that follows once 1–4 have landed.

### Component 1 — Weight EMA + de-noised checkpoint selection

**New file `rfdetr_video/ema.py`:** a `ModelEMA` holding an exponential moving
average of the model's **trainable** parameters (frozen backbone params are
shared by reference, not copied — keeps VRAM down and is correct since they
never change). CPU storage of the EMA state is acceptable if VRAM is tight.
`update(model)` applies `ema = decay·ema + (1-decay)·param`; buffers are copied,
not averaged. Default `decay = 0.999`.

**Edits to `rfdetr_video/train.py`:**
- Instantiate `ModelEMA` after model construction (gated by `cfg.ema_enabled`).
- Call `ema.update(model)` inside the optimizer-step block (every
  `grad_accum_steps`, not every batch).
- Each eval epoch, run `evaluate()` on **both** the raw model and the EMA model.
- **Selection metric** = a smoothed composite. Per eval epoch compute
  `sel = w₃·AP@0.3 + w₅·AP@0.5 + w_f·F1` on the EMA model (default weights
  `(0.5, 0.3, 0.2)`). The selection score is the rolling mean of `sel` over the
  last `cfg.selection_smooth_k` evals (default 3). `best.pth` stores the **EMA**
  weights at the epoch with the best smoothed score.
- `history.json` / `train.csv` gain `ema/AP@0.3`, `ema/AP@0.5`, `ema/F1`,
  `ema/val_loss`, `sel`, `sel_smoothed` columns. `best.txt` reports EMA metrics
  and states the selection score used.
- `last.pth` (raw) continues to be saved; add `last_ema.pth`.

**Config additions (`config.py`):** `ema_enabled: bool = True`,
`ema_decay: float = 0.999`, `selection_smooth_k: int = 3`,
`selection_weights: tuple = (0.5, 0.3, 0.2)`.

### Component 2 — Validation demotion (protocol)

Mostly procedural; minimal code. Codifies that **`valid` is for within-run epoch
selection only** and **cross-run comparison happens on `test`** (via Component 3's
registry).

- `best.txt` header gains a one-line note: "val metrics = within-run selection
  only; cross-run comparison: see runs/results.csv".
- A short protocol section in `rfdetr_video/video_description.txt` (or a new
  `rfdetr_video/EVAL_PROTOCOL.md`) documents the rule.
- No metric is removed; `val_*` columns still exist in the registry as
  within-run diagnostics, just not as the comparison key.

### Component 3 — Reproducible run metadata

**Edits to `rfdetr_video/train.py`:**
- At run start capture `git rev-parse HEAD` and `git status --porcelain`
  (dirty flag) via `subprocess`, wrapped in try/except → `"unknown"` if not a
  repo. Write `git_sha`, `git_dirty`, `train_timestamp` into `config.json`.
- Add `--config FILE` to `parse_args()`: load a base config (JSON) into a dict,
  construct `Config(**base)`, then apply CLI overrides on top. **Precedence
  rule:** value-type CLI args default to `None` and override only when explicitly
  passed (`_maybe` already does this); `store_true` flags can only *enable* on
  top of the base config. This precedence is documented in the `--config` help
  text and covered by a test.

**New directory `rfdetr_video/configs/`** (committed): one JSON file per canonical
experiment — at minimum `etf_consistency.json`, `etf_no_consistency.json`,
`no_consistency_distill_crrcd.json`, `consistency_distill_crrcd.json` (the
current "main"). Each is a full `Config` dump. One-variable-at-a-time is enforced
by diffing these files.

### Component 4 — Results registry + auto test-eval

**Refactor `_eval_stfs_ablations.py`:** extract `run_ablations(run_dir: Path) ->
dict` from `main()` (the per-run logic that loads `best.pth`, runs the ablation
on `dataset2_split_test` + `cadica_50plus_new`, and writes
`ablation_results.{txt,json}`). `main()` keeps working as a script by calling
`run_ablations` with the existing hardcoded run name.

**New file `rfdetr_video/tools/run_experiment.py`:** one command per experiment —
`python -m rfdetr_video.tools.run_experiment --config rfdetr_video/configs/X.json
[--seed N] [--run-name NAME]`. Steps:
1. Build `Config` from the file (+ seed/run-name overrides), run
   `rfdetr_video.train.train(cfg)`.
2. Call `run_ablations(run_dir)` on the produced `best.pth`.
3. Parse `ablation_results.json`, append/upsert one row to
   `rfdetr_video/runs/results.csv`, keyed by `(run_name, seed)`.

**`results.csv` schema:** `run_name, git_sha, git_dirty, seed, timestamp, T,
img_size, etf_enabled, etf_spatial_radius, consistency_enabled,
consistency_weight, distill_enabled, crrcd_enabled, distill_centre_frame_only,
distill_frame_offsets, best_epoch, val_sel, val_AP30, val_AP50, test_AP30,
test_AP50, test_AP75, test_F1, test_P, test_R, test_Frag, test_FragRate,
cadica_AP30, cadica_AP50, cadica_F1, cadica_P, cadica_R`.

`results.csv` is the **canonical cross-run comparison artifact**; its comparison
columns are `test_*` and `cadica_*`. `val_*` are within-run diagnostics only.

### Component 5 — Noise-floor calibration (run plan, not code)

After Components 1–4 land, run 3 canonical configs × 3 seeds (9 trainings) via
`run_experiment.py`:
- `etf_consistency` (no distill — simplest baseline),
- `no_consistency_distill_crrcd`,
- `consistency_distill_crrcd` (current "main").
- Seeds: 42, 43, 44.

A small `rfdetr_video/tools/noise_floor.py` reads `results.csv`, groups by config,
and reports mean ± std of `test_AP30` / `test_F1` / `cadica_AP30`. Output: the
"treat Δ < X as noise" threshold used to interpret all future ablations.

## 5. Testing

Extend `rfdetr_video/tests/test_smoke.py` (unit-level; the existing
`RFDETR_VIDEO_HEAVY=1` gating for forward-pass tests is unchanged):
- `ModelEMA.update` math: EMA of a constant param stays constant; one update
  moves the EMA toward the model by exactly `(1-decay)`; only trainable params
  are tracked.
- Smoothed composite selection: rolling-mean-of-K and the weighted score are
  computed as specified.
- `--config FILE`: a config file round-trips through `Config`; CLI override
  precedence works (value arg overrides base; `store_true` enables; un-passed
  value args leave the base untouched).
- `run_experiment` registry append: a row with the full expected schema is
  written/upserted; a second call with the same `(run_name, seed)` updates in
  place.
- `_eval_stfs_ablations.run_ablations` remains callable and `__main__` still
  works.

## 6. Risks

- **EMA changes `best.pth` contents** → not weight-comparable to past `best.pth`.
  Accepted: the new protocol implies fresh runs anyway.
- **`--config` precedence is subtle** (argparse defaults clobbering the base
  config). Mitigation: value args default to `None`; explicit precedence test.
- **`_eval_stfs_ablations.py` refactor** could break the existing script.
  Mitigation: keep `main()` + `__main__` working; covered by a test.
- **EMA VRAM cost** — second copy of trainable params; distill runs already hold
  a teacher. Mitigation: EMA only trainable params, CPU storage allowed.
- **git capture in a non-repo / CI** → handled with try/except → `"unknown"`.

## 7. Out of scope (deferred — the "sequenced" follow-up)

The overfitting retune: LR schedule (the dead `MultiStepLR(30,40)`), epoch budget
(50 → ~12–15), early stopping, dropout / weight decay, and dropping or retuning
the count-consistency loss (shown to cut recall with no AP gain). These are
deferred until this protocol can measure them reliably.
