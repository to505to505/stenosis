# Post-Network Tuning and Prompt Tuning — alternatives to ETF+KD distillation

## Goal

Add two parameter-efficient temporal-adaptation baselines so we can compare
them to the current Video RF-DETR run
(`rfdetr_video/runs/video_5_etf_consistency_distill_crrcd`). All three share
the *same* dataset, evaluation pipeline, training loop, and 2-D RF-DETR
checkpoint — they differ only in *how* temporal information is injected into
a frozen single-frame detector.

This spec is a pure design document — no training is started by the
implementation that follows it. Sweep scripts can launch the runs later.

## The three setups under comparison

### Method A — current ETF + KD + CRRCD + consistency (no change)

```
frames (B,T,3,H,W)
   │
   ▼ backbone (FROZEN)
features (B*T,C,h,w)
   │
   ▼ ETF (TRAINABLE) — temporal self-attn on backbone feature maps
features' (B*T,C,h,w)
   │
   ▼ transformer + heads (TRAINABLE, KD-DETR + CRRCD with HR teacher)
hs (L,B*T,Q,D) ─► class/bbox heads ─► (B,T,Q,K)
```

Early fusion at the backbone, full detector fine-tuned with KD-DETR-style
distillation from a frozen RF-DETR-Large @ 704 teacher, plus CRRCD relational
distillation on decoder hidden states and a count-consistency loss across the
T-frame window.

### Method B — Post-Network Tuning (new)

```
frames (B,T,3,H,W)
   │
   ▼ backbone (FROZEN)
features (B*T,C,h,w)
   │
   ▼ transformer (FROZEN) — no ETF, no temporal info yet
hs (L,B*T,Q,D)
   │   take hs[-1] → reshape (B,T,Q,D)
   ▼ TemporalPostNet (TRAINABLE) — per-query temporal self-attn
hs' (B*T,Q,D)
   │
   ▼ class/bbox heads (FROZEN)
predictions (B,T,Q,K)
```

The base 2-D model runs each frame independently; the only trainable
component is a small temporal Transformer applied to the decoder's
*last-layer* hidden states. Heads are reused frozen, so post-net gradients
flow through them. Aux outputs are dropped (they would be constants).

Loss: detection loss only on B*T frames. No KD/CRRCD/consistency in the
default configuration (kept off via the same toggles).

### Method C — Prompt Tuning with cross-frame propagation (new)

```
frames (B,T,3,H,W)
   │
   ▼ backbone (FROZEN)
features (B*T,C,h,w) ──► pool & project ──► c_t  ∈ (B,T,D)
   │
   │     P_0 (learnable, n_prompts, D)
   │     P_t = GRUCell(c_t broadcast, P_{t-1})       (TRAINABLE)
   │     tgt_add_t[:n_prompts] = P_t,  rest = 0      (B*T,Q,D)
   ▼
transformer (FROZEN) — decoder receives tgt = original_tgt + tgt_add
hs (L,B*T,Q,D)
   │
   ▼ class/bbox heads (FROZEN)
predictions (B,T,Q,K)
```

The entire detector stays frozen. A small bank of learnable visual prompts
`P ∈ (n_prompts, D)` (e.g. 16) is propagated frame-by-frame via a `GRUCell`
conditioned on the pooled backbone feature of each frame. The propagated
prompts are added to the first `n_prompts` rows of the decoder's `tgt` (query
content) before the decoder runs — implemented via the existing decoder
forward-pre-hook in `model.py`, extended with an additive `tgt_add` mode.

Loss: detection loss only. No KD/CRRCD/consistency in the default
configuration.

## Why these two

They are the two standard parameter-efficient adaptation families called out
in the literature snippet the user supplied. Both freeze the spatial
detector and add a small temporal module that the user predicts will be
*worse* than early fusion. The point of the experiment is to quantify that
gap on dataset2_split test.

## Trainable parameter budgets (approximate)

| Method | Trainable parameters (rough) |
|---|---|
| A — ETF + full detector | ~30M+ (transformer + heads + ETF) |
| B — PostNet only | ~600k (1× temporal MHA + LN at D=256, n_heads=8) |
| C — Prompt only | ~270k (n_prompts·D + GRUCell(D,D) + Linear projector) |

Both new methods are intentionally tiny — the comparison is *temporal
capacity in the right place* vs. *parameter count*, not parameter count
alone.

## Implementation surface

Everything is added under `rfdetr_video/`. We are deliberately not making a
new package — these alternatives are training-mode variants of the same
model class.

### New files

* `rfdetr_video/postnet.py` — `TemporalPostNet` (per-slot temporal MHA, zero-init residual).
* `rfdetr_video/prompt.py` — `TemporalPromptBank` (GRUCell-propagated learnable prompts).
* `rfdetr_video/tests/test_alternatives.py` — smoke tests for both modules.

### Config additions (`rfdetr_video/config.py`)

```python
# adaptation mode — exactly one of these governs how temporal info enters
adapt_mode: str = "full"   # {"full", "postnet", "prompt"}

# Post-Network Tuning
postnet_enabled: bool = False
postnet_heads: int = 8
postnet_layers: int = 1
postnet_dropout: float = 0.0

# Prompt Tuning
prompt_enabled: bool = False
prompt_num_prompts: int = 16
prompt_init_std: float = 0.02
prompt_propagate: str = "gru"   # {"gru", "shared"}
```

`adapt_mode != "full"` automatically:

* sets `freeze_backbone = freeze_decoder = True`
* sets `etf_enabled = False`
* sets `distill_enabled = crrcd_enabled = consistency_enabled = False`
* enables either `postnet_enabled` or `prompt_enabled`

This means a single CLI flag selects the comparison setup without the user
having to remember to flip a dozen toggles.

### Model changes (`rfdetr_video/model.py`)

1. Add `self.postnet` and `self.prompt_bank` modules conditional on config.
2. In `forward`:
   * If `prompt_bank`: compute `tgt_add ∈ (B*T, Q, D)`, set
     `self._inject_decoder_inputs = {"tgt_add": ...}`.
   * Decoder pre-hook extended with an additive branch — when `tgt_add` is
     present, return `(args[0] + tgt_add, *args[1:])`, leave refpoints
     untouched.
   * After transformer call, if `postnet`: refine `hs[-1]`, re-apply frozen
     class/bbox heads with `ref_unsigmoid[-1]`, replace `out["pred_logits"]`
     and `out["pred_boxes"]`. Drop `aux_outputs`.
3. `_param_group_for` already routes new modules (`postnet`, `prompt`) to the
   `"new"` LR bucket — extend the prefix list accordingly.

The teacher/general KD branches are off in these modes, so the existing
decoder-input replacement path is unchanged.

### Train changes (`rfdetr_video/train.py`)

* New CLI: `--adapt-mode {full,postnet,prompt}` plus the per-mode tunables.
* When `adapt_mode != "full"`, force-disable distill/CRRCD/consistency/ETF
  in `cfg_kwargs`. Train loop branches 2 and 3 are already gated on
  `cfg.distill_enabled`, so they no-op automatically.
* No changes to dataset, evaluator, EMA, selection, or scheduler — these are
  shared.

## Testing

`rfdetr_video/tests/test_alternatives.py`:

1. `test_postnet_output_shape` — synthetic `hs (B*T,Q,D)` → refined same shape.
2. `test_postnet_residual_zero_init` — at init the refined hs equals the
   input hs (zero-init out-proj).
3. `test_prompt_bank_shape` — synthetic srcs + (B, T) → `(B, T, N, D)`.
4. `test_prompt_propagation_nontrivial` — `P_T != P_0` after T steps (i.e.
   GRU actually moves the prompt).
5. `test_adapt_mode_force_disables` — `adapt_mode="postnet"` zeros all
   distill/CRRCD/consistency/ETF flags and only post-net params have
   `requires_grad=True`.
6. `test_model_forward_postnet` — heavy test (gated on
   `RFDETR_VIDEO_HEAVY=1`) for one forward pass with postnet wiring.
7. `test_model_forward_prompt` — heavy test for prompt mode wiring.

## Out of scope

* Training launchers / sweep scripts — left for follow-up; the user
  explicitly said "don't start the new training right away".
* Mixing the alternatives with KD or CRRCD — possible later, but the point
  of the comparison is the clean, default baseline of each method.
* Q-Former variants of PostNet, attention-based prompt mixers — could be
  added under the same `adapt_mode` umbrella later.
