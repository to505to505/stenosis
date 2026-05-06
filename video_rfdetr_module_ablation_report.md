# Video RF-DETR Module Ablation Report

## Summary

This report summarizes the module-level evidence from the previously evaluated non-distilled Video RF-DETR model and the newer distilled Video RF-DETR model. The main conclusion is that **ETF is the only consistently strong video-specific component**, **STFS does not provide a robust measurable gain**, and **refinement is useful but must be treated as a special case because the model was trained to rely on that final prediction path**.

The experiments support the following interpretation:

- **ETF** is the most reliable and strongest temporal component. It improves performance in the non-distilled model and in the distilled model across both CADICA and Dataset2.
- **STFS** does not show a consistent positive contribution. Its effect is neutral, mixed, or slightly negative depending on the dataset and metric.
- **Refinement** improves the trained final prediction path, especially localization/AP50, but it should not be interpreted as a clean independent architecture gain because the checkpoint was trained with refinement enabled.

## Evaluated Runs And Result Files

### Non-Distilled Video Model

Result file:

- [_eval_cadica_video_ablate_nodistill_latest.json](_eval_cadica_video_ablate_nodistill_latest.json)

Run:

```text
rfdetr_video/runs/stfs_nodistill_v6_etf
```

Dataset:

```text
cadica_50plus_new
```

### Distilled Video Model

Result files:

- [_eval_cadica_video_module_importance_stfs_crrcd_v6_etf_postref_centreKD_stfsAlign_stfsShifter.txt](_eval_cadica_video_module_importance_stfs_crrcd_v6_etf_postref_centreKD_stfsAlign_stfsShifter.txt)
- [_eval_dataset2_split_test_video_module_importance_stfs_crrcd_v6_etf_postref_centreKD_stfsAlign_stfsShifter.txt](_eval_dataset2_split_test_video_module_importance_stfs_crrcd_v6_etf_postref_centreKD_stfsAlign_stfsShifter.txt)

Run:

```text
rfdetr_video/runs/stfs_crrcd_v6_etf_postref_centreKD_stfsAlign_stfsShifter
```

Important configuration flags:

```text
distill_enabled = true
distill_through_refine = true
distill_centre_frame_only = true
stfs_feature_align_enabled = true
stfs_aggregator_enabled = true
stfs_shifter_enabled = true
consistency_enabled = true
etf_enabled = true
```

Datasets:

```text
cadica_50plus_new
dataset2_split_test
```

## Architectural Context

The Video RF-DETR model processes a temporal window of frames, typically `T=5`. The frames are flattened into the batch dimension before the RF-DETR backbone/transformer/decoder path:

```text
(B, T, 3, H, W) -> (B*T, 3, H, W)
```

The RF-DETR decoder and detection heads are shared across all frames. There are not five separate decoders or five separate detection heads. Instead, the same model weights process each frame, and the frame-specific activations are reshaped back into `(B, T, Q, *)` after prediction.

The video-specific components are:

1. **ETF: Early Temporal Fusion**
   - Applies temporal self-attention over features before the transformer/decoder.
   - Gives a dense differentiable temporal path between neighboring frames and the supervised frame.

2. **STFS: Query-Level Spatio-Temporal Feature Sharing**
   - Tracks confident query slots across frames after the first decoder pass.
   - Injects strong neighbor-frame slots into weaker target-frame slots.
   - Uses a learned aggregator and optional reference-point shifting/candidate refinement.

3. **Refinement**
   - A post-decoder correction stage.
   - Takes first-pass or STFS-enriched query slots and re-attends them to the current-frame memory.
   - Produces the final class logits and boxes.

Refinement is decoder-like, but it is not simply a normal seventh decoder layer. It is placed after the first-pass predictions and after optional STFS injection. Therefore, it should be described as a **post-decoder refinement block** or **final localization correction stage**, not as a standard additional decoder layer.

## Non-Distilled Video Model On CADICA

The non-distilled model was evaluated on `cadica_50plus_new`. The most relevant variants are the full final path, first pass only, refinement without STFS, and ETF-disabled paths.

| Variant | AP30 | AP50 | F1 | Interpretation |
|---|---:|---:|---:|---|
| final | 0.2219 | 0.0477 | 0.1528 | Full runtime path |
| first_pass | 0.1983 | 0.0395 | 0.1421 | No final refinement/STFS path |
| refine_no_stfs | 0.2264 | 0.0475 | 0.1556 | Refinement kept, STFS injection removed |
| no_etf_final | 0.1844 | 0.0447 | 0.1407 | ETF disabled |
| no_etf_first_pass | 0.1600 | 0.0314 | 0.1167 | ETF disabled and no refinement |
| no_etf_refine_no_stfs | 0.1879 | 0.0446 | 0.1415 | ETF disabled, refinement kept, no STFS |

### Non-Distilled Interpretation

ETF is useful in the ordinary non-distilled video model. Removing ETF from the final path reduces AP30 from `0.2219` to `0.1844`, a drop of `0.0375`. It also reduces F1 from `0.1528` to `0.1407`.

STFS does not help in this run. The `refine_no_stfs` variant is slightly better than the full final path:

| Comparison | AP30 | AP50 | F1 |
|---|---:|---:|---:|
| final | 0.2219 | 0.0477 | 0.1528 |
| refine_no_stfs | 0.2264 | 0.0475 | 0.1556 |
| final - refine_no_stfs | -0.0045 | +0.0002 | -0.0028 |

This suggests that, in the non-distilled checkpoint, refinement is useful as a final correction stage, but STFS injection itself is not improving the result.

## Distilled Video Model On CADICA

The distilled model was evaluated on `cadica_50plus_new` with the full module-importance matrix.

| Variant | AP30 | AP50 | F1 | P | R | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| final | 0.3997 | 0.1159 | 0.2441 | 0.3207 | 0.1970 | Full runtime path |
| first_pass | 0.3282 | 0.0814 | 0.2329 | 0.2856 | 0.1966 | No final refinement path |
| refine_no_stfs | 0.4044 | 0.1155 | 0.2436 | 0.3200 | 0.1966 | Refinement kept, STFS removed |
| no_etf_final | 0.2805 | 0.0824 | 0.2020 | 0.1934 | 0.2115 | ETF disabled |
| no_etf_first_pass | 0.2420 | 0.0687 | 0.1929 | 0.1732 | 0.2176 | ETF disabled, no refinement |
| no_etf_refine_no_stfs | 0.2833 | 0.0816 | 0.2011 | 0.1925 | 0.2106 | ETF disabled, refinement kept, no STFS |
| stfs_no_aggregator | 0.4037 | 0.1185 | 0.2474 | 0.3252 | 0.1996 | Learned STFS aggregator removed |
| stfs_no_shifter | 0.3922 | 0.1132 | 0.2414 | 0.3172 | 0.1948 | STFS shifter removed |
| stfs_no_candidate_refine | 0.4059 | 0.1180 | 0.2457 | 0.3229 | 0.1983 | 5-candidate sparse refinement removed |
| stfs_legacy_no_agg_no_shift | 0.4003 | 0.1139 | 0.2408 | 0.3165 | 0.1944 | Legacy no-aggregator/no-shifter path |

### CADICA Importance Deltas

Positive values mean the module helps the final/AP metric relative to the corresponding ablated variant.

| Module | dAP30 | dAP50 | dF1 |
|---|---:|---:|---:|
| ETF | +0.1192 | +0.0335 | +0.0421 |
| ETF on first pass | +0.0863 | +0.0127 | +0.0400 |
| refinement only | +0.0761 | +0.0341 | +0.0107 |
| STFS net | -0.0047 | +0.0005 | +0.0005 |
| STFS aggregator | -0.0040 | -0.0026 | -0.0033 |
| STFS shifter | +0.0075 | +0.0028 | +0.0027 |
| 5-candidate refinement | -0.0062 | -0.0020 | -0.0016 |
| aggregator+shifter stack | -0.0006 | +0.0021 | +0.0033 |

### Distilled CADICA Interpretation

ETF is the dominant module on CADICA. Removing ETF reduces AP30 by `0.1192`, AP50 by `0.0335`, and F1 by `0.0421`. This is by far the strongest module-level signal.

STFS does not show a meaningful net benefit. The full final path is slightly worse than `refine_no_stfs` on AP30:

| Comparison | AP30 | AP50 | F1 |
|---|---:|---:|---:|
| final | 0.3997 | 0.1159 | 0.2441 |
| refine_no_stfs | 0.4044 | 0.1155 | 0.2436 |
| final - refine_no_stfs | -0.0047 | +0.0005 | +0.0005 |

The internal STFS ablations are also mixed. Removing the aggregator improves AP30/AP50/F1 slightly, removing candidate refinement improves AP30/AP50/F1 slightly, while the shifter has a small positive effect. The result is not a strong case that STFS is driving the model's performance.

## Distilled Video Model On Dataset2

The same distilled checkpoint was also evaluated on `dataset2_split_test`.

| Variant | AP30 | AP50 | F1 | P | R | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| final | 0.5657 | 0.2711 | 0.3657 | 0.3836 | 0.3493 | Full runtime path |
| first_pass | 0.5239 | 0.1744 | 0.3540 | 0.3677 | 0.3413 | No final refinement path |
| refine_no_stfs | 0.5690 | 0.2673 | 0.3614 | 0.4774 | 0.2908 | Refinement kept, STFS removed |
| no_etf_final | 0.5163 | 0.2496 | 0.3437 | 0.4338 | 0.2846 | ETF disabled |
| no_etf_first_pass | 0.4566 | 0.1871 | 0.3279 | 0.3454 | 0.3121 | ETF disabled, no refinement |
| no_etf_refine_no_stfs | 0.5183 | 0.2631 | 0.3441 | 0.4372 | 0.2837 | ETF disabled, refinement kept, no STFS |
| stfs_no_aggregator | 0.5698 | 0.2632 | 0.3640 | 0.3821 | 0.3475 | Learned STFS aggregator removed |
| stfs_no_shifter | 0.5663 | 0.2588 | 0.3610 | 0.3788 | 0.3449 | STFS shifter removed |
| stfs_no_candidate_refine | 0.5607 | 0.2557 | 0.3619 | 0.3797 | 0.3457 | 5-candidate sparse refinement removed |
| stfs_legacy_no_agg_no_shift | 0.5693 | 0.2608 | 0.3573 | 0.3749 | 0.3413 | Legacy no-aggregator/no-shifter path |

### Dataset2 Importance Deltas

Positive values mean the module helps the final/AP metric relative to the corresponding ablated variant.

| Module | dAP30 | dAP50 | dF1 |
|---|---:|---:|---:|
| ETF | +0.0493 | +0.0216 | +0.0220 |
| ETF on first pass | +0.0674 | -0.0126 | +0.0261 |
| refinement only | +0.0450 | +0.0928 | +0.0074 |
| STFS net | -0.0033 | +0.0039 | +0.0042 |
| STFS aggregator | -0.0041 | +0.0079 | +0.0017 |
| STFS shifter | -0.0006 | +0.0123 | +0.0046 |
| 5-candidate refinement | +0.0050 | +0.0155 | +0.0037 |
| aggregator+shifter stack | -0.0036 | +0.0103 | +0.0084 |

### Distilled Dataset2 Interpretation

ETF remains beneficial on Dataset2. Compared with `no_etf_final`, the full final path improves AP30 by `0.0493`, AP50 by `0.0216`, and F1 by `0.0220`.

STFS is slightly more favorable on Dataset2 than on CADICA, but still weak. The full final path has slightly lower AP30 than `refine_no_stfs`, but slightly higher AP50 and F1:

| Comparison | AP30 | AP50 | F1 |
|---|---:|---:|---:|
| final | 0.5657 | 0.2711 | 0.3657 |
| refine_no_stfs | 0.5690 | 0.2673 | 0.3614 |
| final - refine_no_stfs | -0.0033 | +0.0039 | +0.0042 |

This means STFS gives a small AP50/F1 benefit on Dataset2, but it is not a strong driver of performance and it does not consistently improve AP30.

## Cross-Run Comparison

### ETF

ETF is consistently useful.

| Model / Dataset | ETF dAP30 | ETF dAP50 | ETF dF1 |
|---|---:|---:|---:|
| non-distilled CADICA | +0.0375 | +0.0030 | +0.0122 |
| distilled CADICA | +0.1192 | +0.0335 | +0.0421 |
| distilled Dataset2 | +0.0493 | +0.0216 | +0.0220 |

ETF is the only module whose benefit is clear across all evaluated settings. It provides a dense temporal feature fusion path before the decoder, allowing the model to use neighboring-frame information early in the visual representation.

Recommended claim:

> Early Temporal Fusion is the most robust video-specific component. It consistently improves detection performance across non-distilled and distilled Video RF-DETR checkpoints.

### STFS

STFS is not consistently useful.

| Model / Dataset | STFS net dAP30 | STFS net dAP50 | STFS net dF1 |
|---|---:|---:|---:|
| non-distilled CADICA | -0.0045 | +0.0002 | -0.0028 |
| distilled CADICA | -0.0047 | +0.0005 | +0.0005 |
| distilled Dataset2 | -0.0033 | +0.0039 | +0.0042 |

Across the evaluated models, STFS does not produce a strong or stable gain. It slightly hurts AP30 in all three comparisons. It gives a tiny AP50/F1 improvement in the distilled Dataset2 case, but the magnitude is small compared with ETF and refinement.

Recommended claim:

> STFS provides only marginal and mixed improvements in the current implementation. It may slightly improve AP50/F1 in some settings, but it is not the main source of the video model's performance.

### Refinement

Refinement behaves differently from ETF and STFS. It is useful, but it is not a pure video module and not a clean retraining-controlled architectural ablation.

| Model / Dataset | first_pass AP30 | refine_no_stfs AP30 | first_pass AP50 | refine_no_stfs AP50 | F1 gain |
|---|---:|---:|---:|---:|---:|
| non-distilled CADICA | 0.1983 | 0.2264 | 0.0395 | 0.0475 | +0.0135 |
| distilled CADICA | 0.3282 | 0.4044 | 0.0814 | 0.1155 | +0.0107 |
| distilled Dataset2 | 0.5239 | 0.5690 | 0.1744 | 0.2673 | +0.0074 |

The refinement block improves the trained final prediction path, especially localization. On Dataset2, AP50 increases from `0.1744` in the first pass to `0.2673` with refinement but without STFS.

However, this should be interpreted carefully. The distilled checkpoint was trained with refinement enabled, including `distill_through_refine=true`. Therefore, the first-pass output is not necessarily trained to be the final optimal detector output. Removing refinement at inference time measures how much the trained model relies on the refinement path, not how much a retrained no-refinement architecture would lose.

Recommended claim:

> The refinement block is an important post-decoder correction stage for the trained final prediction path, especially for localization. However, because the model was trained with refinement enabled, the ablation should be interpreted as inference-path dependence rather than a clean standalone architecture gain.

## How To Present This In Slides

Use the following claim hierarchy:

| Component | Claim Strength | Suggested Slide Wording |
|---|---|---|
| ETF | Strong | Main effective temporal component |
| Refinement | Medium | Important trained final localization correction stage |
| STFS | Weak / mixed | Sparse slot transfer with marginal measured benefit |

Suggested slide conclusion:

> Across both non-distilled and distilled Video RF-DETR experiments, ETF is the only module that provides a consistent and substantial temporal gain. The refinement block is important for the trained final prediction path and improves localization, but it should be interpreted as a post-decoder correction stage rather than an independently validated temporal module. STFS provides only marginal and mixed improvements, suggesting that the current model obtains most of its video benefit from early temporal feature fusion rather than sparse slot-level temporal transfer.

## Limitations

The refinement ablation is an inference-only ablation on a model trained with refinement. Therefore, it cannot prove that adding refinement would improve a separately trained no-refinement baseline by the same amount.

The strongest possible experiment would be:

```text
train same model without refinement
train same model with refinement
compare both under the same data, KD, schedule, and evaluation protocol
```

There is no time for that retraining in the current workflow, so the presentation should keep refinement in a secondary role and avoid overclaiming.

## Final Takeaway

The video model's robust temporal gain comes mainly from **ETF**. **STFS did not provide a consistent improvement** in the current checkpoints. **Refinement improves the trained final prediction path**, especially AP50/localization, but it is a separate post-decoder correction mechanism and should not be presented as a clean retraining-controlled architecture gain.

The safest and most accurate interpretation is:

> ETF is the primary effective video component. STFS is currently marginal and mixed. Refinement is useful as a trained final correction stage, but its contribution should be presented separately from the temporal-module claims.