============================================================
Distillation in rfdetr_video — implementation reference
============================================================

Goal
----
Transfer per-frame detection knowledge from the frozen 2-D
RF-DETR-Large teacher (HR, single-frame) into the temporal
VideoRFDETR student (LR, T-frame window with STFS + ETF).
The design mirrors rfdetr_temporal/distill so all three
losses (KD-spec, KD-general, CRRCD) co-exist with the
temporal-specific modules (ETF, FeatureAggregator track-
queries, RefPointShift) without interfering with them.

Three training branches per step
--------------------------------
For each (B, T, 3, S, S) student window plus a paired
(B, T, 3, 704, 704) HR teacher window (replay-augmented to
match the student's geometric augmentations):

  Branch 1 — Student detection
    out = model(images, query_mode="student")
    Full pipeline: ETF → backbone → projector → transformer
    → STFS (track_queries from FeatureAggregator +
    RefPointShift on previous-frame refpoints) → refinement.
    Losses: criterion(pred, targets) (cls + box + giou)
            + λ_num · L_num (multi-frame count consistency)

  Branch 2 — KD specific + CRRCD (per-frame)
    t_out = teacher.forward_video(teacher_frames_HR)
    student_kd_spec = model(
        images, query_mode="teacher",
        decoder_inputs={tgt, refpoints} from teacher,
    )
    distill_spec = distillation_loss(student_kd_spec, t_out)
    loss_crrcd  = CRRCD(teacher_hs, student_hs_spec,
                        foreground_weight)
    Loss: + β_distill · loss_distill_spec
          + β_crrcd   · loss_crrcd

  Branch 3 — KD general (per-frame)
    gen_q = model.sample_general_queries(Q_g)   # 100 queries
    t_out_gen = teacher.forward_video_general(
        teacher_frames, gen_q.refpoint, gen_q.query_feat,
        min_weight=...)
    student_kd_gen = model(images, query_mode="general",
        general_queries=gen_q,
        decoder_inputs={teacher tgt, refpoints})
    Loss: + β_distill · β_gen · loss_distill_gen

KD-DETR query-slot alignment
----------------------------
Both teacher and student decoders are wrapped with hooks so
that, in branches 2 and 3, the student's transformer.decoder
sees *the same* (tgt, refpoints) the teacher saw.

  - Teacher: pre-hook captures decoder_tgt + decoder_refpoints
    from the live forward, post-hook captures decoder_hs.
  - Student: forward_pre_hook on transformer.decoder
    (`_inject_decoder_inputs`) overwrites tgt and refpoints
    with the supplied teacher tensors before the decoder
    runs. A post-hook captures `_captured_decoder_hs` for
    CRRCD; a pre-hook on decoder.layers[-1] also captures
    the cross-attention inputs (kept for future use).
  - Slot-permutation invariance: `_swap_group_detr` keeps
    matching consistent across DETR group-detr query splits.

Why STFS is bypassed in KD branches
-----------------------------------
STFS (track_queries + RefPointShift) is a temporal-only
mechanism that has no analogue in the static teacher; running
it would corrupt the slot alignment with the teacher. In
VideoRFDETR.forward, both `query_mode == "teacher"` and
`query_mode == "general"` early-return immediately after the
transformer call, before STFS injection and refinement.
ETF (Early Temporal Fusion on backbone features) runs
unconditionally in all three branches, because the temporal
window itself is shared.

Components
----------
rfdetr_video/distill/teacher.py
  VideoFrozenRFDETRTeacher(FrozenRFDETRTeacher):
    - forward_video(frames_hr):
        flattens (B,T,3,S,S) → (B*T,3,S,S), runs base
        forward, returns per-frame {decoder_tgt,
        decoder_refpoints, decoder_hs, foreground_weight,
        pred_logits, pred_boxes, …}.
    - forward_video_general(frames_hr, refpoint_w,
                            query_feat_w, min_weight):
        same flatten + runs the teacher with externally
        supplied general queries; used for Branch 3.

rfdetr_video/distill/__init__.py
  Re-exports CRRCDLoss and distillation_loss from
  rfdetr_temporal.distill (single source of truth for the
  loss math) and exports VideoFrozenRFDETRTeacher.

rfdetr_video/model.py — VideoRFDETR additions:
  - register_teacher_queries(refpoint_w, query_feat_w):
        stores teacher specific queries used in branch 2.
  - sample_general_queries(Q_g, device, dtype):
        refpoint = zeros(Q_g, 4),
        query_feat ~ N(0, 0.02²) of shape (Q_g, D).
  - _inject_decoder_inputs (forward_pre_hook on
        self.transformer.decoder): overrides (tgt, refpoints)
        when `decoder_inputs=...` is passed.
  - _captured_decoder_hs  (forward_hook on the decoder):
        last hidden states for CRRCD.
  - _captured_cross_inputs (forward_pre_hook on
        decoder.layers[-1]): reserved for future losses.
  - In `forward(...)`: KD branches early-return after the
        transformer, skipping STFS + refinement.

rfdetr_video/dataset.py
  When with_teacher_frame=True, the dataset returns a
  parallel HR (T, 3, 704, 704) window using replay-aware
  geometric augmentation so teacher and student see the
  same crop / flip / rotate per timestep — mandatory for
  per-frame slot-aligned KD.

rfdetr_video/train.py
  - Builds VideoRFDETR + VideoFrozenRFDETRTeacher; loads the
    same `checkpoint_best_total.pth` into both.
  - Registers teacher queries on the student.
  - Builds CRRCDLoss and assigns it to `model.crrcd` so its
    two _RelationMLPs (F_t, F_ts) are picked up by
    `model.get_param_groups()` (no "backbone" in the name →
    decoder lr=1e-4).
  - Optimizer is built *after* this assignment.
  - Per-step loss: see Branches 1–3 above. Single
    backward() over the summed loss (one optimizer step
    per `grad_accum_steps`).

CRRCD loss details (inherited unchanged from rfdetr_temporal)
-------------------------------------------------------------
  - Two relation MLPs:
        F_t  : teacher–teacher  hs differences
        F_ts : teacher–student  hs differences
  - For each frame:
        sample K_fg foreground anchors (by teacher
        foreground_weight) + K_bg background anchors,
        compute pairwise relations, score by cosine
        similarity, optimise Sigmoid-NCE with n_neg
        in-batch negatives at temperature τ.
  - Defaults used here: K_fg=16, K_bg=32, n_neg=16,
        τ=0.1, β_crrcd=2.0.

Hyper-parameters used for `stfs_crrcd_v6_etf`
---------------------------------------------
  bs=2, grad_accum=2, T=5, img_size=512, epochs=50,
  ETF on, distill on, distill_general on (Q_g=100),
  CRRCD on (K_fg=16, K_bg=32, n_neg=16, τ=0.1, β=2.0),
  consistency on (λ=0.5, threshold=0.3),
  AdamW lr=1e-4 (decoder) / 1e-5 (backbone),
  wd=1e-4, MultiStepLR(30,40), warmup 500, AMP,
  grad_clip=0.1.

What does NOT receive grads
---------------------------
  - All teacher params: requires_grad=False, frozen in
    eval(), `.grad is None` after backward (asserted in
    the heavy smoke test).
  - The student backbone is unfrozen but trains at lr=1e-5.

Smoke test
----------
rfdetr_video/tests/test_smoke.py::test_distill_one_step
(gated on RFDETR_VIDEO_HEAVY=1) runs one full step
exercising all three branches and asserts non-zero grads
on `refine_layer`, `model.etf`, and `model.crrcd`, plus
that every teacher param has `.grad is None`.
