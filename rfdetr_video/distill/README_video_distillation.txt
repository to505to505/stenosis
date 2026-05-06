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
            + λ_stfs_align · L_stfs_align (optional teacher
              feature-space alignment on STFS-injected embeddings)

  Branch 2 — KD specific + CRRCD (per-frame or centre-frame)
    If distill_centre_frame_only (E2): slice kd_images and
    kd_teacher_frames to the centre frame c=T//2 before
    all Branch-2/3 teacher and student calls (T_kd=1).
    t_out = teacher.forward_video(kd_teacher_frames)
    student_kd_spec = model(
        kd_images, query_mode="teacher",
        decoder_inputs={tgt, refpoints} from teacher,
    )
    If distill_through_refine (E1): student_hs_spec comes
    from model._captured_refined_hs (post-refine_norm).
    Otherwise: model._captured_decoder_hs (pre-refine).
    distill_spec = distillation_loss(student_kd_spec, t_out)
    loss_crrcd  = CRRCD(teacher_hs, student_hs_spec,
                        foreground_weight)
    Loss: + β_distill · loss_distill_spec
          + β_crrcd   · loss_crrcd

  Branch 3 — KD general (per-frame or centre-frame)
    Same centre-frame slicing as Branch 2 when E2 active.
    gen_q = model.sample_general_queries(Q_g)   # 100 queries
    t_out_gen = teacher.forward_video_general(
        kd_teacher_frames, gen_q.refpoint, gen_q.query_feat,
        min_weight=...)
    student_kd_gen = model(kd_images, query_mode="general",
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
transformer call, before STFS injection. ETF (Early Temporal
Fusion on backbone features) runs unconditionally in all
three branches, because the temporal window itself is shared.

E1 — KD through refinement (distill_through_refine=True)
---------------------------------------------------------
Problem: pre-E1, the KD branches early-returned before the
refinement layer, so the teacher supervised the pre-refine
decoder output whereas inference always used the post-refine
tensor. This misalignment caused the distilled final AP50 to
be *lower* than the un-distilled final AP50.

Fix: when `distill_through_refine=True`, after computing the
first-pass (pre-STFS) `hs[-1]` and `ref_unsigmoid`, the KD
branches call `_refinement_pass` with those tensors. STFS
stays bypassed (temporal slot alignment is not needed —
teacher and student use the same query slots). The
post-refinement output replaces `pred_logits`/`pred_boxes`
in the returned dict and `_captured_refined_hs` is captured
for CRRCD after `refine_norm`.

E2 — Centre-frame-only KD (distill_centre_frame_only=True)
----------------------------------------------------------
Problem: per-frame KD over all T frames penalises temporal-
only detections that the H-FN recovery mechanism in STFS is
designed to make. The 2-D teacher has no knowledge of these
frames and produces near-zero foreground weights for them,
effectively penalising STFS-sourced detections.

Fix: when `distill_centre_frame_only=True`, Branches 2 and 3
slice both `images` and `teacher_frames` to the centre frame
(c = T//2, T_kd = 1) before any teacher or student call.
Branch 1 (detection) still sees the full T-frame window.

E3 — STFS feature alignment (stfs_feature_align_enabled=True)
-------------------------------------------------------------
Problem: E1 makes KD supervise the refinement layer, but KD
branches still bypass STFS to preserve teacher query-slot
alignment. Therefore the refinement layer sees two feature
distributions during training: clean per-frame `hs[-1]` in KD
branches, and STFS-enriched embeddings in Branch 1 / inference.
The KD-heavy signal can make refinement overfit to the clean
single-frame distribution, while FeatureAggregator receives no
direct teacher-space regularisation.

Fix: Branch 1 captures `model._captured_stfs_hs` immediately
after `inject_features(...)` and before `_refinement_pass(...)`.
It also captures `model._captured_stfs_mask`, a boolean mask of
slots whose embedding changed due to STFS injection. When
`stfs_feature_align_enabled=True`, Branch 1 adds:

  L_stfs_align = mean(1 - max_cos(student_stfs_slot,
                  teacher_decoder_slot))

Only STFS-modified slots contribute. The teacher pool is the
top-k foreground teacher decoder slots per frame (from
`foreground_weight`) so background queries do not dominate. This
avoids assuming 1:1 query-slot identity while still keeping STFS
outputs geometrically close to the teacher's robust decoder
feature space.

If `distill_centre_frame_only=True`, this loss is also computed
only on the centre frame and reuses the same KD-specific teacher
forward, preserving the low-VRAM centre-frame KD path.

E4 — 5-point proposal-shifted refinement (RefPointShift)
--------------------------------------------------------
Problem: the old RefPointShift used an MLP to regress an exact
`Δcxcywh` offset from embeddings alone. That is geometrically
blind: it decides where the artery moved before looking at local
visual evidence. On OOD motion patterns the regressed refpoint can
push deformable cross-attention into background.

Fix: RefPointShift no longer has trainable MLP parameters. For each
STFS-injected H-FN slot it builds five explicit reference boxes from
the strong source refpoint:

  centre, up, down, left, right

The wh of every candidate is `src_wh * stfs_shifter_padding_alpha`.
The side candidates shift cx/cy by half of that expanded wh. During
student refinement, normal queries run the standard refinement pass.
Only STFS-injected slots get a sparse second refinement over the 5
candidates. The candidate outputs are softly collapsed by foreground
confidence, so the final tensor keeps shape `(B,T,Q,*)` while the
deformable cross-attention has sampled visual features from all five
spatial hypotheses.

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
        pre-refinement hidden states; used by CRRCD when
        distill_through_refine=False.
  - _captured_refined_hs  (captured inside _refinement_pass
        immediately after refine_norm): post-refinement
        hidden states; used by CRRCD when
        distill_through_refine=True (E1).
    - _captured_stfs_hs and _captured_stfs_mask (captured after
      inject_features): STFS-enriched embeddings and the mask
      of modified slots used by L_stfs_align (E3). They remain
      None when stfs_enabled=False / --no-stfs.
  - RefPointShift: deterministic 5-point proposal grid with zero
        trainable parameters. `_refinement_pass` uses it only for
        STFS-injected slots via sparse candidate refinement (E4).
  - _captured_cross_inputs (forward_pre_hook on
        decoder.layers[-1]): reserved for future losses.
  - In `forward(...)`: KD branches early-return after the
        transformer, skipping STFS. When
        distill_through_refine=True they additionally call
        _refinement_pass on hs[-1] reshaped to (B,T,Q,D)
        and replace pred_logits/pred_boxes with refined
        outputs (aux_outputs dropped on this path).

rfdetr_video/dataset.py
  When with_teacher_frame=True, the dataset returns a
  parallel HR (T, 3, 704, 704) window using replay-aware
  geometric augmentation so teacher and student see the
  same crop / flip / rotate per timestep — mandatory for
  per-frame slot-aligned KD.

rfdetr_video/train.py
  - Builds VideoRFDETR + VideoFrozenRFDETRTeacher; loads the
    student checkpoint and the teacher checkpoint.
  - Registers teacher queries on the student.
  - Builds CRRCDLoss and assigns it to `model.crrcd` so its
    two _RelationMLPs (F_t, F_ts) are picked up by
    `model.get_param_groups()` (no "backbone" in the name →
    decoder lr=1e-4).
  - Optimizer is built *after* this assignment.
  - Prints `[KD] distill_through_refine=... 
    distill_centre_frame_only=... stfs_feature_align=...
    stfs_feature_align_weight=...` and
    `[Video] stfs_enabled=... refinement_enabled=...` at startup.
  - Argparse flags: --distill-through-refine (E1),
    --distill-centre-frame-only (E2), --stfs-feature-align,
    --stfs-feature-align-weight, --stfs-feature-align-teacher-topk,
    --stfs-shifter-padding-alpha, --no-stfs for full STFS bypass,
    --no-refinement for first-pass decoder-output training/eval.
    --stfs-shifter-hidden-dim is deprecated and ignored because
    RefPointShift has no MLP.
  - Per-step loss: see Branches 1–3 above. Centre-frame
    slicing (E2) applied before teacher/student calls in
    Branches 2+3. CRRCD student_hs source switches to
    model._captured_refined_hs when E1 active. STFS feature
    alignment (E3) is added after the KD-specific teacher forward
    and before the KD-specific student forward resets model
    captures. Single backward() over the summed loss (one
    optimizer step per `grad_accum_steps`).

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

Hyper-parameters used for `stfs_crrcd_v6_etf_postref_centreKD`
--------------------------------------------------------------
  Same as above plus:
    distill_through_refine=True  (E1)
    distill_centre_frame_only=True  (E2)
  Both flags activate together; either can be used alone.

Optional STFS feature alignment add-on
--------------------------------------
  Add:
    --stfs-feature-align
    --stfs-feature-align-weight 0.1
    --stfs-feature-align-teacher-topk 16

  Recommended run-name suffix:
    stfs_crrcd_v6_etf_postref_centreKD_stfsAlign

What does NOT receive grads
---------------------------
  - All teacher params: requires_grad=False, frozen in
    eval(), `.grad is None` after backward (asserted in
    the heavy smoke test).
  - The student backbone is unfrozen but trains at lr=1e-5.

Smoke tests
-----------
rfdetr_video/tests/test_smoke.py::test_distill_one_step
(gated on RFDETR_VIDEO_HEAVY=1) runs one full step
exercising all three branches and asserts non-zero grads
on `refine_layer`, `model.etf`, and `model.crrcd`, plus
that every teacher param has `.grad is None`.

rfdetr_video/tests/test_smoke.py::
    test_distill_through_refine_and_centre_only[E1]
    test_distill_through_refine_and_centre_only[E1+E2]
(gated on RFDETR_VIDEO_HEAVY=1) runs Branches 2+3 only
(no detection loss) and asserts:
  - refine_layer receives grad from KD branch alone (E1).
  - CRRCD MLPs receive grad.
  - teacher params have .grad is None.
  - E1+E2 variant: pred_logits.shape[0] == B*1 (T_kd=1).

rfdetr_video/tests/test_smoke.py::test_stfs_feature_alignment_loss_masked_topk
validates the pure feature-alignment loss: only masked STFS slots
receive gradient and teacher matching is restricted to top-k
foreground teacher slots.

rfdetr_video/tests/test_smoke.py::test_forward_shapes now also
asserts that a student forward captures `_captured_stfs_hs` and
`_captured_stfs_mask` with shape (B,T,Q,*).

rfdetr_video/tests/test_smoke.py::test_refpoint_shift_generates_five_point_grid
asserts that RefPointShift emits centre/up/down/left/right candidates
and has zero trainable parameters.
