============================================================
Distillation in rfdetr_video
============================================================

Goal
----
Transfer per-frame detection knowledge from a frozen 2-D RF-DETR teacher
into the Video RF-DETR student. The student sees a T-frame low-resolution
window; the teacher sees geometrically aligned high-resolution frames.

Training branches
-----------------
Branch 1 - student detection
  out = model(images, query_mode="student")
  outputs: pred_logits / pred_boxes with shape (B,T,Q,*)
  loss: standard RF-DETR detection loss over all B*T frames
        plus optional count consistency loss

Branch 2 - KD specific sampling
  t_out = teacher.forward_video(kd_teacher_frames)
  s_out = model(
      kd_images,
      query_mode="teacher",
      decoder_inputs={
          "tgt": t_out["decoder_tgt"],
          "refpoints": t_out["decoder_refpoints"],
      },
  )
  loss: KD-DETR distillation loss
        plus optional CRRCD loss on decoder hidden states

Branch 3 - KD general sampling
  gen_q = model.sample_general_queries(Q_g)
  t_out_gen = teacher.forward_video_general(kd_teacher_frames, ...)
  s_out_gen = model(
      kd_images,
      query_mode="general",
      general_queries=gen_q,
      decoder_inputs={...},
  )
  loss: weighted KD-DETR distillation loss

Centre-frame KD option
----------------------
When distill_centre_frame_only=True, branches 2 and 3 slice the student and
teacher windows to the centre frame before teacher/student KD calls. Branch 1
still trains on the full T-frame window.

Query-slot alignment
--------------------
The teacher captures the decoder input tensors it used. The student decoder
has a forward pre-hook that replaces its decoder tgt and refpoints with those
teacher tensors in KD branches. This keeps teacher and student slot indices
aligned for KD-DETR losses.

CRRCD
-----
CRRCD uses:
  teacher_hs: t_out["decoder_hs"]
  student_hs: model._captured_decoder_hs
  weights:    t_out["foreground_weight"]

The relation MLPs are attached to the student model before optimizer
construction so they are included in model.get_param_groups().

Teacher frames
--------------
VideoStenosisDataset returns high-resolution teacher windows when
with_teacher_frame=True. Geometric replay is shared with the student window,
so teacher and student frames remain spatially aligned.

CLI flags
---------
Core training:
  --epochs
  --batch-size
  --lr
  --T
  --dataset
  --checkpoint
  --output-dir
  --run-name
  --freeze-decoder
  --no-wandb
  --num-workers
  --grad-accum
  --img-size

Distillation:
  --distill
  --no-general
  --num-general-queries
  --distill-teacher-ckpt
  --distill-centre-frame-only

CRRCD:
  --crrcd
  --crrcd-weight
  --crrcd-num-fg
  --crrcd-num-bg
  --crrcd-num-negatives
  --crrcd-temperature

Consistency:
  --consistency-weight
  --consistency-threshold
  --no-consistency

ETF:
  --etf
  --etf-heads
  --etf-dropout

Smoke tests
-----------
rfdetr_video/tests/test_smoke.py covers:
  - no pixel-level unfold call
  - source invariants for the video model and ETF
  - count consistency gradients
  - dataset T-frame target shape
  - model forward shape checks when RFDETR_VIDEO_HEAVY=1
  - ETF gradient flow when RFDETR_VIDEO_HEAVY=1
  - CRRCD and end-to-end KD branch smoke tests when RFDETR_VIDEO_HEAVY=1
