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
