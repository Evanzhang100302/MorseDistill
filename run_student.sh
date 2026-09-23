#!/bin/bash
# Distillation: MLP history encoder distilled from a trained teacher.
# Usage: bash run_student.sh DATASET GPU SEED EPOCHS [WINDOW] [EXTRA_FLAGS...]
#
# To reproduce the paper's reported model (MorseDistill / "x4" configuration),
# pass:
#   --no_vlm --dual_morse --alpha 0 --beta_topo_st 0.001 \
#   --lambda_out 0.1 --lambda_cond 5.0 --lambda_mpred 0.1 --lambda_mfeat 0.1
# and set TEACHER_EXP to match the --exp_name used for run_teacher.sh (the
# teacher must also have been trained with --no_vlm --dual_morse).
set -e
DS=${1:-Wildfire_CA}; GPU=${2:-0}; SEED=${3:-42}; EP=${4:-100}; W=${5:-16}
shift 5 2>/dev/null || shift $#
PY=${PYTHON:-python}
source "$(dirname "$0")/ds_config.sh"
ds_config "$DS" || exit 1
EXP=${EXP_NAME:-student_w${W}}
$PY train_distill.py \
  --dataset "$DS" --zoom "$VZ" --emb_file "$EMB" \
  --st_morse_file "$STF" \
  --teacher_dir "checkpoints/${DS}_${TEACHER_EXP:-teacher}_seed${SEED}" \
  --total_epochs "$EP" --warmup_cond_epochs 5 --window "$W" \
  --timesteps 500 --samplingsteps 500 \
  --lambda_out 1.0 --lambda_cond 1.0 --lambda_fuse 0.5 \
  --alpha 0.05 --beta_topo_st 0.001 \
  --max_val_seqs 200 --eval_every 10 \
  --cuda_id "$GPU" --seed "$SEED" --exp_name "$EXP" "$@"
