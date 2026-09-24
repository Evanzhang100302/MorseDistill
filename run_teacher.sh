#!/bin/bash
# Teacher training (unified ST Morse, transformer history encoder).
# Usage: bash run_teacher.sh DATASET GPU SEED EPOCHS [EXTRA_FLAGS...]
# EXP_NAME overrides the checkpoint directory suffix (default: teacher).
set -e
DS=${1:-Wildfire_CA}; GPU=${2:-0}; SEED=${3:-42}; EP=${4:-100}
shift 4 2>/dev/null || shift $#
PY=${PYTHON:-python}
source "$(dirname "$0")/ds_config.sh"
ds_config "$DS" || exit 1

[ -f "$STF" ] || $PY build_st_morse_features.py --dataset "$DS" --zoom "$SZ" --n_bins 16 --k 5 --w_time 1.0 \
  --out_file "$STF"

$PY train_teacher.py \
  --dataset "$DS" --zoom "$VZ" --emb_file "$EMB" --st_morse_file "$STF" \
  --total_epochs "$EP" --timesteps 500 --samplingsteps 500 \
  --alpha 0.05 --beta_topo_st 0 \
  --max_val_seqs 200 --eval_every 10 \
  --cuda_id "$GPU" --seed "$SEED" --exp_name "${EXP_NAME:-teacher}" "$@"
