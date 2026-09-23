#!/bin/bash
# Evaluate a trained checkpoint (teacher or student).
# Usage: bash run_test.sh DATASET CKPT_DIR ENCODER GPU [SAMPLINGSTEPS] [OUT_JSON] [WINDOW] [DDIM_ETA] [TRAIN_SEED]
#   ENCODER is "transformer" for a teacher checkpoint, "mlp" for a student.
# NOTE: the eval seed is fixed at 42 on purpose so every model sees identical
# sampling noise; TRAIN_SEED is recorded for bookkeeping only.
set -e
DS=$1; CK=$2; ENC=${3:-transformer}; GPU=${4:-0}; SS=${5:-500}; OUT=${6:-}; W=${7:-16}; ETA=${8:-1.0}; TSEED=${9:-}
PY=${PYTHON:-python}
source "$(dirname "$0")/ds_config.sh"
ds_config "$DS" || exit 1
# HIDDEN / NHID let evaluation match a student whose trunk is not the 256x2
# default; they must match how the checkpoint was trained or load_state_dict
# fails. NOVLM / NOSTM / DUALM (set to any non-empty value) mirror the flags
# the checkpoint was trained with.
$PY test_st.py --dataset "$DS" --ckpt_dir "$CK" --encoder "$ENC" --window "$W" \
  --hidden "${HIDDEN:-256}" --n_hidden "${NHID:-2}" ${NOVLM:+--no_vlm} ${NOSTM:+--no_st_morse} \
  ${DUALM:+--dual_morse} \
  --st_in_dim "${STIN:-2}" \
  --emb_file "$EMB" --st_morse_file "$STF" \
  --zoom "$VZ" --timesteps 500 --samplingsteps "$SS" --n_samples 3 --ddim_eta "$ETA" \
  --cuda_id "$GPU" --seed 42 ${TSEED:+--train_seed $TSEED} ${OUT:+--out_json $OUT}
