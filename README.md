# MorseDistill

Code for distilling a transformer-based spatio-temporal point process model
into a lightweight MLP, using a unified spatio-temporal discrete-Morse
structure to guide the distillation. This release ships three datasets:

- **Wildfire/CA** — per-state cut of a Kaggle US wildfire panel.
- **Canada** — 2023 wildfire hotspots, a Kaggle mirror of NASA FIRMS
  satellite hotspot detections.
- **eMAS** — NASA eMAS (enhanced MODIS Airborne Simulator) fire detections.

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10 and PyTorch 2.4 (CUDA 12.1). A GPU is strongly
recommended; all commands below take a `--cuda_id` argument.

## Repository layout

```
model_mm.py                  teacher/student diffusion model, ST-Morse encoders, fusion gate
model_student.py              MLP history encoder (the distillation target)
transformer_st.py             transformer history encoder (the teacher)
morse_function.py             discrete Morse function construction on a clique complex
st_morse.py                   spatio-temporal cell binning and k-NN graph utilities
build_st_morse_features.py    builds the unified ST-Morse complex for a dataset
train_teacher.py              trains the teacher (transformer encoder + diffusion)
train_distill.py              distills the teacher into the MLP student
test_st.py                    evaluates a teacher or student checkpoint
ds_config.sh, run_*.sh        per-dataset config and thin wrappers around the above
dataset/                      Wildfire/CA, Canada, and eMAS event sequences (train/val/test)
st_morse_features/            precomputed ST-Morse complexes (rebuildable, see below)
checkpoints/                  one trained teacher + student per dataset (seed noted below)
```

## Quickstart: evaluate the shipped checkpoints

```bash
# Teacher (transformer encoder)
NOVLM=1 DUALM=1 bash run_test.sh Wildfire_CA \
    checkpoints/Wildfire_CA_teacher_dual_seed42 transformer 0

# Student (MLP encoder, distilled)
NOVLM=1 DUALM=1 bash run_test.sh Wildfire_CA \
    checkpoints/Wildfire_CA_x4_seed42 mlp 0
```

Swap `Wildfire_CA` for `Canada_fire` (seed 42) or `eMAS_fire` (seed 2026 —
see caveat below) to evaluate the other datasets. Each run prints spatial
MAE and temporal MAE/RMSE, and writes a JSON summary if `--out_json` is
appended.

## Training from scratch

### 1. Build the ST-Morse complex (optional — a prebuilt file is already in
`st_morse_features/`; delete it to force a rebuild)

```bash
python build_st_morse_features.py --dataset Wildfire_CA --zoom 8 \
    --n_bins 16 --k 5 --w_time 1.0 --out_file st_morse_features/st_morse_features_Wildfire_CA_z8_b16.pt
```

### 2. Train the teacher

```bash
EXP_NAME=teacher_dual bash run_teacher.sh Wildfire_CA 0 42 100 --no_vlm --dual_morse
```

### 3. Distill the student

The reported configuration (`--dual_morse`, no VLM branch, λ_pred=0.1,
λ_feat=5.0, λ_Morse-pred=0.1, λ_Morse-feat=0.1, β_topo=0.001):

```bash
EXP_NAME=x4 TEACHER_EXP=teacher_dual bash run_student.sh Wildfire_CA 0 42 100 16 \
    --no_vlm --dual_morse --alpha 0 --beta_topo_st 0.001 \
    --lambda_out 0.1 --lambda_cond 5.0 --lambda_mpred 0.1 --lambda_mfeat 0.1
```

### 4. Evaluate

```bash
NOVLM=1 DUALM=1 bash run_test.sh Wildfire_CA checkpoints/Wildfire_CA_x4_seed42 mlp 0 \
    500 results.json
```

Paper results are averaged over seeds 42, 888, and 2026; repeat steps 2–3
with `--seed 888` / `--seed 2026` (and matching `EXP_NAME`/`TEACHER_EXP`) to
reproduce the full table.

## Notes

- `--no_vlm` is used throughout: the released model does not depend on a
  vision-language embedding. The `EMB` path in `ds_config.sh` is an unused
  placeholder kept only because `--emb_file` is a required argument.
- `--dual_morse` enables the two-branch architecture (one encoder over all
  ST cells, one over the topologically critical cells only), which is fused
  before conditioning the diffusion model.
- **eMAS caveat.** The shipped `eMAS_fire` split is chronological: the
  held-out region is only partially covered by the training period. This
  makes the transformer teacher's spatial accuracy sensitive to the random
  seed — of {42, 888, 2026}, only 2026 trains a stable teacher on this
  split, which is why `checkpoints/eMAS_fire_*_seed2026` is the one shipped
  here. This is a property of the split, not of the method; the
  distilled MLP student is comparatively stable across seeds. See the paper
  appendix for details.
