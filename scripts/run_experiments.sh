#!/usr/bin/env bash
# Centralizes launching a batch of train.py runs (the 3-seed baseline today,
# a Tier 1/2 parameter sweep later) so there is one place to edit the params
# and one switch to flip between running them one after another (the common
# case: one GPU) and running them all at once (when several GPUs are free).
#
# Every run gets exactly one GPU, never more — sharing a GPU across
# concurrent runs has caused problems before.
#
# Usage:
#   ./scripts/run_experiments.sh sequential
#   ./scripts/run_experiments.sh parallel
#
# Edit GPUS and EXPERIMENTS below before each use.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-sequential}"   # sequential | parallel
if [[ "$MODE" != "sequential" && "$MODE" != "parallel" ]]; then
  echo "Usage: $0 [sequential|parallel]" >&2
  exit 1
fi

# GPU ids to use, in order. Check `nvidia-smi` and set these by hand.
# - sequential: only one entry is ever in use at a time, so a single id
#   (e.g. GPUS=(0)) is reused for every experiment.
# - parallel: each experiment gets its own entry, so this needs at least
#   as many ids as EXPERIMENTS below.
GPUS=(1 2 3)

# One entry per experiment: "name_exp  <extra args appended to COMMON_ARGS>".
# This is the loop-over-params spot — add/edit lines here for a sweep.
# --- Tier 1, E7: loss_w_aux sweep (PROJECT_PLAN.md §3.4) ---------------------
# Eq. 11 weights L_bezier at 1.0; options.py defaults it to 0.01, a factor of
# 100 below. No code change needed, the flag already exists.
#
# The 0.01 point of this sweep is already measured: it IS seedfloor_1111_chn,
# same seed, same COMMON_ARGS, default loss_w_aux. Do not re-run it.
# One factor at a time, so every run below holds seed 1111.
EXPERIMENTS=(
  "e7_aux01_chn --seed 1111 --loss_w_aux 0.1"
  "e7_aux03_chn --seed 1111 --loss_w_aux 0.3"
  "e7_aux10_chn --seed 1111 --loss_w_aux 1.0"
)

# Previous batch, kept for provenance (3-seed noise floor, run 2026-08-03,
# L1 spread 0.0077 — see PROJECT_PLAN.md §3.2):
#   "seedfloor_1111_chn --seed 1111"
#   "seedfloor_2222_chn --seed 2222"
#   "seedfloor_3333_chn --seed 3333"

# Args shared by every experiment in this batch. Identical to the seed-floor
# batch, which is what makes e7_* comparable to seedfloor_1111_chn.
COMMON_ARGS="--mode train --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 2"

if [[ "$MODE" == "parallel" && ${#GPUS[@]} -lt ${#EXPERIMENTS[@]} ]]; then
  echo "parallel mode needs ${#EXPERIMENTS[@]} GPU ids in GPUS, only ${#GPUS[@]} given." >&2
  exit 1
fi

pids=()
for i in "${!EXPERIMENTS[@]}"; do
  entry="${EXPERIMENTS[$i]}"
  name="${entry%% *}"
  extra_args="${entry#* }"
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  logfile="nohup_${name}.out"

  echo "[$name] GPU $gpu -> $logfile"
  CUDA_VISIBLE_DEVICES=$gpu nohup python train.py $COMMON_ARGS --name_exp "$name" $extra_args \
    > "$logfile" 2>&1 &
  pid=$!
  pids+=("$pid")

  if [[ "$MODE" == "sequential" ]]; then
    wait "$pid"
    echo "[$name] done"
  fi
done

if [[ "$MODE" == "parallel" ]]; then
  echo "Launched ${#pids[@]} runs in parallel: ${pids[*]}"
  wait "${pids[@]}"
  echo "All runs finished."
fi
