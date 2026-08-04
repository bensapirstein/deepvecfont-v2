#!/usr/bin/env bash
# Centralizes launching a batch of train.py runs (the 3-seed baseline today,
# a Tier 1/2 parameter sweep later) so there is one place to edit the params
# and one switch to flip between running them one after another (the common
# case: one GPU) and running them all at once (when several GPUs are free).
#
# Every run gets exactly one GPU, never more — sharing a GPU across
# concurrent runs has caused problems before.
#
# parallel mode runs EXPERIMENTS in waves of len(GPUS): it launches up to
# that many at once, waits for the wave to finish, then launches the next
# wave. EXPERIMENTS can be longer than GPUS; it doesn't need editing between
# waves.
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
# - sequential: only GPUS[0] is ever used, one experiment at a time.
# - parallel: EXPERIMENTS runs in waves of len(GPUS) -- one experiment per id
#   per wave, however many waves it takes to get through the whole array.
GPUS=(1 2 3)

# One entry per experiment: "name_exp  <extra args appended to COMMON_ARGS>".
# This is the loop-over-params spot — add/edit lines here for a sweep.
#
# Full re-run, 2026-08-03 night: all of Tier 1 in one batch. The seedfloor and
# E7 runs below already trained once, but under the pre-fix compute_val_loss
# (see PROJECT_PLAN.md §1.4 "Fixed 2026-08-03") — val_metric silently omitted
# the refinement-decoder loss, so prune_checkpoints could have kept the wrong
# checkpoint as "best" for every one of them. Their old experiment dirs were
# moved to experiments/archive_pre_metricfix/ rather than deleted. Re-running
# them here selects checkpoints under the corrected metric.
EXPERIMENTS=(
  # 3-seed noise floor (PROJECT_PLAN.md §3.2). Re-run under the fixed metric;
  # previous L1 spread 0.0077 was measured on possibly-mis-selected checkpoints.
  "seedfloor_1111_chn --seed 1111"
  "seedfloor_2222_chn --seed 2222"
  "seedfloor_3333_chn --seed 3333"

  # E7: loss_w_aux sweep (PROJECT_PLAN.md §3.4). Eq. 11 weights L_bezier at 1.0;
  # options.py defaults it to 0.01, a factor of 100 below. No code change,
  # the flag already exists. The 0.01 point IS seedfloor_1111_chn (same seed,
  # same COMMON_ARGS, default loss_w_aux) — not repeated here.
  "e7_aux01_chn --seed 1111 --loss_w_aux 0.1"
  "e7_aux03_chn --seed 1111 --loss_w_aux 0.3"
  "e7_aux10_chn --seed 1111 --loss_w_aux 1.0"

  # E9, encoder noise. sigma=1.0 is the baseline (seedfloor_1111_chn), not
  # repeated here. Test sigma stays at 1.0 throughout: it is the only source
  # of stochasticity at inference, so setting it to 0 makes all n_samples
  # candidates identical. It gets its own sweep later, on the winning train sigma.
  "e9_sigma000_chn --seed 1111 --enc_noise_std_train 0.0"
  "e9_sigma010_chn --seed 1111 --enc_noise_std_train 0.1"
  "e9_sigma025_chn --seed 1111 --enc_noise_std_train 0.25"
  "e9_sigma050_chn --seed 1111 --enc_noise_std_train 0.5"

  # E10, dropout. 0.0 is the baseline, again already measured.
  "e10_drop01_chn --seed 1111 --dropout 0.1"
  "e10_drop02_chn --seed 1111 --dropout 0.2"

  # E1, terminal LayerNorm on both encoder stacks. Run alone and with the E9
  # winner: normalizing the residual stream is what makes sigma a meaningful
  # quantity rather than an arbitrary one, so the interaction is the point.
  "e1_norm_chn --seed 1111 --enc_final_norm True"
)

# Args shared by every experiment in this batch. Identical across all of them,
# which is what makes every row comparable to seedfloor_1111_chn.
COMMON_ARGS="--mode train --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 2"

# Appends the launched PID to the global `pids` array. Must be called directly
# (not via `$(launch_one ...)`) -- command substitution forks a subshell, and a
# background job started inside one is not a child of *this* shell, so `wait`
# on its pid later fails with "not a child of this shell".
launch_one() {
  local name="$1" extra_args="$2" gpu="$3"
  local logfile="nohup_${name}.out"
  echo "[$name] GPU $gpu -> $logfile"
  CUDA_VISIBLE_DEVICES=$gpu nohup python train.py $COMMON_ARGS --name_exp "$name" $extra_args \
    > "$logfile" 2>&1 &
  pids+=("$!")
}

if [[ "$MODE" == "sequential" ]]; then
  # One GPU (GPUS[0]) reused for every experiment, one at a time.
  for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%% *}"
    extra_args="${entry#* }"
    pids=()
    launch_one "$name" "$extra_args" "${GPUS[0]}"
    wait "${pids[0]}"
    echo "[$name] done"
  done
else
  # parallel: waves of len(GPUS), so EXPERIMENTS can be longer than GPUS
  # without editing the array between waves.
  n_gpu=${#GPUS[@]}
  n_exp=${#EXPERIMENTS[@]}
  wave=1
  for (( start=0; start<n_exp; start+=n_gpu )); do
    pids=()
    for (( j=0; j<n_gpu && start+j<n_exp; j++ )); do
      entry="${EXPERIMENTS[$((start + j))]}"
      name="${entry%% *}"
      extra_args="${entry#* }"
      launch_one "$name" "$extra_args" "${GPUS[$j]}"
    done
    echo "Wave $wave: launched ${#pids[@]} runs: ${pids[*]}"
    wait "${pids[@]}"
    echo "Wave $wave done."
    wave=$((wave + 1))
  done
  echo "All ${n_exp} runs finished."
fi
