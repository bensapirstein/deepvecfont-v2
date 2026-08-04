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
# Tier 2, 2026-08-04 (PROJECT_PLAN.md §3.5). Eight runs, one seed each, on the
# breadth-over-depth allocation: cover the whole tier first, deepen whichever
# candidate leads afterwards. Three GPUs, three waves, ~1 h per run.
#
# Every entry stays at --seed 1111 and shares COMMON_ARGS with the Tier 1 batch,
# so each row is comparable to seedfloor_1111_chn (L1 0.1724) on exactly one
# changed factor. Read the caveat first: the re-measured seed floor is 0.0093 and
# no Tier 1 candidate cleared it, so a single point here is unlikely to either.
# The eval-noise decomposition in scripts/eval_noise.sh runs alongside this batch
# to work out how much of that 0.0093 is decode noise rather than seed noise.
EXPERIMENTS=(
  # E8, ordinal label smoothing on the argument head. The 128-way cross-entropy
  # is permutation-invariant in the bin index — predicting bin 5 for a target of
  # 60 costs what predicting 61 costs — so the head has no notion that coordinates
  # live on a line. sigma is in bins. sigma=0 is the baseline, not repeated.
  # Screens on the rendered metric: it changes the cross-entropy scale, so
  # val_metric is not comparable across these rows (§1.5).
  "e8_ls05_chn --seed 1111 --args_label_smooth_sigma 0.5"
  "e8_ls10_chn --seed 1111 --args_label_smooth_sigma 1.0"
  "e8_ls20_chn --seed 1111 --args_label_smooth_sigma 2.0"

  # E13, quantization. Split into two one-factor runs rather than the one bundled
  # change §3.5 describes, because bin count and padding_idx are independent and
  # the protocol is one factor at a time. Gate 2 (the oracle) is deliberately
  # skipped: it runs after, to explain the result rather than to license it.
  # Also screens on the rendered metric — 256 bins changes the CE scale too.
  "e13_bins256_chn --seed 1111 --n_args_bins 256"
  "e13_nopad_chn --seed 1111 --arg_embed_pad_idx False"

  # E3, self-refinement decoder depth. Sec. 3.3 says 2 layers, the code clones 1.
  # This is the decoder whose output is actually scored (§1.1), which is why one
  # character earns a slot.
  "e3_refine2_chn --seed 1111 --n_layers_refine 2"
  "e3_refine3_chn --seed 1111 --n_layers_refine 3"

  # E14, warmup + cosine. ExponentialLR(0.997) over 150 epochs multiplies the lr
  # by 0.64, so the released schedule is effectively constant at this budget, and
  # there is no warmup at all. Disproportionately helps short runs, which is what
  # the entire matrix consists of. If it wins, it applies to every run after it
  # and the protocol section has to say so.
  "e14_wucos_chn --seed 1111 --lr_schedule warmup_cosine"
)

# Tier 1 batch, 2026-08-03 night, kept for provenance. Re-run under the fixed
# compute_val_loss (PROJECT_PLAN.md §1.5) after val_metric was found to omit the
# refinement-decoder loss. Results in §3.2: noise floor 0.0093, nothing cleared it.
#   "seedfloor_1111_chn --seed 1111"
#   "seedfloor_2222_chn --seed 2222"
#   "seedfloor_3333_chn --seed 3333"
#   "e7_aux01_chn --seed 1111 --loss_w_aux 0.1"
#   "e7_aux03_chn --seed 1111 --loss_w_aux 0.3"
#   "e7_aux10_chn --seed 1111 --loss_w_aux 1.0"
#   "e9_sigma000_chn --seed 1111 --enc_noise_std_train 0.0"
#   "e9_sigma010_chn --seed 1111 --enc_noise_std_train 0.1"
#   "e9_sigma025_chn --seed 1111 --enc_noise_std_train 0.25"
#   "e9_sigma050_chn --seed 1111 --enc_noise_std_train 0.5"
#   "e10_drop01_chn --seed 1111 --dropout 0.1"
#   "e10_drop02_chn --seed 1111 --dropout 0.2"
#   "e1_norm_chn --seed 1111 --enc_final_norm True"

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
