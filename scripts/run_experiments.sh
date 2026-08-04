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
# Restricted to 2 GPUs as of 2026-08-04 (user directive) -- GPU 3 is off limits
# going forward, not just for this run. The already-running Tier 2 batch that
# used GPU 3 in waves 1-2 was left to finish rather than killed.
GPUS=(1 2)

# One entry per experiment: "name_exp  <extra args appended to COMMON_ARGS>".
# This is the loop-over-params spot — add/edit lines here for a sweep.
#
# Tier 3 + multi-seed replication, 2026-08-04 (PROJECT_PLAN.md §3.6, §8 items 2-3).
# Sixteen runs in two batches. Two GPUs, ~1 h per run, so ~8 h — one overnight.
#
# The batch is split because the two halves answer different questions and the
# second is the one that survives a null:
#
#   Batch A (10 runs, waves 1-5). Tier 3 breadth, all at --seed 1111, each one
#   factor off seedfloor_1111_chn. Expect these to land inside the floor, same as
#   Tier 1 and Tier 2 did; they are in the table for coverage of the assignment's
#   change categories, not because a single point is expected to resolve.
#
#   Batch B (6 runs, waves 6-8). The three largest deltas measured so far, each
#   re-run at seeds 2222 and 3333 so it has a mean against the baseline's mean
#   (0.1724 / 0.1631 / 0.1680) rather than a point against a point. §3.2's
#   decomposition put ~88% of the 0.0093 floor in training-seed variance, so this
#   is the only lever that can shrink the bar, and it is what §8 item 3 leans to.
#
# Run A then B in one go: leave both blocks uncommented and launch parallel. To
# stop after A, comment out the Batch B block.
EXPERIMENTS=(
  # ---- Batch A: Tier 3 breadth ------------------------------------------------

  # E12, KL weight. kl_beta has always been 0.01 and never tuned. It interacts
  # with E9 head-on: with a sigma=1.0 additive perturbation already on the encoder
  # output, the reparameterization trick is close to decorative, so the KL term may
  # be regularizing something that is already noisy by construction. Bracketed wide
  # (0 and 10x) rather than sampled finely — nothing in two tiers has resolved at
  # this scale, so a fine sweep would buy resolution the setup does not have.
  "e12_kl000_chn --seed 1111 --kl_beta 0.0"
  "e12_kl100_chn --seed 1111 --kl_beta 0.1"

  # E11, AdamW. torch.optim.AdamW is imported in train.py and unused. Adam applies
  # --weight_decay as an L2 term inside the gradient, where the adaptive per-
  # parameter rate rescales it; AdamW decouples it. At the released weight_decay=0
  # the two are identical, so this run is the first time the flag does anything.
  "e11_adamw_chn --seed 1111 --optimizer adamw --weight_decay 0.01"

  # E2, image-stack normalization. The released norm is a spatial LayerNorm over
  # [C, H, W], which pools channel and spatial statistics together and so discards
  # per-channel scale. All three alternatives normalize per channel. This is the
  # assignment's "add normalization layers" bullet done on the image branch, where
  # E1 did it on the sequence branch.
  "e2_groupnorm_chn --seed 1111 --img_norm group"
  "e2_batchnorm_chn --seed 1111 --img_norm batch"
  "e2_instancenorm_chn --seed 1111 --img_norm instance"

  # E4, width. ngf 16 -> 32 doubles both image encoder and decoder. Unlike every
  # other row in this batch it changes capacity, so read it as a capacity control
  # rather than as a like-for-like architectural factor, and say so in the report.
  "e4_ngf32_chn --seed 1111 --ngf 32"

  # E15, weight EMA. The one candidate here with a reliable prior in the
  # literature, and the only one that changes nothing about training — the EMA is
  # evaluated and checkpointed, the optimizer still steps the raw weights. If
  # anything in Tier 3 clears the floor it is most likely this.
  "e15_ema999_chn --seed 1111 --ema_decay 0.999"

  # E5, latent width. --bottleneck_bits looked like a free flag and was not: the
  # latent is written into a 512-wide slot in ModalityFusion, so any other value
  # crashed until the z_proj projection added 2026-08-04. Bracketed either side of
  # 512. Note both rows change parameter count in the image decoder too, whose
  # input_nc is bottleneck_bits + char_num.
  "e5_bneck256_chn --seed 1111 --bottleneck_bits 256"
  "e5_bneck1024_chn --seed 1111 --bottleneck_bits 1024"

  # E6 has no row here on purpose. The dead Perceiver cross-attention parameters
  # are constructed before several live modules, and every construction draws from
  # the global RNG stream, so deleting them shifts the initialization of everything
  # after them. A "dead params removed" run is numerically a seed change, and the
  # seed floor is larger than anything the sweep is chasing, so the run could not
  # be read either way. It is a reconstruction finding instead:
  #   python scripts/dead_params.py

  # ---- Batch B: multi-seed replication of the three largest deltas -------------
  # Seed 1111 is already trained for all three; only 2222 and 3333 are missing.
  # Deltas vs the seed-1111 baseline at screening: E9 sigma=0.5 -0.0059,
  # E1 -0.0054, E13 bins256 -0.0050. All roughly half the 0.0093 floor, and
  # statistically indistinguishable from each other, which is exactly why picking
  # one would have been arbitrary and all three get the same treatment.
  "e9_sigma050_2222_chn --seed 2222 --enc_noise_std_train 0.5"
  "e9_sigma050_3333_chn --seed 3333 --enc_noise_std_train 0.5"
  "e1_norm_2222_chn --seed 2222 --enc_final_norm True"
  "e1_norm_3333_chn --seed 3333 --enc_final_norm True"
  "e13_bins256_2222_chn --seed 2222 --n_args_bins 256"
  "e13_bins256_3333_chn --seed 3333 --n_args_bins 256"
)

# Tier 2 batch, 2026-08-04, kept for provenance. Results in §3.5: nothing cleared
# the 0.0093 floor; E13 bins256 was the largest single delta at -0.0050.
#   "e8_ls05_chn --seed 1111 --args_label_smooth_sigma 0.5"
#   "e8_ls10_chn --seed 1111 --args_label_smooth_sigma 1.0"
#   "e8_ls20_chn --seed 1111 --args_label_smooth_sigma 2.0"
#   "e13_bins256_chn --seed 1111 --n_args_bins 256"
#   "e13_nopad_chn --seed 1111 --arg_embed_pad_idx False"
#   "e3_refine2_chn --seed 1111 --n_layers_refine 2"
#   "e3_refine3_chn --seed 1111 --n_layers_refine 3"
#   "e14_wucos_chn --seed 1111 --lr_schedule warmup_cosine"

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
