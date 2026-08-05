#!/usr/bin/env bash
# Companion to run_experiments.sh, for the test/eval half of a batch: runs
# test_few_shot.py then eval_reconstruction_error.py for each experiment,
# auto-picking each one's best-val_metric checkpoint. Selection reads
# experiments/<name>_main_model/logs/checkpoint_metrics.csv (written by train.py at
# every checkpoint save -- see checkpoint_log.py and prune_checkpoints() in train.py,
# which keeps the lowest-val_metric checkpoints on disk for the same reason), via
# scripts/best_checkpoint.py. Falls back to legacy filename parsing for experiment
# dirs trained before that manifest existed. Same sequential/parallel switch, one
# GPU per run.
#
# Today's use: score the full Tier 1 re-run (3-seed floor + E7 + E9 + E10 +
# E1, PROJECT_PLAN.md §3.2-§3.4) at the screening budget. Edit EXPERIMENTS for
# later Tier 1/2 batches.
#
# NOTE: PROJECT_PLAN.md §3.2 specifies the screening eval as 8 fixed test
# fonts. test_few_shot.py has no font-subsetting flag -- it always runs the
# full 34-font test split -- so this only cuts cost via --n_samples 3, not
# by font count. Fine for now (minutes either way); revisit if it isn't.
#
# parallel mode runs EXPERIMENTS in waves of len(GPUS), same as
# run_experiments.sh -- EXPERIMENTS can be longer than GPUS.
#
# Usage:
#   ./scripts/test_experiments.sh sequential
#   ./scripts/test_experiments.sh parallel

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-sequential}"   # sequential | parallel
if [[ "$MODE" != "sequential" && "$MODE" != "parallel" ]]; then
  echo "Usage: $0 [sequential|parallel]" >&2
  exit 1
fi

# GPU ids to use, in order. Check `nvidia-smi` and set these by hand.
# Restricted to 2 GPUs as of 2026-08-04 (user directive) -- GPU 3 is off limits
# going forward, not just for this run.
GPUS=(1 2)

# One entry per experiment: "name_exp  <extra args, if any>". Extra args are
# needed whenever the candidate's flag changes what modules ModelMain
# constructs (not just a training-time weighting) -- ModelMain(opts) is built
# the same way for test as for train, and ckpt loading is strict, so a
# mismatch is a hard crash, not a silent wrong answer. E1 (--enc_final_norm)
# adds LayerNorm parameters and needs the flag repeated here; E7 (loss_w_aux,
# a loss weight only) and E9/E10 (enc_noise_std_train, dropout -- neither
# adds parameters, and eval() already disables dropout's effect) do not.
# Matches scripts/run_experiments.sh's EXPERIMENTS for the 2026-08-03 night re-run.
#
# Tier 2, 2026-08-04. Which flags have to be repeated here, and why:
#   --n_args_bins      YES. Resizes arg_embed and args_fcn, and denumericalize
#                      reads it at decode time. Omitting it is a load crash.
#   --n_layers_refine  YES. Adds DecoderLayer modules, so state_dict keys change.
#   --arg_embed_pad_idx  YES, though only for the record: padding_idx changes no
#                      shape and has no effect without gradients, so test would
#                      run correctly without it. Passed so opts.txt is honest.
#   --args_label_smooth_sigma  no. Loss-only, never read outside Transformer.loss.
#   --lr_schedule      no. Optimizer-only.
#
# Tier 3, 2026-08-04. Same question, answered per flag:
#   --img_norm         YES. Swaps LayerNorm for GroupNorm/BatchNorm2d/InstanceNorm2d
#                      in both image stacks. Different modules, different state_dict
#                      keys and shapes. Omitting it is a load crash.
#   --ngf              YES. Every conv width in both image stacks, plus fc_fusion.
#   --bottleneck_bits  YES. fc_fusion output width, the image decoder's input_nc,
#                      and it decides whether z_proj exists at all.
#   --ema_decay        no. Training-time only; the EMA weights are what got saved
#                      into the checkpoint, under the ordinary key names.
#   --optimizer        no. Optimizer-only, never reaches ModelMain.
#   --weight_decay     no. Same.
#   --kl_beta          no. Loss weight only.
EXPERIMENTS=(
  # Re-screened, not re-trained, in the same batch as the candidates. The recorded
  # baseline came from a different screening session, and test_few_shot.py's
  # best-of-n_samples decoding is unseeded, so it moves ~0.001-0.003 between
  # sessions (§3.2). Scoring it here removes that as a confound. Batch B compares
  # against all three seed-floor runs, so all three are re-screened.
  "seedfloor_1111_chn"
  "seedfloor_2222_chn"
  "seedfloor_3333_chn"

  # Batch A: Tier 3 breadth.
  "e12_kl000_chn"
  "e12_kl100_chn"
  "e11_adamw_chn"
  "e2_groupnorm_chn --img_norm group"
  "e2_batchnorm_chn --img_norm batch"
  # e2_instancenorm_chn dropped -- see run_experiments.sh's E2 note.
  "e4_ngf32_chn --ngf 32"
  "e15_ema999_chn"
  "e5_bneck256_chn --bottleneck_bits 256"
  "e5_bneck1024_chn --bottleneck_bits 1024"

  # Batch B: multi-seed replication. Seed 1111 of each was screened in an earlier
  # session; re-screened here so all three seeds of a candidate come from one
  # session and the decode-noise term is common to the whole mean.
  "e9_sigma050_chn"
  "e9_sigma050_2222_chn"
  "e9_sigma050_3333_chn"
  "e1_norm_chn --enc_final_norm True"
  "e1_norm_2222_chn --enc_final_norm True"
  "e1_norm_3333_chn --enc_final_norm True"
  "e13_bins256_chn --n_args_bins 256"
  "e13_bins256_2222_chn --n_args_bins 256"
  "e13_bins256_3333_chn --n_args_bins 256"
)

# Tier 2 screening batch, 2026-08-04, kept for provenance:
#   "seedfloor_1111_chn"  "e8_ls05_chn"  "e8_ls10_chn"  "e8_ls20_chn"
#   "e13_bins256_chn --n_args_bins 256"   "e13_nopad_chn --arg_embed_pad_idx False"
#   "e3_refine2_chn --n_layers_refine 2"  "e3_refine3_chn --n_layers_refine 3"
#   "e14_wucos_chn"

# Tier 1 batch, 2026-08-03 night, kept for provenance:
#   "seedfloor_1111_chn"  "seedfloor_2222_chn"  "seedfloor_3333_chn"
#   "e7_aux01_chn"  "e7_aux03_chn"  "e7_aux10_chn"
#   "e9_sigma000_chn"  "e9_sigma010_chn"  "e9_sigma025_chn"  "e9_sigma050_chn"
#   "e10_drop01_chn"  "e10_drop02_chn"
#   "e1_norm_chn --enc_final_norm True"

# Screening budget per PROJECT_PLAN.md §3.2: n_samples 3, not the n_samples 50
# confirmation eval.
COMMON_ARGS="--mode test --model_name main_model --language chn --max_seq_len 71 --batch_size 1 --n_samples 3 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29"

# entry is "name_exp" or "name_exp <extra args>" -- ${entry#* } leaves entry
# unchanged (wrong) when there's no space to strip, so check first.
entry_name() { echo "${1%% *}"; }
entry_args() { if [[ "$1" == *" "* ]]; then echo "${1#* }"; else echo ""; fi; }

NAMES=()
for entry in "${EXPERIMENTS[@]}"; do
  NAMES+=("$(entry_name "$entry")")
done

best_ckpt() {
  # Delegates to checkpoint_log.py so selection logic lives in exactly one place,
  # shared with prune_checkpoints() in train.py.
  python3 scripts/best_checkpoint.py "experiments/$1_main_model"
}

declare -A CKPTS
for name in "${NAMES[@]}"; do
  CKPTS["$name"]="$(best_ckpt "$name")"
done

# Appends the launched PID to the global `pids` array. Must be called directly
# (not via `$(launch_one ...)`) -- command substitution forks a subshell, and a
# background job started inside one is not a child of *this* shell, so `wait`
# on its pid later fails with "not a child of this shell".
launch_one() {
  local name="$1" ckpt="$2" gpu="$3" extra_args="${4:-}"
  local logfile="nohup_test_${name}.out"
  echo "[$name] ckpt=$ckpt GPU $gpu -> $logfile"
  CUDA_VISIBLE_DEVICES=$gpu nohup python test_few_shot.py $COMMON_ARGS --name_exp "$name" --name_ckpt "$ckpt" $extra_args \
    > "$logfile" 2>&1 &
  pids+=("$!")
}

if [[ "$MODE" == "sequential" ]]; then
  for entry in "${EXPERIMENTS[@]}"; do
    name="$(entry_name "$entry")"
    extra_args="$(entry_args "$entry")"
    pids=()
    launch_one "$name" "${CKPTS[$name]}" "${GPUS[0]}" "$extra_args"
    wait "${pids[0]}"
    echo "[$name] done"
  done
else
  # parallel: waves of len(GPUS), so EXPERIMENTS can be longer than GPUS.
  n_gpu=${#GPUS[@]}
  n_exp=${#EXPERIMENTS[@]}
  wave=1
  for (( start=0; start<n_exp; start+=n_gpu )); do
    pids=()
    for (( j=0; j<n_gpu && start+j<n_exp; j++ )); do
      entry="${EXPERIMENTS[$((start + j))]}"
      name="$(entry_name "$entry")"
      extra_args="$(entry_args "$entry")"
      launch_one "$name" "${CKPTS[$name]}" "${GPUS[$j]}" "$extra_args"
    done
    echo "Wave $wave: launched ${#pids[@]} runs: ${pids[*]}"
    wait "${pids[@]}"
    echo "Wave $wave done."
    wave=$((wave + 1))
  done
  echo "All ${n_exp} runs finished."
fi

echo
echo "=== Evaluation ==="
declare -A L1S IOUS RENDER
for name in "${NAMES[@]}"; do
  ckpt="${CKPTS[$name]}"
  echo "--- $name ($ckpt) ---"
  out="$(python eval_reconstruction_error.py --exp_dir "experiments/${name}_main_model" --name_ckpt "$ckpt")"
  echo "$out" | tail -5
  L1S["$name"]="$(echo "$out" | grep -oP "Reconstruction Error \(L1.*?: \K[0-9.]+")"
  IOUS["$name"]="$(echo "$out" | grep -oP "Mean IOU \(s-IoU\): \K[0-9.]+")"
  RENDER["$name"]="$(echo "$out" | grep -oP "Renderability: \K[0-9.]+ glyphs \(\d+/\d+\)")"
done

# PROJECT_PLAN.md / docs/tier1-launch.md §5 report format: per run, L1 minus
# the baseline's L1, and whether that clears the noise-floor bar. Only
# meaningful once BASELINE has an L1 -- set to the empty string to skip.
BASELINE="seedfloor_1111_chn"

# The bar a candidate has to clear to be a result rather than seed variance.
# Was 0.008 for Tier 1's first pass; re-measured to 0.0093 on 2026-08-04 after the
# val_metric fix changed which checkpoint seed 3333 selects (PROJECT_PLAN.md §3.2).
# scripts/eval_noise.sh is measuring how much of this is decode noise rather than
# seed noise; lower it only when that lands, and say so in the plan when you do.
NOISE_FLOOR=0.0093

echo
echo "=== Per-run summary (docs/tier1-launch.md §5) ==="
printf "%-20s %-16s %-8s %-8s %-24s %-9s %s\n" "name" "checkpoint" "L1" "s-IoU" "renderability" "delta" "clears_${NOISE_FLOOR}"
baseline_l1="${L1S[$BASELINE]:-}"
for name in "${NAMES[@]}"; do
  l1="${L1S[$name]:-NA}"
  iou="${IOUS[$name]:-NA}"
  render="${RENDER[$name]:-NA}"
  if [[ "$name" == "$BASELINE" ]]; then
    delta="--"; clears="baseline"
  elif [[ -n "$baseline_l1" && "$l1" != "NA" ]]; then
    delta="$(python3 -c "print(f'{$l1 - $baseline_l1:+.4f}')")"
    clears="$(python3 -c "print('yes' if ($baseline_l1 - $l1) > $NOISE_FLOOR else 'no')")"
  else
    delta="n/a"; clears="n/a"
  fi
  printf "%-20s %-16s %-8s %-8s %-24s %-9s %s\n" "$name" "${CKPTS[$name]}" "$l1" "$iou" "$render" "$delta" "$clears"
done
echo "delta = candidate L1 - $BASELINE L1 (negative = better). clears: candidate beats baseline by more than $NOISE_FLOOR L1."

seed_names=(seedfloor_1111_chn seedfloor_2222_chn seedfloor_3333_chn)
have_all_seeds=1
for s in "${seed_names[@]}"; do
  [[ -n "${L1S[$s]:-}" ]] || have_all_seeds=0
done
if [[ "$have_all_seeds" == 1 ]]; then
  echo
  echo "=== Seed-noise floor (screening metric spread) ==="
  python3 - "${L1S[seedfloor_1111_chn]}" "${L1S[seedfloor_2222_chn]}" "${L1S[seedfloor_3333_chn]}" <<'EOF'
import sys
vals = [float(v) for v in sys.argv[1:]]
print(f"L1 per seed: {vals}")
print(f"spread (max-min): {max(vals)-min(vals):.4f}")
EOF
fi
