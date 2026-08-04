#!/usr/bin/env bash
# Splits the screening noise floor into its two components.
#
# PROJECT_PLAN.md §3.2 records a seed-noise floor of 0.0093 L1 across seeds
# 1111/2222/3333, and no Tier 1 candidate cleared it. But that 0.0093 is not
# purely seed noise. Two things vary between any two screening numbers:
#
#   1. training-seed noise -- different seed, different weights;
#   2. eval-decode noise   -- test_few_shot.py never calls setup_seed, and the
#      only source of stochasticity at inference is the sigma=1.0 encoder
#      perturbation, so best-of-n_samples picks a different candidate each run.
#      §3.2 already observed this: seeds 1111 and 2222 selected the *same*
#      checkpoint under both the old and new val_metric, yet their L1 still moved
#      by 0.001-0.003 when re-screened.
#
# This script measures (2) alone, by re-screening ONE fixed checkpoint several
# times. Whatever spread comes back is entirely decode noise, because the weights
# never change. Subtract it and what remains is the seed contribution.
#
# It matters because the two have different fixes. If decode noise dominates,
# raising --n_samples shrinks the floor for eval cost only -- no retraining. If
# seed noise dominates, the only fix is running candidates at multiple seeds,
# which multiplies the training matrix. Guessing wrong costs a day either way.
#
# The n_samples ladder at the end shows how fast the decode component shrinks, so
# the screening budget can be set from a measurement rather than from taste. Note
# that raising n_samples also lowers L1 systematically (it is a best-of-N), so
# numbers are comparable *within* an n_samples column only, never across.
#
# Costs no training GPU. Run it alongside the Tier 2 training batch.
#
# Usage:
#   ./scripts/eval_noise.sh                       # defaults below
#   ./scripts/eval_noise.sh seedfloor_1111_chn 0  # experiment, gpu

set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-seedfloor_1111_chn}"
GPU="${2:-0}"

# Repetitions at the current screening budget, then one run at each larger value.
# Three reps is the minimum that gives a spread; five is better if the GPU is free.
REPS=3
LADDER=(10 20)

EXP_DIR="experiments/${NAME}_main_model"
CKPT="$(python3 scripts/best_checkpoint.py "$EXP_DIR")"
STASH="${EXP_DIR}/results_noise"

COMMON_ARGS="--mode test --model_name main_model --language chn --max_seq_len 71 --batch_size 1 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29"

echo "Experiment : $NAME"
echo "Checkpoint : $CKPT   (fixed for every run below -- the weights never change)"
echo "GPU        : $GPU"
echo

mkdir -p "$STASH"

# test_few_shot.py always writes to results/<name_ckpt>/, so consecutive runs
# would overwrite each other. Move the tree aside after each eval.
run_once() {
  local n_samples="$1" tag="$2"
  local res_dir="${EXP_DIR}/results/${CKPT}"

  rm -rf "$res_dir"
  CUDA_VISIBLE_DEVICES=$GPU python test_few_shot.py $COMMON_ARGS \
    --name_exp "$NAME" --name_ckpt "$CKPT" --n_samples "$n_samples" \
    > "nohup_noise_${tag}.out" 2>&1

  local out
  out="$(python eval_reconstruction_error.py --exp_dir "$EXP_DIR" --name_ckpt "$CKPT")"
  local l1 iou
  l1="$(echo "$out" | grep -oP "Reconstruction Error \(L1.*?: \K[0-9.]+")"
  iou="$(echo "$out" | grep -oP "Mean IOU \(s-IoU\): \K[0-9.]+")"
  printf "  n_samples=%-3s %-12s L1=%-8s s-IoU=%s\n" "$n_samples" "$tag" "$l1" "$iou"
  echo "$n_samples $l1" >> "${STASH}/results.txt"

  # Keep the tree only if something needs inspecting later; otherwise it is ~15 MB
  # per rep of SVGs nobody will read again.
  rm -rf "$res_dir"
}

: > "${STASH}/results.txt"

echo "=== Decode noise: $REPS repetitions of the same checkpoint, n_samples 3 ==="
for (( r=1; r<=REPS; r++ )); do
  run_once 3 "rep${r}"
done

echo
echo "=== n_samples ladder: one run each ==="
for n in "${LADDER[@]}"; do
  run_once "$n" "n${n}"
done

echo
echo "=== Summary ==="
python3 - "${STASH}/results.txt" <<'EOF'
import sys
from collections import defaultdict

by_n = defaultdict(list)
for line in open(sys.argv[1]):
    n, l1 = line.split()
    by_n[int(n)].append(float(l1))

print(f"{'n_samples':>10} {'runs':>5} {'mean L1':>9} {'spread':>8}")
for n in sorted(by_n):
    vals = by_n[n]
    spread = max(vals) - min(vals) if len(vals) > 1 else float('nan')
    print(f"{n:>10} {len(vals):>5} {sum(vals)/len(vals):>9.4f} {spread:>8.4f}")

reps = by_n.get(3, [])
if len(reps) > 1:
    decode = max(reps) - min(reps)
    floor = 0.0093   # PROJECT_PLAN.md 3.2, re-measured 2026-08-04
    print()
    print(f"Decode-only spread at n_samples 3, fixed weights : {decode:.4f}")
    print(f"Full seed-floor spread across 3 seeds           : {floor:.4f}")
    if decode >= floor * 0.5:
        print("=> Decode noise is at least half the floor. Raising --n_samples shrinks it")
        print("   for eval cost only; the ladder above says how far. Record the new bar in")
        print("   PROJECT_PLAN.md 3.2 and re-screen Tier 1 at the new budget so the table")
        print("   stays one comparison.")
    else:
        print("=> The floor is mostly training-seed variance. More n_samples will not help;")
        print("   resolving effects this size needs candidates run at multiple seeds, which")
        print("   is a 3x training matrix. That trade belongs in PROJECT_PLAN.md 8.")
EOF

echo
echo "Raw per-run L1 values: ${STASH}/results.txt"
