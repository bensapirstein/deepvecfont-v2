#!/usr/bin/env bash
# E9, second half: the TEST-time sigma sweep.
#
# PROJECT_PLAN.md §3.4 says, of the encoder perturbation:
#
#     "Keep test-time sigma above zero. It is the *only* source of stochasticity at
#      inference, and therefore the only reason the N_s candidates differ from each
#      other at all. [...] Sweep it separately, second, on the winning train sigma."
#
# and §4 scheduled it for days 6-8, "plus the test-time sigma sweep on the E9
# winner". It was never run. Tier 2 launched without it and Tier 3 was scoped
# without it. This script is that sweep.
#
# It matters more than a leftover to-do, for three reasons:
#
#   1. E9 train sigma=0.5 is the single best candidate measured so far -- rank 1 of
#      18 on rendered L1 (-0.0059) and one of only two rows that also moved s-IoU
#      the right way. Its companion knob has never been touched.
#
#   2. sigma_test is applied at eval REGARDLESS of what sigma_train was. In
#      Transformer.forward the scale is `opts.enc_noise_std_train if self.training
#      else opts.enc_noise_std_test`, and both compute_val_loss and test_few_shot.py
#      run under .eval(). So every E9 row was trained at its own sigma and then
#      validated and tested at sigma=1.0. For the sigma_train=0 row that is a
#      complete train/test mismatch -- a model that never saw noise, evaluated under
#      unit-variance noise -- which is a plausible reason it was the worst E9 row
#      (+0.0016) and says nothing about whether sigma_train=0 is a bad idea.
#
#   3. sigma_test sets how much the n_samples candidates differ from each other, so
#      it trades off directly against best-of-N. Too low and all N samples collapse
#      to the same glyph (at exactly 0 they are identical, which §3.4 warns about);
#      too high and each individual sample is worse. There is an optimum and nobody
#      has looked for it.
#
# THIS NEEDS NO RETRAINING. models/transformers.py parses opts at module import, so
# --enc_noise_std_test on the test command line reaches the encoder directly. Same
# cost class as eval_noise.sh -- eval only, runs alongside a training batch.
#
# Usage:
#   ./scripts/sigma_test_sweep.sh                            # defaults below
#   ./scripts/sigma_test_sweep.sh e9_sigma050_chn 0          # experiment, gpu
#   ./scripts/sigma_test_sweep.sh seedfloor_1111_chn 1       # the baseline arm
#
# Run it on BOTH the E9 winner and the baseline. If the optimum sigma_test is the
# same for both, it is a property of the eval procedure and applies to every row in
# the results table -- which would mean every number measured so far was taken at an
# arbitrary point on this curve. If it differs, it interacts with sigma_train and
# belongs in the E9 discussion specifically. Either answer is worth the GPU hour.

set -euo pipefail
cd "$(dirname "$0")/.."

NAME="${1:-e9_sigma050_chn}"
GPU="${2:-0}"

# 1.0 is the released value and the one every number so far was measured at, so it
# is in the ladder as the reference point rather than as a candidate.
SIGMAS=(0.1 0.25 0.5 0.75 1.0 1.5)
N_SAMPLES=3

EXP_DIR="experiments/${NAME}_main_model"
CKPT="$(python3 scripts/best_checkpoint.py "$EXP_DIR")"
STASH="${EXP_DIR}/results_sigma_test"

COMMON_ARGS="--mode test --model_name main_model --language chn --max_seq_len 71 --batch_size 1 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29"

echo "Experiment  : $NAME"
echo "Checkpoint  : $CKPT   (fixed -- the weights never change, only eval sigma)"
echo "GPU         : $GPU"
echo "n_samples   : $N_SAMPLES"
echo
echo "NOTE: decode noise at n_samples 3 was measured at 0.0011 (PROJECT_PLAN 3.2)."
echo "A difference across this ladder smaller than that is not a difference."
echo

mkdir -p "$STASH"
: > "${STASH}/results.txt"

run_one() {
  local sigma="$1"
  local res_dir="${EXP_DIR}/results/${CKPT}"

  rm -rf "$res_dir"
  CUDA_VISIBLE_DEVICES=$GPU python test_few_shot.py $COMMON_ARGS \
    --name_exp "$NAME" --name_ckpt "$CKPT" --n_samples "$N_SAMPLES" \
    --enc_noise_std_test "$sigma" \
    > "nohup_sigmatest_${NAME}_${sigma}.out" 2>&1

  local out l1 iou
  out="$(python eval_reconstruction_error.py --exp_dir "$EXP_DIR" --name_ckpt "$CKPT")"
  l1="$(echo "$out" | grep -oP "Reconstruction Error \(L1.*?: \K[0-9.]+")"
  iou="$(echo "$out" | grep -oP "Mean IOU \(s-IoU\): \K[0-9.]+")"
  printf "  sigma_test=%-5s L1=%-8s s-IoU=%s\n" "$sigma" "$l1" "$iou"
  echo "$sigma $l1 $iou" >> "${STASH}/results.txt"
  rm -rf "$res_dir"
}

echo "=== test-time sigma ladder ==="
for s in "${SIGMAS[@]}"; do
  run_one "$s"
done

echo
echo "=== Summary ==="
python3 - "${STASH}/results.txt" <<'EOF'
import sys

rows = []
for line in open(sys.argv[1]):
    s, l1, iou = line.split()
    rows.append((float(s), float(l1), float(iou)))

if not rows:
    sys.exit("no results")

ref = next((r for r in rows if abs(r[0] - 1.0) < 1e-9), None)
best = min(rows, key=lambda r: r[1])
DECODE_NOISE = 0.0011   # PROJECT_PLAN.md 3.2, measured 2026-08-04

print(f"{'sigma_test':>11} {'L1':>9} {'s-IoU':>9} {'vs sigma=1.0':>14}")
for s, l1, iou in rows:
    d = f"{l1 - ref[1]:+.4f}" if ref else "--"
    mark = "  <- best" if (s, l1, iou) == best else ""
    print(f"{s:>11} {l1:>9.4f} {iou:>9.4f} {d:>14}{mark}")

if ref:
    gain = ref[1] - best[1]
    print()
    print(f"Best sigma_test  : {best[0]}  (L1 {best[1]:.4f})")
    print(f"Released sigma   : 1.0  (L1 {ref[1]:.4f})")
    print(f"Gain over 1.0    : {gain:+.4f}")
    if gain <= DECODE_NOISE:
        print()
        print(f"=> Inside the {DECODE_NOISE} decode-noise band. sigma_test=1.0 is fine and")
        print("   this knob is closed. Record it as a measured null -- it is worth a line")
        print("   in the report precisely because the plan flagged it as untuned.")
    else:
        print()
        print("=> Larger than decode noise, on an EVAL-ONLY change. Two consequences:")
        print("   1. Re-screen at the better sigma_test before comparing anything else,")
        print("      because every number in the results table was taken at 1.0.")
        print("   2. Run this same ladder on the baseline. If the optimum matches, this")
        print("      is a property of the eval procedure rather than of E9, and it")
        print("      shifts the whole table rather than promoting one candidate.")
EOF

echo
echo "Raw rows: ${STASH}/results.txt"
