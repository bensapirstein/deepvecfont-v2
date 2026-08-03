#!/usr/bin/env bash
# Companion to run_experiments.sh, for the test/eval half of a batch: runs
# test_few_shot.py then eval_reconstruction_error.py for each experiment,
# auto-picking each one's highest-epoch checkpoint. Same sequential/parallel
# switch, one GPU per run.
#
# Today's use: score the three seed-floor runs at the screening budget
# (PROJECT_PLAN.md §3.2) to get the seed-noise floor. Edit EXPERIMENTS for
# later Tier 1/2 batches.
#
# NOTE: PROJECT_PLAN.md §3.2 specifies the screening eval as 8 fixed test
# fonts. test_few_shot.py has no font-subsetting flag -- it always runs the
# full 34-font test split -- so this only cuts cost via --n_samples 3, not
# by font count. Fine for now (minutes either way); revisit if it isn't.
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
GPUS=(1 2 3)

# One entry per experiment: the name_exp to test (its highest-epoch
# checkpoint is picked automatically from experiments/<name>_main_model/checkpoints/).
EXPERIMENTS=(
  "seedfloor_1111_chn"
  "seedfloor_2222_chn"
  "seedfloor_3333_chn"
)

# Screening budget per PROJECT_PLAN.md §3.2: n_samples 3, not the n_samples 50
# confirmation eval.
COMMON_ARGS="--mode test --model_name main_model --language chn --max_seq_len 71 --batch_size 1 --n_samples 3 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29"

if [[ "$MODE" == "parallel" && ${#GPUS[@]} -lt ${#EXPERIMENTS[@]} ]]; then
  echo "parallel mode needs ${#EXPERIMENTS[@]} GPU ids in GPUS, only ${#GPUS[@]} given." >&2
  exit 1
fi

latest_ckpt() {
  ls "experiments/$1_main_model/checkpoints" | sort -t_ -k1,1nr | head -1
}

declare -A CKPTS
for name in "${EXPERIMENTS[@]}"; do
  CKPTS["$name"]="$(latest_ckpt "$name")"
done

pids=()
for i in "${!EXPERIMENTS[@]}"; do
  name="${EXPERIMENTS[$i]}"
  ckpt="${CKPTS[$name]}"
  gpu="${GPUS[$(( i % ${#GPUS[@]} ))]}"
  logfile="nohup_test_${name}.out"

  echo "[$name] ckpt=$ckpt GPU $gpu -> $logfile"
  CUDA_VISIBLE_DEVICES=$gpu nohup python test_few_shot.py $COMMON_ARGS --name_exp "$name" --name_ckpt "$ckpt" \
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

echo
echo "=== Evaluation ==="
l1s=()
for name in "${EXPERIMENTS[@]}"; do
  ckpt="${CKPTS[$name]}"
  echo "--- $name ($ckpt) ---"
  out="$(python eval_reconstruction_error.py --exp_dir "experiments/${name}_main_model" --name_ckpt "$ckpt")"
  echo "$out" | tail -5
  l1="$(echo "$out" | grep -oP "Reconstruction Error \(L1.*?: \K[0-9.]+")"
  l1s+=("$l1")
done

echo
echo "=== Seed-noise floor (screening metric spread) ==="
python3 - "${l1s[@]}" <<'EOF'
import sys
vals = [float(v) for v in sys.argv[1:]]
print(f"L1 per seed: {vals}")
print(f"spread (max-min): {max(vals)-min(vals):.4f}")
EOF
