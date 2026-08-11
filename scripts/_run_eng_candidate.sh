#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="$1"
GPU="$2"
EXTRA_ARGS="${3:-}"
RESUME_ARGS="${4:-}"
COMMON_ARGS="--mode train --model_name main_model --language eng --max_seq_len 51 --ref_nshot 4 --batch_size 32 --n_epochs 631 --freq_ckpt 20 --max_ckpt_keep 3 --wandb_project deepvecfont-v2-eng"
CUDA_VISIBLE_DEVICES=$GPU nohup python train.py $COMMON_ARGS --name_exp "$NAME" $EXTRA_ARGS $RESUME_ARGS \
  > "nohup_${NAME}.out" 2>&1 &
echo "$!"
