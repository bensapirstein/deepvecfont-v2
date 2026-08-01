# Project Commands

Commands I've adjusted/use for this project. Fillers in `[ ]` — fill in before running.

Notes:
- `[CKPT]` = checkpoint filename, found in `experiments/<name_exp>_<model_name>/checkpoints/`.
  List available ones with: `ls experiments/<name_exp>_main_model/checkpoints/`
- Pick a free GPU id with `nvidia-smi` before setting `CUDA_VISIBLE_DEVICES`.

## Training

### Chinese (chn)

```
CUDA_VISIBLE_DEVICES=1 nohup python train.py --mode train --name_exp dvf_base_exp_chn --model_name main_model --batch_size 32 --max_seq_len 71 --language chn --ref_nshot 8 --resume True --name_ckpt [CKPT] --freq_ckpt 5 --max_ckpt_keep 3 --n_epochs 201
```

Last used: `--name_ckpt 100_4040_valloss4.0329.ckpt`

### English (eng)

```
CUDA_VISIBLE_DEVICES=1 nohup python train.py --mode train --name_exp dvf_base_exp_eng --model_name main_model --batch_size 32 --max_seq_len 51 --language eng --ref_nshot 4 --resume True --name_ckpt [CKPT] --freq_ckpt 5 --max_ckpt_keep 3 --n_epochs 201
```

Latest checkpoint available: `600_192921_valloss2.0824.ckpt`

## Testing (few-shot)

Results land in `experiments/{name_exp}/results/{name_ckpt}/{font_id}/svgs_single` (candidates) and `svgs_merge` (selected, by IOU) — one subfolder per checkpoint, so reruns with a different `--name_ckpt` don't overwrite prior results.

### Chinese (chn)

```
CUDA_VISIBLE_DEVICES=2 python test_few_shot.py --mode test --name_exp dvf_base_exp_chn --language chn --max_seq_len 71 --model_name main_model --batch_size 1 --n_samples 50 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29 --name_ckpt [CKPT]
```

Runs so far: `100_4040_valloss4.0329.ckpt`, `125_5040_valloss3.8273.ckpt` (2026-08-01)

### English (eng)

```
CUDA_VISIBLE_DEVICES=2 python test_few_shot.py --mode test --name_exp dvf_base_exp_eng --language eng --max_seq_len 51 --model_name main_model --batch_size 1 --n_samples 20 --ref_nshot 4 --ref_char_ids 0,1,26,27 --name_ckpt [CKPT]
```

`ref_char_ids` picks which characters are used as references (default `0,1,26,27` = A, B, a, b); adjust to taste.
