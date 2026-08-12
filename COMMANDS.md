# Project Commands

Commands I've adjusted/use for this project. Fillers in `[ ]` — fill in before running.

Notes:
- `[CKPT]` = checkpoint filename, found in `experiments/<name_exp>_<model_name>/checkpoints/`.
  List available ones with: `ls experiments/<name_exp>_main_model/checkpoints/`. For the
  best one by `val_metric` (not just the latest), don't eyeball the filename — as of
  2026-08-03 checkpoint filenames are plain `{epoch}_{step}.ckpt` and the metric lives in
  `logs/checkpoint_metrics.csv` instead; run
  `python scripts/best_checkpoint.py experiments/<name_exp>_main_model`.
- Pick a free GPU id with `nvidia-smi` before setting `CUDA_VISIBLE_DEVICES`.
- Conda env: `dvf_v2` (`conda activate dvf_v2`). Needed for all commands below (has torch, cairosvg, etc.).
  In a fresh non-interactive shell (e.g. a plain `bash -c`, not a login shell), plain
  `conda activate dvf_v2` can fail with `command not found` because conda's shell function
  was never sourced. Fix: `source /opt/anaconda3/etc/profile.d/conda.sh && conda activate dvf_v2`.
- `data/` and `experiments/` are symlinks to `/data/bens/deepvecfont-v2/` (repointed 2026-08-03, previously `~/gpufs`). This is real separate storage, unlike the old target. Both must be gitignored via slash-less `data` / `experiments` lines — a trailing-slash pattern wouldn't match a symlink, and `.gitignore` currently still has `experiments/`, so fix that before the first `git add`.

## Storage

`/data/bens` quota: **200 GB**. Planning arithmetic in `PROJECT_PLAN.md` §7.3.

| Item | Size |
|---|---|
| Dataset, unzipped | ~30 GB |
| One checkpoint | 1.2 GB (roughly ⅓ model, ⅔ Adam optimizer state) |
| Confirmation eval, 34 fonts @ `n_samples 50` | ~1 GB |
| Screening eval, 8 fonts @ `n_samples 3` | ~15 MB |

About 33 training runs are planned, so at `--max_ckpt_keep 1` (best + latest = 2 files) that is ~79 GB of checkpoints. `2` costs ~119 GB and still fits; `3` costs ~158 GB and does not, once data and results are counted.

Check headroom before launching anything long:

```
df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
```

Housekeeping: delete checkpoints of any candidate that has been screened and dropped, and delete screening eval trees once the numbers are recorded. Neither is needed again.

## Batch sweeps (Tier 1 and later)

`scripts/run_experiments.sh` (training) and `scripts/test_experiments.sh` (screening eval)
centralize a batch of runs instead of hand-typing `train.py`/`test_few_shot.py` per
candidate. Both take `sequential` or `parallel`; `parallel` runs in waves of `len(GPUS)`,
so the `EXPERIMENTS` array can be longer than the number of GPUs available — no need to
edit the array between waves.

1. Edit `GPUS` and `EXPERIMENTS` at the top of each script (`GPUS` is a plain list of ids,
   e.g. `(1 2 3)`; check `nvidia-smi` first). `test_experiments.sh`'s `EXPERIMENTS` should
   list the same `name_exp` values that were just trained. **If a candidate's flag adds or
   removes model parameters (e.g. E1's `--enc_final_norm`), the same flag must be repeated
   in `test_experiments.sh`'s entry** (`"name_exp --enc_final_norm True"`) — `ModelMain(opts)`
   is rebuilt from `opts` at test time too, and checkpoint loading is strict, so a mismatch
   is a hard crash on `load_state_dict`, not a silently wrong answer. Flags that only affect
   the loss (`loss_w_aux`) or have no effect once `.eval()` is called (`dropout`) don't need
   repeating.

2. Pre-flight, every time before launching training (checks the flags actually wire into the
   model, catches the class of bug where a flag reaches `opts.txt` but never reaches
   `models/`):
   ```
   python scripts/check_infra.py
   ```
   Expect all checks to pass, 0 failed.

3. Launch training (each run is ~1h for 150 epochs on a 3090; a batch of N runs on G GPUs
   is ceil(N/G) waves):
   ```
   nohup ./scripts/run_experiments.sh parallel > run_experiments_batch.log 2>&1 &
   disown
   ```
   Track progress: `tail -f run_experiments_batch.log` (wave boundaries) or
   `tail -f nohup_<name_exp>.out` (one training run's stdout).

4. Once training finishes, launch screening (test + eval for every entry in
   `test_experiments.sh`'s `EXPERIMENTS`; a few minutes total, `--n_samples 3`):
   ```
   nohup ./scripts/test_experiments.sh parallel > test_experiments_batch.log 2>&1 &
   disown
   ```
   It prints a per-run table (checkpoint, L1, s-IoU, renderability, delta vs baseline, and
   whether that delta clears the 0.008 noise floor — `docs/tier1-launch.md` §5) at the end,
   after `tail -f test_experiments_batch.log` stops moving.

5. To check which checkpoint is "best" for one experiment without running the whole batch:
   ```
   python scripts/best_checkpoint.py experiments/<name_exp>_main_model
   ```
   Reads `logs/checkpoint_metrics.csv` (written by `train.py` at every checkpoint save,
   ranked by `val_metric`). Falls back to parsing the legacy `..._valloss{x}.ckpt` filename
   for experiment dirs trained before 2026-08-03, when that manifest didn't exist yet.

## Training

### Chinese (chn)

```
CUDA_VISIBLE_DEVICES=1 nohup python train.py --mode train --name_exp dvf_base_exp_chn --model_name main_model --batch_size 32 --max_seq_len 71 --language chn --ref_nshot 8 --resume True --name_ckpt [CKPT] --freq_ckpt 5 --max_ckpt_keep 3 --n_epochs 201
```

Last used: `--name_ckpt 100_4040_valloss4.0329.ckpt`

### English (eng)

```
CUDA_VISIBLE_DEVICES=1 nohup python train.py --mode train --name_exp dvf_base_exp_eng --model_name main_model --batch_size 32 --max_seq_len 51 --language eng --ref_nshot 4 --freq_ckpt 20 --max_ckpt_keep 3 --n_epochs 801 --wandb_project deepvecfont-v2-eng
```

Latest checkpoint available: `600_192921_valloss2.0824.ckpt`

**Every English `train.py` run takes `--wandb_project deepvecfont-v2-eng`** (2026-08-06),
so English stays in its own wandb project instead of mixed into the Chinese dashboard.
`test_few_shot.py` has no wandb calls, so this only applies to training commands.

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

## Evaluation (reconstruction error, Table 2 metric)

Run after `test_few_shot.py` so `experiments/{name_exp}/results/{name_ckpt}/*/svgs_merge/*.html` exist.

```
python eval_reconstruction_error.py --exp_dir experiments/dvf_base_exp_chn_main_model --name_ckpt [CKPT]
```

For English, swap `--exp_dir` to `experiments/dvf_base_exp_eng_main_model`.

Results so far:
- chn `125_5040_valloss3.8273.ckpt`: L1=0.1668, mean IOU=0.2550 (34 fonts, 1768 glyphs) (2026-08-01)

## Results table (`RESULTS.csv`)

One row per scored checkpoint -- name, seed, checkpoint, L1, s-IoU, renderability,
eval budget. Regenerate after any new `eval_reconstruction_error.py` run (it reads
every `experiments/*/results/*.csv` on disk, so it's always a full rebuild, not an
append):

```
python scripts/build_results_table.py
git add RESULTS.csv && git commit -m "update RESULTS.csv" && git push
```

New experiment directories are picked up automatically. If the script warns
`not in BATCH map, tagged 'unclassified'`, add the new `name_exp` to the `BATCH`
dict (and `N_SAMPLES`/`NOTES` if relevant) at the top of the script.

## Rendered-metric checkpoint selection (added 2026-08-12)

Full runbook: `docs/review-response.md`. The short version, for when you know what you
are doing and just need the flags.

Everything below is opt-in. `--render_val_freq 0` and `--ckpt_select val_metric` are the
defaults, so every command above this section reproduces exactly as it did before.

**Carve the held-out split first.** `train.py` validates on the *test* split, so without
this there is no set to select on that is not the set being scored.

```
python scripts/make_val_split.py --language chn --n_val 20          # dry run
python scripts/make_val_split.py --language chn --n_val 20 --apply
```

**Train with the rendered pass on.** `--render_val_freq` is in epochs and is snapped up
to a multiple of `--freq_ckpt`; a rendered score on an epoch with no checkpoint selects
nothing.

```
CUDA_VISIBLE_DEVICES=<gpu> python train.py --mode train --name_exp <name> \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --ref_char_ids 0,1,2,3,26,27,28,29 --batch_size 32 \
  --seed <seed> --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 10 \
  --render_val_freq 25 --render_val_fonts 0 --render_val_samples 1 \
  --ckpt_select val_render_l1
```

`--ref_char_ids` is required on a *training* command once `--render_val_freq` is on: the
rendered pass runs at `mode='test'`, which reads its references from that flag and
asserts there are `--ref_nshot` of them. English uses the default `0,1,26,27`, Chinese
needs the eight above. `train.py` refuses to start rather than failing at the first
checkpoint.

`--render_val_fonts 0` means all of them; `--render_val_samples 1` is one decode per
glyph, not the test protocol's best-of-50, which is unaffordable at every checkpoint.

**Select and score.**

```
python scripts/best_checkpoint.py experiments/<name>_main_model --criterion val_render_l1
```

Runs trained with `--render_val_freq 0` have no such column and this fails loudly rather
than falling back to `val_metric`. That is deliberate: a silent fallback would hide the
difference the flag exists to measure.

**Compare the two selection rules, no GPU.** Needs `--max_ckpt_keep` large enough that
the checkpoints are still on disk.

```
python scripts/selection_disagreement.py --csv_out selection_audit.csv \
  --floor 0.0093 experiments/rv_*_chn_main_model
```

**Rebuild the Chinese dataset at the paper's 10× augmentation.** `--n_aug 9` is the font
plus nine transformed copies. Run the val split *before* this, never after.

```
cd data_utils
python augment.py --language chn --split train --n_aug 9 --max_len 71 --n_chars 52 \
  --img_size 64 --output_path ../data/vecfont_dataset
python relax_rep.py --language chn --split train --max_len 71 --n_chars 52 \
  --output_path ../data/vecfont_dataset
python relax_rep.py --language chn --split val --max_len 71 --n_chars 52 \
  --output_path ../data/vecfont_dataset
```

`augment.py` writes `sequence.npy` and `rendered_64.npy`; the dataloader reads
`sequence_relaxed.npy` and `pts_aux.npy`. Skipping `relax_rep.py` trains on stale
sequences without any error.
