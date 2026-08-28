# Reproducing this project

Every number in `report/REPORT.md` comes from the commands below. They are the working
commands, not idealised ones: the flags that look redundant are the ones that fail
silently when they are missing, and each is noted where it appears.

Upstream's [README](https://github.com/yizhiwang96/deepvecfont-v2/blob/main/README.md)
covers installation, the dataset build and the custom-dataset pipeline. This file covers
what we added on top: the reconstruction, the improved model, the measuring instrument,
and the sweep.

**Hardware.** One RTX 3090 per run, never shared. A 150-epoch Chinese run is about one
GPU-hour; a 630-epoch English run is about 17.5. Everything below assumes `data/` and
`experiments/` at the repository root — symlink them somewhere with room if your home
directory is small.

---

## 1. Environment

```bash
conda create -n dvf_v2 python=3.9 && conda activate dvf_v2
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install tensorboardX einops timm scikit-image cairosvg pandas scipy
```

`cairosvg` is not optional here as it is upstream: the evaluation rasterizes generated
SVGs with it, and `render_val.py` uses it at training time.

Optional but recommended: `pip install wandb`. Training mirrors its metrics to Weights &
Biases alongside TensorboardX; without it the run still trains and still writes
`checkpoint_metrics.csv`, which is what checkpoint selection actually reads.

Before a long batch, `python scripts/check_infra.py` asserts the flags, the data and the
harness against each other. 224 of its checks pass with no GPU and no torch; the norm-factory
group needs both, and the script says so rather than counting a skip as a pass. It is faster
than discovering a typo six hours in.

## 2. Data

Download and unpack the authors' dataset as in the [README](../README.md#dataset), so that
`data/vecfont_dataset/{chn,eng}/{train,test}` exists.

For the main results that is all. Two rebuilds appear later: §6 raises Chinese augmentation
to the paper's 10×, and §5 carves a held-out validation split. Neither is needed to
reproduce §3 or §5.1 of the report.

## 3. The reconstruction (baseline)

Chinese. **`--max_seq_len 71` and `--ref_nshot 8` are required**; the defaults are the
English values and Chinese silently truncates without them.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode train \
  --name_exp chn_baseline_1111 --model_name main_model \
  --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 \
  --seed 1111 --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 3
```

English. 630 epochs, chosen by measuring convergence on three seeds and freezing the
budget before any candidate was looked at.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode train \
  --name_exp eng_baseline_1111 --model_name main_model \
  --language eng --max_seq_len 51 --ref_nshot 4 --batch_size 32 \
  --seed 1111 --n_epochs 631 --freq_ckpt 20 --max_ckpt_keep 3 \
  --wandb_project deepvecfont-v2-eng
```

`--seed` is ours; upstream has no seeding, which is why the seed floor in §5 could not be
measured before we added it.

## 4. The improved model

The improved configuration is one flag. `--enc_noise_std_train` is the σ of the Gaussian
perturbation applied to the sequence-encoder output during training; upstream hardcodes
1.0, and E9 halves it.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode train \
  --name_exp chn_e9_sigma050_1111 --model_name main_model \
  --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 \
  --enc_noise_std_train 0.5 \
  --seed 1111 --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 3
```

Swap the English block from §3 the same way for the English arm.

All 21 candidates work like this: a flag on the same entry point, defaulting to the
released behaviour. `scripts/check_infra.py` asserts that every candidate's default
reproduces the baseline exactly, so a candidate that appears to help because it also
changed something else cannot pass unnoticed. `python options.py --help` lists them, each
tagged with its experiment ID; the report's §4.2 says what each one tests.

Two that need care at test time:

- **E1 (`--enc_final_norm True`)** adds LayerNorm parameters. The flag must be repeated on
  `test_few_shot.py` or `load_state_dict` hard-crashes on the extra keys.
- **E7, E8, E13** change terms inside the loss, so their `val_metric` is not comparable
  with any other run's. They are screened on the rendered metric only.

## 5. Testing and evaluation

Decode, then score. Decoding writes candidates to
`experiments/<name>_main_model/results/<ckpt>/<font_id>/svgs_single` and the IoU-selected
one to `svgs_merge`.

```bash
CUDA_VISIBLE_DEVICES=0 python test_few_shot.py --mode test \
  --name_exp chn_baseline_1111 --model_name main_model \
  --language chn --max_seq_len 71 --ref_nshot 8 \
  --ref_char_ids 0,1,2,3,26,27,28,29 \
  --batch_size 1 --n_samples 50 --name_ckpt <ckpt>

python eval_reconstruction_error.py \
  --exp_dir experiments/chn_baseline_1111_main_model \
  --name_ckpt <ckpt> --gt_source raster --csv_out eval_chn_baseline_1111.csv
```

English uses `--language eng --max_seq_len 51 --ref_nshot 4 --ref_char_ids 0,1,26,27`, and
`--max_fonts 34` on the evaluation: the English test split is ~1,386 fonts and a full
sweep is unaffordable. That 34-font subset is optimistic by a measured 0.0074, reported in
§3 rather than absorbed.

Three things to hold fixed, because each one moves the number more than most of the
architecture changes do:

- **`--gt_source`.** `raster` compares against the dataset's own rendering; `svg` re-renders
  the ground-truth outline through the same rasterizer as the prediction. Chinese moves
  **0.0455** between them, English 0.0074. Every number in this project is `raster`, so
  the term cancels in paired comparisons. A reported error is undefined until this is stated.
- **`--n_samples`.** Screening uses 3, confirmation uses 50, and the paper uses 10 on
  English and 50 on Chinese. Numbers from different values are not comparable.
- **`--name_ckpt`.** Pass it explicitly at a matched epoch when comparing runs. Letting the
  harness auto-select confounds a candidate's delta with its training epoch — which is
  exactly what happened to E1 and is written up in §5.5.

## 6. The seed-noise floor

This is the instrument the whole report rests on, and it is cheap: train the unmodified
baseline three times, change nothing but `--seed`, and measure the spread.

```bash
for s in 1111 2222 3333; do
  CUDA_VISIBLE_DEVICES=0 python train.py --mode train \
    --name_exp seedfloor_${s}_chn --model_name main_model \
    --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 \
    --seed $s --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 3
done
bash scripts/eval_noise.sh          # split that spread into decode noise vs seed variance
```

Chinese comes back at **L1 0.0093, s-IoU 0.0315**; English at **L1 0.0038, s-IoU 0.0129,
SSIM 0.0140**. Decomposed, the decode noise is 0.0011 of the Chinese figure and training
seed variance is the other 88%, so raising `--n_samples` cannot narrow the bar — only more
seeds can.

Read every candidate against this, never against a single baseline run. Measured on our
own table: 22 of 26 candidates beat a single-seed anchor, 6 of 26 beat the three-seed mean,
and the top three single-seed leaders replicated once in three.

## 7. The sweep

Batch training and testing go through two drivers rather than hand-rolled loops. Edit
`GPUS`, then the `EXPERIMENTS` array, then `COMMON_ARGS` at the top of each.

```bash
./scripts/run_experiments.sh parallel     # one GPU per experiment, in waves
./scripts/test_experiments.sh parallel    # decode + score the batch
```

Analysis, all CPU-only:

```bash
python scripts/build_results_table.py                  # rebuild RESULTS.csv from disk
python scripts/recompute_deltas.py                     # every delta, regenerated not typed
python scripts/paired_wilcoxon.py                      # per-seed paired test
python scripts/quantization_oracle.py                  # what the 128-bin grid costs (§6.5)
python scripts/val_metric_correlation.py               # val_metric vs the rendered metric
```

`build_results_table.py` is a full rebuild, not an append: it reads every
`experiments/*/results/*.csv` on disk. If it warns that a run is `not in BATCH map`, add
the `name_exp` to the `BATCH` dict at the top.

Deltas are never typed by hand. `recompute_deltas.py` regenerates all of them from
`RESULTS.csv` against both the single-seed anchor and the three-seed mean, which is how the
discrepancy in §5.2 was found.

## 8. The re-evaluation in §5.6

An external review of an earlier draft challenged two things: checkpoints selected on a
criterion that correlates with the reported metric at ρ = 0.125, and a reproduction left at
6× Chinese augmentation against the paper's 10×. Section 5.6 answers both. Reproducing it
takes three steps, in this order.

**Carve a held-out validation split first.** `train.py` as released validates on the *test*
split, so without this there is no set to select on that is not the set being scored.

```bash
python scripts/make_val_split.py --language chn --n_val 20            # dry run
python scripts/make_val_split.py --language chn --n_val 20 --apply
```

The resulting splits are committed in `data_splits/`. Re-running without `--apply` checks
your data against them and exits non-zero if they disagree.

**Rebuild Chinese at 10×.** `--n_aug 9` is the font plus nine transformed copies. Never run
this before the val split, or held-out fonts leak in as augmented copies.

```bash
cd data_utils
python augment.py   --language chn --split train --n_aug 9 --max_len 71 \
  --n_chars 52 --img_size 64 --output_path ../data/vecfont_dataset
python relax_rep.py --language chn --split train --max_len 71 --n_chars 52 \
  --output_path ../data/vecfont_dataset
python relax_rep.py --language chn --split val   --max_len 71 --n_chars 52 \
  --output_path ../data/vecfont_dataset
```

`relax_rep.py` is not optional. `augment.py` writes `sequence.npy`, the dataloader reads
`sequence_relaxed.npy`, and skipping the second step trains on stale sequences with no
error anywhere.

The released `data_utils/augment.py` ends its five rules in a bare `else`, so any
`aug_idx >= 4` silently returns `rotate(-5)` — `--n_aug 9` would have written five
identical copies and reported itself as 10×. Our version has nine distinct rules, indices
0–4 byte-identical to the released ones, and raises on an unknown index.

**Train with rendered-metric selection.** `--render_val_freq` is in epochs and is snapped
up to a multiple of `--freq_ckpt`; a rendered score on an epoch with no checkpoint selects
nothing.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --mode train \
  --name_exp rv_chn_baseline_1111 --model_name main_model \
  --language chn --max_seq_len 71 --ref_nshot 8 \
  --ref_char_ids 0,1,2,3,26,27,28,29 --batch_size 32 \
  --seed 1111 --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 10 \
  --render_val_freq 25 --render_val_fonts 0 --render_val_samples 1 \
  --ckpt_select val_render_l1

python scripts/best_checkpoint.py experiments/rv_chn_baseline_1111_main_model \
  --criterion val_render_l1
```

`--ref_char_ids` becomes required on a *training* command once `--render_val_freq` is on,
because the rendered pass runs at `mode='test'` and asserts it has `--ref_nshot` references.
`train.py` refuses to start rather than failing at the first checkpoint.

`best_checkpoint.py` fails loudly on a run trained without the rendered pass rather than
falling back to `val_metric`. A silent fallback would hide the difference the flag exists
to measure.

Comparing the two selection rules needs no GPU:

```bash
python scripts/selection_disagreement.py --csv_out selection_audit.csv \
  --floor 0.0093 experiments/rv_*_chn_main_model
```

Both defaults are off — `--render_val_freq 0` and `--ckpt_select val_metric` — so every
command in §3 through §7 reproduces exactly as it did before this section existed.

## 9. Rebuilding the report

```bash
bash report/build.sh
```

Regenerates all nine figures from `RESULTS.csv`, re-derives every number quoted in
`REPORT.md` and checks it (81 assertions, non-zero exit on any mismatch), then runs pandoc.
The verifier gates the build, so a number that has drifted from the results table stops the
PDF instead of reaching the submission. Needs `pandoc` and `xelatex` on PATH; the figure
steps need only `numpy` and `matplotlib`.

Nothing in the report is typed twice. If a number changes, it changes in `RESULTS.csv` and
everything downstream follows.
