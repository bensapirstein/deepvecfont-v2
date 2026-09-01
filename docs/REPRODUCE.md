# Reproducing this project

Every number in the report comes from the commands below. They are the working
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

For the main results that is all. Section 8 carves a held-out validation split and raises
Chinese augmentation to the paper's 10×; neither is needed for any number in the report,
which uses the released data throughout.

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

`--seed` is ours; upstream has no seeding, which is why the margin of error in §6 could not
be measured before we added it.

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
`--max_fonts 34` on the evaluation: the English test split is ~1,386 fonts and a full sweep
is unaffordable. The subset is optimistic, and by how much is measured rather than assumed:
the released epoch-500 checkpoint scores 0.0645 on those 34 fonts and 0.0719 over 862, a
difference of 0.0074. Every English number here uses the 34-font subset, so the term cancels
in paired comparisons.

Three things to hold fixed, because each one moves the number more than most of the
architecture changes do:

- **`--gt_source`.** `raster` compares against the dataset's own rendering; `svg` re-renders
  the ground-truth outline through the same rasterizer as the prediction. Chinese moves
  **0.0455** between them, English 0.0074. Every number in this project is `raster`, so
  the term cancels in paired comparisons. A reported error is undefined until this is stated.
- **`--n_samples`.** Screening uses 3, confirmation uses 50, and the paper uses 10 on
  English and 50 on Chinese. Numbers from different values are not comparable.
- **`--name_ckpt`.** Pass it explicitly at a matched epoch when comparing runs. Letting the
  harness auto-select confounds a candidate's delta with its training epoch: E1's first
  reading came off checkpoints at epochs 100 and 125 against baselines at 150, and halved
  when the runs were re-read at a matched epoch. Every delta in the report is matched-epoch.

## 6. The margin of error

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

The spread depends on the sampling budget, so state which one a bar came from. At the
screening budget (`--n_samples 3`) Chinese comes back at **L1 0.0097, s-IoU 0.0315**, and
those are the bars the report commits to. At the confirmation budget (`--n_samples 50`) the
same three runs give L1 0.0093 and s-IoU 0.0236. English is **L1 0.0038, s-IoU 0.0129,
SSIM 0.0140**. Decomposed by `eval_noise.sh`, decode noise accounts for 0.0011 of the
Chinese figure and training-seed variance for the other 88%, so raising `--n_samples`
cannot narrow the bar — only more seeds can.

Read every candidate against this, never against a single baseline run. Measured on our
own table: 22 of 26 candidates beat a single-seed anchor, 6 of 26 beat the three-seed mean,
and the top three single-seed leaders replicated once in three. Report §6.3 has that
argument in full.

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
22-of-26 against 6-of-26 discrepancy in report §6.3 was found.

## 8. Optional: rendered-metric selection and the 10× rebuild

Two properties of the protocol above are worth being able to vary, so the repository ships
the machinery for both: checkpoints are selected on `val_metric`, a criterion that
correlates with the reported metric at ρ = 0.125, and the released Chinese data is built at
6× augmentation against the paper's 10×. **No number in the report uses either**, and every
default here is off, so §3 through §7 reproduce unchanged whether or not this section is
run. Turning them on takes three steps, in this order.

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
command in §3 through §7 reproduces exactly as it did before this section existed. The
`aug_rules` fix is the exception and is unconditional: it corrects a released bug rather
than adding an option, and at the released `--n_aug 5` it is byte-identical to upstream.

## 9. The report

The written report (`REPORT.md` / `REPORT.pdf`, its figures, and its build and
verification scripts) is not part of this repository. It is submitted separately, and its
own verifier re-derives every quoted number from a copy of `RESULTS.csv` and fails the
build on any drift, so a number that has drifted from this repository's results table
cannot reach the submitted PDF.

Nothing in the report is typed twice. If a number in `RESULTS.csv` here changes, the
report's copy needs to be refreshed and rebuilt to match.
