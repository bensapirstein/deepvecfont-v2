# Tier 2 launch runbook

Day 2, 2026-08-04. Written on the Mac, run on the cluster. Companion to
`docs/tier1-launch.md`; the batch scripts are already edited, so this is mostly
paste-and-watch.

**Eight training runs, one seed each, ~1 h per run, three GPUs, three waves ≈ 3 h.**
Plus one eval-only job that costs no training GPU and answers a question the Tier 1
result raised.

## What Tier 1 concluded, and what it means for today

Every Tier 1 candidate — E7 at three weights, E9 at four sigmas, E10 at two dropouts,
E1 — landed within ±0.006 L1 of the seed-1111 baseline, against a re-measured
seed-noise floor of **0.0093**. Nothing cleared it. Renderability was 100% throughout,
so nothing is confounded there. `PROJECT_PLAN.md` §3.2 has the table.

Two consequences worth holding in mind while this batch runs:

1. **Do not expect Tier 2 to behave differently by default.** These effect sizes are
   at or below what the current screening setup can resolve. The batch is still worth
   running — E3 and E13 are paper-versus-code corrections that belong in the report
   regardless of sign, and E8 is the tier's real modelling idea — but plan for
   "recorded, not claimed" and be pleasantly surprised.
2. **The floor itself is the more valuable target.** `scripts/eval_noise.sh` splits it
   into decode noise and seed noise. Run it today, on a spare GPU, while the batch
   trains.

## 0. Sync and pre-flight

```bash
cd ~/deepvecfont-v2
git pull origin repro
conda activate dvf_v2

df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
nvidia-smi

python scripts/check_infra.py     # expect 127 passed, 0 failed
```

Section 7 of `check_infra.py` is new and covers every Tier 2 flag. It is the guard
against a declared-but-unread flag: a run that trains happily, writes the flag into
`opts.txt` and wandb, and returns the baseline number. That looks like a result, which
is worse than a crash. E13 makes this sharper than Tier 1 did, because it resizes
tensors in seven places and a missed one is either a load error or a head predicting
over 256 bins against a target built at 128.

Storage: eight runs at `--max_ckpt_keep 2` is roughly 8 × 3.6 GB ≈ 29 GB, on top of
Tier 1's dirs and `experiments/archive_pre_metricfix/`. Check the quota above before
launching; if it is tight, the archive is the thing to reap, not the Tier 1 results.

## 1. Rung 2 — two epochs each of the shape-changing candidates

E8 and E14 only touch a loss term and the optimizer, so they cannot crash on load.
E13 and E3 change tensor shapes and `state_dict` keys, so smoke-test those two before
committing three hours to them.

```bash
# E13, 256 bins: resizes arg_embed and args_fcn
CUDA_VISIBLE_DEVICES=1 python train.py --mode train --name_exp smoke_e13 \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --n_epochs 2 --freq_ckpt 1 --max_ckpt_keep 1 \
  --n_args_bins 256 --wandb False

# E3, 2 refinement layers: adds DecoderLayer modules
CUDA_VISIBLE_DEVICES=1 python train.py --mode train --name_exp smoke_e3 \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --n_epochs 2 --freq_ckpt 1 --max_ckpt_keep 1 \
  --n_layers_refine 2 --wandb False

# E8 at the largest sigma, cheapest of the three to sanity check
CUDA_VISIBLE_DEVICES=1 python train.py --mode train --name_exp smoke_e8 \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --n_epochs 2 --freq_ckpt 1 --max_ckpt_keep 1 \
  --args_label_smooth_sigma 2.0 --wandb False
```

What to look for:

- **All three**: a checkpoint written, and a row appended to
  `experiments/smoke_*/logs/checkpoint_metrics.csv` with a finite `val_metric`.
- **E13**: the run does not crash on the first backward pass. A shape mismatch surfaces
  immediately.
- **E8**: `val_svg_args` in the manifest is *higher* than the baseline's at the same
  epoch. That is expected and correct — cross-entropy against a smoothed target carries
  the target's own entropy, so the number is on a different scale. It is also exactly
  why E8 screens on the rendered metric and not on `val_metric` (§1.5).
- **E14**: check the printed `[E14] warmup_cosine: ...` line and that the `lr:` field in
  the training log rises over the first epochs rather than starting flat at 2e-4.

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --mode train --name_exp smoke_e14 \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --n_epochs 2 --freq_ckpt 1 --max_ckpt_keep 1 \
  --lr_schedule warmup_cosine --wandb False
grep -o "lr: [0-9.]*" nohup_smoke_e14.out 2>/dev/null | head    # or read the console
```

Then clean up: `rm -rf experiments/smoke_e{13,3,8,14}_main_model`.

## 2. The batch

`scripts/run_experiments.sh` is already staged with all eight. Confirm `GPUS` matches
what `nvidia-smi` says is free, then:

```bash
cd ~/deepvecfont-v2
sed -n '30,80p' scripts/run_experiments.sh    # eyeball GPUS and EXPERIMENTS
./scripts/run_experiments.sh parallel
```

Wave-based launching means the array can be longer than `GPUS` without editing between
waves. Eight runs on three GPUs is three waves; the last wave runs two.

The eight, and what each one is:

| Run | Flag | What it tests |
|---|---|---|
| `e8_ls05_chn` | `--args_label_smooth_sigma 0.5` | Ordinal label smoothing, narrow |
| `e8_ls10_chn` | `--args_label_smooth_sigma 1.0` | …σ = 1 bin |
| `e8_ls20_chn` | `--args_label_smooth_sigma 2.0` | …σ = 2 bins |
| `e13_bins256_chn` | `--n_args_bins 256` | Paper's Sec. 3.1 bin count |
| `e13_nopad_chn` | `--arg_embed_pad_idx False` | Bin 0 gets a trainable embedding |
| `e3_refine2_chn` | `--n_layers_refine 2` | Paper's Sec. 3.3 refinement depth |
| `e3_refine3_chn` | `--n_layers_refine 3` | …one deeper |
| `e14_wucos_chn` | `--lr_schedule warmup_cosine` | Warmup + cosine to the epoch budget |

`COMMON_ARGS` is byte-identical to the Tier 1 batch, which is what keeps every row
comparable to `seedfloor_1111_chn` on exactly one changed factor.

**A note on the E13 split.** §3.5 describes E13 as one change bundling 256 bins with
dropping `padding_idx`. They are run separately here because the protocol is one factor
at a time and the two are independent. Worth knowing while reading the result: the plan
says `padding_idx=0` gives bin 0 "a frozen zero embedding", and that is not quite what
the code does. `nn.Embedding(..., padding_idx=0)` zeroes row 0 at construction, but
`_init_embeddings` then runs `kaiming_normal_` over the whole weight and overwrites it.
Row 0 ends up frozen at a *random* vector, not at zero — `padding_idx` survives only as
a zero gradient. The defect is real either way, and the corrected description belongs in
the report.

**E13's second gate is deliberately skipped.** §3.5 gates E13 on the quantization
oracle, which has never been written or run. Running E13 now means the oracle explains
the result afterwards instead of licensing it beforehand. Write the oracle while the
batch trains (§2.3), and read the two together.

## 3. Alongside the batch — the noise-floor decomposition

This is the highest-value item today and it costs no training GPU. Run it on whichever
GPU is idle, or after the first wave.

```bash
./scripts/eval_noise.sh seedfloor_1111_chn 0
```

It re-screens one fixed checkpoint three times at `n_samples 3`, then once each at 10
and 20. Because the weights never change across those runs, the spread it reports is
entirely decode noise. Subtracting it from the 0.0093 seed floor gives the training-seed
contribution.

The script prints its own verdict, and the two branches lead different places:

- **Decode noise dominates** → raising `--n_samples` shrinks the floor for eval cost
  alone. Set the new screening budget, re-screen Tier 1 at it (13 eval runs, minutes
  each, no retraining), and the whole table becomes one comparison at a finer
  resolution. This is the good outcome.
- **Seed noise dominates** → more samples will not help, and resolving effects this size
  needs each candidate at multiple seeds, which triples the training matrix. That is a
  §8 decision and it probably means cutting Tier 3 to pay for it.

Either way the number goes into `PROJECT_PLAN.md` §3.2 as a dated **Measured** paragraph,
because the report's discussion section needs it: "which candidates fell inside the seed
noise" is a much weaker sentence than "here is what the noise was made of".

## 4. Screening eval

`scripts/test_experiments.sh` is staged to match, including `seedfloor_1111_chn` as a
re-screened baseline in the same session. That last part matters: `test_few_shot.py`
never calls `setup_seed`, so the baseline drifts 0.001–0.003 between sessions, and
comparing today's candidates against a number measured yesterday folds that drift into
every delta.

```bash
./scripts/test_experiments.sh parallel
```

`NOISE_FLOOR` in that script is set to 0.0093. Lower it only when `eval_noise.sh` has
landed, and record why in the plan when you do.

## 5. What to report back

For each of the eight, plus the re-screened baseline:

- run name, selected checkpoint, L1, s-IoU, renderability
- delta against the same-session baseline, and whether it clears `NOISE_FLOOR`

Plus, separately:

- the `eval_noise.sh` summary table and its verdict
- for E8, whether `val_svg_args` moved the way the smoothing predicts (a scale shift, not
  a quality signal)
- for E13 at 256 bins, whether renderability held — a finer grid decoded by the same
  best-of-N could plausibly produce more degenerate paths, and that would confound the L1

Then §3.2's rank-correlation check between `val_metric` and screening Error is finally
computable across enough runs to mean something. Tier 1 gave thirteen and Tier 2 adds
eight; exclude E8 and E13 from the correlation, since both change the cross-entropy and
their `val_metric` is on a different scale by construction.

## 6. If the batch finishes early

In priority order:

1. The quantization oracle (§2.3). Pure eval, and it is the last open Stage 1 item
   alongside SSIM.
2. SSIM in `eval_reconstruction_error.py`, then rescore. Stage 1 then closes and the
   reconstruction section can be written.
3. Whichever Tier 2 candidate leads, re-run at seeds 2222 and 3333 so it has a mean
   rather than a point. This is the depth pass, and it is worth more than starting
   Tier 3.
