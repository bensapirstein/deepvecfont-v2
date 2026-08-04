# Tier 1 launch runbook

Day 1 evening, 2026-08-03. Paste into the Claude session on the cluster, or run by hand.

Context: the day-1 blocking list in `PROJECT_PLAN.md` §9 is closed. The seed-noise floor
is measured (L1 spread **0.0077**, §3.2), one 150-epoch Chinese run costs **~1.0 h** on a
3090 (§3.3), and the Tier 1 code diff landed on the Mac today. This runbook launches the
Tier 1 sweep, which §4 schedules for days 3 to 5. Running it tonight buys back the report
time §4 flags as tight.

**The rule that governs everything below:** a candidate must beat the baseline by more
than **0.008 L1** to be a result. Anything smaller gets recorded, not claimed.

---

## 0. Before anything

```bash
cd ~/deepvecfont-v2
git pull origin repro
df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
nvidia-smi
```

Storage: each run at `--max_ckpt_keep 2` costs roughly what a seed-floor run cost. Check
there is room for six more before launching both batches. Quota is 200 GB.

Then re-run rung 1. It now has a section 5 covering the new wiring, and it is the thing
that catches a flag that was declared but never read:

```bash
conda activate dvf_v2
python scripts/check_infra.py
```

Expect **82 passed, 0 failed** (81 plus the wandb round-trip, which skips on the Mac and
runs on the cluster). That count went up from 69 on 2026-08-03: a section 6 was added
covering the `val_metric` checkpoint-selection fix (see §1.4 in `PROJECT_PLAN.md` — it was
missing the refinement-decoder loss). Do not launch on a failure in section 5 or 6.

---

## 1. Batch A — E7, no code change, launch first

`scripts/run_experiments.sh` is already staged for this batch. Eq. 11 weights `L_bézier`
at 1.0; `options.py` defaults `loss_w_aux` to 0.01, a factor of 100 below (§1.4, §3.4).

The 0.01 point of the sweep is already measured: it **is** `seedfloor_1111_chn`, same
seed, same `COMMON_ARGS`, default `loss_w_aux`. Do not re-run it.

```bash
# check GPUS=(1 2 3) matches what nvidia-smi says is free, then:
./scripts/run_experiments.sh parallel
```

Three runs, one GPU each, ~1 h wall clock.

If the GPU ids differ, edit `GPUS` at the top of the script rather than passing them some
other way, so the batch stays reproducible from the file.

---

## 2. Rung 2 for the new code, before Batch B

Batch A exercises no new code, which is exactly why it goes first. Batch B does. Two
epochs on one GPU, three configurations, to prove the wiring runs before an hour of
compute rides on it:

```bash
# E9 at a non-default sigma
CUDA_VISIBLE_DEVICES=0 python train.py --mode train --name_exp smoke_e9 \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --n_epochs 2 --freq_ckpt 1 --max_ckpt_keep 1 --wandb False \
  --enc_noise_std_train 0.25

# E1, which adds parameters
CUDA_VISIBLE_DEVICES=0 python train.py --mode train --name_exp smoke_e1 \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --n_epochs 2 --freq_ckpt 1 --max_ckpt_keep 1 --wandb False \
  --enc_final_norm True

# E10
CUDA_VISIBLE_DEVICES=0 python train.py --mode train --name_exp smoke_e10 \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --n_epochs 2 --freq_ckpt 1 --max_ckpt_keep 1 --wandb False \
  --dropout 0.1
```

What to look for, in order of how badly it bites:

- All three complete two epochs without a shape error. E1 changes the parameter count, so
  it is the one that can fail at construction.
- `experiments/smoke_e1_main_model/opts.txt` records `enc_final_norm: True`. If a flag
  does not reach `opts.txt` it did not reach the model either.
- The loss is finite and in the same order of magnitude as the baseline's first epochs. A
  much larger loss on `smoke_e1` is expected and fine, since the terminal LayerNorm
  changes the scale of what the decoder sees.

Then delete them: `rm -rf experiments/smoke_e{1,9,10}_main_model`.

---

## 3. Batch B — E9, E1, E10

Edit `scripts/run_experiments.sh`, replacing the `EXPERIMENTS` array with the block below,
and leave `COMMON_ARGS` untouched. Identical `COMMON_ARGS` across batches is what makes
every row comparable to `seedfloor_1111_chn`, so it is not a detail.

```bash
EXPERIMENTS=(
  # E9, encoder noise. sigma=1.0 is the baseline and is already measured as
  # seedfloor_1111_chn, so it is not repeated here. Test sigma stays at 1.0
  # throughout: it is the only source of stochasticity at inference, so setting
  # it to 0 makes all n_samples candidates identical. It gets its own sweep
  # later, on the winning train sigma.
  "e9_sigma000_chn --seed 1111 --enc_noise_std_train 0.0"
  "e9_sigma010_chn --seed 1111 --enc_noise_std_train 0.1"
  "e9_sigma025_chn --seed 1111 --enc_noise_std_train 0.25"
  "e9_sigma050_chn --seed 1111 --enc_noise_std_train 0.5"

  # E1, terminal LayerNorm on both encoder stacks. Run alone and with the E9
  # winner: normalizing the residual stream is what makes sigma a meaningful
  # quantity rather than an arbitrary one, so the interaction is the point.
  "e1_norm_chn --seed 1111 --enc_final_norm True"

  # E10, dropout. 0.0 is the baseline, again already measured.
  "e10_drop01_chn --seed 1111 --dropout 0.1"
  "e10_drop02_chn --seed 1111 --dropout 0.2"
)
```

Seven runs. With three GPUs that is three waves, so **sequential** on three GPUs is wrong
here; `parallel` mode requires at least as many ids as experiments. Two options:

```bash
# option 1: three at a time, editing the array between waves
./scripts/run_experiments.sh parallel      # with 3 entries in EXPERIMENTS

# option 2: split by GPU, one long sequential chain each, launched in three shells
./scripts/run_experiments.sh sequential    # GPUS=(1), 3 entries
```

Option 2 is less babysitting: roughly 3 h per chain, all seven done overnight. Whichever
you pick, never put two runs on the same GPU concurrently.

---

## 4. Screening eval

After each batch. `test_experiments.sh` auto-selects the best-val-loss checkpoint off the
filename, which is what the seed floor was measured on, so leave that alone.

```bash
# edit the EXPERIMENTS array in test_experiments.sh to match the batch, then:
./scripts/test_experiments.sh sequential
```

Screening budget is `--n_samples 3`, all 34 fonts. Note the caveat carried in that
script: `test_few_shot.py` has no font-subsetting flag, so the 8-font screening subset in
§3.2 is not actually wired up and screening saves cost only through `n_samples`. Say
"screening" next to every number this produces.

**One exception to the auto-selection.** `val_metric` is comparable across E7, E9, E1 and
E10, because none of them changes the cross-entropy. It will stop being comparable at E8
and E13 in Tier 2. Nothing to do tonight, but do not carry the habit forward.

---

## 5. What to report back

Per run: the experiment name, the checkpoint the eval picked, L1, s-IoU, and the
renderability count. Then, for each candidate, L1 minus the `seedfloor_1111_chn` L1 of
**0.1734**, and whether that delta clears **0.008**.

That last column is the whole experiment. A candidate at −0.004 is not a win, it is a run
that landed inside the noise, and the report says so.

These go into `PROJECT_PLAN.md` §5 and the vault's `Run Log.md` and
`Experiment Tracker.md`, repo first.

---

## 6. Then, on day 2

Stage 1 is not closed yet. Still open, and all of it is Mac work except the oracle:

- SSIM in `eval_reconstruction_error.py` (§2.2). The only missing piece of the metric
  triple; renderability and the per-font CSV already landed.
- The quantization oracle (§2.3), on the 128-bin grid, now that §1.5 has established the
  training data is not on the 64-bin grid. No script exists for this yet.
- Rescore the epoch-100 and epoch-125 Chinese results with the extended script.
- Answer §2.4 and write the reconstruction section while it is fresh.
