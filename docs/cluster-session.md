# Day 1 cluster session

Paste this into the Claude session running on the cluster, or work through it by hand.
Everything here needs the GPU, the dataset, or both, which is why it did not happen on the Mac.

Context: `../PROJECT_PLAN.md`. Infrastructure change log: `infra-upgrade.md`.
Env: `conda activate dvf_v2`. Repo: `~/deepvecfont-v2`. Branch: `repro`.

Work top to bottom. Steps 1 to 4 gate everything else. Step 5 is the long pole and wants to be
launched as early in the session as possible, because it runs unattended.

---

## 1. Sync and install

```bash
cd ~/deepvecfont-v2
git stash list                 # check for local work before pulling
git checkout repro
git pull origin repro

conda activate dvf_v2
pip install wandb
wandb login                    # key from https://wandb.ai/authorize
```

Also fix the one-character `.gitignore` issue while you are here. `data` and `experiments` are
symlinks to `/data/bens/deepvecfont-v2`, and the current pattern has a trailing slash, which does
not match symlinks:

```bash
sed -i 's|^experiments/$|experiments|' .gitignore
git diff --stat .gitignore     # expect 1 file changed, 1 insertion, 1 deletion
```

Check the storage headroom before launching anything long. Quota is 200 GB, the dataset is ~30 GB,
and each checkpoint is 1.2 GB. `PROJECT_PLAN.md` §7.3 has the arithmetic.

```bash
df -h /data/bens
du -sh /data/bens/deepvecfont-v2/* 2>/dev/null
```

## 2. Pre-flight, no GPU

```bash
python scripts/check_infra.py
```

Expect `59 passed, 0 failed`. If anything fails here, stop and fix it before burning GPU time.
This runs in seconds and catches a bad merge.

## 3. GPU smoke test, 2 epochs

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --n_epochs 2 --name_exp smoke_test \
  --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 \
  --freq_log 5 --freq_val 20 --freq_ckpt 1
```

The frequency flags are not optional. Two epochs is roughly 80 steps, and the defaults
(`freq_log 50`, `freq_val 500`) would log one point and zero validation points, which tests nothing.
`--max_seq_len 71` matters too: it defaults to 51, the English value, and Chinese would truncate.

Confirm in the wandb run: config carries `seed`, `wandb`, `enc_noise_std_train`,
`enc_noise_std_test`, `dropout`, `max_ckpt_keep`; the run is tagged `chn`; `Loss/*` has several
points; `VAL/*` exists including `VAL/val_metric`; `CKPT/val_metric` has a point per checkpoint
epoch; and **no files or images were uploaded**.

Then confirm the off switch works:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --n_epochs 1 --name_exp smoke_test_nowandb \
  --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 --freq_log 5 --wandb False
```

No wandb run should be created. Then `rm -rf experiments/smoke_test_main_model experiments/smoke_test_nowandb_main_model`.

## 4. The bin histogram — do this before anything else that costs time

This decides whether E13 stays in the experiment list, and it takes minutes.

```bash
python scripts/bin_histogram.py --language chn --plot
python scripts/bin_histogram.py --language eng
```

The script prints a VERDICT. Read it:

- **COMB** means the training sequences were round-tripped through the 64-bin grid during
  preprocessing (`data_utils/relax_rep.py:27` writes back through a numpy view). The 128-bin head
  is modelling data with only 64 bins of information. **Drop E13** from the Tier 2 list, use the
  64-bin row of the oracle table in `PROJECT_PLAN.md` §2.3, and note it in the report as a
  reconstruction finding. It is a good finding.
- **FILLED** means the grids never met, the model genuinely runs at 128 bins, and E13 stays,
  gated only on the oracle.

Record the verdict in the vault at `Alef Alif/02-Theory-Research/DeepVecFont-v2/Experiments/Experiment Tracker.md`.

## 5. Launch the three-seed baseline

The highest-value thing in the whole plan, and it runs unattended. Get it going before the
smaller tasks. Three seeds, identical otherwise, launched via `scripts/run_experiments.sh`
(also the template for later Tier 1/2 sweeps — edit its `GPUS`/`EXPERIMENTS` arrays per batch).

Check `nvidia-smi` first and set the GPU ids in that script by hand. With three free GPUs:

```bash
./scripts/run_experiments.sh parallel
```

With only one GPU free (the common case), set `GPUS=(0)` (or whichever id is free) in the
script and run sequentially instead:

```bash
./scripts/run_experiments.sh sequential
```

Either way each run still gets exactly one GPU — never more than one at a time per run.

`--max_ckpt_keep 2` rather than the new default of 1, because §3.3 wants one candidate scored at
two epoch budgets to validate the 60-epoch screening assumption, which needs both checkpoints.

The spread of the screening metric across these three runs is the resolution limit. Any Stage 2
candidate whose improvement falls inside it is not a result.

## 6. Time five epochs, size the matrix

While the baseline runs, read the wall-clock off the first seed's log and extrapolate to 150 epochs.

| Cost of one 150-epoch Chinese run | Matrix |
|---|---|
| ≤ 3 h | Everything in §3.4 to §3.6 |
| 3–8 h | Tiers 1 and 2, single sweep points |
| > 8 h | Tier 1 only, 60-epoch screening budget |

Write the answer into `PROJECT_PLAN.md` §3.3 and into the vault tracker.

## 7. Settle the eval-script question

`eval_reconstruction_error.py` was rewritten on the Mac to handle both results layouts, take
`--name_ckpt`, and report renderability plus a per-font CSV. It was tested against synthetic
fixtures covering both layouts, but never against real output. Run it on what exists:

```bash
python eval_reconstruction_error.py --exp_dir experiments/dvf_base_exp_chn_main_model
```

It prints the detected layout on the first line. That line answers the open question in
`PROJECT_PLAN.md` §1.5:

- `Layout: flat` means the 0.1668 came from a pre-per-checkpoint tree with no record of which
  checkpoint produced it. Re-run `test_few_shot.py` for `125_5040_valloss3.8273.ckpt` to attribute it.
- `Layout: per-checkpoint` means pass `--name_ckpt` and the number is attributable as-is.

Either way you now get renderability and a per-font CSV, which is most of §2.2. **SSIM is still
missing** and needs `scikit-image` in the env; that is the next piece of work, not a day 1 item.

---

## What to report back

1. Pre-flight and smoke test: pass or fail.
2. **The bin histogram verdict.** Comb or filled. This is the one that changes the plan.
3. Detected results layout, and the re-scored Chinese numbers with renderability.
4. Measured cost of a 150-epoch run, and therefore the matrix size.
5. That the three seed-floor runs are alive.

## Do not

- Touch `main`. All work is on `repro`.
- Wire `--enc_noise_std_*` or `--dropout` into the model tonight. That is day 3 work, and doing it
  before the seed floor exists means the first measurements have nothing to be compared against.
- Chase the gap between 0.1668 and 0.080. It sits on DeepSVG's published Chinese number (0.167),
  the assignment permits reporting a failed reproduction, and Stage 2 is measured against your own
  baseline. See `PROJECT_PLAN.md` §2.4.
