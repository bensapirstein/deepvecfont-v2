# Infrastructure upgrade — 2026-08-02

Branch `repro`. Four changes to `options.py` and `train.py`, plus a pre-flight check script.
Nothing here changes model behaviour: every new default reproduces what the code did before.

Plan context: [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md) §7.

---

## Change log

### 1. `--seed`

`options.py` gains `--seed` (int, default `1111`). `train_main_model` now calls
`setup_seed(opts.seed)` instead of the hardcoded `setup_seed(1111)`.

Default matches the old hardcoded value, so an unflagged run is byte-identical to before.
This exists for the three-seed baseline in `PROJECT_PLAN.md` §3.2, which establishes the
resolution limit every Stage 2 comparison is measured against.

```
--seed 1111   # the old behaviour
--seed 2222
--seed 3333
```

### 2. wandb alongside TensorboardX

`options.py` gains `--wandb` (bool, default `True`). When enabled, `train_main_model` calls

```python
wandb.init(project="deepvecfont-v2", name=opts.name_exp, config=vars(opts), tags=[opts.language])
```

then mirrors every `writer.add_scalar` with a `wandb.log`, and calls `wandb.finish()` at the end.
TensorboardX is untouched and still writes to `experiments/<name_exp>/logs/`.

**Metrics only.** `writer.add_image` is deliberately not mirrored, so no artifacts are uploaded.
A check in `scripts/check_infra.py` fails the build if `wandb.Image`, `wandb.save`,
`wandb.Artifact` or `log_artifact` ever appear in `train.py`.

**Tag names are identical to the TensorboardX tags** (`Loss/loss`, `Loss/svg_args`,
`VAL/loss_img_l1`, …), so the two dashboards read the same.

**One `wandb.log` per logging block, not per scalar.** `wandb.log` takes a dict, and issuing
twelve separate calls at the same step is a known way to lose keys: an explicit `step=` stays
open until a higher step arrives, and repeated calls at that step accumulate rather than commit.
Batching into one dict per block sidesteps the question. The train block and the validation
block can both fire at the same `batches_done`; they merge correctly, which
`scripts/check_infra.py` verifies against a live offline run.

Two additions beyond a strict mirror, both one line, both easy to revert:

- `lr` and `epoch` go into the train dict. TensorboardX never logged them, and they cost nothing.
- `CKPT/val_metric` is logged at each checkpoint. This is the quantity `prune_checkpoints`
  selects on, and `PROJECT_PLAN.md` §3.2 needs it to test whether `val_metric` predicts the
  rendered test metric. Without it, that correlation has to be scraped out of checkpoint filenames.

**Update, 2026-08-03: `val_metric` no longer lives in the filename, and it was fixed at the
same time.** `compute_val_loss` was silently dropping the refinement-decoder loss
(`svg_para`) out of `val_metric` even though that decoder produces what `test_few_shot.py`
scores — see `PROJECT_PLAN.md` §1.4's fixed note. Checkpoints are now named plain
`{epoch}_{step}.ckpt`, and every save appends its metrics to
`experiments/<name>/logs/checkpoint_metrics.csv` (`checkpoint_log.py`). `prune_checkpoints`
and `scripts/test_experiments.sh` (via `scripts/best_checkpoint.py`) both select the best
checkpoint from that manifest, so there is one selection code path instead of two things
that can drift apart. Pre-fix experiment dirs (no manifest) still resolve via the old
filename-embedded score.

**Degradation.** The import is wrapped in `try/except ImportError`. If wandb is missing, the run
prints a warning and continues on TensorboardX alone. If `--wandb False`, no wandb code runs.
Every `wandb.*` call site sits behind `if use_wandb`, verified statically by the check script.

**`str2bool`.** `--wandb` uses a `str2bool` converter rather than `type=bool`, because argparse's
`type=bool` maps any non-empty string to `True`, which would make `--wandb False` silently enable
the flag. The pre-existing boolean flags (`--tboard`, `--resume`, `--multi_gpu`) still have that
behaviour; `scripts/check_infra.py` prints a `[note]` for each. Migrating them is a separate,
optional change.

### 3. `--max_ckpt_keep` default `-1` → `1`

One character in `options.py`. `prune_checkpoints` already keeps the latest checkpoint on top of
the N best, so `1` means two files on disk per run: best and latest.

The training commands in `COMMANDS.md` pass `--max_ckpt_keep 3` explicitly, so they are unaffected.
`PROJECT_PLAN.md` §8.6 flags the one case where 1 is too few: scoring the same candidate at two
different epoch budgets needs both checkpoints retained.

### 4. Stage 2 experiment flags

Declared in `options.py`, **not wired into the model**. Defaults reproduce current behaviour exactly.

| Flag | Default | Wires to, when implemented |
|---|---|---|
| `--enc_noise_std_train` | `1.0` | `models/transformers.py:450`, `x = x + torch.randn_like(x)` (sigma is currently implicit) |
| `--enc_noise_std_test` | `1.0` | the same perturbation at val/test time |
| `--dropout` | `0.0` | `MultiHeadedAttention`, `PositionwiseFeedForward`, `attn_dropout`, `ff_dropout` |

Declaring them now means the baseline runs and the later candidate runs share one config schema in
the wandb UI, with no gap in the record where a flag existed for some runs and not others.
Rationale in `PROJECT_PLAN.md` §7.2; the experiments themselves are E9 and E10 in §3.4.

### 5. `eval_reconstruction_error.py` — results layout fix

The script globbed `<exp_dir>/results/*` and expected font directories, but `test_few_shot.py` now
writes `results/<name_ckpt>/<font_idx>/`. Against the current layout every font was skipped and the
script reported nothing scored. It also had no `--name_ckpt` although `COMMANDS.md` documents one.

Now it detects both layouts, prints which one it found on the first line, takes `--name_ckpt` to
disambiguate when several checkpoints are present, and exits with a clear error rather than
silently scoring zero fonts. Two additions from `PROJECT_PLAN.md` §2.2 came along for free because
the loop was already open: **renderability** (glyphs rendered over glyphs attempted, and fonts
scored over fonts found) and a **per-font CSV** for the paired Wilcoxon. SSIM is still missing; it
needs `scikit-image` in the env and is the next piece of work.

Tested on the Mac against synthetic fixtures covering the per-checkpoint layout, the flat layout,
an ambiguous multi-checkpoint tree, and a bad `--name_ckpt`. Not yet run against real output.

### 6. `scripts/bin_histogram.py` — new

Histograms the quantized coordinate arguments of the training data, numpy only, no torch or GPU.
Prints a verdict on whether the data sits on the 64-bin or the 128-bin grid, which decides whether
E13 stays in the Stage 2 experiment list. Rationale in `PROJECT_PLAN.md` §1.5 and §2.3.

Validated against synthetic fixtures built both ways: a dataset round-tripped through n=64 is
correctly called a COMB, one that was not is correctly called FILLED.

### 7. Also touched

- `.gitignore`: added `wandb/`, the local run directory wandb creates in the repo root.
- `scripts/check_infra.py`: new, see below.
- `STAGE2_EXPERIMENTS.md` moved to `archive/` with a superseded header; `PROJECT_PLAN.md` created
  at repo root covering both stages.
- `docs/cluster-session.md`: new, the day 1 runbook for the GPU session.

Still not touched, and flagged in `PROJECT_PLAN.md` §1.5: `.gitignore` has `experiments/` with a
trailing slash, which will not match the symlink on the cluster. One character, and
`docs/cluster-session.md` step 1 has the `sed` for it.

---

## Syncing the cluster

Cluster clone: `~/deepvecfont-v2`, conda env `dvf_v2`.

### Preferred: git

Keeps the `main..repro` diff intact, which is a graded report section.

```bash
# on the Mac  (tilde must sit outside the quotes to expand)
cd ~/"Projects/ACADEMIC/MLDS/Generative Models for Text and Images/deepvecfont-v2"
git add options.py train.py .gitignore PROJECT_PLAN.md docs/ scripts/ archive/
git commit -m "Add --seed and wandb logging, declare stage 2 flags, unify the project plan"
git push origin repro

# on the cluster
cd ~/deepvecfont-v2
git stash list                 # check for local work first
git checkout repro
git pull origin repro
```

If the cluster clone has uncommitted local edits, `git stash` them before pulling and inspect the
stash afterwards rather than dropping it. The cluster copy is the one that has actually run
training, so a local edit there may be a fix that never made it back.

### Fallback: rsync

For pushing work-in-progress without a commit. Excludes everything large or machine-specific.

```bash
rsync -avz --progress \
  --exclude '.git/' --exclude 'data' --exclude 'data.zip' --exclude 'experiments' \
  --exclude 'wandb/' --exclude '__pycache__/' --exclude '.DS_Store' --exclude 'nohup.out' \
  ~/"Projects/ACADEMIC/MLDS/Generative Models for Text and Images/deepvecfont-v2/" \
  [CLUSTER]:~/deepvecfont-v2/
```

Fill in `[CLUSTER]` with the ssh host, matching the `[ ]` filler convention in `COMMANDS.md`.
Note the trailing slash on the source path: without it rsync nests the directory.

`--exclude 'data'` and `--exclude 'experiments'` are slash-less on purpose. Both are symlinks on
the cluster, and a trailing-slash pattern does not match a symlink. This is the same subtlety as
the `.gitignore` issue in `PROJECT_PLAN.md` §1.5.

### wandb setup on the cluster

Not in the `dvf_v2` env yet.

```bash
conda activate dvf_v2
pip install wandb
wandb login          # paste the key from https://wandb.ai/authorize
```

If you would rather not log in on a shared node, `export WANDB_MODE=offline` writes runs to
`wandb/` locally and `wandb sync wandb/offline-run-*` uploads them later.

---

## The dry-run ladder

Three rungs, cheapest first. Do not skip a rung.

### Rung 1 — local, no GPU, no data

```bash
python scripts/check_infra.py
```

59 checks in a few seconds. Verifies flag names, types and defaults; verifies `--wandb False`
actually disables the flag; statically verifies the `train.py` wiring via AST (seeding, `wandb.init`
arguments, every call guarded, no artifact uploads, loss-item lists hoisted out of the `--tboard`
branch); and runs a live wandb init/log/finish in offline mode.

Status as of 2026-08-02: **59 passed, 0 failed.**

Worth re-running on the cluster after the sync, before rung 2, since it needs no GPU and catches a
bad merge in seconds.

What it cannot check: that `setup_seed` actually makes a run reproducible, that the mirrored
scalars carry sane values, and that a run appears in the wandb web UI. That is rung 2.

### Rung 2 — cluster, GPU, 2 epochs

```bash
conda activate dvf_v2
CUDA_VISIBLE_DEVICES=1 python train.py --n_epochs 2 --name_exp smoke_test \
  --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 \
  --freq_log 5 --freq_val 20 --freq_ckpt 1
```

The three frequency flags matter. At roughly 40 steps per epoch, two epochs is about 80 steps,
while the defaults are `freq_log 50` and `freq_val 500`. Left alone the run would log one scalar
point and zero validation points, which is a weak test of a logging change. `--max_seq_len 71`
matters too: it defaults to 51, which is the English value, and Chinese sequences would truncate.

Confirm, in order:

1. Console prints `Training on experiment smoke_test_main_model...` and a wandb run URL.
2. The wandb run's **Config** tab shows all of `seed`, `wandb`, `enc_noise_std_train`,
   `enc_noise_std_test`, `dropout`, `max_ckpt_keep`, and that the run is tagged `chn`.
3. The **Charts** tab has `Loss/loss` with several points, plus `Loss/svg_*` and `Loss/img_*`.
4. `VAL/*` series exist, including `VAL/val_metric`.
5. `CKPT/val_metric` has a point per checkpoint epoch.
6. No images or files were uploaded.
7. `experiments/smoke_test_main_model/checkpoints/` holds the expected checkpoints, and
   `experiments/smoke_test_main_model/logs/` still has TensorboardX event files.
8. `experiments/smoke_test_main_model/opts.txt` lists the new flags.

Then re-run once with `--wandb False` and confirm the training loop is unaffected and no wandb
run is created.

Delete `experiments/smoke_test_main_model/` afterwards.

### Rung 3 — the real thing

The three-seed baseline in `PROJECT_PLAN.md` §3.2, on the schedule in §4.

---

## Rollback

Every change is additive except the `--max_ckpt_keep` default. To revert the whole set:

```bash
git revert <commit>
```

To disable only the wandb path without touching code, pass `--wandb False`, or uninstall wandb and
let the `ImportError` guard take over.
