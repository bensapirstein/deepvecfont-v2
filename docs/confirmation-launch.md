# Confirmation session — σ_test ladder, then E9's confirmation eval

Day 3, 2026-08-05. Paste into the Claude session on the cluster.
Companion to `docs/tier1-launch.md`, `tier2-launch.md`, `tier3-launch.md`.

**This session trains nothing.** Every step is CPU or eval-only, which is why it
comes before any further training batch: it is the cheapest work left, and two of
its four steps can invalidate or re-point the expensive work that would follow.

Plan sections: §3.6 (Tier 3 result), §3.7 (confirmation), §8 items 8 and 9.

---

## Why this session, and not another training batch

Tier 3 returned fifteen runs and settled Stage 2's headline: **E9 `enc_noise_std_train=0.5`
is the only candidate of twenty-one whose paired per-seed difference has the same sign at
all three seeds**, and (see §3.6, added today) it does so on L1 *and* s-IoU independently.
E1 and E13 both flipped under replication. §3.6's own reading rule says a same-sign
candidate "earns the confirmation eval even if the means overlap." So E9 has earned it,
and the confirmation eval is what §5's table and the report's section 5 are waiting on.

Two things must happen before that eval, in this order:

1. **The σ_test ladder has never run.** `test_few_shot.py` applies `--enc_noise_std_test`
   regardless of what σ_train was, and every number in the project was taken at the
   released σ_test = 1.0 without anyone checking that 1.0 is a sensible point on that
   curve. Running the confirmation eval first and the ladder afterwards risks having to
   run the confirmation eval twice.
2. **`val_metric_correlation.py` has never run**, and it gates the seven staged E14-deep
   rows. It needs no GPU at all, so it costs nothing to answer here rather than later.

---

## Pre-committed decision rules

Written down before any number is seen, so the rule cannot be chosen to suit the result.
This is the same discipline `docs/english-arm.md` applies to the epoch budget.

### Rule 1 — what a σ_test shift licenses (§8 item 9, resolved 2026-08-05)

**Re-screen the finalists only; caveat the rest.**

Let σ\* be the L1-minimizing point of a ladder, and compare it to that ladder's own σ = 1.0
row. The measured decode noise at `n_samples 3` is **0.0011** (§3.2), so a difference
smaller than that is not a difference.

| What the two ladders show | What it means | What to do |
|---|---|---|
| Neither ladder improves on σ=1.0 by > 0.0011 | The released default is fine | One sentence in §6. Confirmation runs at σ_test = 1.0 |
| **Both** ladders improve, at the same σ\* (or adjacent grid points) | A property of the **eval procedure**, not of any candidate | Confirmation eval runs at σ\*. Tiers 1–3 stay as measured at 1.0, with a stated caveat in §3.2. Only the six finalist rows are re-measured |
| Only **one** ladder improves | A σ_train × σ_test **interaction**, not a free win | Confirmation runs at σ_test = **1.0** regardless. Tuning the eval on the candidate's own ladder would confirm E9 on a footing it was never screened on. Report the interaction in §6 as its own finding |

The third row is the trap. `sigma_test_sweep.sh` is run on `seedfloor_1111_chn` as well as
on `e9_sigma050_chn` specifically so that case is detectable.

### Rule 2 — what the confirmation eval can conclude

The confirmation eval is `n_samples 50`, all 34 fonts, per-font CSV. It is **not**
comparable to any screening number in §3.2, §3.5 or §3.6 — different budget, and the
`n_samples` ladder in §3.2 showed a systematic best-of-N drop (0.1722 → 0.1691 → 0.1678 at
n = 3 → 10 → 20) far larger than any candidate delta. Baseline and candidate are both
re-scored at 50 in this session; nothing is compared across the budget line.

The claim E9 is allowed to support, if it holds: *a paired per-font test favours E9 at all
three seeds at confirmation budget.* Not "E9 beats the noise floor" — the 0.0093 floor is
the baseline's own seed spread and is the quantity the paired design replaces.

---

## Step 0 — sync and check

```bash
cd ~/deepvecfont-v2
git pull origin repro
python scripts/check_infra.py          # expect 127 passed, 0 failed
df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
```

Budget note: a confirmation eval writes ~1 GB of SVGs and PNGs (§7.3). Six of them is
~6 GB. Check there is room before step 4, and delete the screening eval trees of any
candidate already recorded in `RESULTS.csv` — they are not needed again.

## Step 1 — the two CPU items (no GPU, minutes)

```bash
python scripts/val_metric_correlation.py
python scripts/val_metric_correlation.py --exclude e8 e13 e7   # loss-scale changers
python scripts/dead_params.py
```

`--exclude e7` is new and deliberate: §3.5's audit found `loss_w_aux` is a weight *inside*
the sum `val_metric` is, appearing twice, swept over 100×. E7 belongs in the same excluded
class as E8 and E13 and was never flagged that way. Run it both ways and report both.

What each one decides:

- **The correlation** gates E14-deep. High rank correlation → the five control rows mean
  something and the batch runs as seven. Low → the batch shrinks to the two peak-lr rows
  (`--lr 4e-4`, `--lr 8e-4`), which screen on the rendered metric regardless. Either way it
  is a reportable §6 finding: whether the cheap proxy predicted the expensive metric.
- **`dead_params.py`** produces the E6 count for §1.4. Seconds, and it closes an item that
  has been open since day 1.

## Step 2 — the σ_test ladder (GPU, eval only, both GPUs in parallel)

Six σ values per ladder at `n_samples 3`, no retraining, fixed checkpoint.

```bash
./scripts/sigma_test_sweep.sh e9_sigma050_chn   1 > sigma_e9.log   2>&1 &
./scripts/sigma_test_sweep.sh seedfloor_1111_chn 2 > sigma_base.log 2>&1 &
wait
cat experiments/e9_sigma050_chn_main_model/results_sigma_test/results.txt
cat experiments/seedfloor_1111_chn_main_model/results_sigma_test/results.txt
```

The script stashes `sigma L1 s-IoU` triples and cleans up each results tree as it goes, so
this costs almost no disk. Read both ladders against **Rule 1** above before touching
step 4.

Note while reading: the ladder is at `n_samples 3`, and σ_test is precisely the knob that
governs how much the N decoded candidates differ from each other. So the ladder's shape is
partly a property of N = 3. If σ\* lands at an endpoint (0.1 or 1.5) rather than in the
interior, say so — an endpoint optimum means the bracket was too narrow, not that the
extreme is best.

## Step 3 — set σ_test for the confirmation eval

From Rule 1. Record the chosen value and which of the three cases produced it; it goes in
§3.2 and in the §5 table caption.

```bash
SIGMA_TEST=1.0     # or sigma* -- set from step 2, per Rule 1
```

## Step 4 — confirmation eval, six runs

Baseline at three seeds and E9 σ=0.5 at three seeds, `n_samples 50`, all 34 fonts.
E9 needs **no extra flag at test time** — `enc_noise_std_train` adds no parameters and
`test_few_shot.py` runs under `.eval()`, so the train-time σ never applies here. That is
the whole reason this pair is comparable.

**Time one run before launching all six.** The cost of an `n_samples 50` eval has never
been recorded, and §3.3's rule is to measure before sizing. Start with the seed-1111 pair,
which is the pair that yields the headline Wilcoxon; if the two of them take more than
about 90 minutes each, run only that pair this session and take seeds 2222 and 3333 next.

```bash
run_confirm () {   # $1 name_exp, $2 gpu
  local exp="experiments/$1_main_model"
  local ckpt; ckpt="$(python3 scripts/best_checkpoint.py "$exp")"
  echo "[$1] ckpt=$ckpt sigma_test=$SIGMA_TEST GPU $2"
  CUDA_VISIBLE_DEVICES=$2 python test_few_shot.py \
    --mode test --model_name main_model --language chn --max_seq_len 71 \
    --batch_size 1 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29 \
    --n_samples 50 --enc_noise_std_test "$SIGMA_TEST" \
    --name_exp "$1" --name_ckpt "$ckpt" > "nohup_confirm_$1.out" 2>&1
  python eval_reconstruction_error.py --exp_dir "$exp" --name_ckpt "$ckpt" \
    --csv_out "$exp/results/eval_${ckpt}_n50.csv"
}

# wave 1 -- the headline pair. Time these two before committing to the rest.
run_confirm seedfloor_1111_chn 1 & run_confirm e9_sigma050_chn 2 & wait
# wave 2
run_confirm seedfloor_2222_chn 1 & run_confirm e9_sigma050_2222_chn 2 & wait
# wave 3
run_confirm seedfloor_3333_chn 1 & run_confirm e9_sigma050_3333_chn 2 & wait
```

**The `--csv_out` suffix is not optional.** `eval_reconstruction_error.py` defaults to
`results/eval_<ckpt>.csv`, which is exactly where the `n_samples 3` screening CSV already
sits — without the suffix the confirmation run silently overwrites the screening row that
every Tier 1–3 delta was computed from. `scripts/build_results_table.py` was taught the
`_n<N>` suffix on 2026-08-05 and reads the budget off the filename, so a suffixed CSV
lands in `RESULTS.csv` as its own row with `n_samples=50` and the same checkpoint label.

## Step 5 — the paired Wilcoxon

New script, `scripts/paired_wilcoxon.py`, added 2026-08-05. Self-checked against exact
enumeration with `--selftest`.

```bash
python scripts/paired_wilcoxon.py --selftest

B=experiments; R=results
python scripts/paired_wilcoxon.py --metric l1 --labels 1111 2222 3333 \
  --baseline  $B/seedfloor_1111_chn_main_model/$R/eval_*_n50.csv \
              $B/seedfloor_2222_chn_main_model/$R/eval_*_n50.csv \
              $B/seedfloor_3333_chn_main_model/$R/eval_*_n50.csv \
  --candidate $B/e9_sigma050_chn_main_model/$R/eval_*_n50.csv \
              $B/e9_sigma050_2222_chn_main_model/$R/eval_*_n50.csv \
              $B/e9_sigma050_3333_chn_main_model/$R/eval_*_n50.csv

# and again on the second metric -- E9's s-IoU case is independent of its L1 case
python scripts/paired_wilcoxon.py --metric iou --labels 1111 2222 3333 \
  --baseline ... --candidate ...     # same paths
```

The script runs **one test per seed and does not pool them**. Per-font differences from
different training seeds share the same fonts and the same data, so they are not
independent draws; pooling 102 rows and calling it n = 102 would inflate significance by
roughly the seed correlation. The across-seed summary is the sign pattern of the three
Hodges–Lehmann shifts, which is the same criterion §3.6 used to promote E9 — now at
per-font resolution instead of one aggregate number per seed.

Report the Hodges–Lehmann shift next to every p-value. A p-value alone says nothing about
size, and every effect in this project lives inside a 0.0093 floor.

## Step 6 — record

```bash
python scripts/build_results_table.py     # picks up the _n50 rows automatically
git add -A && git commit -m "Confirmation eval on E9 sigma=0.5; sigma_test ladder; paired Wilcoxon"
git push origin repro
```

Then, repo first:

- `PROJECT_PLAN.md` §3.2 — the σ_test ladder, as a **Measured (2026-08-05)** paragraph, and
  the Rule 1 case it fell into
- `PROJECT_PLAN.md` §3.7 and **§5** — the confirmation table. §5 has been empty on purpose
  since day 1 waiting for a finalist; E9 is the finalist
- `PROJECT_PLAN.md` §8 items 8 and 9 — strike them through with the resolution
- Vault `Runs/Run Log.md` — one row per eval; `Experiments/Experiment Tracker.md` — E9
  `screened` → `confirming` → `done`; `_Open Tasks.md` — checkboxes and `updated:`

---

## If the session runs long

Cut in this order, protecting what cannot be recovered cheaply later:

1. Drop confirmation seeds 2222 and 3333 (keep the 1111 pair — it is the headline Wilcoxon).
2. Drop the `--metric iou` Wilcoxon pass (recomputable from the same CSVs at any time, no GPU).
3. Drop the σ_test ladder on `seedfloor_1111_chn` — **but if you do, Rule 1 collapses to its
   third row and the confirmation must run at σ_test = 1.0.** A ladder on the candidate
   alone cannot distinguish an eval-procedure property from a candidate interaction.

Do not cut step 1. It needs no GPU and it gates a seven-run batch.

## What is queued behind this session

- **E14-deep**, seven runs, gated on step 1's correlation. Staged and commented out in
  `run_experiments.sh`.
- **English arm**, `docs/english-arm.md`. First action is a 6-epoch timing run, which can
  share a GPU wave with any of the eval steps above since evals leave a GPU idle between waves.
- **Stage 1 closeout**: SSIM in `eval_reconstruction_error.py`, the quantization oracle on
  the 128-bin grid, and §2.4 in prose. All Mac-side, no cluster needed.
