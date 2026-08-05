# Stage 1 closeout — SSIM rescore and the quantization oracle

Day 3, 2026-08-05. Paste into the Claude session on the cluster.
Companion to `docs/confirmation-launch.md`.

**This session trains nothing and needs no GPU for anything except the rescore's
rendering, which is CPU-bound anyway.** It closes the two Stage 1 items that have
been open since day 2 and fills the SSIM column of the §5 table.

Both scripts were written and self-tested on the Mac on 2026-08-05. Run their
`--selftest` first: it needs no dataset and no results tree, so a failure there is
an environment problem rather than a data problem, and it takes seconds to find out.

---

## Step 0 — sync and self-test

```bash
cd ~/deepvecfont-v2
git pull origin repro
python eval_reconstruction_error.py --selftest      # expect 9 passed, 0 failed
python scripts/quantization_oracle.py --selftest    # expect 18 passed, 0 failed
```

The eval self-test cross-checks the new SSIM against `skimage` to 1e-12 **if skimage
is importable**. It is not a dependency and the check skips cleanly when absent — on
the Mac the two agreed to 1.1e-16 across eight noise levels and a glyph-like mask pair,
so if skimage happens to be in `dvf_v2` the check should pass rather than skip. If it
fails, stop and report the number; do not rescore anything.

## Step 1 — the quantization oracle (CPU, one pass over the test set)

```bash
python scripts/quantization_oracle.py \
    --language chn --img_size 64 --grids 64 128 256 \
    --csv_out oracle_chn.csv | tee oracle_chn.log
```

Time an abbreviated pass first if the full one looks slow — it renders
`34 fonts × 52 glyphs × 4 grids` = 7072 glyphs through cairosvg:

```bash
python scripts/quantization_oracle.py --language chn --max_fonts 2 --grids 128
```

**What to read, and in this order.** The script prints a `grid` table then three
derived lines. The derived lines are the result; the raw table is not.

1. **`Pipeline floor (no quantization)`.** Rendering the true outline through this
   pipeline and scoring it against the dataset's own raster. If this is large — say
   above 0.05 — then a substantial part of the 0.1668 baseline is representation and
   rasterizer loss that no model could ever have avoided, and that is a §2.4 finding
   in its own right that the four candidate explanations did not anticipate. Report
   it either way.
2. **`cost of quantization at n=128`.** The pipeline floor subtracted out. This is
   the number §2.3 has been waiting for. Expect it to be small: §1.2 says the
   rasterized metric is largely blind to sub-pixel placement and a bin is 0.625 px.
3. **`Ceiling on E13 (128 -> 256 bins)`.** An upper bound on a candidate that already
   screened null at −0.0053 and flipped sign under replication. **Compare it to the
   0.0097 seed floor before reading anything into it.** If the ceiling is below the
   floor, E13 was unresolvable by construction and that is worth stating plainly:
   the experiment could not have produced a detectable result no matter how it landed.

Record all three in `PROJECT_PLAN.md` §2.3 as a dated **Measured (2026-08-05)**
paragraph, and put the `n=inf` and `n=128` rows into the §5 table's
*Quantization oracle (floor)* line, which has been empty since day 1.

## Step 2 — rescore with SSIM

`eval_reconstruction_error.py` now emits SSIM on the anti-aliased grayscale render
and `ssim_bin` on the same binary masks L1 and s-IoU use. Rescoring re-renders from
the existing merge HTMLs, so **no `test_few_shot.py` run is needed and nothing is
re-decoded** — the numbers are scored on exactly the same SVGs as before, and L1 and
s-IoU must come back bit-identical. That is the check: if L1 moves, something other
than SSIM changed and the rescore is not trustworthy.

The six confirmation rows are what §5 needs. Use the same `--csv_out` suffix the
confirmation eval used, or `build_results_table.py` will file them as screening rows.

```bash
B=experiments
for exp in seedfloor_1111_chn seedfloor_2222_chn seedfloor_3333_chn \
           e9_sigma050_chn e9_sigma050_2222_chn e9_sigma050_3333_chn ; do
  d="$B/${exp}_main_model"
  ckpt="$(python3 scripts/best_checkpoint.py "$d")"
  echo "=== $exp  ckpt=$ckpt"
  python eval_reconstruction_error.py --exp_dir "$d" --name_ckpt "$ckpt" \
      --csv_out "$d/results/eval_${ckpt}_n50.csv"
done
```

**Confirm L1 is unchanged** against `RESULTS.csv` before going further:
0.1662 / 0.1569 / 0.1632 for the baselines and 0.1588 / 0.1531 / 0.1623 for E9.

Then the screening rows, which are cheap and make the SSIM column complete across
every tier rather than only the finalists:

```bash
./scripts/test_experiments.sh sequential      # re-eval only; edit EXPERIMENTS first
```

If that is more time than it is worth, skip it. §5 only needs the six.

## Step 3 — rebuild and record

```bash
python scripts/build_results_table.py    # now fills the ssim column where present
python scripts/recompute_deltas.py       # regenerates every delta from the CSV
git add -A && git commit -m "Quantization oracle, SSIM rescore, Stage 1 closed"
git push origin repro
```

`build_results_table.py` was taught on 2026-08-05 to average an `ssim` column when the
per-font CSV has one and to leave it **blank** when it does not. Blank rather than
zero is deliberate: a missing measurement and a measured zero are different things,
and a zero would silently enter any mean computed over the column later.

Then, repo first:

- `PROJECT_PLAN.md` §2.3 — the three oracle numbers, dated
- `PROJECT_PLAN.md` §2.4 — whether the pipeline floor changes the reading. The
  2026-08-05 answer there attributes the gap to training budget and rules out the
  other three explanations; a large pipeline floor would be a fifth explanation that
  paragraph does not currently contain, so update it rather than appending to it
- `PROJECT_PLAN.md` §5 — the SSIM column and the oracle row
- `PROJECT_PLAN.md` §2.5 — **Stage 1 closes here.** Strike the section through with
  the date if the oracle row and the SSIM rescore both landed
- Vault `Runs/Run Log.md` and `_Open Tasks.md`

## What is queued behind this session

Nothing that blocks the report. Both remaining candidates for cluster time are
optional and neither is on the critical path:

- **E14-deep's two peak-lr rows** (`--lr 4e-4`, `--lr 8e-4`). The other five rows
  were cut when `val_metric_correlation.py` returned ρ = 0.125.
- **The English arm** (`docs/english-arm.md`), as a generalization test carrying E9
  across rather than a second sweep.
- **One 600-epoch Chinese baseline**, if the GPUs are otherwise idle. §2.4 names the
  training budget as the leading explanation for the gap to 0.080 and currently
  supports it with three arguments rather than a measurement. Four hours converts it
  into a measurement, and §5 has a row waiting. It is a Stage 1 question, so it ranks
  below both items above.
