# Day 6: the final GPU push

Runbook for the 2026-08-08 cluster session (day 6 of 13). Five jobs, ordered by value
per GPU-hour, not by interest. Submission is Saturday 15 August, seven days out.

Read `PROJECT_PLAN.md` §0, §8 items 10 and 12, and §9 first. Everything below is a
consequence of those.

**GPU allocation, set by Ben 2026-08-08: use GPUs 3, 2 and 1. Leave GPU 0 free.**
This supersedes the `GPUS=(1 2)` line and the "GPU 3 is off limits going forward"
comment in `scripts/run_experiments.sh` and `scripts/test_experiments.sh`, both of
which date from a 2026-08-04 directive that no longer applies. Edit both scripts to
`GPUS=(3 2 1)` before the first launch and delete the stale comment while you are
there.

---

## 0. The situation, stated before anything else

**Nothing in the report is blocked on a GPU.** Day 5 closed the last two open
questions: the training budget stays eliminated on Chinese, and E9 does not replicate
on English. The critical path is §4, §5 and §6 of `REPORT.md`, all three unwritten,
and that is Mac-side work.

So every job below is additive. **None of it may delay the writing, and none of it may
reopen a closed conclusion.** Jobs run overnight against Ben's sleep, not against his
report time. If a job overruns, kill it; nothing here is load-bearing for a
conclusion that is already recorded.

One job is an exception to "additive", and it is Job A. It does not add a finding, it
checks whether an existing one is real. That is why it runs first.

---

## 1. Job A: de-confound E1, and audit checkpoint selection

**~4 GPU-hours. Highest value per hour in this document. Run it first.**

### 1.1 What is wrong

`REPORT.md` §5 carries E1 (terminal encoder LayerNorm) as the project's one effect
that exceeds its own floor: s-IoU degraded at all three seeds, mean −0.0760, 2.4× the
0.0315 s-IoU floor, with the seed-3333 leg at 0.1095, the lowest value anywhere in
`RESULTS.csv`. It is finding 3 in §6 and the only non-null result in Stage 2 besides
E9.

Checkpoint selection is by `val_metric`, via `prune_checkpoints` and
`scripts/best_checkpoint.py`. Reading the selected epoch off `RESULTS.csv`:

| Run | Selected epoch |
|---|---|
| `seedfloor_1111_chn`, `seedfloor_2222_chn`, `seedfloor_3333_chn` | 150, 150, 150 |
| `e1_norm_chn` (seed 1111) | 150 |
| `e1_norm_2222_chn` | **125** |
| `e1_norm_3333_chn` | **100** |
| `e4_ngf32_chn`, `e5_bneck256_chn`, `e8_ls20_chn` | **125** |

**Every baseline was scored at 150. E1's two degrading legs are the two scored
earliest.** The seed-1111 leg, the only one scored at a matched 150, is also the
mildest (s-IoU 0.2101 against a 0.2240 anchor). The effect and the checkpoint epoch
are confounded, and the report currently attributes all of it to the LayerNorm.

This cannot be corrected arithmetically. The baseline's own s-IoU across epochs runs
0.2550 (125), 0.2467 (135), 0.2545 (150), which is not monotonic enough to support a
correction. It has to be measured.

### 1.2 The wider question the same runs answer

Selection is on `val_metric`; scoring is on rendered L1 and s-IoU. From `train.py:102`:

```
val_metric = 10·img_l1 + 0.01·vggpt + svg_total + svg_para_total
```

`img_l1` is the **image decoder branch's** L1 against the ground-truth raster. What
gets scored is the **refinement decoder's** SVG, rasterized. Different head.
`svg_para_total` is the refinement decoder, but it is cross-entropy over command types
and quantized bins, not pixels. **No term in the selection criterion is the quantity
being reported.**

`scripts/val_metric_correlation.py` measured the cross-run version of this at ρ =
0.125 and stated that within-run selection "was never affected", on the grounds that a
constant loss-scale factor cancels inside one run. That argument is about scale and
says nothing about ordering, and ordering is what `prune_checkpoints` acts on. It is
untested. Job A tests it as a by-product.

Two further signals, already in `RESULTS.csv` and costing nothing: the
`seedfloor600_*_chn` runs trained to 600 epochs and `val_metric` selected epochs 200 /
150 / 200; and the authors' own English checkpoints get monotonically worse on the
reported metric as they train (see Job B).

### 1.3 Train

Three runs, one wave, ~1 h. **`--max_ckpt_keep 10` is the point of this job**: it
keeps every checkpoint at `--freq_ckpt 25`, so epochs 25/50/75/100/125/150 all survive
and can be scored against each other. Every other flag matches the originals exactly,
so the new runs are comparable to the existing table.

```bash
cd ~/deepvecfont-v2
git pull origin repro
python scripts/check_infra.py          # expect 196 passed, 0 failed
df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
```

Edit `scripts/run_experiments.sh`: set `GPUS=(3 2 1)`, replace `EXPERIMENTS` with the
block below, and change `--max_ckpt_keep 2` to `--max_ckpt_keep 10` in `COMMON_ARGS`.

```bash
EXPERIMENTS=(
  # Job A, 2026-08-08. Retrains of two E1 legs plus one matched baseline, all with
  # every checkpoint kept, so E1 can be read at 150 against a baseline at 150 and
  # the val_metric-vs-rendered ordering can be checked within a run.
  "a_e1_norm_2222_chn --seed 2222 --enc_final_norm True"
  "a_e1_norm_3333_chn --seed 3333 --enc_final_norm True"
  "a_seedfloor_3333_chn --seed 3333"
)

COMMON_ARGS="--mode train --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 10"
```

```bash
./scripts/run_experiments.sh parallel
```

The `a_` prefix keeps these separate from the originals in `experiments/` and in
`RESULTS.csv`. Do not overwrite the originals; the confound is itself a finding and
the old rows are its evidence.

`a_seedfloor_3333_chn` is a retrain of an existing baseline, and it is here because
seed 3333 is the leg where E1 looks worst. Its epoch-150 number should land within the
0.0093 floor of the existing `seedfloor_3333_chn` (0.1680 screening / 0.1632 at
`n_samples 50`). **If it does not, stop and report before scoring anything else** —
that would mean the retrain is not reproducing, and nothing downstream is readable.

### 1.4 Score

Screening budget, `n_samples 3`, matching how every E1 number in the table was
produced. Three checkpoints per run, nine decodes. Do **not** use
`test_experiments.sh` here: it auto-selects the best-val checkpoint, which is exactly
the behaviour under audit.

```bash
for RUN in a_e1_norm_2222_chn a_e1_norm_3333_chn a_seedfloor_3333_chn; do
  GPU=3
  for CKPT in 100_4040.ckpt 125_5040.ckpt 150_6040.ckpt; do
    CUDA_VISIBLE_DEVICES=$GPU python test_few_shot.py --mode test --name_exp $RUN \
      --language chn --max_seq_len 71 --model_name main_model --batch_size 1 \
      --n_samples 3 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29 --name_ckpt $CKPT
    python eval_reconstruction_error.py --exp_dir experiments/${RUN}_main_model \
      --name_ckpt $CKPT --csv_out eval_${RUN}_${CKPT%.ckpt}.csv
  done
done
```

Confirm the exact checkpoint filenames with `ls experiments/<run>_main_model/checkpoints/`
before running the loop; the step counts above are the ones the existing Chinese runs
used and should hold, but they are derived from batch count and are worth one `ls`.

**Landmine, from §3.7.** `test_few_shot.py`'s resume-skip check ignores `--n_samples`.
If a results tree already exists for a checkpoint it will silently rescore stale
output. These are fresh `a_`-prefixed experiments so it should not fire, but if you
re-run any decode, delete `experiments/<run>_main_model/results/<ckpt>/` first.

### 1.5 Reading rule, pre-committed 2026-08-08 before any of these runs launch

Compare E1 to baseline **at matched epoch 150 only**. The 100 and 125 rows exist to
measure the epoch effect, not to pick a favourable comparison.

1. **E1's s-IoU deficit holds at matched 150, at both seeds, beyond the 0.0315
   floor.** The finding stands as written. Add one paragraph noting it was checked for
   the epoch confound and survived.
2. **The deficit shrinks inside the floor at matched 150.** E1 becomes a null like
   everything else, and §5, §6 finding 3 and the abstract are rewritten. Losing it is
   not a loss: "our one apparent effect turned out to be an artifact of checkpoint
   selection, and here is the measurement" is a stronger §6 paragraph than the effect
   was.
3. **Mixed across the two seeds.** Report as inconclusive with both numbers shown. No
   third seed, no re-tuning.

In all three cases, the epoch curve gets reported regardless of what it does to E1,
because it is the evidence for the selection audit.

---

## 2. Job B: does `val_metric` rank checkpoints the way the metric does

**~15 GPU-minutes. Run it while Job A trains, on whichever GPU is free first.**

The authors' released English checkpoints, one training run, three checkpoints, in
`RESULTS.csv`:

| Official English ckpt | rendered L1 (raster) | rendered L1 (svg) |
|---|---|---|
| 500 | 0.0645 | 0.0569 |
| 550 | 0.0649 | 0.0573 |
| 600 | 0.0658 | 0.0584 |

Monotonically worse the longer it trains, on both conventions. They shipped 600.

`scripts/val_on_checkpoint.py`, added 2026-08-08, computes `val_metric` for any
checkpoint including ones we did not train. It is a validation pass with no decoding,
which is why this is minutes rather than hours.

```bash
cd ~/deepvecfont-v2
for E in 500_160821 550_176871 600_192921; do
  CUDA_VISIBLE_DEVICES=3 python scripts/val_on_checkpoint.py \
    --ckpt_path experiments/official_eng_main_model/checkpoints/${E}.ckpt \
    --language eng --max_seq_len 51 --ref_nshot 4 \
    --tag official_eng_${E%%_*} --csv_out val_metric_audit.csv
done
```

Adjust `--ckpt_path` to wherever the released English weights actually sit; check with
`find /data/bens/deepvecfont-v2 -name '*_192921.ckpt'`. The script warns loudly if the
architecture flags do not match the checkpoint, so read any `load_state_dict` noise as
"wrong flags" before reading it as "bad checkpoint".

Do the same for the three Chinese official checkpoints if they are still on disk.

**What it decides.** If `val_metric` ranks 500 < 550 < 600 (best to worst), matching
the rendered order, selection is sound and the authors simply released their last
checkpoint. If it prefers 600, then the released code selects checkpoints on a
criterion that does not track its own reported metric — and every checkpoint in this
project, including all 37 rows of `RESULTS.csv`, was chosen the same way. That is a
§6 finding of the same class as the two-rasterizer result: a defect in the evaluation
apparatus rather than in any model.

**Scope guard.** Whatever this returns, we are **not** switching the project to
rendered-L1 checkpoint selection. That means decoding the validation split at every
checkpoint and it invalidates every number in `RESULTS.csv`, with seven days left.
Report it as a measured protocol defect, and name proper selection as future work
alongside the flow-matching head.

---

## 3. Job C: three more English baseline seeds

**~17.5 GPU-hours each, three GPUs, one overnight. Launch when Job A's training
finishes.**

The English floor (L1 0.0038, s-IoU 0.0129, SSIM 0.0140) rests on three points. §6's
own closing argument is that a reference computed from few samples is itself a sample,
and the s-IoU floor already moved from 0.0401 to 0.0315 on re-measurement. Six points
is a materially better floor, and the floor is what every English claim is quoted
against.

```bash
EXPERIMENTS=(
  "eng_seedfloor_4444 --seed 4444"
  "eng_seedfloor_5555 --seed 5555"
  "eng_seedfloor_6666 --seed 6666"
)

COMMON_ARGS="--mode train --model_name main_model --language eng --max_seq_len 51 --ref_nshot 4 --batch_size 32 --n_epochs 631 --freq_ckpt 20 --max_ckpt_keep 3 --wandb_project deepvecfont-v2-eng"
```

`--n_epochs 631` is the budget frozen on 2026-08-07 in `docs/english-arm.md` Step 3,
before any candidate was looked at. **It is not re-derived per seed here.** The three
original baselines used per-seed `E_conv` values, and re-deriving would make the new
seeds incomparable to the old ones. Score at the nearest surviving checkpoint to 630,
the same way §5.2 handled 640 / 580 / 640.

Score with `n_samples 50` on the 34-font subset, raster convention, matching every
existing English row:

```bash
CUDA_VISIBLE_DEVICES=<gpu> python test_few_shot.py --mode test --name_exp <run> \
  --language eng --max_seq_len 51 --model_name main_model --batch_size 1 \
  --n_samples 50 --ref_nshot 4 --ref_char_ids 0,1,26,27 --max_fonts 34 --name_ckpt <ckpt>
```

### 3.1 Reading rule, pre-committed 2026-08-08

**These seeds re-measure the floor. They do not reopen §5.2.** E9's English deltas
were computed as paired same-seed, same-epoch differences, so additional baseline
seeds cannot change any of the six deltas — they change only the yardstick those
deltas are held against.

If the floor **widens**, the mixed-sign reading in §5.2 is reinforced and nothing is
rewritten. If the floor **narrows** far enough that some individual E9 deltas clear
it, the reading still stands, because **mixed sign is mixed sign at any floor**: one
seed favours E9 on both metrics and two disfavour it on both, and no floor changes
that. Update the floor numbers wherever §5.2 quotes them, and say in the same
paragraph that the floor was re-measured after the fact and what it did.

---

## 4. Job D: Tier 4, transformer capacity

**~6 GPU-hours, two waves of three, Chinese. Launch after Job C returns.**

### 4.1 Why this reopens a closed scope decision

§8 item 10 closed further breadth on 2026-08-05, on measured grounds: candidate spread
is about half seed spread, and the leading single-seed candidate survived replication
1 time in 3. That reasoning holds and is not being overturned. **What it declined was
single-seed breadth. This is a three-seed batch, which is not that**, and it is
recorded as a reopening in §8 item 12 rather than slipped in quietly, on the same
principle as the day-5 English overrule.

The substantive reason: the assignment's example list has seven bullets and Tiers 1 to
3 cover six of them cleanly. **"Add residual or attention layers"** is covered only by
E3 (refinement decoder 1→2→3 layers). The sequence transformer's own width and depth,
which is the model's actual capacity, was never varied. E4 widened the image stacks
(`ngf` 16→32, null at 0.1691) and E5 the latent (`bottleneck_bits` 256/1024, null at
0.1687/0.1699); neither touched the transformer.

Two flags added 2026-08-08, both defaulting to the released value, both asserted by
`check_infra.py` section 9:

| Flag | Released | Candidate | Site |
|---|---|---|---|
| `--enc_depth` (E16) | 6, giving 12 self-attention blocks | 8, giving 16 | `models/model_main.py` |
| `--dec_d_ff` (E17) | 1024, a 2× expansion on d_model 512 | 2048, the literature-default 4× | `models/transformers.py:209` |

E17 is the better-motivated of the two. `ff` is deep-copied into both decoder stacks,
so one flag widens the autoregressive decoder and the refinement decoder that actually
produces the scored output.

### 4.2 Three seeds, and this is not optional

Both flags change parameter count, so both shift the global RNG stream for every
module constructed after them. That is the exact objection that kept E6 off the
candidate list: a capacity change is partly an initialization change, and the seed
floor exceeds anything the sweep is chasing. **Three seeds is what answers it** — an
init shift becomes a seed draw, and averaging over three of them is what makes a
capacity change readable at all. Never run either flag at one seed.

```bash
EXPERIMENTS=(
  "e17_dff2048_1111_chn --seed 1111 --dec_d_ff 2048"
  "e17_dff2048_2222_chn --seed 2222 --dec_d_ff 2048"
  "e17_dff2048_3333_chn --seed 3333 --dec_d_ff 2048"
  "e16_depth8_1111_chn --seed 1111 --enc_depth 8"
  "e16_depth8_2222_chn --seed 2222 --enc_depth 8"
  "e16_depth8_3333_chn --seed 3333 --enc_depth 8"
)

COMMON_ARGS="--mode train --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 2"
```

Score via `./scripts/test_experiments.sh parallel` at `n_samples 3`, the screening
budget, comparing each seed's mean against the three-seed baseline mean (0.1621 L1 /
0.2681 s-IoU at `n_samples 50`; use the screening anchor 0.1728 / 0.2240 for
screening-budget comparisons). Watch for the epoch-selection confound Job A is
about — if `best_checkpoint.py` selects below 150 for any of these six, score at 150
as well and report both.

### 4.3 Reading rule, pre-committed 2026-08-08

**Expect null.** Twenty-six single-factor changes have already been shown to perturb
this metric less than re-seeding does, and there is no reason capacity is special.
This batch is in the report for coverage of the assignment's change categories and to
close the one partial line in it, not because a win is expected. Say that in §4 rather
than letting a null read as a disappointment.

A candidate counts only if its three-seed mean clears the floor (L1 0.0097, s-IoU
0.0315) **and** the sign is the same at all three seeds, the same bar E9 had to clear.
No promotion to confirmation budget for anything that misses it.

---

## 5. Job E: the two optional cleanups

**Only after A through D. Neither changes a conclusion.**

1. **Finish English checkpoint 500's decode from font 862.** `test_few_shot.py`
   resumes where it stopped. Converts the ‡ rows in §5.1 from estimate to
   measurement, sharpening the 34-font subset's known optimism (0.0074 / 0.0106).

   **Note for the record:** Ben asked on 2026-08-08 whether this should switch to
   checkpoint 600. It should not. 500 is the **best**-scoring of the three released
   English checkpoints (0.0645 raster / 0.0569 svg) and 600 the worst; the partial
   decode was already run on 500; and the purpose of the 862-font decode is measuring
   subset bias, not model quality, so consistency with the existing partial run is
   what matters. Switching would cost a full re-run and buy nothing.

2. **E14-deep's two peak-lr rows**, `e14_lr4e4_chn` and `e14_lr8e4_chn`. Already
   staged and commented out in `run_experiments.sh`. Rows 1–5 of that block stay cut:
   ρ = 0.125 killed them on 2026-08-05.

---

## 6. Suggested wall-clock order

| When | Job | GPUs | Wall clock |
|---|---|---|---|
| Sat evening | A train | 3, 2, 1 | ~1 h |
| Sat evening | B | one free GPU | ~15 min |
| Sat evening | A score | 3, 2, 1 | ~2–3 h |
| Sat night → Sun evening | C train | 3, 2, 1 | ~18 h |
| Sun evening | C score | 3, 2, 1 | ~3 h |
| Sun night | D train, two waves | 3, 2, 1 | ~2 h |
| Mon morning | D score | 3, 2, 1 | ~2 h |
| Mon onward | E, if wanted | any | — |

Mac-side over the same period: §6, then §4 and §5, then the vault recording checklist.
The GPU column and the writing column do not touch.

---

## 7. Recording checklist

Repo first, then vault, per the established pattern.

**Repo**

- `RESULTS.csv` via `python scripts/build_results_table.py` — new rows for Job A's nine
  scored checkpoints, Job C's three baselines, Job D's six candidates.
- `PROJECT_PLAN.md` §0 status rows; §3.6 for the E1 outcome; §5.2's floor numbers if
  Job C moves them; a new §3.8 or §5.3 for Tier 4; §8 item 12's outcome.
- `val_metric_audit.csv` committed alongside `RESULTS.csv`.
- `REPORT.md` §5 and §6 finding 3, **only if Job A changes the E1 reading**.
- `docs/day6-gpu-push.md` — mark each job done inline, the way this project's other
  runbooks carry their outcomes.

**Vault** (`Alef Alif/02-Theory-Research/DeepVecFont-v2/`)

- `Runs/Run Log.md` — one row per training and per eval run. This is already six rows
  behind from day 5's English arm.
- `Experiments/Experiment Tracker.md` — E1 status transition, E16 and E17 added as
  `planned` → whatever they reach.
- `_Open Tasks.md` — checkboxes and the `updated:` frontmatter date.

---

## 8. What is explicitly not happening

Listed so it cannot be reopened at 2 a.m. with a free GPU.

- **No re-run of E9 on English** at a longer budget, a different checkpoint, or a
  fourth seed. Declined 2026-08-08 in §9 item 5, before this session. Hardware
  becoming available is not new evidence.
- **No new single-seed breadth batch.** §8 item 10, on measured grounds.
- **No switch to rendered-L1 checkpoint selection** for the project's existing
  numbers. §2 above.
- **No chasing the published 0.080.** §2.4 settled this; the reproduction is faithful
  and the gap is a property of the evaluation.
- **No commits to `main`.** The `main..repro` diff is a graded report section.
