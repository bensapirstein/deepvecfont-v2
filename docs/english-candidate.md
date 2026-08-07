# The English candidate run

Runbook for the 2026-08-07 cluster session (day 5). Two jobs, one of them carried
over from day 4. Companion to `docs/english-arm.md`, which owns the budget rule this
document deliberately breaks.

Read `PROJECT_PLAN.md` §0 and §8 items 4 and 10 first. Everything below is a
consequence of those.

---

## 0. The overrule, stated before anything else

`docs/english-arm.md` pre-committed a cut-off: *">6h per run: drop English."* On
2026-08-07 that rule fired. `E_conv` came out at 580 across the three seeds, the frozen
budget is 630, and at ~100 s/epoch a single English run costs **~17.5 GPU-hours**. Step
1's timing measurement and Step 2's convergence measurement agree independently, and
they agree on "drop it."

**We are running it anyway, as a deliberate overrule, decided 2026-08-07 by Ben with
three GPUs free.** The reasons, so the report can state them rather than have a reader
find the contradiction in the diff:

- The rule was written on 2026-08-04 to protect report time against GPU time. Six days
  of schedule slack have since appeared (`PROJECT_PLAN.md` §4, "Actual, as of
  2026-08-05"), so the tradeoff the rule priced no longer holds at that price.
- The cost is wall-clock, not contention. Three runs across three GPUs is one
  overnight, not three days.
- What it buys is the only generalization claim in the project. Everything else in
  Stage 2 is one script.

**This is a rule broken with the reason recorded, not a rule that failed to apply.**
The report says so in the methods section. A pre-committed rule that gets quietly
dropped the one time it produces an inconvenient answer is worth less than no rule, and
this project's entire Stage 2 argument rests on rules that were fixed before the
numbers arrived.

What is *not* overruled: the frozen budget itself, the eval convention, σ_test = 1.0,
and the reading rule in §4 below. Those stay pre-committed.

---

## 1. Three seeds of one candidate, not one seed of three candidates

The choice was live this morning and the project's own record settles it.

**Against breadth.** `PROJECT_PLAN.md` §0 records two measurements from 2026-08-05.
Every single-seed delta in the document was quoted against `seedfloor_1111_chn`, the
worst of three baseline draws, and 22 of 26 candidates "beat" that anchor while only 6
beat the mean. Separately, 26 candidates span 0.0101 in L1 while three baseline seeds
span 0.0097, so candidate-induced spread is roughly half the seed-induced spread. Then
Batch B tested three single-seed leaders at two more seeds each and **one of the three
survived.** §8 item 10 declined to fund another breadth batch on exactly this evidence.
Running three different English candidates at one seed each would reproduce, on a new
script, the failure mode this project has now measured twice.

**For replication.** The English L1 floor is **0.0038** (§3 below). E9's Chinese
confirmation delta was **−0.0040**. The expected effect is therefore about 1.05× the
floor it has to clear. A single English seed cannot resolve that; three can, by the
same same-sign paired test that promoted E9 on Chinese in the first place.

So: **E9 `enc_noise_std_train=0.5`, seeds 1111 / 2222 / 3333, paired to the three
existing English baselines.** One factor, one script, three seeds.

---

## 2. Design

### 2.1 Pairing

Each E9 run uses the same seed as the baseline it is compared against, so the
comparison is one factor and not one factor plus a draw. `scripts/paired_wilcoxon.py`
then runs per seed and never pools, per its own docstring.

### 2.2 Per-seed `--n_epochs`, and why it differs across the three

The baselines were scored at **640** (seeds 1111, 3333) and **580** (seed 2222). That
split was not a choice: `max_ckpt_keep=3` retains the three lowest-`val_metric`
checkpoints plus the latest, so which epochs survived pruning came down to each seed's
own noise, and 640 simply did not survive for 2222.

To avoid inheriting that accident on the candidate side, each E9 run declares an
`--n_epochs` whose **last** epoch is the epoch its baseline was scored at. `max_ckpt_keep`
always keeps the latest checkpoint, so the one we need is retained by construction:

| run | pairs against | baseline epoch | `--n_epochs` | est. cost |
|---|---|---|---|---|
| `e9_sigma050_1111_eng` | `eng_seedfloor_1111` @ 640 | 640 | **641** | ~17.8 h |
| `e9_sigma050_2222_eng` | `eng_seedfloor_2222` @ 580 | 580 | **581** | ~16.1 h |
| `e9_sigma050_3333_eng` | `eng_seedfloor_3333` @ 640 | 640 | **641** | ~17.8 h |

Every pair is then same-seed and same-epoch, which is stronger than the baselines
manage among themselves.

**Why varying `--n_epochs` across the three is safe here and would not be for E14.**
These runs use the default `ExponentialLR`, whose per-epoch gamma is fixed and
independent of `--n_epochs`, so the LR trajectory up to epoch 640 is identical whether
the run was scheduled to stop at 641 or 801. E14's `warmup_cosine` derives `total_steps`
from `--n_epochs` and does not have this property. The same argument appears in
`docs/english-arm.md` under "Cut-off computed"; it is the reason that document's
checkpoint substitution was defensible, and it is load-bearing again here. **If an E14
English run is ever launched, none of this transfers and it needs a real run at one
fixed budget.**

### 2.3 Naming

`e9_sigma050_<seed>_eng`, seed in the name for all three including 1111. The Chinese
arm left the seed out of `e9_sigma050_chn` and had to annotate the row afterwards
(`NOTES` in `scripts/build_results_table.py`). Not repeating that.

---

## 3. The English floor, measured 2026-08-07

From `RESULTS.csv`, three baseline seeds, `n_samples 50`, 34-font subset, raster
convention:

| metric | 1111 | 2222 | 3333 | mean | **spread (floor)** |
|---|---|---|---|---|---|
| L1 | 0.0583 | 0.0621 | 0.0587 | 0.0597 | **0.0038** |
| s-IoU | 0.7371 | 0.7242 | 0.7314 | 0.7309 | **0.0129** |
| SSIM | 0.7427 | 0.7287 | 0.7408 | 0.7374 | **0.0140** |

Two readings worth carrying into the report:

- **Relative noise is almost identical across the two scripts.** 0.0038 on a mean of
  0.0597 is 6.4%; Chinese is 0.0093 on 0.1621, or 5.7%. The absolute floor is a
  quarter the size and the instrument is no sharper.
- **The paper's own English ablation is finer than this floor.** Its published English
  ablation spans 0.0069 in total with individual steps of 0.003 or less, so the
  individual steps sit **below** 0.0038. §8 item 4 flagged this as a risk on
  2026-08-04 and asked whether the English floor would land near the Chinese one. It
  did not, and the answer is more pointed than the question: this instrument could not
  have resolved the paper's English ablation one row at a time. That belongs in §6 as
  a finding about the benchmark, not as an excuse.

---

## 4. Reading rule, pre-committed 2026-08-07 before any run launches

Fixed now, in this file, because §0 above already spent this project's credibility once
today.

**Primary test.** Same-sign paired L1 difference against the same-seed baseline at all
three seeds, which is the bar E9 cleared on Chinese (6 of 6 across L1 and s-IoU).
Magnitude read against the floors in §3, never bare. Per-seed paired Wilcoxon on the
per-font L1 column, never pooled across seeds.

**The three outcomes, and what each one is written up as:**

1. **3/3 same-sign on L1, magnitude at or above 0.0038.** E9 generalizes to a second
   script. Report with the floor, the three per-seed Wilcoxon p-values, and the
   Chinese confirmation delta beside it.
2. **3/3 same-sign, magnitude below 0.0038.** Directionally consistent, magnitude
   unresolved by this instrument. Sign agreement across three independent seeds is
   still worth 1/8 under a null, so this is reported as weak support with the
   magnitude explicitly declared unresolvable, not rounded up into case 1.
3. **Mixed sign.** E9 does not replicate on English. This is a **result and it is
   reported as one**, not as a failed run. Batch B put the survival rate of a
   single-seed Chinese leader at 1 in 3, so a candidate that replicated on one script
   and not another is well within what this project has already measured, and it
   sharpens §6's argument about what a sweep at this resolution can and cannot
   resolve.

**Not permitted after seeing the result:** re-tuning σ_test (settled at 1.0, §8 item
9), selecting a different checkpoint, scoring a different font subset, or adding a
fourth seed to break a 2/1 split. Any of those turns the confirmation into a search.

---

## 5. Preflight

```bash
cd ~/deepvecfont-v2
git pull                      # picks up bb79e57 and this file
nvidia-smi                    # pick free GPU ids by hand
df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
```

Disk was 75 GB of 200 GB on 2026-08-06. Three English runs at `max_ckpt_keep 3` plus
decode trees is comfortably inside that, but check rather than assume.

**GPU note.** GPU 3 was placed off limits by directive on 2026-08-04 and
`scripts/run_experiments.sh` still ships `GPUS=(1 2)`. The English baselines ran on
three GPUs, so ids 0, 1, 2 are the expected set. Confirm against `nvidia-smi`.

---

## 6. Job A, first: score `seedfloor600_*_chn`

Carried over from day 4, and it runs **before** the English launch rather than after.
It is bounded, it is the last open item from the previous session, and §9 item 4 makes
it a gate on the report's central claim: if it comes back materially better than
0.1629, §2.4 changes again and the report's argument changes with it. Better to know
that before committing 18 hours of GPU to an additive result.

Three seeds were trained on 2026-08-06 and never decoded. Both GT conventions,
`n_samples 50`, 34 fonts, to match §5.1.

```bash
cd ~/deepvecfont-v2
GPU=0

for seed in 1111 2222 3333 ; do
  NAME=seedfloor600_${seed}_chn
  CK=$(python scripts/best_checkpoint.py --name_exp ${NAME})   # prints the filename
  echo "######## $NAME  $CK"
  CUDA_VISIBLE_DEVICES=$GPU python test_few_shot.py --mode test \
    --name_exp ${NAME} --model_name main_model --language chn \
    --max_seq_len 71 --batch_size 1 --ref_nshot 8 \
    --ref_char_ids 0,1,2,3,26,27,28,29 --n_samples 50 \
    --name_ckpt ${CK} > nohup_${NAME}_eval.out 2>&1

  D=experiments/${NAME}_main_model
  python eval_reconstruction_error.py --exp_dir $D --name_ckpt ${CK} \
      --gt_source raster --csv_out $D/results/eval_${CK}_n50.csv
  python eval_reconstruction_error.py --exp_dir $D --name_ckpt ${CK} \
      --gt_source svg    --csv_out $D/results/eval_${CK}_n50_gtsvg.csv
done
```

**Expected: ~0.162 raster.** The official 600-epoch checkpoint at 0.1629 already bounds
what a 600-epoch run of ours returns through this harness. Anything near that is
confirmation. **If it comes back materially better, stop and re-read §2.4 before
launching Job B** — it would mean the released checkpoint is not what it appears to be.

---

## 7. Job B: E9 English, three seeds

Launch the moment Job A's decode releases its GPU, or in parallel on the other two ids
if you would rather not serialize. The three runs are independent, so seed 3333 landing
a few hours behind the other two costs nothing.

### 7.1 Via `run_experiments.sh`

Edit in place, then launch `parallel`:

```bash
GPUS=(0 1 2)

EXPERIMENTS=(
  "e9_sigma050_1111_eng --seed 1111 --n_epochs 641 --enc_noise_std_train 0.5"
  "e9_sigma050_2222_eng --seed 2222 --n_epochs 581 --enc_noise_std_train 0.5"
  "e9_sigma050_3333_eng --seed 3333 --n_epochs 641 --enc_noise_std_train 0.5"
)

# English block. --n_epochs is deliberately NOT here: it varies per run, §2.2.
COMMON_ARGS="--mode train --model_name main_model --language eng --max_seq_len 51 --ref_nshot 4 --batch_size 32 --freq_ckpt 20 --max_ckpt_keep 3 --wandb_project deepvecfont-v2-eng"
```

```bash
./scripts/run_experiments.sh parallel
```

`--n_epochs` appears in the per-experiment args rather than `COMMON_ARGS`, which is a
departure from every previous batch. `run_experiments.sh` appends extra args after
`COMMON_ARGS`, so a duplicated flag would resolve to the later one, but leaving it out
of the shared block makes the difference visible instead of silent.

### 7.2 Or as three explicit commands

Equivalent, and preferable if the cluster's copy of the script has drifted from the
repo's:

```bash
cd ~/deepvecfont-v2
EARGS="--mode train --model_name main_model --language eng --max_seq_len 51 --ref_nshot 4 --batch_size 32 --freq_ckpt 20 --max_ckpt_keep 3 --wandb_project deepvecfont-v2-eng --enc_noise_std_train 0.5"

CUDA_VISIBLE_DEVICES=0 nohup python train.py $EARGS \
  --name_exp e9_sigma050_1111_eng --seed 1111 --n_epochs 641 \
  > nohup_e9_sigma050_1111_eng.out 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python train.py $EARGS \
  --name_exp e9_sigma050_2222_eng --seed 2222 --n_epochs 581 \
  > nohup_e9_sigma050_2222_eng.out 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python train.py $EARGS \
  --name_exp e9_sigma050_3333_eng --seed 3333 --n_epochs 641 \
  > nohup_e9_sigma050_3333_eng.out 2>&1 &
```

**Sanity check within the first 10 minutes**, before walking away: confirm from the
wandb run config or the log header that `enc_noise_std_train` reads 0.5 and
`enc_noise_std_test` still reads 1.0. E9 varies the training-time sigma only; the
test-time sigma stays at the released default, which is what §8 item 9 settled.

---

## 8. Job C: score the three E9 runs

After training. Raster convention only, matching the baselines exactly. English needs
`--char_num 52` on the eval and `--max_fonts 34` on both halves.

```bash
cd ~/deepvecfont-v2
GPU=0

for pair in "1111 640_205761" "2222 580_186501" "3333 640_205761" ; do
  set -- $pair ; SEED=$1 ; CKSTEM=$2
  NAME=e9_sigma050_${SEED}_eng
  # the candidate's own checkpoint at that epoch; step count will differ from the
  # baseline's, so read it off disk rather than reusing CKSTEM
  CK=$(ls experiments/${NAME}_main_model/checkpoints/ | grep -E "^${CKSTEM%%_*}_" | head -1)
  echo "######## $NAME  $CK"

  CUDA_VISIBLE_DEVICES=$GPU python test_few_shot.py --mode test \
    --name_exp ${NAME} --model_name main_model --language eng \
    --max_seq_len 51 --batch_size 1 --ref_nshot 4 --ref_char_ids 0,1,26,27 \
    --n_samples 50 --max_fonts 34 --enc_noise_std_train 0.5 \
    --name_ckpt ${CK} > nohup_${NAME}_eval.out 2>&1

  D=experiments/${NAME}_main_model
  python eval_reconstruction_error.py --exp_dir $D --name_ckpt ${CK} --char_num 52 \
      --gt_source raster --max_fonts 34 \
      --csv_out $D/results/eval_${CK}_n50_subset34.csv
done
```

`--enc_noise_std_train 0.5` is repeated on `test_few_shot.py` deliberately.
`ModelMain(opts)` is constructed the same way for test as for train and checkpoint
loading is strict, so any flag that changes what modules get built has to be passed to
both. `test_experiments.sh` documents this in its own header.

Then the paired test, per seed, never pooled:

```bash
python scripts/paired_wilcoxon.py --metric l1 \
  --baseline \
    experiments/eng_seedfloor_1111_main_model/results/eval_640_205761.ckpt_n50_subset34.csv \
    experiments/eng_seedfloor_2222_main_model/results/eval_580_186501.ckpt_n50_subset34.csv \
    experiments/eng_seedfloor_3333_main_model/results/eval_640_205761.ckpt_n50_subset34.csv \
  --candidate \
    experiments/e9_sigma050_1111_eng_main_model/results/eval_*_n50_subset34.csv \
    experiments/e9_sigma050_2222_eng_main_model/results/eval_*_n50_subset34.csv \
    experiments/e9_sigma050_3333_eng_main_model/results/eval_*_n50_subset34.csv \
  --labels 1111 2222 3333

# and again with --metric iou
```

Then rebuild the table:

```bash
python scripts/build_results_table.py
```

`scripts/build_results_table.py` already carries `BATCH` and `N_SAMPLES` entries for the
three `e9_sigma050_*_eng` names, added on the Mac 2026-08-07, so no edit is needed on
the cluster. The `_n50_subset34` suffix does not match `CKPT_NSAMPLES_RE`, which is why
the `N_SAMPLES` fallback entries exist.

---

## 9. Recording checklist

Repo first, then vault, per the project convention.

- [ ] `PROJECT_PLAN.md` §5, the confirmation table, gains the English rows
- [ ] `PROJECT_PLAN.md` §0 gains a one-line row for the outcome, dated
- [ ] `PROJECT_PLAN.md` §8 item 4 closed with which of §4's three cases fired
- [ ] `docs/english-arm.md` Step 3 marked as executed, with a pointer here
- [ ] `RESULTS.csv` rebuilt, three or six new rows
- [ ] Vault `Runs/Run Log.md`, one row per training and per eval run
- [ ] Vault `Experiments/Experiment Tracker.md`, E9 English status transition
- [ ] Vault `_Open Tasks.md` checkboxes and `updated:` frontmatter
- [ ] `REPORT.md` §5 and §6: the generalization result, whichever way it lands
