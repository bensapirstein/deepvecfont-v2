# Review response: rendered-metric checkpoint selection, on a corrected dataset

Runbook for the cluster session that answers `docs/review-gemini.md`. Staged Mac-side
2026-08-12; nothing in it has been run. Scope decision is `PROJECT_PLAN.md` §8 item 14.

Submission is 1 September 2026, so this has about nineteen days of slack and three GPUs
(3, 2, 1; GPU 0 stays free). The Chinese arm is roughly twelve hours of wall clock, the
English arm roughly fifty-three. Neither is on the critical path in the sense that the
report already exists; both are on it in the sense that the review's answer is
"not ready to submit" and this is what changes that.

---

## 1. What the review asked for, and what this actually does

Its single highest-priority fix, verbatim: *"Re-evaluate the baseline and the top
candidates (E9, and the degrading E1) using rendered-metric checkpoint selection."*

Taken literally that is not runnable, and the reason is worth stating before anything
launches, because it is the finding this session starts from.

**`train.py` has always validated on the test split.** Line 129:
`get_loader(..., 'test')`. What this codebase calls validation is the set the report
scores. So `val_metric` was computed on the test fonts, and "select checkpoints by the
rendered metric" — computed the same way — would mean picking, for every run, whichever
checkpoint happens to score best on the very fonts we then report. That is oracle
selection. It would produce better numbers than the review is asking for and it would be
worth less than the numbers we have.

So the batch does three things instead, and only the second is what was asked:

1. **A real held-out validation split**, 20 Chinese base fonts carved out of train by
   `scripts/make_val_split.py`, never trained on and never scored in any table.
2. **Rendered-metric selection on it.** `render_val.py` decodes the val fonts at every
   checkpoint, rasterizes with cairosvg, binarizes at the same threshold
   `eval_reconstruction_error.py` uses, and logs the resulting L1 and s-IoU into
   `checkpoint_metrics.csv`. `--ckpt_select val_render_l1` makes both pruning and
   `scripts/best_checkpoint.py` rank on it. Same decoder, same rasterizer, same units as
   the reported number — everything `val_metric` was not.
3. **The dataset the paper describes.** Sec. 4.1 augments Chinese 10×; ours was built at
   6×. That is the review's third critical issue, and Ben's call on 2026-08-12 was to
   rebuild at 10× and make it the baseline rather than carry the mismatch as a caveat.

Two corrections to the review, for the record and for the report's reply, neither of
which changes what it asked for:

- It quotes the Chinese floor as 0.0097 and E9's L1 delta as 0.0040. Both are ours;
  §5.3 of the report already says in those words that E9 does not clear its floor and
  rests on the same-sign criterion alone. The review reads that as a pivot made to save
  the candidate. It was made against it — `PROJECT_PLAN.md` §0's "Correction made while
  drafting §5" records catching and removing a *narrower* floor that E9 would have
  cleared. The report should say this once, plainly, and not argue further.
- "Invalidates the fine-grained comparisons between all 26 candidates" is a hypothesis,
  not a measurement. §8 below measures it, on this batch's own data, for free.

**Nothing in this batch is comparable to any row in `RESULTS.csv` from before today.**
Different training set, different training-set size, different selection rule. Every
comparison here is internal: candidates against *these* three baseline seeds, at a floor
re-measured from *these* three. Say so in the report rather than splicing the tables.

---

## 2. Preflight, on the cluster, before anything trains

```bash
cd ~/deepvecfont-v2
git pull origin repro
conda activate dvf_v2

python scripts/check_infra.py          # expect 220 passed, 0 failed
df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
nvidia-smi
```

Disk is the one that can bite. This batch keeps every checkpoint of 27 Chinese runs
(`--max_ckpt_keep 10` against six checkpoints per run), which is the largest thing this
project has ever held at once. If `/data/bens` is over 120 GB used before starting, prune
old `experiments/*/results` trees first — the `eval_*.csv` summaries are what
`RESULTS.csv` was built from and the decode trees underneath them are not needed again.

---

## 3. Rebuild the Chinese dataset

Order matters and is not reversible without a re-download. **Back up first**, then do the
split before the augmentation, so no held-out font's sheared twin is left in train.

```bash
cd /data/bens/deepvecfont-v2/data/vecfont_dataset
cp -a chn chn_6x_backup            # ~3 GB, and the only way back
```

**3.1 Carve the val split.** Dry run first; it prints what it would move and moves
nothing.

```bash
cd ~/deepvecfont-v2
python scripts/make_val_split.py --language chn --n_val 20
python scripts/make_val_split.py --language chn --n_val 20 --apply
```

Expect: 1272 directories over 212 base fonts before, 20 base fonts to `chn/val`, 100
augmented copies to `chn/_val_aug_discard`, 1152 directories over 192 base fonts left in
train. The manifest lands in `data_splits/chn_val_split.json` (tracked in git) and beside
the data. Delete `_val_aug_discard` once the counts look right.

**3.2 Re-augment train at 10×.** `data_utils/augment.py` runs from inside its own
directory and skips any directory whose name contains `_`, so it re-augments the 192
remaining base fonts and leaves the `_k` copies alone until it overwrites them.

```bash
cd ~/deepvecfont-v2/data_utils
python augment.py --language chn --split train --n_aug 9 \
  --max_len 71 --n_chars 52 --img_size 64 \
  --output_path ../data/vecfont_dataset
```

**`--n_aug 9`, not 10.** The font itself plus nine transformed copies is the paper's 10×.

**This only works because of a fix made today.** The released `aug_rules` had five
branches ending in a bare `else`, so every index ≥ 4 returned `rotate(-5)`: `--n_aug 9`
against the original code would have written five identical copies and reported itself as
10× augmentation. `data_utils/augment.py` now defines nine distinct transforms
(`N_AUG_RULES = 9`) and raises on anything beyond them. Indices 0–4 are byte-identical to
the released rules, so the 6× dataset is a strict subset of this one.

**3.3 Relax the representation.** `augment.py` writes `sequence.npy` and
`rendered_64.npy`; the dataloader reads `sequence_relaxed.npy` and `pts_aux.npy`. Those
come from `relax_rep.py` and are stale for the copies just rewritten.

```bash
python relax_rep.py --language chn --split train --max_len 71 --n_chars 52 \
  --output_path ../data/vecfont_dataset
python relax_rep.py --language chn --split val --max_len 71 --n_chars 52 \
  --output_path ../data/vecfont_dataset
```

**3.4 Verify before training anything on it.**

```bash
cd ~/deepvecfont-v2
ls data/vecfont_dataset/chn/train | wc -l        # expect 1920 = 192 x 10
ls data/vecfont_dataset/chn/val   | wc -l        # expect 20
ls data/vecfont_dataset/chn/test  | wc -l        # unchanged, 34
python -c "
import json, os, numpy as np
val = set(json.load(open('data_splits/chn_val_split.json'))['val_fonts'])
train = {d.split('_')[0] for d in os.listdir('data/vecfont_dataset/chn/train')}
assert not (val & train), f'LEAK: {sorted(val & train)}'
print('no val font appears in train, augmented or otherwise')
for d in sorted(os.listdir('data/vecfont_dataset/chn/train'))[:1]:
    p = f'data/vecfont_dataset/chn/train/{d}'
    print(d, {f: np.load(os.path.join(p, f)).shape
              for f in ['sequence_relaxed.npy', 'pts_aux.npy', 'rendered_64.npy']})
"
```

Then check that the four new augmentation rules produced four *different* fonts, which is
the failure the old `else` branch would have hidden:

```bash
python -c "
import numpy as np
base = '000'   # or whatever the first base id in train is
a = [np.load(f'data/vecfont_dataset/chn/train/{base}_{k}/sequence.npy') for k in range(9)]
import itertools
dupes = [(i,j) for i,j in itertools.combinations(range(9),2) if np.array_equal(a[i],a[j])]
print('duplicate augmentation pairs:', dupes or 'none')
assert not dupes
"
```

**3.5 English.** No augmentation on the English arm (the affine helpers in
`data_utils/common_utils.py` hardcode the Chinese sequence length of 71), so English needs
the val split only. Check the train split size first and scale `--n_val` to roughly 10%
of the base fonts.

```bash
ls data/vecfont_dataset/eng/train | wc -l
python scripts/make_val_split.py --language eng --n_val 40
python scripts/make_val_split.py --language eng --n_val 40 --apply
```

---

## 4. Reading rules. Fixed here, before a single number exists

This is the section the whole batch is worth. The review's charge is that a rule was
pre-committed and then bent when it produced an inconvenient answer. The only reply that
means anything is a rule written down before the run and followed afterwards, including
when it hurts.

**4.1 The floor is re-measured, from this batch, before any candidate is looked at.**
Train the three `rv_seedfloor_*_chn` seeds, score them, take the spread of their L1 and
s-IoU. That spread is the floor for everything here. The old 0.0093 / 0.0315 do not carry
over: different training set, different training-set size, different selection rule.
`scripts/test_experiments.sh`'s `NOISE_FLOOR` is a placeholder marked as one and must be
replaced with the measured value before any candidate row is read.

**4.2 A candidate is a result only if its three-seed mean clears the re-measured floor
AND its sign agrees at all three seeds.** Both conditions, on the metric being claimed.
This is the same bar every earlier tier used. Same-sign agreement alone is not a result —
that is precisely the move the review objected to, and it is not available here.

**4.3 The three outcomes for E9, written now.**

- *Clears the re-measured floor on L1, same sign at three seeds.* E9 survives honest
  checkpointing and a corrected dataset. It stays in the improved column, and the report
  says the earlier reading was under-powered rather than wrong.
- *Inside the floor, whatever the sign.* **E9 comes out of the improved column.** The
  sweep is reported as a null result, in those words, and the report's contribution is
  the instrument and the protocol, not a winning candidate. This is the review's own
  requested outcome and there is to be no third framing invented for it.
- *Clears the floor in the degrading direction.* Reported as a degrading result, same as
  E1 and `cen_e3_refine2`. That is a finding.

**4.4 The same three outcomes apply to E1 and to all six category representatives.** An
effect that evaporates under correct selection is as much a finding as one that survives.
Do not write up a disappearing E1 as "reduced to null by better methodology" — write it
as: E1's degrading effect was partly an artifact of checkpoint selection, and here is by
how much.

**4.5 What would make this batch unreadable, and what to do about it.** If the three
baseline seeds come back with a spread materially larger than the old 0.0093, the 10×
dataset has made the instrument blunter, not sharper, and no candidate here can be read
at all. In that case: report the floor measurement itself as the finding, keep the old
6× tables as the substantive results with their selection caveat stated, and do not run
the English arm. Decide this on the three baseline seeds alone, before scoring a single
candidate.

**4.6 Hardware is not evidence.** Three free GPUs and nineteen days do not license a
fourth seed, a longer budget, or a second candidate after seeing a near-miss. If a rule
above is overruled, it is overruled *in writing, with the reason*, the way
`docs/english-arm.md` Step 3 records its overrule. Never quietly.

---

## 5. Chinese: launch

`scripts/run_experiments.sh` is already edited — `GPUS=(3 2 1)`, the 27-run `EXPERIMENTS`
array, and `COMMON_ARGS` carrying `--render_val_freq 25 --ckpt_select val_render_l1
--max_ckpt_keep 10`. Read it once before launching; do not re-derive the commands.

**5.1 Smoke test first, and do not skip it.** Nothing in the rendered-validation path has
ever run on a GPU. One run, three epochs, one checkpoint, four val fonts:

```bash
CUDA_VISIBLE_DEVICES=3 python train.py --mode train --name_exp smoke_rv_chn \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --ref_char_ids 0,1,2,3,26,27,28,29 \
  --batch_size 32 --seed 1111 --n_epochs 3 --freq_ckpt 1 --max_ckpt_keep 10 \
  --render_val_freq 1 --render_val_fonts 4 --render_val_samples 1 \
  --ckpt_select val_render_l1
```

**`--ref_char_ids` on a training command is not a typo.** The rendered pass runs the
model at `mode='test'`, which takes its reference glyphs from that flag instead of
drawing them at random, and asserts there are exactly `--ref_nshot` of them. The default
is the English set of four, so a Chinese run without it dies at the first checkpoint —
an hour in, with a bare `AssertionError`. `train.py` now refuses before epoch 1 instead,
and both `COMMON_ARGS` arrays already carry the right ids. Passing the same ids the test
run will use also means the validation decode is conditioned exactly as the scored
decode is, which is the point.

What to check, in order:

```bash
cat experiments/smoke_rv_chn_main_model/logs/checkpoint_metrics.csv
```

- `val_render_l1` is populated on all three rows, not blank.
- It is in the right neighbourhood. On the raster convention an untrained Chinese model
  should land somewhere around 0.2–0.4, and the pipeline floor is 0.1422, so anything
  below ~0.14 means the ground-truth side is wrong, not that the model is brilliant.
- `val_render_renderability` is near 1.0. A low value at epoch 3 is expected; a value of
  0.0 means `render()` is failing on every sequence and the pass is measuring nothing.
- Three checkpoints on disk, and `python scripts/best_checkpoint.py
  experiments/smoke_rv_chn_main_model --criterion val_render_l1` returns one of them.

Delete `experiments/smoke_rv_chn_main_model` before the real batch.

**5.2 Time one real run** before committing three GPUs for twelve hours. The rendered pass
is new cost: 20 fonts × 52 glyphs of autoregressive decode plus cairosvg, six times per
run. If it adds more than ~15 minutes to a ~60-minute run, drop `--render_val_fonts` to
10 in `COMMON_ARGS` and note the change here — the split stays 20 fonts, the pass just
reads the first 10 of them.

**5.3 Launch.**

```bash
./scripts/run_experiments.sh parallel
```

Nine waves of three. Watch the first wave's `nohup_rv_seedfloor_*.out` for the
`rendered val L1` line at epoch 25 before walking away.

---

## 6. Chinese: score

```bash
./scripts/test_experiments.sh parallel
```

Already set to `CRITERION="val_render_l1"` and `--n_samples 50` — the confirmation
budget, because the argument is about which checkpoint gets scored and the score has to be
the one the report quotes. Then, in order:

1. Read the three baseline seeds. Compute the floor. Put it in `NOISE_FLOOR` and in
   `PROJECT_PLAN.md` as a `**Measured (2026-08-…)**` paragraph. Apply §4.5's gate.
2. Only then read the candidates.
3. `python scripts/build_results_table.py` — add an `rv-review` batch entry for the new
   names, and keep the new rows visibly separate from the pre-2026-08-12 ones. They are
   not the same experiment.
4. `python scripts/recompute_deltas.py` for the deltas. Never type one.
5. `python scripts/paired_wilcoxon.py` on E9 and E1, same as the earlier confirmation.

---

## 7. English

Only after the Chinese arm has been read and §4.5's gate has not fired. Nine runs at
~17.5 GPU-h is ~53 hours of wall clock on three GPUs.

Swap `EXPERIMENTS` for `EXPERIMENTS_ENG` and `COMMON_ARGS` for `COMMON_ARGS_ENG` in both
scripts (both arrays are already written, sitting directly below the Chinese ones), then
the same launch and score. The English scoring pass runs at `--n_samples 10`, which is
the paper's own Sec. 4.1 English protocol and closes the review's §2 point in the same
motion.

English floor is re-measured from `rv_seedfloor_*_eng` exactly as §4.1 requires. The old
0.0038 does not carry over either.

---

## 8. The audit that costs nothing

Because `--max_ckpt_keep 10` kept every checkpoint and every one carries both columns,
the review's question 2 can be answered without a GPU:

```bash
python scripts/selection_disagreement.py --csv_out selection_audit.csv \
  --floor <the floor measured in §6 step 1> \
  experiments/rv_*_chn_main_model
```

For each run: which epoch `val_metric` would have picked, which epoch the rendered metric
picks, whether they differ, and what the difference costs in rendered L1. The reading rule
is printed by the script and is fixed here too — a mean cost inside the floor means
checkpoint selection was not what decided the earlier sweep, and the review's question 2
is answered in the negative *with our own data*. A mean cost above the floor means it was,
and the old tables are reported as selection-confounded.

Run this before writing the report section. It is the strongest single piece of evidence
this batch can produce, and it is free.

---

## 9. Not in scope

Listed so they stay out.

- **No fourth seed, no longer budget, no new candidate**, whatever the numbers do. §4.6.
- **No re-scoring of the old 6× rows under the new selection rule.** Their checkpoints
  were pruned on 2026-08-06 and most no longer exist; the ones that do would give a
  partial, self-selected sample. The old tables keep their caveat.
- **No chasing the gap to 0.080.** `PROJECT_PLAN.md` §2.4 closed it and §8 item 13's
  Chinese-rendering question is a separate session, not this one. If the 10× dataset
  happens to move the absolute number, record it as a measurement and resist the
  temptation to reopen the argument around it.
- **No English augmentation.** The affine helpers hardcode 71-length sequences and the
  paper only claims 10× on Chinese.

---

## 10. Recording checklist

Repo first, then the vault.

- `PROJECT_PLAN.md`: §0 status rows for the dataset rebuild, the new floor and each arm's
  reading; the measured numbers inline in §3 and §5 as dated `**Measured**` paragraphs;
  §8 item 14 closed with what actually happened.
- `RESULTS.csv` via `scripts/build_results_table.py`, `rv-review` batch.
- `data_splits/chn_val_split.json` and `eng_val_split.json` committed — which fonts were
  held out has to survive the next `/data/bens` cleanup.
- `selection_audit.csv` committed.
- `report/REPORT.md`: §5.1 gains standard deviations per the review's §4; the arrow
  glyphs in the metric headers get fixed; §4's heading stops calling E9 an architectural
  change and calls it what it is, a training-dynamics intervention; §6 gains the
  selection-disagreement measurement; and §5.3 says whatever §4.3 says it says.
- Vault: `Runs/Run Log.md` one row per training and per eval run;
  `Experiments/Experiment Tracker.md` status transitions; `_Open Tasks.md` checkboxes and
  the `updated:` frontmatter.
