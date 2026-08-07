# Official checkpoints, the metric-definition test, and the 600-epoch retrain

Day 3 evening, 2026-08-05. Paste into the Claude session on the cluster.
Follows `docs/stage1-closeout.md`.

**This is the session that decides whether Stage 1 is a reproduction or a near-miss**,
and the assignment's wording makes that worth real GPU time: *"you must compare your
results to the reported results to within negligible differences."*

Two jobs, and they run at the same time on different GPUs. Job A is eval-only and
takes about half an hour. Job B is four hours of training. **Start B first so it runs
while you read A**, then act on A when it lands.

---

## Why this session exists

`scripts/quantization_oracle.py` returned a pipeline floor of **L1 = 0.1422** on
Chinese: that is what you score when you render the *unmodified ground-truth outline*
through this evaluation and compare it against the dataset's stored raster. Fourteen
percent of pixels disagree before a model is involved at all.

The paper reports **0.080**. That is below our floor. **Under the scoring definition
this project has used for thirteen days, the paper's number is not reachable in
principle**, by any model, however well trained. So one of two things is true:

1. **The metric is defined differently.** The paper likely renders both sides through
   one rasterizer, where we render the candidate through cairosvg and take the ground
   truth from the dataset's pre-rendered PNG. Two rasterizers, two conventions, and a
   0.1422 disagreement between them that has nothing to do with model quality.
2. **The reproduction is undertrained**, and the floor is simply irrelevant because we
   are nowhere near it.

These make opposite predictions about the released checkpoints, which is what makes
this cheap to settle.

| | official 600 scores ~0.08 | official 600 scores ~0.16 |
|---|---|---|
| **under `--gt_source raster`** | our metric matches theirs; we are undertrained; job B fixes it | see the next column |
| **under `--gt_source svg`** | the metric was a definition difference; report both and match the paper | neither explains it; escalate, see "if both fail" |

`--gt_source svg` is new today. `test_few_shot.py` has always written `2 × char_num`
SVGs into each merge HTML, candidates first and ground-truth outlines second, and the
evaluation only ever read the first half. The second half is the ground truth rendered
through the *same* rasterizer as the candidate, so scoring against it removes the
cross-rasterizer term entirely.

---

## Job B — launch this first, it is the long pole

Three Chinese seeds to 600 epochs, matching the released English budget. About four
hours wall-clock across three GPUs, twelve GPU-hours.

Check `nvidia-smi` and set the ids to whatever is actually free.

```bash
cd ~/deepvecfont-v2
git pull origin repro
python scripts/check_infra.py          # expect all pass
df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
```

**Storage first, because this is the largest batch in the project.** Three runs at 600
epochs with `--freq_ckpt 25` and `--max_ckpt_keep 2` is the same retention as the
150-epoch batches, so disk should be flat. If `du` shows the screening result trees
still present from Tiers 1 to 3, delete them: they are all recorded in `RESULTS.csv`
and are not needed again.

```bash
for s in 1111 2222 3333 ; do
  gpu=$(( ${s:0:1} ))     # 1,2,3 -- override by hand if nvidia-smi says otherwise
  CUDA_VISIBLE_DEVICES=$gpu nohup python train.py --mode train \
    --name_exp seedfloor600_${s}_chn --model_name main_model \
    --language chn --max_seq_len 71 --ref_nshot 8 --batch_size 32 \
    --seed $s --n_epochs 601 --freq_ckpt 25 --max_ckpt_keep 2 \
    > nohup_seedfloor600_${s}_chn.out 2>&1 &
done
```

**`--n_epochs 601`, not 600.** The released English command uses `--n_epochs 801` for a
600-epoch checkpoint, and the existing Chinese runs used 151 for 150. Keep the
convention or the last checkpoint will not be where the naming implies.

Note this is a **new experiment name**, not a resume of the 150-epoch runs. Those stay
exactly as they are, because every Stage 2 number in the report is referenced to them
and a resumed run would silently change the baseline underneath thirteen days of
results.

Two things to watch in the first ten minutes:

- **Steady-state seconds per epoch.** The 150-epoch runs held 22.8 to 24.7 s. If 600
  epochs projects past six hours, say so before committing the night to it.
- **`e9_sigma050` is not in this batch.** Deliberate. Confirm the baseline reaches the
  paper first; whether E9 gets a 600-epoch arm is a decision for after job A lands.

---

## Job A — the decisive eval, on a spare GPU

### A0. Place the checkpoints where the harness expects them

`train.py` and `test_few_shot.py` both append `_main_model` to `--name_exp`, so the
directory they look in is `experiments/<name_exp>_main_model/checkpoints/`. The upload
path has no suffix. Use a **distinct name** so the official weights cannot be confused
with anything trained here.

```bash
cd ~/deepvecfont-v2
for lang in chn eng ; do
  src=/data/bens/deepvecfont-v2/experiments/dvf_base_exp_${lang}/checkpoints
  dst=experiments/official_${lang}_main_model/checkpoints
  mkdir -p "$dst" && ln -sfn "$src"/*.ckpt "$dst"/ && ls -l "$dst"
done
```

Expect `500_20040 / 550_22040 / 600_24040` for Chinese and
`500_160821 / 550_176871 / 600_192921` for English.

### A1. Confirm they load strictly

This is the check that the thirteen days of flag-adding did not break backward
compatibility. Every Stage 2 flag was written to construct its modules only when
enabled, so the defaults should produce the released architecture exactly. If this
fails, the failing key names say which flag broke the contract.

```bash
python - <<'PY'
import glob, torch
for p in sorted(glob.glob('experiments/official_*_main_model/checkpoints/*.ckpt')):
    sd = torch.load(p, map_location='cpu')
    sd = sd.get('model', sd)
    print(f"{p}: {len(sd)} tensors, {sum(v.numel() for v in sd.values() if hasattr(v,'numel')):,} params")
PY
```

Cross-check the parameter count against `scripts/dead_params.py`'s 136,055,207 for the
full model. A match means the released weights and our model are the same architecture,
which is the precondition for everything below.

### A2. Decode the Chinese official checkpoints

`--n_samples 50` and the frozen `ref_char_ids`, so the numbers are directly comparable
to the six confirmation rows already in `RESULTS.csv`.

```bash
GPU=0    # whichever job B did not take
for ck in 500_20040 550_22040 600_24040 ; do
  CUDA_VISIBLE_DEVICES=$GPU python test_few_shot.py --mode test \
    --name_exp official_chn --model_name main_model --language chn \
    --max_seq_len 71 --batch_size 1 --ref_nshot 8 \
    --ref_char_ids 0,1,2,3,26,27,28,29 --n_samples 50 \
    --name_ckpt ${ck}.ckpt > nohup_official_chn_${ck}.out 2>&1
done
```

**Clear `results/<ckpt>/` before any re-run at a different `--n_samples`.** The resume
check keys on the merge HTML existing, not on the budget that produced it. This cost
two silently mislabeled runs during the confirmation session.

### A3. Score each one BOTH ways — this is the whole point

```bash
D=experiments/official_chn_main_model
for ck in 500_20040 550_22040 600_24040 ; do
  echo "######## $ck  raster"
  python eval_reconstruction_error.py --exp_dir $D --name_ckpt ${ck}.ckpt \
      --gt_source raster --csv_out $D/results/eval_${ck}.ckpt_n50.csv
  echo "######## $ck  svg"
  python eval_reconstruction_error.py --exp_dir $D --name_ckpt ${ck}.ckpt \
      --gt_source svg --csv_out $D/results/eval_${ck}.ckpt_n50_gtsvg.csv
done
```

Then the same on English, which has its own protocol. English is the arm the released
checkpoint was most likely tuned for, so it is the stronger test of the two.

```bash
GPU=0
for ck in 600_192921 ; do
  CUDA_VISIBLE_DEVICES=$GPU python test_few_shot.py --mode test \
    --name_exp official_eng --model_name main_model --language eng \
    --max_seq_len 51 --batch_size 1 --ref_nshot 4 --ref_char_ids 0,1,26,27 \
    --n_samples 50 --name_ckpt ${ck}.ckpt > nohup_official_eng_${ck}.out 2>&1
  D=experiments/official_eng_main_model
  python eval_reconstruction_error.py --exp_dir $D --name_ckpt ${ck}.ckpt --char_num 52 \
      --gt_source raster --csv_out $D/results/eval_${ck}.ckpt_n50.csv
  python eval_reconstruction_error.py --exp_dir $D --name_ckpt ${ck}.ckpt --char_num 52 \
      --gt_source svg --csv_out $D/results/eval_${ck}.ckpt_n50_gtsvg.csv
done
```

The paper's English number is **0.052** and its Chinese number is **0.080**. Two
languages times two scoring modes gives four numbers, and the pattern across them is
more informative than any one of them.

### A4. The English oracle, for the same reason

```bash
python scripts/quantization_oracle.py --language eng --img_size 64 \
    --max_seq_len 51 --grids 128 --csv_out oracle_eng.csv
```

If English's pipeline floor is also far above 0.052, the cross-rasterizer term is
systematic rather than a Chinese quirk, and that settles the interpretation on its own.

---

## Reading it — filled 2026-08-06

**Chinese numbers are the full 34-font test set** (as planned). **English numbers are
NOT a full run** — see "Why English stopped short" below before quoting these.

| | paper | official 600, raster | official 600, svg | our 150-epoch baseline |
|---|---|---|---|---|
| Chinese | 0.080 | 0.1629 | 0.1174 | 0.1621 (3-seed mean) |
| English | 0.052 | 0.0658 (34-font subset) | 0.0584 (34-font subset) | 0.0597 (3-seed mean, 34-font subset, raster; `docs/english-arm.md`) |
| pipeline floor | — | 0.1422 (chn, full 34) | 0.0266 (eng, 34-font subset) | — |

None of the three preset buckets below fit cleanly — **the answer is language-dependent**,
which the original three options didn't anticipate:

- **Chinese fits none of the three cleanly.** Raster (0.163) and svg (0.117) are both
  well above 0.080 — a ~2x and ~46% gap respectively. Same shape as before this session:
  neither "we were undertrained" nor "it's a metric definition" survives on its own.
- **English lands much closer to the paper under both conventions** — 0.0658/0.0584
  against 0.052, a 15-27% gap, not 2x. And the raster/svg gap barely matters for
  English (0.0074) versus Chinese (0.0455), which the pipeline floor explains
  mechanically: English's floor is 0.0266, Chinese's is 0.1422. The cross-rasterizer
  disagreement that dominates Chinese's raster/svg split is just much smaller in
  English's dataset/rendering pipeline — independent of the model or checkpoint.
- **So: same harness, same eval code, same "official checkpoint, no training here"
  setup — English reproduces close to the paper, Chinese doesn't.** That's a real
  finding for §2.4, not an artifact of one convention choice. Job B (600-epoch
  Chinese retrain, three seeds, landed 2026-08-06) is the direct test of whether more
  Chinese training closes the remaining gap; not yet scored against these numbers.
- **Our own English baseline (0.0597, three seeds, `docs/english-arm.md`) scores
  *better* than the official released checkpoint (0.0651 mean) on the same 34-font
  subset**, and is closer to the paper (0.052) than the official checkpoint is. Unlike
  Chinese, where the official checkpoint essentially equals our from-scratch baseline,
  English training here is not undertrained relative to the release — the remaining
  gap to the paper looks like a protocol/metric question, not a training one, on
  English specifically.

### Why English stopped short (2026-08-06)

English's test set is 1,386 fonts against Chinese's 34 — a 40x larger decode workload
that nobody had sized before starting. At the measured ~43 s/font, a full 3-checkpoint
run projected to ~40 GPU-hours and the in-progress single-checkpoint decode alone hit
27 GB before being stopped partway (a full 1,386-font, 3-checkpoint sweep would have
cost well over 100 GB against a 200 GB quota already sitting near the cap from
project-wide screening history). Scope was cut to:

- **Checkpoint 500**: scored on an 862-font partial decode (62% of the test set,
  stopped and scored as-is) — the closest thing to a full number we have.
- **Checkpoints 500/550/600**: scored on a matched, deterministic 34-font subset
  (`--max_fonts`, added to `options.py`/`test_few_shot.py`/`eval_reconstruction_error.py`
  this session; the test split is unshuffled, so the first N fonts are the same set
  every time) — chosen to mirror the Chinese test set size exactly, for comparability.
- All three checkpoints landed within 0.0013 (raster) / 0.0015 (svg) of each other on
  the subset — statistically flat, same pattern as Chinese's 500/550/600. The subset
  reads optimistic relative to the larger sample by +0.0074 (raster) / +0.0106 (svg)
  (34-font vs 862-font, checkpoint 500) — real but modest, ~11-19% relative. That gap is
  the evidence for whether a small-subset screening protocol is defensible elsewhere in
  this project on time grounds; it is documented here rather than closed by a full run.
- The English pipeline floor (`oracle_eng_subset34.csv`) is also on the 34-font subset,
  not the full set, for the same reason.

**Bottom line: treat the English row as a scoped, reasoned estimate, not the load-bearing
number Job A originally set out to produce.** If cluster time and disk quota allow later,
completing checkpoint 500's decode (resume from font 862; the skip-if-done logic in
`test_few_shot.py` picks up where it left off) is the cheapest way to firm this up.

Whatever the eventual full-scope number says, **the Stage 2 result is unaffected.** E9 is
measured against our own baseline under one fixed convention, at three seeds, with a
paired per-font test. A change in the absolute scale of the metric moves both arms of
that comparison equally.

---

## Recording

```bash
python scripts/build_results_table.py
python scripts/recompute_deltas.py
git add -A && git commit -m "Evaluate official checkpoints under both GT conventions" && git push origin repro
```

`build_results_table.py` does not yet know about the `_gtsvg` suffix. Either teach it
the same way it learned `_n<N>`, or keep the gt-svg rows in a separate small table in
§2.4 and say so. Do not let two scoring conventions land in one column of `RESULTS.csv`
unlabelled.

Then:

- `PROJECT_PLAN.md` §1.2 — the metric section needs the two-rasterizer point, which is
  a property of the metric and belongs there rather than in §2.4
- `PROJECT_PLAN.md` §2.4 — rewrite around whichever branch the table selects. The
  current text argues for training budget from three indirect observations; replace the
  argument with the measurement
- `PROJECT_PLAN.md` §5 — official-checkpoint rows, labelled with their convention
- **Correct the English baseline record.** `Run Log.md` lists
  `dvf_base_exp_eng` at epoch 600 as a trained run. The checkpoint cannot be found and
  its step count matches the released `600_192921.ckpt` exactly, so it was almost
  certainly the released checkpoint all along. **There is no self-trained English
  baseline.** Say so, and drop the row rather than leaving an unverifiable one
- Vault `Runs/Run Log.md`, `_Open Tasks.md`, `_MOC DeepVecFont-v2.md`
