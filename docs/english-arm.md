# The English arm

Staged 2026-08-04. Nothing here launches until the Chinese Tier 3 batch is off the
GPUs. §8 item 4 had English out of scope; days 3 to 5 came free, so it is affordable
again — but it is a much larger budget than Chinese and the size has never been
measured, so this follows §3.3's discipline: **measure, then size, then commit.**

## Why English is not just "the same batch with a different flag"

| | Chinese | English |
|---|---|---|
| `--char_num` | 52 (per the repo default) | 52 |
| `--max_seq_len` | 71 | 51 |
| `--ref_nshot` | 8 | 4 |
| `--ref_char_ids` | `0,1,2,3,26,27,28,29` | `0,1,26,27` (4 ids, must match `ref_nshot`) |
| Epoch budget in use | 151 | 801 in `COMMANDS.md`; the existing baseline checkpoint is epoch 600 |
| Confirmation `--n_samples` | 50 | 10 per §4, 20 in `COMMANDS.md` — **unresolved, see below** |

Shorter sequences and half the reference shots make an English epoch cheaper, and the
budget is four times longer. Those pull in opposite directions and neither has been
timed, so the cost of one English run is genuinely unknown. Guessing it is how a day
of GPU gets spent on a matrix that does not fit.

**Resolve the `n_samples` discrepancy before the confirmation eval, not after.** §4
says 10 to match the paper; `COMMANDS.md` line 126 uses 20. They are not
interchangeable — the `n_samples` ladder in §3.2 showed a systematic best-of-N drop
from 0.1691 at 10 to 0.1678 at 20 on Chinese, which is larger than most candidate
deltas. Whichever is chosen, every English number in the report has to use it.

## Step 1 — time it

```bash
cd ~/deepvecfont-v2 && conda activate dvf_v2
CUDA_VISIBLE_DEVICES=1 python train.py --mode train --name_exp eng_timing \
  --model_name main_model --language eng --max_seq_len 51 --ref_nshot 4 \
  --batch_size 32 --seed 1111 --n_epochs 6 --freq_ckpt 5 --max_ckpt_keep 1
```

Read steady-state s/epoch off epochs 2→5, the same way §3.3 did for Chinese
(excluding startup). Then size against this, which is §3.3's table re-derived for a
budget that is now shared with the combination runs in §3.7:

| Cost of one English run at the chosen budget | What fits |
|---|---|
| ≤ 1.5 h | 3-seed baseline + 3 candidates + the combination |
| 1.5 – 3 h | 3-seed baseline + 2 candidates |
| 3 – 6 h | 3-seed baseline + the combination only |
| > 6 h | Drop English. Say so in §8 item 4 and spend the time on the report |

`rm -rf experiments/eng_timing_main_model` afterwards.

## Step 2 — the 3-seed English baseline, and the cut-off

Three seeds, because an English delta is no more interpretable without an English
noise floor than a Chinese one was. The Chinese floor does not transfer: it is a
property of this dataset, this budget and this metric.

```bash
# EXPERIMENTS in run_experiments.sh, with COMMON_ARGS switched to the English block
"eng_seedfloor_1111 --seed 1111"
"eng_seedfloor_2222 --seed 2222"
"eng_seedfloor_3333 --seed 3333"
```

**Freeze the cut-off from these three curves before looking at any candidate.** The
rule, stated now so it cannot be chosen to suit a result later:

> Take each seed's `checkpoint_metrics.csv`. Find the earliest epoch `E_conv` after
> which the best-so-far `val_metric` improves by less than 0.5% over the following
> 100 epochs. Take the **latest** `E_conv` across the three seeds, round up to the
> next multiple of `freq_ckpt`, add 50. That is the budget, and it is the same
> `--n_epochs` for every English run including the baselines.

Two reasons the rule is written down rather than eyeballed. Different budgets across
runs would confound the comparison outright, since a candidate trained longer is not
one factor away from the baseline. And E14 in particular *cannot* be compared across
budgets at all: `warmup_cosine` derives `total_steps` from `--n_epochs`, so changing
the budget changes the schedule itself.

Re-run the three baselines at the frozen budget if `E_conv + 50` differs from what
they were trained to. That is the honest version and it costs three runs; scoring the
baselines at a checkpoint the candidates never saw is the shortcut, and it puts a
budget difference inside every delta.

## Step 3 — candidates

Which candidates depends on Tier 3 Batch B, which is what the whole Chinese arm has
been building toward. The ordering:

1. **Whatever Batch B promotes.** A candidate with three Chinese seeds and a
   consistent paired sign is the one worth a second language. E9 σ=0.5 and E1 are the
   two leading on rendered L1 (ranks 1 and 2 of 18); E13 bins256 is third.
2. **The combination** from §3.7, if it beats its parts on Chinese.
3. **Not E14**, unless the peak-LR rows change the picture — see below.

English is a *generalization* test, not a second sweep. The claim it supports is
"this held on a second script", which needs two or three candidates carried across,
not eighteen. Running the full matrix again would spend the budget and add nothing
the Chinese arm has not already said.

## On E14 and English

Tempting, and worth resisting for now. E14's Chinese rendered delta was −0.0010 —
rank 10 of 18, and at the 0.0011 decode-noise level — with the worst s-IoU in Tier 2.
The lower val curve is real; what it means is not yet established, and
`scripts/val_metric_correlation.py` plus the `e14_ctrl_gamma_chn` control settle that
on Chinese for free before any English GPU time is spent on it.

There is one honest reason E14 might do better on English than on Chinese, and it
belongs in the report either way: warmup and cosine decay disproportionately help
*short* runs, and English trains 4× longer, so if anything the English budget is
where E14 should matter **less**. If it nonetheless helps more there, that is
evidence the effect is not the schedule-shape story at all.

## Reference rows for the English column

`dvf_base_exp_eng`, checkpoint `600_192921_valloss2.0824.ckpt`, has never been
tested. Score it early — it is a free reference row and it is also the only existing
check that the English data path works end to end.

```bash
CUDA_VISIBLE_DEVICES=1 python test_few_shot.py --mode test --name_exp dvf_base_exp_eng \
  --language eng --max_seq_len 51 --model_name main_model --batch_size 1 \
  --n_samples 3 --ref_nshot 4 --ref_char_ids 0,1,26,27 \
  --name_ckpt 600_192921_valloss2.0824.ckpt
python eval_reconstruction_error.py --exp_dir experiments/dvf_base_exp_eng_main_model \
  --name_ckpt 600_192921_valloss2.0824.ckpt
```

That run predates the `val_metric` fix, so its checkpoint was selected under the
`svg_para`-blind metric. It is a reference row, not a baseline — the 3-seed runs in
step 2 are the baseline.

Paper English rows for the §5 table, from §1.3: DeepSVG 0.125, DeepVecFont 0.056,
DeepVecFont-v2 **0.052**. The §2.4 discussion of the gap to the paper applies here
too and should not be re-litigated per language.

One thing to watch that the Chinese arm did not have to: the paper's English ablation
in Tab. 1 runs 0.0588 → 0.0557 → 0.0529 → 0.0519, so the *whole* published English
ablation spans 0.0069 absolute, and its individual steps are 0.003 or less. If the
English noise floor from step 2 comes out anywhere near the Chinese 0.0093, it will
be wider than the entire effect range the paper reports for English — which is worth
stating plainly in §6 rather than discovering at the end.
