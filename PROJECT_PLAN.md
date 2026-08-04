# DeepVecFont-v2 — Project Plan

Final project, Generative Models for Text and Images, Reichman University.
Paper: Wang, Wang, Yu, Zhu, Lian, *DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality*, CVPR 2023.
Fork of `yizhiwang96/deepvecfont-v2`. All work on branch `repro`; `main` stays a pristine upstream mirror.

**Submission: Saturday 15 August 2026.** Day 1 is Monday 3 August. Thirteen days.

This document covers both graded stages. It absorbs and replaces `archive/STAGE2_EXPERIMENTS.md` and carries forward the Stage 1 material from `archive/FLOW_MATCHING_PLAN.md`. Those two files stay in `archive/` for provenance: the flow-matching plan is cited directly in the report's future-work paragraph.

---

## 0. Where things stand

| | |
|---|---|
| Chinese baseline | `dvf_base_exp_chn_main_model`, trained from scratch, checkpoints at epochs 100 and 125 (`125_5040_valloss3.8273.ckpt`) |
| English baseline | `600_192921_valloss2.0824.ckpt` available |
| Few-shot test runs | Chinese, epochs 100 and 125, `n_samples 50`, `ref_nshot 8` |
| Best measured number | Chinese Error (L1) **0.1668**, mean IoU 0.2550, over 34 fonts and 1768 glyphs, at epoch 125 |
| Paper's Chinese number | **0.080** |
| Paper metric implemented | Yes, `eval_reconstruction_error.py` |
| Missing for Stage 1 | SSIM, quantization oracle (renderability, per-font CSV and the bin histogram landed) |
| Stage 2, Tier 1 | Run and screened 2026-08-04. Nothing cleared the noise floor; see §3.2 |
| Stage 2, Tier 2 | Run and screened 2026-08-04. Nothing cleared the noise floor either; see §3.5. E8's s-IoU shift was flagged as the one follow-up worth budget, then **closed 2026-08-04** without a GPU: it sits at the s-IoU seed floor, not above it |
| Stage 2, Tier 3 | Coded and staged 2026-08-04. Sixteen runs in two batches — breadth at seed 1111, plus the three largest measured deltas at two more seeds each. Not yet launched; see §3.6 and `docs/tier3-launch.md` |
| Seed-noise floor | L1 spread **0.0093** (re-measured 2026-08-04). Decomposed 2026-08-04: decode noise 0.0011 (12%), seed variance ~0.0082 (88%) dominates — see §3.2. **s-IoU spread 0.0401** (2026-08-04), roughly three times noisier than L1 in relative terms — see §3.5 |

0.1668 sits essentially on DeepSVG's published Chinese number (0.167), so the baseline is inside the benchmark's range even though it does not reach the paper. §2.4 says how to write that up. It is not a blocker for Stage 2, which is measured against your own baseline rather than against 0.080.

Infrastructure, as of 2026-08-04: `--seed`, wandb mirroring alongside TensorboardX, `--max_ckpt_keep`, a `checkpoint_metrics.csv` manifest replacing the filename-embedded val loss, and all of Tier 1's and Tier 2's flags wired and covered by `check_infra.py` (127 checks). Details in `docs/infra-upgrade.md`.

---

## 1. Shared context

### 1.1 The architecture, in the terms the report will use

Two encoders read the reference glyphs. A CNN reads the reference glyph *images*; a 12-block self-attention stack reads the reference glyph *SVG sequences*. `ModalityFusion` merges them into a VAE latent, and two decoders consume it: an image decoder producing a 64×64 raster, and an autoregressive transformer decoder producing the drawing commands. A second, shallower decoder then refines the sequence in one parallel pass with full context.

The paper's headline contribution is the **relaxation representation**. Every drawing command carries eight coordinate arguments (four control points, two axes each) whether or not the command type uses all of them, which is what lets a single uniform head serve moves, lines and curves. `cmd_args_mask` selects the live slots per command type: all eight for `CurveFromTo`, indices 0, 1, 6, 7 for `MoveFromTo` and `LineFromTo`, none for `EOS`.

The loss has six terms: image L1 and VGG perceptual, KL on the latent, cross-entropy on the command type, cross-entropy on the quantized arguments, Bézier alignment (Eq. 9), and relaxation consistency (Eq. 10). Both decoders contribute their own command and argument terms, appearing in the code as `loss_dict['svg']` and `loss_dict['svg_para']`.

Two facts about the code that shape everything downstream:

**The argument head is shared between the two decoders.** `self.args_fcn = nn.Linear(512, 8 * 128)` is instantiated once and called from both `Transformer_decoder.forward` (the autoregressive decoder) and `Transformer_decoder.parallel_decoder` (the refinement decoder). `command_fcn` is shared the same way.

**The refinement decoder produces what actually gets scored.** `test_few_shot.py` selects candidates on `syn_{i}_{sample}_refined.svg`, which comes from `sampled_svg_2`, and the merge HTML that `eval_reconstruction_error.py` reads is built from those refined files. Any change that stops short of the refinement pass does not reach the reported number.

### 1.2 The metric, and what it can and cannot see

The paper reports one number, "Error" (Sec. 4.1): the average L1 distance between the rasterized synthesized glyph and the ground-truth glyph image at 64×64. Both sides are binary masks, so the mean absolute difference is the fraction of the 4096 pixels that disagree. `Error = 0.080` reads as "8.0% of pixels are wrong".

It is a rasterized proxy for vector fidelity. It says nothing about command count, self-intersection, or control-point placement, only whether the inked region lands in the right place. Two further properties belong in the report:

1. It is scored after best-of-N_s selection, so it measures the best candidate rather than the average sample.
2. Sub-pixel coordinate accuracy is largely invisible to it. This caps how much any coordinate-level improvement can show up, and the discussion section should say so rather than let a small delta look like a small idea.

### 1.3 Paper numbers

| Model | Error-EN ↓ | Error-CN ↓ |
|---|---|---|
| DeepSVG | 0.125 | 0.167 |
| DeepVecFont | 0.056 | 0.086 |
| DeepVecFont-v2 | 0.052 | 0.080 |

Ablation (Tab. 1, English): base 0.0588 → +relaxation 0.0557 → +Bézier alignment 0.0529 → +self-refinement 0.0519.

Note the scale. The paper's entire margin over its own predecessor on Chinese is **0.006 absolute**. Every Stage 2 candidate will produce a delta in that neighbourhood or smaller, and the whole protocol in §3.2 exists because of that one number.

### 1.4 Paper versus code

Four places where the released code departs from the paper text. All four are legitimate reconstruction findings, and three of them became experiments.

| Where | Paper says | Code does | Becomes |
|---|---|---|---|
| Sec. 3.1 | one-hot arguments of **256** dimensions (`δp ∈ R^256×8`, `W_args^b ∈ R^dE×256`) | quantizes to **128** bins; the stale comment on `args_fcn` still reads `# shape: bs, max_len, 8, 256` | E13 |
| Sec. 3.3 | self-refinement is "a **2-layer** Transformer decoder" | `decoder_layers_parallel = clones(DecoderLayer(...), 1)` | E3 |
| Eq. 11 | weights `L_bézier` at **1.0** | `loss_w_aux = 0.01`, a factor of 100 below; and `L_img` is split into `loss_w_l1 = 10` and `loss_w_pt_c = 0.01` rather than a single 1.0 | E7 |
| Sec. 4.1 | the `N(0,I)` perturbation is an **inference-time** device simulating "the feature distortion caused by the human-designing uncertainty" | `x = x + torch.randn_like(x)` at `models/transformers.py:450`, applied unconditionally in training, validation and test | E9 |

A fifth item is not a paper discrepancy but belongs in the same discussion. In `Transformer.forward` the loop unpacks `cross_attn, cross_ff, self_attns` and uses only `self_attns`. The cross-attention modules and `self.latents = nn.Parameter(torch.randn(256, 512))` are constructed, handed to the optimizer, and never called. With `depth=6` and `self_per_cross_attn=2` the encoder is 12 self-attention blocks and zero cross-attention blocks, so the model is not the Perceiver its configuration block advertises. The authors left a comment marking this as known.

### 1.5 Known issues in the harness

These are not experiments. They are things that will produce wrong numbers if left alone, and the first two need resolving before any Stage 2 measurement.

**The eval script and the results layout were out of sync. Fixed 2026-08-02.** `test_few_shot.py` writes to `experiments/<name_exp>/results/<name_ckpt>/<font_idx>/svgs_merge/`, while `eval_reconstruction_error.py` globbed `<exp_dir>/results/*` and expected each match to be a font directory. Against the per-checkpoint layout that glob returned the checkpoint directory, one level too shallow, and every font was skipped. The script also had no `--name_ckpt` argument although `COMMANDS.md` documents one.

The script now detects both layouts, accepts `--name_ckpt` to disambiguate when several checkpoints are present, and errors loudly rather than silently scoring nothing. It was tested against synthetic fixtures covering both layouts, ambiguous and missing checkpoints included, but not yet against real output.

**The open question is what the 0.1668 was measured on.** That figure was produced on 2026-08-01, the same day the per-checkpoint commit landed. Running the fixed script prints the detected layout on its first line, which answers it. A `flat` result means the number came from a tree with no record of which checkpoint produced it, and `test_few_shot.py` should be re-run to attribute it. See `docs/cluster-session.md` step 7.

**Resolved (2026-08-03).** `python eval_reconstruction_error.py --exp_dir experiments/dvf_base_exp_chn_main_model` (no `--name_ckpt`) prints `Layout: per-checkpoint | checkpoint: 135_5440_valloss3.8618.ckpt`. Per-checkpoint means the 0.1668 figure is attributable as-is, no re-run needed — the harness bug in §2.4's candidate list is ruled out. Note the 135-checkpoint tree (the only one still on disk; 100/125 were pruned by `max_ckpt_keep`) scores L1=0.1641, mean IOU=0.2467, renderability 33/34 fonts (font 0028 failed to render, 0 svgs). That's a different checkpoint than the recorded 0.1668/0.2550 at epoch 125, not a regression — expected epoch-to-epoch noise, consistent with §2.4's "still falling val loss" point.

**`val_metric` is not comparable across candidates that change the loss.** `compute_val_loss` builds it as `loss_w_l1 · img_l1 + loss_w_pt_c · vggpt + svg_total`, and the checkpoint filename embeds it. Any candidate that alters the cross-entropy itself changes what the number means, and `prune_checkpoints` will then be selecting on a different quantity. E8 and E13 are the two affected candidates; §3.2 handles them.

**Fixed (2026-08-03): `val_metric` was also missing the refinement decoder's loss entirely, independent of the E8/E13 issue above.** `test_few_shot.py` scores `sampled_svg_2`, which comes from the parallel/refinement decoder (`loss_dict['svg_para']`), not the teacher-forced sequential one (`loss_dict['svg']`). `compute_val_loss` declared `loss_val['svg_para']` accumulators but its accumulate/average loops only ever iterated `['img', 'svg']`, so `svg_para` stayed at its zero initializer forever and never reached `val_metric`. Checkpoint selection was therefore blind to the exact decoder pass that produces the scored SVGs. `val_metric` now is `loss_w_l1 · img_l1 + loss_w_pt_c · vggpt + svg_total + svg_para_total`. This changes its scale, so it is not comparable to any `val_metric`/embedded-filename number recorded before this date (the seed-floor and baseline figures elsewhere in this doc are L1/s-IoU from rendered output, not `val_metric`, so they are unaffected).

Checkpoint filenames also no longer embed the metric (`{epoch}_{step}.ckpt`, not `..._valloss{x}.ckpt`). Every checkpoint save now appends a row (`epoch, step, checkpoint, val_metric`, and the l1/vggpt/svg/svg_para breakdown) to `experiments/<name>/logs/checkpoint_metrics.csv`, and `prune_checkpoints` / `scripts/test_experiments.sh` select the best checkpoint from that manifest instead of parsing a filename. `scripts/best_checkpoint.py` (via `checkpoint_log.py`) falls back to the legacy filename-embedded score for experiment dirs trained before this fix (baseline, the three seedfloor runs), so those don't need retraining. See `checkpoint_log.py`.

**`.gitignore` has `experiments/` with a trailing slash**, which does not match a symlink. `COMMANDS.md` describes the entry as slash-less. Both `data` and `experiments` are symlinks on the cluster, repointed to `/data/bens/deepvecfont-v2` on 2026-08-03, so the current pattern will not ignore them. One character, worth fixing before the first `git add`. `COMMANDS.md` line 10 still describes the old `~/gpufs` target and needs the same correction.

**Renderability is silently rewarded.** `test_few_shot.py` wraps `render()` in a bare `except: continue`, and `eval_reconstruction_error.py` then skips any font whose merge HTML lacks exactly `2 × char_num` SVGs. A model that fails on hard glyphs currently gets those fonts dropped from its average. §2.2 turns this into a reported quantity.

**There are two `numericalize` / `denumericalize` pairs, on different grids.** This one is new, it was not in either archived plan, and it may change what E13 is worth.

| Definition | Default | Reached from |
|---|---|---|
| `models/transformers.py:596` | `n=128` | the model path. `model_main.py` does `from .transformers import *`, so every model-time call resolves here |
| `data_utils/relax_rep.py:6` | `n=64` | preprocessing only, inside `cal_aux_bezier_pts` |

The preprocessing copy is used in one place, `relax_rep.py:27`:

```python
stroke_seq = char_seq[k]                                   # a numpy view, not a copy
stroke_seq[4:] = denumericalize(numericalize(stroke_seq[4:]))   # round-trip at n=64
```

Because `stroke_seq` is a view into `char_seq` and thence into `font_seq`, that assignment writes back through the view. The function's stated job is computing auxiliary Bézier supervision points, but the round-trip mutates the sequence array itself as a side effect. Whether that mutation reached the persisted training sequences depends on what the preprocessing driver does with `font_seq` afterwards, which is a question about a dataset that is already built.

If it did, the training data sits on a 64-bin grid while the head predicts over 128 bins, and half the bins are structurally unreachable. That would explain part of the gap in §2.4 on its own, and it would make E13 a waste of a Tier 2 slot. The bin histogram in §2.3 settles it in a few minutes: look for a comb pattern. Do that before committing to E13.

**Measured (2026-08-03).** `scripts/bin_histogram.py` reports neither a clean comb nor a clean fill: **38.5% of mass in odd bins for Chinese, 36.0% for English** (30/32 and 32/32 odd bins occupied respectively). The script's own verdict was `PARTIAL comb ... Investigate before scoping E13`, which `docs/cluster-session.md` step 4 didn't anticipate — it only has instructions for the two clean cases.

**Resolved (2026-08-03), by reading the preprocessing driver rather than the histogram.** The mutation never reaches disk. In `relax_rep.relax_rep.process` the two statements are in this order:

```python
np.save(os.path.join(font_dir, 'sequence_relaxed.npy'), ret.reshape(opts.n_chars, -1))
pts_aux = cal_aux_bezier_pts(ret, opts)      # mutates `ret` in place, afterwards
```

`sequence_relaxed.npy` is written **before** `cal_aux_bezier_pts` is called, and `ret` is never saved again. The in-place round-trip through the n=64 grid therefore corrupts only the in-memory array that `cal_aux_bezier_pts` is about to consume for its own auxiliary points, and `dataloader.py:37` loads `sequence_relaxed.npy`, the untouched file. `cal_aux_bezier_pts` has no other caller in the repo.

Two consequences:

1. **The training sequences are at full float resolution**, not on a 64-bin grid. The 128-bin head is modelling data that genuinely carries information below 64-bin resolution, so E13's first gate passes. It remains gated on the oracle: doubling the bins is only worth a Tier 2 slot if the §2.3 oracle floor turns out to be a material fraction of the gap, and §1.2 caps how much any sub-pixel change can show up in a rasterized L1 at 64×64.
2. **A "partial comb" was never a reachable outcome.** Either every persisted sequence went through the n=64 round-trip or none did, because one code path writes all of them. The 38.5% figure is not evidence of partial corruption; it is the natural shape of font coordinate distributions, which cluster on round design-grid values and so favour even bins without being confined to them. `bin_histogram.py`'s `odd_frac < 0.4` threshold is what manufactured the PARTIAL verdict, and it is arbitrary. The discriminating test is `odd_frac ≈ 0` versus `odd_frac > 0`, and 30/32 odd bins occupied answers it.

The histogram still earns its place in the report as the figure showing the mass at bins 0 and 127, which is the `.clip(min=0, max=n-1)` and `padding_idx=0` point in §2.3. It just does not decide E13.

---

## 2. Stage 1 — reproduction plus metrics

Stage 1 is close to done. What remains is the measuring instrument, which is both a graded deliverable in its own right and the thing that makes every Stage 2 comparison legible.

### 2.1 Done

Chinese trained from scratch, few-shot tested at two checkpoints, and the paper's Error metric implemented and run. English has a checkpoint at epoch 600 and no test run yet.

### 2.2 What to add to `eval_reconstruction_error.py`

All of these run on the same rendered 64×64 mask pair the script already produces, so each is one function.

**SSIM.** The metric from class. Gaussian window, `data_range=1.0`. It complements L1 rather than duplicating it: L1 counts disagreeing pixels wherever they fall, SSIM penalizes disagreement that breaks local structure. A glyph with a uniformly slightly-too-thick stem and a glyph with one mangled stroke can share an L1 score and separate cleanly on SSIM.

**IoU, relabelled.** Already computed. Call it s-IoU when comparing to the vector-font literature. DualVector (CVPR 2023) reports SSIM / L1 / s-IoU on this task, so the triple is the field's convention and using it makes the results table directly comparable.

**Renderability rate.** Log the fraction of glyphs that fail to render and the fraction of fonts skipped, per candidate. If two candidates differ here, their Error values are not comparable, and with a sweep of fifteen variants this will bite at some point.

**Per-font CSV.** The script already accumulates per-font means; write them out. With 34 Chinese test fonts you can then run a **paired Wilcoxon signed-rank test** on per-font Error, candidate against baseline. Given expected deltas near 0.005, comparing two grand means is not evidence, and saying so in the report is worth marks on its own.

Optional, only if the schedule loosens: FID on rendered glyph images. Defensible for English (1425 test fonts × 52 glyphs ≈ 74k images). For Chinese it is 34 × 52 = 1768 images, far below where FID is stable, so either skip it or report it with an explicit caveat about small-sample bias.

### 2.3 Two one-off measurements

**Bin histogram.** Histogram `numericalize` output over the ground-truth training data and count the mass in bins 0 and 127, for Chinese and for English. `numericalize` does `.clip(min=0, max=n-1)`, so any coordinate outside `[0, 30]` is destroyed. Separately, `SVGEmbedding.arg_embed = nn.Embedding(128, 128, padding_idx=0)` freezes bin 0's embedding row (at its kaiming init value, not at zero — see the correction in §3.5), so a coordinate at the lower boundary is truncated on the way out and unlearnable on the way in. One script, trivially checkable, and it makes a good figure.

This histogram also answers the open question in §1.5 about the two quantization grids, which is the thing that actually decides E13. Plot the full 128-bin histogram and look at its shape:

- A **comb**, with every odd bin empty, means the training sequences were round-tripped through the 64-bin grid during preprocessing. The 128-bin head is then modelling data that carries no information below the 64-bin resolution, and E13 is pointless without raising the preprocessing `n` and rebuilding the dataset, which is a multi-hour job on a 5 GB archive.
- A **filled** histogram means the two grids never met, the model genuinely operates at 128 bins, and E13 is worth its Tier 2 slot.

Run this before anything else in §2.3. It is a few minutes of work and it either keeps or kills an experiment.

**Quantization oracle.** Push ground-truth sequences through `numericalize` then `denumericalize`, render, and score with the identical pipeline. This is the floor no model with the current head can beat. Use whichever grid the histogram says is actually in force, and report which:

| Grid | Bin width (viewBox units) | At 64×64 | Rounding error per coordinate |
|---|---|---|---|
| n = 128 (`models/transformers.py`) | `30/128 = 0.2344` | 0.625 px | uniform on ±0.3125 px |
| n = 64 (`data_utils/relax_rep.py`) | `30/64 = 0.4688` | 1.25 px | uniform on ±0.625 px |

One pass over the test set, and it becomes the reference row that tells you what fraction of your gap to 0.080 is even addressable by a coordinate-level change. At the 64-bin grid that fraction is roughly four times larger than at 128, so the answer matters.

### 2.4 The gap to 0.080, and how to write about it

Your 0.1668 against the paper's 0.080 is roughly double. **This is not a crisis and it does not need solving.** The assignment allows reporting that the paper's numbers were not reproduced, and 0.1668 sits almost exactly on DeepSVG's published Chinese result of 0.167, so the model is landing inside the benchmark's own range rather than somewhere unexplainable. Report the number, say what plausibly accounts for it, and move on.

What the report should list, without spending days chasing any of it:

1. **Undertrained.** Epoch 125 against whatever budget the paper used. The val loss was still falling. This is the most likely single cause and the cheapest to test: score epoch 100 and epoch 125 and read the direction.
2. **Different test protocol.** `ref_char_ids`, `n_samples`, and the font list all move this number, and the paper does not fully specify them.
3. **Harness bug.** See §1.5. If the eval script scored a stale results tree, the number means something other than what it says.
4. **Coarser quantization than assumed.** Also §1.5. If the training sequences sit on the 64-bin grid, the oracle floor is roughly four times the rounding error the 128-bin arithmetic predicts, and some fraction of the gap was never winnable.

Of these, only the last two cost anything to check, and both are already on the day 1 list for other reasons: the results-glob question has to be settled before any Stage 2 comparison is trustworthy, and the bin histogram has to run before E13 can be scoped. Neither is being run to explain the gap. The explanation falls out for free.

A reproduction that lands at 0.1668 with a paragraph of honest accounting is a fine Stage 1. What matters for the grade is that the measurement setup is sound from here on, because Stage 2 is a comparison against your own baseline, not against 0.080.

### 2.5 Stage 1 closes when

The extended eval script runs on the existing Chinese results and emits Error, SSIM, s-IoU, renderability, and a per-font CSV; the oracle row and bin histogram exist; and §2.4 has an answer. It can then be written up while Stage 2 trains.

---

## 3. Stage 2 — a controlled sweep

### 3.1 Why a sweep rather than one large change

The assignment asks for one meaningful architectural change and for a report covering "what worked, what did not, and what you learned". A single large bet answers the first and risks having nothing to say for the second. A controlled sweep of single-factor modifications, each drawn from a category the assignment lists, each measured against a seed-noise floor, with the winners combined into the submitted improved model, answers both by construction. It also removes the failure mode where the one idea does not land and there is nothing left to write.

The flow-matching coordinate head in `archive/FLOW_MATCHING_PLAN.md` was the alternative. It was scoped, costed, and rejected on schedule grounds: a single change touching the loss, the sampler, the AR feedback path and the refinement decoder at once, with one shot at getting it right, does not fit thirteen days. That decision is itself reportable, and the report's future-work paragraph writes itself from that document.

### 3.2 The protocol matters more than the experiment list

**Measure the seed-noise floor before anything else.** Until today `train.py` called `setup_seed(1111)` with nothing varying it. Run the **baseline three times, seeds 1111 / 2222 / 3333**, at the screening budget. The spread of the screening metric across those three runs is your resolution limit. Any candidate whose improvement falls inside it is not a result, and reporting it as one is the most likely way to lose marks in the discussion. This costs three screening runs and it is the highest-value item in this document. Launch it on day 1, in parallel with the §2 metric work.

**Measured (2026-08-03).** All three seeds trained 150 epochs, `scripts/run_experiments.sh parallel`, one GPU each. Scored with `scripts/test_experiments.sh` at the screening budget (`--n_samples 3`, all 34 fonts — see the caveat in that script about the 8-font subset not being wired up), on each seed's **best-val-loss checkpoint** (`prune_checkpoints` in `train.py` keeps that one specifically, alongside the latest — for seed 3333 that's epoch 100, not the final epoch 150):

| Seed | Checkpoint | L1 | s-IoU |
|---|---|---|---|
| 1111 | `150_6040_valloss4.0166.ckpt` | 0.1734 | 0.2253 |
| 2222 | `150_6040_valloss3.7663.ckpt` | 0.1657 | 0.2474 |
| 3333 | `100_4040_valloss4.0104.ckpt` | 0.1694 | 0.1913 |

**Seed-noise floor: L1 spread = 0.0077.** Any Stage 2 candidate's improvement over this baseline needs to clear ~0.008 in L1 to be a result rather than noise. All three fonts sets rendered fully (34/34 fonts, 1768/1768 glyphs, 0 skipped).

**Re-measured (2026-08-04), after the `val_metric` fix below.** `compute_val_loss` was missing the refinement-decoder loss (`svg_para`) from `val_metric` entirely — see the fixed note further down this section. Because that bug predates every run trained so far, it could have affected which checkpoint `prune_checkpoints` kept as "best" for *any* of them, baseline included, not just Stage-2 candidates. The three seed-floor runs and the three E7 runs were re-trained from scratch under the fix (old dirs preserved at `experiments/archive_pre_metricfix/`, not deleted) and re-screened with `scripts/test_experiments.sh`:

| Seed | Checkpoint | L1 | s-IoU |
|---|---|---|---|
| 1111 | `150_6040.ckpt` | 0.1724 | 0.2154 |
| 2222 | `150_6040.ckpt` | 0.1631 | 0.2410 |
| 3333 | `150_6040.ckpt` | 0.1680 | 0.2555 |

**New spread: L1 = 0.0093 — larger than the 0.008 bar, not smaller.** Confirmed cause, not guessed: comparing `logs/checkpoint_metrics.csv` (new, corrected `val_metric`) against the archived runs' filenames (old, `svg_para`-blind `val_metric`) shows the fix changed which checkpoint was selected as best for seed 3333 specifically — old picked epoch 100 (`100_4040_valloss4.0104.ckpt`, lower on the broken metric), new picks epoch 150 (`val_metric` 5.894603 at epoch 150 vs 5.905838 at epoch 100, a difference of 0.011 that only appears once `svg_para` is counted). Seeds 1111 and 2222 picked epoch 150 under *both* the old and new metric, yet their L1 still moved by ~0.001-0.003 on re-run — `test_few_shot.py` does stochastic best-of-`n_samples` decoding, so even a fixed checkpoint isn't bit-for-bit reproducible across two screening runs. The 0.0016 spread increase is therefore a mix of both effects, not attributable to either alone, but the checkpoint-selection change for seed 3333 is the one piece that's directly verifiable from the manifest rather than inferred.

**Consequence for Tier 1 (§3.4): re-screened under the fix, nothing clears the new bar.** E7 (`loss_w_aux` ∈ {0.1, 0.3, 1.0}), E9 (`enc_noise_std_train` ∈ {0, 0.1, 0.25, 0.5}), E10 (`dropout` ∈ {0.1, 0.2}) and E1 (`enc_final_norm`) were all re-trained and re-screened in the same batch. Every candidate's delta from the seed-1111 baseline falls within ±0.006 L1 — smaller than the 0.0093 noise floor itself:

| Candidate | L1 | s-IoU | delta vs seed 1111 |
|---|---|---|---|
| E7 `loss_w_aux=0.1` | 0.1679 | 0.2263 | −0.0045 |
| E7 `loss_w_aux=0.3` | 0.1702 | 0.2307 | −0.0022 |
| E7 `loss_w_aux=1.0` | 0.1746 | 0.2234 | +0.0022 |
| E9 σ=0 | 0.1740 | 0.2470 | +0.0016 |
| E9 σ=0.1 | 0.1696 | 0.2273 | −0.0028 |
| E9 σ=0.25 | 0.1720 | 0.2538 | −0.0004 |
| E9 σ=0.5 | 0.1665 | 0.2522 | −0.0059 |
| E10 dropout=0.1 | 0.1724 | 0.2376 | +0.0000 |
| E10 dropout=0.2 | 0.1722 | 0.2081 | −0.0002 |
| E1 `enc_final_norm` | 0.1670 | 0.2101 | −0.0054 |

Renderability was 100% (1768/1768) for every row, so nothing here is confounded by the renderability caveat in §1.5. **This is itself the Tier 1 result: none of E7/E9/E10/E1 individually clear a noise floor that grew past the bar meant to screen them.** Two live implications for §3.5-§3.7 and the report: (1) don't spend more budget re-running Tier 1 single-factor points expecting a different sign, the effect sizes here are smaller than what this screening setup can resolve; (2) either the screening budget (`n_samples 3`, 60-epoch option in §3.3) needs to grow to shrink the noise floor below 0.008, or the bar itself needs revisiting — both are §8 decisions, not something to silently paper over in the results table.

**The floor is not all seed noise, and the decomposition is now the priority (2026-08-04).** Two things vary between any two screening numbers: the training seed, and the eval decode. `test_few_shot.py` never calls `setup_seed`, and the σ=1.0 encoder perturbation is the only source of stochasticity at inference, so best-of-`n_samples` picks a different candidate on every run. The re-measurement above already caught this — seeds 1111 and 2222 selected the same checkpoint under both the old and the new `val_metric`, and their L1 still moved by 0.001–0.003.

The two components have different remedies, which is why separating them is worth doing before spending more training budget:

| Dominant component | Remedy | Cost |
|---|---|---|
| Decode noise | Raise `--n_samples` for screening, re-screen Tier 1 at the new budget | Eval only, no retraining |
| Seed noise | Run each candidate at 3 seeds | 3× the training matrix, and Tier 3 pays for it |

`scripts/eval_noise.sh` measures the decode component directly, by re-screening one fixed checkpoint several times: the weights never change, so the spread it reports is decode noise alone. It also runs an `n_samples` ladder at 10 and 20 to show how fast that component shrinks. Costs no training GPU; it runs alongside the Tier 2 batch. Record the result here as a dated **Measured** paragraph, and only then move `NOISE_FLOOR` in `scripts/test_experiments.sh` off 0.0093.

**Measured (2026-08-04), alongside the Tier 2 launch.** `scripts/eval_noise.sh seedfloor_1111_chn 0`, three reps of the fixed `150_6040.ckpt` checkpoint at `n_samples 3`: L1 = 0.1722 / 0.1717 / 0.1711, spread **0.0011**. The `n_samples` ladder (one run each): 0.1691 at 10, 0.1678 at 20 — both a systematic best-of-N drop, not noise, and not comparable to the n=3 column.

**Decode noise is 0.0011 against a 0.0093 seed-noise floor — about 12% of it.** The remaining ~0.0082 is training-seed variance. This settles the branch: raising `--n_samples` for screening would buy almost nothing, since decode noise was never the dominant term. Resolving effects in the size class Tier 1 and Tier 2 are chasing needs each candidate run at multiple seeds, which is a 3× training matrix — the harder, more expensive branch, and now the confirmed one rather than a guess. That is a §8 decision (item 3), and it now has a number behind it instead of an open question.

The report needs this either way. "Which candidates fell inside the seed noise" (§6, discussion) is a much weaker sentence than an account of what the noise was made of.

**Checked (2026-08-04): the wandb val curves and the rendered metric disagree, and only about E14.** The `val_metric` curves show `warmup_cosine` consistently below baseline, with E1 and E9 σ=0.5 also looking favourable. Ranking all eighteen Tier 1 and Tier 2 candidates on the rendered screening metric:

| Rank | Candidate | rendered L1 | delta | s-IoU delta |
|---|---|---|---|---|
| 1 | E9 σ=0.5 | 0.1665 | −0.0059 | +0.0368 |
| 2 | E1 `enc_final_norm` | 0.1670 | −0.0054 | −0.0053 |
| 3 | E13 `n_args_bins=256` | 0.1675 | −0.0050 | +0.0050 |
| … | | | | |
| 10 | E14 `warmup_cosine` | 0.1715 | −0.0010 | **−0.0178** |

**E1 and E9 σ=0.5 are confirmed** as the two leading candidates on the metric that gets reported, which is what Batch B already runs at three seeds. **E14 is not.** Its delta of −0.0010 sits at the 0.0011 decode-noise level, so it is indistinguishable from baseline on the rendered metric, and its s-IoU is the worst of any Tier 2 row.

The disagreement is worth more than either number alone, because `val_metric` is by far the **lower-variance instrument**: it is deterministic given a checkpoint, computed over the whole validation set, and logged at every checkpoint, where the rendered metric carries 0.0011 of decode noise from unseeded best-of-N decoding and is measured once. A consistent separation in a val curve genuinely is stronger evidence than a single rendered point — *provided the two agree about which checkpoint is better*. That is §3.2's rank-correlation question, still open, and it now has a concrete reason to be answered: `scripts/val_metric_correlation.py`, no GPU.

There is a specific mechanism that would make E14 the one candidate where the proxy misleads. `warmup_cosine` ends at 0.05× the base lr; the released `ExponentialLR(0.997)` ends at 0.997¹⁵⁰ = 0.635×. That is a **12.7× difference in terminal step size**, and late-training validation loss falls as the lr decays and the weights stop bouncing around the minimum, whether or not autoregressive rollout improves. `val_metric` is teacher-forced; the reported metric is an autoregressive rollout through the refinement decoder, and teacher forcing cannot see exposure bias. `--lr_gamma` was added 2026-08-04 so the control exists: an `exp` run at γ = 0.05^(1/150) = 0.98023 lands on the same terminal lr and isolates the schedule *shape* from plain annealing.

E14 also bundles two factors — warmup and decay shape — which breaks the one-at-a-time rule the rest of the sweep follows. Both are separable with existing flags: `--lr_min_factor 1.0` collapses the cosine to a constant and gives warmup only, `--lr_warmup_steps 1` gives cosine only. The E14-deep block in `scripts/run_experiments.sh` stages all of this, gated on the correlation result.

**Audited 2026-08-04: which other candidates could fall through the same gap.** E14 prompted a sweep of every candidate against what `val_metric` actually measures. It is, from `train.py`:

```
val_metric = loss_w_l1*img_l1 + loss_w_pt_c*vggpt + svg['total'] + svg_para['total']
svg['total'] = loss_w_cmd*cmd + loss_w_args*args + loss_w_aux*aux + loss_w_smt*smt
```

Three distinct failure modes fall out, and each hits different candidates.

**(a) Loss-scale changers — `val_metric` not comparable across runs at all.** E8 and E13 were already known and excluded. **E7 was not, and should have been:** `loss_w_aux` is a weight *inside the sum that val_metric is*, appearing twice (via `svg` and `svg_para`), and the sweep spans 0.01 → 1.0, a 100× reweighting. E7 was screened on the rendered metric so no reported number is wrong, but any reading of E7's val curve against baseline is meaningless, and it was never flagged in the way E8 and E13 were. All three are now excluded by default in `scripts/val_metric_correlation.py`.

Checkpoint *selection* is unaffected for all three: `prune_checkpoints` compares checkpoints within one run, where a constant scale factor cancels. Only cross-run comparison breaks.

Past runs cannot be rescaled retrospectively — `checkpoint_metrics.csv` recorded `val_svg_total` but not the aux term separately. `val_svg_aux` and `val_svg_para_aux` were added to the manifest 2026-08-04, with the append path guarded so a run in flight writing under the old header does not get its columns shifted.

**(b) Settling-confounded — comparable in scale, but lower val loss need not mean better rollout.** E14 is the known case. Two more are in Tier 3 and running now:

| Candidate | Why it is in this class |
|---|---|
| E14 `warmup_cosine` | Ends at 0.05× lr against `ExponentialLR(0.997)`'s 0.635×, a 12.7× gap in terminal step size |
| **E15 weight EMA** | Averaging weights lowers validation loss essentially by construction — that is what an EMA does. The highest-risk row in Batch A |
| **E11 AdamW, wd 0.01** | Weight decay shrinks weights: part genuine regularization, part settling |

These stay in the correlation and are reported as residuals, because whether they diverge *is* the finding. Read E15's val curve against its rendered number specifically before promoting it.

**(c) `val_metric` is blind to what the candidate changes.** The training loss carries `kl_beta * kl`; `val_metric` does not include the KL term at all. This is the same class of omission as the `svg_para` gap found and fixed on 2026-08-04, and it is still present. It is defensible as a *choice* — KL does not affect rendered output directly — but it has consequences worth stating: **E12's** checkpoint selection is blind to the very quantity E12 varies, and `val_metric` is therefore not the training objective, which the section above implicitly assumed when it was fixed. Document it as a deliberate choice or fix it; leaving it undecided is how the `svg_para` bug survived as long as it did.

**The dropped experiment.** §3.4 said of E9: *"Sweep [test-time σ] separately, second, on the winning train σ,"* and §4 scheduled it for days 6–8. **It never ran.** Tier 2 launched without it and Tier 3 was scoped without it, and E9 σ_train=0.5 is the rank-1 candidate of all eighteen. It matters for three reasons: σ_test is applied at eval regardless of σ_train, so every E9 row was trained at its own σ and then validated *and tested* at σ=1.0 — for the σ_train=0 row that is a full train/test mismatch and a plausible reason it was the worst E9 row; σ_test governs how much the `n_samples` candidates differ from each other, so it trades directly against best-of-N and has an optimum nobody has looked for; and it needs **no retraining at all**, since `models/transformers.py` parses opts at import and `--enc_noise_std_test` on the test command line reaches the encoder. `scripts/sigma_test_sweep.sh` runs it, eval-only, alongside a training batch. Run it on the E9 winner *and* the baseline: if the optimum is the same for both, it is a property of the eval procedure and shifts the whole results table rather than promoting one candidate.

**Screen cheap, confirm expensive.** Do not run `test_few_shot.py --n_samples 50` over all 34 test fonts for every candidate.

- *Screening eval*: 8 fixed test fonts, `--n_samples 3`, at a fixed epoch. Minutes.
- *Confirmation eval*: all 34 fonts, `--n_samples 50`, matching the paper. Finalists only.

Freeze the screening font list and `ref_char_ids` before the first run and never change them.

**Check whether `val_metric` predicts the test metric.** *(Script written 2026-08-04: `scripts/val_metric_correlation.py`. Costs no GPU, runs against the existing Tier 1 and Tier 2 experiment dirs, and gates the E14-deep batch — see the E14 note above.)* `compute_val_loss` produces `val_metric` free at every checkpoint, and it now also goes to wandb as `CKPT/val_metric`. It is a teacher-forced loss; the test metric is a rendered, autoregressively decoded, best-of-N L1. How well they correlate on this model is unknown. After the first six experiments, compute the **rank correlation between `val_metric` and screening Error** across those six. High correlation means you screen everything else for free and spend the saved time on more candidates. Low correlation means the validation loss is not a usable proxy, which is a reportable finding in itself, and you screen on the rendered metric from then on.

The caveat from §1.5 applies: **E8 and E13 change the cross-entropy itself**, so those two are always screened on the rendered metric regardless of what the correlation says.

**One factor at a time, then combine.** Every run differs from the baseline in exactly one flag. Combine winners only at the end, and verify the combination beats each part alone. Interactions are real here: dropout, KL weight and encoder noise are all regularizers pulling on the same slack, and stacking three of them will likely underperform the best one.

### 3.3 Calibrate the budget before sizing the matrix

Time five epochs of the Chinese baseline and extrapolate. The checkpoint name `125_5040` implies about 40 steps per epoch, so a 150-epoch Chinese run is roughly 6,000 steps. What that costs in wall-clock depends on throughput nobody has measured yet.

| Measured cost of one 150-epoch Chinese run | Matrix to run |
|---|---|
| ≤ 3 h | Full: everything in §3.4 to §3.6, all sweep points |
| 3–8 h | Tiers 1 and 2, single sweep points, seeds on the baseline only |
| > 8 h | Tier 1 only, and drop the screening budget to 60 epochs |

Relative ordering between variants usually shows up well before convergence, so a **60-epoch screening budget with a 150-epoch confirmation for finalists** is a legitimate way to double the candidate count. Validate that assumption once: take one Tier 1 candidate, score it at 60 and at 150, and check the sign of the delta agrees.

**Measured (2026-08-03), from the three seed-floor runs' checkpoint timestamps, epoch 25→50 steady state (excludes one-time startup):** 22.8–24.7 s/epoch across the three seeds (seed 2222 slowest). Extrapolated to 150 epochs: **~1.0 h per run**, on an RTX 3090. That puts this in the **≤ 3 h** bucket: full matrix, everything in §3.4 to §3.6.

`COMMANDS.md` shows `CUDA_VISIBLE_DEVICES` 1 and 2 in use, so at least three GPUs are in play. Run three candidates concurrently.

### 3.4 Tier 1 — run first

Highest expected value per GPU-hour. Together these need one flag flip and two one-line code changes.

**E9 — Encoder noise scale.** *Maps to: change the noise schedule.*

At `models/transformers.py:450`, at the end of the sequence encoder's forward pass:

```python
x = x + torch.randn_like(x)   # add a perturbation
```

Unit-variance Gaussian noise, added unconditionally, in training and validation and test. The scale σ=1 appears arbitrary and has never been tuned. It is applied to a 512-dimensional residual stream whose scale is unknown (see E1), so the effective perturbation could be 20% or 200%.

Wire `--enc_noise_std_train` and `--enc_noise_std_test`, already declared in `options.py`. Sweep train σ ∈ {0, 0.1, 0.25, 0.5, 1.0} with test σ held at 1.0.

Keep test-time σ above zero. It is the *only* source of stochasticity at inference, and therefore the only reason the N_s candidates differ from each other at all. Setting it to zero makes `--n_samples 50` produce fifty identical glyphs. Sweep it separately, second, on the winning train σ.

Optional extra: anneal train σ from 1.0 to 0.1 over training. That is a noise schedule in the literal sense the assignment means.

This is the biggest untuned knob in the model and it is a one-line change.

**E1 — Final LayerNorm on the sequence encoder.** *Maps to: add normalization layers.*

Both transformer stacks are pre-norm: `PreNorm` in the encoder, `SublayerConnection` applying `norm` before the sublayer in the decoder. A pre-norm stack needs a terminal LayerNorm after the last block, otherwise the residual stream leaves the encoder unnormalized and its scale grows with depth.

The decoder has this, as `decoder_norm` and `decoder_norm_parallel`. **The encoder does not.** `Transformer.forward` runs its 12 self-attention blocks and goes straight to the noise injection with no final norm, and `att_residual` has the same gap.

Add `nn.LayerNorm(512)` at the end of both. One line each. It is a real architectural omission, it is the assignment's first bullet, and it interacts with E9 in a principled way: normalizing the stream is what makes σ a meaningful quantity rather than an arbitrary one. Run E1 alone, E9 alone, and E1+E9 together.

**E7 — Bézier alignment loss weight.** *Maps to: modify the loss function.*

Eq. 11 sets the weight of `L_bézier` to 1.0; `options.py` sets `loss_w_aux = 0.01`. The paper's own ablation credits this loss with 0.0028 of its total 0.0069 gain, the second-largest single contribution.

Sweep `loss_w_aux` ∈ {0.01, 0.1, 0.3, 1.0}. The flag already exists, so this costs no code at all.

Frame it honestly in the report: it is simultaneously a reconstruction-fidelity correction and a Stage 2 loss modification. Saying so is better than pretending it is only one of them.

**E10 — Dropout.** *Maps to: add regularization.*

Every dropout in the model is zero: `MultiHeadedAttention(dropout=0.0)`, `PositionwiseFeedForward(dropout=0.0)`, `attn_dropout=0.`, `ff_dropout=0.`. The Chinese training set is 212 fonts expanded 10× by affine augmentation, which is small for a 512-wide 12-block encoder plus a 6-layer decoder.

Wire the `--dropout` flag, already declared, and sweep {0, 0.1, 0.2}. Note the overlap with E9: both are regularizers, so run E10 at the baseline noise σ and re-check the winner jointly at the end.

### 3.5 Tier 2 — small structural diffs

Good ideas, slightly more code. §8 asks you which of these to keep.

**Coded and staged 2026-08-04.** All four are wired behind flags whose defaults reproduce the released behaviour exactly, on the same discipline as Tier 1 (§7.2), so the Tier 1 table and the seed floor stay valid references. `check_infra.py` section 7 asserts each flag reaches the model or the optimizer; 127 checks pass. The batch is eight runs at seed 1111, three GPUs, three waves, ~3 h. Runbook: `docs/tier2-launch.md`.

| Run | Flag | Candidate |
|---|---|---|
| `e8_ls05_chn` / `e8_ls10_chn` / `e8_ls20_chn` | `--args_label_smooth_sigma` ∈ {0.5, 1.0, 2.0} | E8 |
| `e13_bins256_chn` | `--n_args_bins 256` | E13, bin count |
| `e13_nopad_chn` | `--arg_embed_pad_idx False` | E13, `padding_idx` |
| `e3_refine2_chn` / `e3_refine3_chn` | `--n_layers_refine` ∈ {2, 3} | E3 |
| `e14_wucos_chn` | `--lr_schedule warmup_cosine` | E14 |

Two deviations from what this section originally specified, both deliberate:

1. **E13 is split into two runs** rather than bundling the bin count with `padding_idx`. They are independent factors and the protocol is one at a time.
2. **E13's oracle gate is skipped.** It runs after the batch and explains the result rather than licensing it. The alternative was holding a training slot idle behind a script that had not been written.

Read the Tier 1 outcome before reading these results: every candidate there landed inside a floor that grew past the bar meant to screen it. Tier 2's effect sizes are in the same neighbourhood, so the realistic expectation is "recorded, not claimed", and the value of E3 and E13 to the report is partly independent of their sign — both are paper-versus-code corrections from §1.4.

**Measured (2026-08-04).** All eight trained (seed 1111, 150 epochs) and screened in the same session as a re-screened `seedfloor_1111_chn` baseline (`scripts/test_experiments.sh parallel`, GPUs 1-2, `n_samples 3`, all 34 fonts):

| Run | Checkpoint | L1 | s-IoU | delta vs baseline | clears 0.0093 |
|---|---|---|---|---|---|
| Baseline, seed 1111 (this session) | `150_6040.ckpt` | 0.1725 | 0.2165 | — | — |
| E8 `sigma=0.5` | `150_6040.ckpt` | 0.1727 | 0.2571 | +0.0002 | no |
| E8 `sigma=1.0` | `150_6040.ckpt` | 0.1729 | 0.2510 | +0.0004 | no |
| E8 `sigma=2.0` | `125_5040.ckpt` | 0.1720 | 0.2574 | −0.0005 | no |
| E13 `n_args_bins=256` | `150_6040.ckpt` | 0.1675 | 0.2215 | −0.0050 | no |
| E13 `arg_embed_pad_idx=False` | `150_6040.ckpt` | 0.1706 | 0.2462 | −0.0019 | no |
| E3 `n_layers_refine=2` | `150_6040.ckpt` | 0.1705 | 0.2187 | −0.0020 | no |
| E3 `n_layers_refine=3` | `150_6040.ckpt` | 0.1715 | 0.2288 | −0.0010 | no |
| E14 `warmup_cosine` | `150_6040.ckpt` | 0.1715 | 0.1987 | −0.0010 | no |

Renderability was 100% (1768/1768, 34/34 fonts) for every row, so nothing here is confounded by the renderability caveat in §1.5, same as Tier 1. **None of the eight clear the 0.0093 floor** — consistent with, and reinforcing, the Tier 1 result. E13 at 256 bins is the largest single delta (−0.0050) but is still roughly half the floor.

One result worth carrying into the discussion section regardless of the floor: **all three E8 rows move s-IoU by +0.03 to +0.04 over baseline (0.2165 → 0.251-0.257) while L1 barely moves (±0.0005).** That is a much larger, consistent shift on a different metric than the one the floor was measured on. It is plausibly real rather than noise — three sigmas, one direction, an effect size four to eight times any L1 delta measured in either tier — but it has not been checked against a seed-noise floor for s-IoU specifically, so it should be reported as an observation, not a claim, until it is. If time allows, it is the single most promising follow-up in this tier: measure the s-IoU seed-noise floor (three seeds, baseline only, s-IoU column) and see if E8 clears it there even though it doesn't on L1. §1.2's point about L1 being largely blind to sub-pixel accuracy is one plausible mechanism — ordinal label smoothing softens the argument head's near-miss errors, which could plausibly tighten structural overlap (s-IoU) without changing how many pixels disagree (L1) by a comparable amount.

Note `e8_ls20_chn` selected `125_5040.ckpt` as its best-val checkpoint rather than `150_6040.ckpt` like every other row — `best_checkpoint.py` picked it from the manifest, not hand-chosen; it is why that row's checkpoint column differs.

**Measured (2026-08-04), and it closes the E8 follow-up without a GPU.** The paragraph above proposed measuring an s-IoU seed-noise floor before deciding whether E8's shift is real. That measurement does not need new runs: the three seed-floor runs were already screened with an s-IoU column, and it was sitting in the re-measured table further up this section.

| Seed | L1 | s-IoU |
|---|---|---|
| 1111 | 0.1724 | 0.2154 |
| 2222 | 0.1631 | 0.2410 |
| 3333 | 0.1680 | 0.2555 |

**s-IoU seed-noise floor: spread 0.0401.** E8's shift is +0.034 to +0.041 over the 0.2165 baseline — at the floor, not above it. So the one Tier 2 observation flagged as "plausibly real rather than noise" does not survive being measured against the right bar, and the three-seed E8 follow-up that §3.5 called the most promising use of remaining budget is not worth running.

Two things are worth keeping from it anyway. First, s-IoU is a **four times noisier** metric than L1 in relative terms here — 0.0401 on a baseline of ~0.23 is a 17% relative spread, against 0.0093 on ~0.168, which is 5.5%. Any result reported on s-IoU needs that stated, and the §5 results table should carry the s-IoU floor next to the L1 one rather than only the latter. Second, the mechanism proposed for E8 — label smoothing tightening structural overlap without moving pixel disagreement — is still a coherent story; it is just not one this screening setup can evidence. That belongs in the discussion as a hypothesis with its measurement cost stated, not as a finding.

This is also a small methodological lesson worth a sentence in §6: the floor was measured on one metric and then used to judge candidates on a second, which is how the E8 observation came to look stronger than it was.

**E8 — Ordinal label smoothing on the argument head.** *Maps to: modify the loss function, add regularization.*

The 128-way cross-entropy over quantized coordinates is permutation-invariant in the bin index: predicting bin 5 when the target is 60 costs exactly what predicting bin 61 costs. The head has no notion that coordinates live on a line, which is why the Bézier and smoothness losses have to reach back through a temperature-0.1 softmax and a straight-through estimator to recover geometry.

Replace the one-hot target `F.one_hot(tgt_args, 128)` in `Transformer.loss` with a discretized Gaussian centred on the true bin, σ ≈ 1–2 bins, renormalized. Five lines. No architecture change, no inference change, no extra cost.

This is the cheap version of the argument the flow-matching plan was built on, and it is the most interesting idea here from a modelling standpoint. Sweep σ ∈ {0.5, 1.0, 2.0} bins. Screen on the rendered metric, since it changes the loss scale.

**E13 — 256 quantization bins, and drop `padding_idx`.** *Maps to: change the encoder or decoder.*

Sec. 3.1 specifies 256; the model path uses 128. Doubling the bins halves the ±0.3125 px rounding error from §2.3.

Touch points: `numericalize(n=128)` and `denumericalize(n=128)` in `models/transformers.py`, `SVGEmbedding.arg_embed`, `args_fcn` output width, the `reshape(N, S, 8, 128)` calls, and `F.one_hot(tgt_args, 128)`. Five constants, all findable by grepping `128`. Careful: `arg_embed` is `nn.Embedding(128, 128)`, where the two 128s mean different things, vocabulary and embedding width. Only the first changes.

While there, drop `padding_idx=0` from `arg_embed`, or shift arguments by +1. Bin 0 is a legitimate coordinate and its embedding never trains.

**Corrected 2026-08-04.** This document said elsewhere that bin 0 "gets a frozen zero embedding". It does not. `nn.Embedding(..., padding_idx=0)` zeroes row 0 at construction, but `SVGEmbedding._init_embeddings` then runs `kaiming_normal_` over the whole weight and overwrites that zero. `padding_idx` survives only as a zero gradient, so row 0 ends up frozen at a *random* kaiming vector. The defect is the same size either way — a legitimate coordinate whose representation is fixed at initialization — but the report should describe it accurately, and "frozen at noise" is a slightly worse failure than "frozen at zero".

**Gated twice. The first gate is cleared (2026-08-03).** The n=64 preprocessing grid never reached the persisted sequences: `relax_rep` saves `sequence_relaxed.npy` before `cal_aux_bezier_pts` mutates its argument, and the dataloader reads that file. See §1.5. The training data is at full resolution, so adding bins to the head adds resolution the data actually has.

**The second gate is the oracle**, still unrun: E13 is worth a Tier 2 slot only if the §2.3 quantization floor turns out to be a material fraction of the gap. Temper the expectation with §1.2 — the metric is a rasterized L1 at 64×64 and is largely blind to sub-pixel coordinate accuracy, which is exactly what halving the rounding error buys. Screen on the rendered metric.

**E3 — Self-refinement decoder, 1 layer → 2.** *Maps to: add residual or attention layers, change the decoder.*

Sec. 3.3 states the refinement module "is actually a 2-layer Transformer decoder". The code has `clones(DecoderLayer(...), 1)`. Change `1` to `2`. One character.

It earns its slot because, per §1.1, the refinement decoder produces the output that is actually scored. Try 3 layers while you are there.

**E14 — Warmup and cosine LR.** *Maps to: etc.*

`ExponentialLR(gamma=0.997)` stepped per epoch. Over 150 epochs that is `0.997^150 = 0.64`, so the learning rate barely moves and the schedule is effectively constant at the budget you are training to. There is no warmup, on a 6-layer transformer decoder with Adam at 2e-4.

Add linear warmup over the first ~500 steps and cosine decay to the epoch budget. Standard transformer practice, and it disproportionately helps short runs, which is what your entire matrix consists of. If it wins, apply it to every subsequent run and say so in the protocol section.

### 3.6 Tier 3 — if the budget allows

**Coded and staged 2026-08-04.** Same discipline as Tiers 1 and 2 (§7.2): every new flag's default reproduces the released behaviour, so the Tier 1 table, the Tier 2 table and the seed floor all stay valid references. `check_infra.py` section 8 asserts each flag reaches the model or the optimizer, and asserts the defaults jointly. Runbook: `docs/tier3-launch.md`. New code: `models/norms.py` (E2), `WeightEMA` in `train.py` (E15), `z_proj` in `models/modality_fusion.py` (E5), `scripts/dead_params.py` (E6).

**Scope decision: sixteen runs in two batches, not seven single points.** §8 item 2 framed Tier 3 as competing with multi-seed replication for the same GPU time. Two GPUs at ~1 h per run makes that a false choice at this batch size, so the tier runs as both:

| Batch | Runs | What it is | Expected outcome |
|---|---|---|---|
| A | 10 | Tier 3 breadth, seed 1111, one factor each | Null, like Tiers 1 and 2. Reportable as coverage of the assignment's change categories |
| B | 6 | The three largest measured deltas at seeds 2222 and 3333 | A mean against a mean, which is interpretable whichever way it lands |

Batch A: E12 `kl_beta` ∈ {0, 0.1}, E11 AdamW at `weight_decay` 0.01, E2 `img_norm` ∈ {group, batch, instance}, E4 `ngf` 32, E15 `ema_decay` 0.999, E5 `bottleneck_bits` ∈ {256, 1024}.

Batch B: E9 σ=0.5 (−0.0059), E1 `enc_final_norm` (−0.0054), E13 `n_args_bins=256` (−0.0050), each at two additional seeds. Those three are the largest deltas measured across both tiers and are indistinguishable from each other at this resolution, which is why picking one to deepen would have been arbitrary. §3.2's decomposition put ~88% of the 0.0093 floor in seed variance, so this is the only lever that shrinks the bar, and Batch B is the half to protect if the night is cut short.

**Batch B is not read against the 0.0093 floor.** That number is the baseline's own spread across seeds and is the quantity being replaced. Read it as baseline mean and spread against candidate mean and spread, plus the paired per-seed difference, since the seeds are matched. A candidate whose three per-seed differences share a sign earns the confirmation eval even if the means overlap — a statement neither earlier tier could support.

**E6 is an audit, not a run, and the reason is worth stating.** The dead Perceiver parameters are constructed before several live modules, and every `nn.Linear` and `nn.Parameter` construction draws from the global RNG stream, so deleting them shifts the initialization of everything built afterwards. A "dead parameters removed" run therefore differs from baseline by an effective seed change, and the seed floor exceeds any effect in play, so it could not be read in either direction. `python scripts/dead_params.py` produces the count for §1.4 instead.

**E4 is a capacity control, not a like-for-like factor.** `ngf` 16 → 32 doubles both image stacks, so unlike every other row in the batch it changes parameter count. Report it as the capacity axis rather than folding it in with the architectural factors.

| ID | Change | Maps to | Note |
|---|---|---|---|
| E5 | `bottleneck_bits` ∈ {128, 256, 512, 1024} | change the latent dimension | Gotcha below |
| E2 | Image encoder/decoder norm: spatial `LayerNorm([C,H,W])` → `GroupNorm` / `BatchNorm2d` / `InstanceNorm2d` | add normalization layers | The current norm couples channel and spatial statistics, discarding per-channel scale |
| E12 | `kl_beta` ∈ {0, 0.003, 0.01, 0.03, 0.1} | add regularization | Interacts with E9: with σ=1 additive noise on the encoder output, the reparameterization is nearly decorative |
| E11 | `Adam` → `AdamW`, `weight_decay` 0 → 0.01 | add regularization | `AdamW` is already imported in `train.py` and unused |
| E4 | `ngf` 16 → 32 | change the encoder or decoder | Widens both image encoder and decoder; check `fc_fusion` still matches |
| E15 | EMA of weights for evaluation | etc. | Reliable small gain, ~15 lines, no effect on training dynamics |
| E6 | Delete or wire the dead Perceiver cross-attention path | change the encoder | See §1.4 |

**E5 gotcha.** `--bottleneck_bits` looks like a free flag and is not. `ModalityFusion` does `seq_feat_[:, 0] = z`, and `seq_feat_` has last dimension 512 fixed by `fc_merge = nn.Linear(seq_latent_dim * ref_nshot, 512)`. Any `bottleneck_bits ≠ 512` fails there. Add a `nn.Linear(bottleneck_bits, 512)` projection before the assignment. Small, but it is a code change rather than a flag flip, so budget for it.

**Resolved 2026-08-04.** `z_proj` added, constructed only when `bottleneck_bits != 512` so the default adds no module and no state_dict key. Only the sequence-side slot is projected; the image decoder keeps receiving the unprojected latent, since its `input_nc` is `bottleneck_bits + char_num` and already tracks the flag. Worth noting for the report that the released code has therefore never run at any other latent width — the flag was declared, documented, and unusable.

**E6 note.** Deleting the dead parameters is a clean finding for the reconstruction section and slightly reduces optimizer state. Wiring the cross-attention to the image features is a real architecture change and belongs here only if everything else is done.

### 3.7 Combination

Take the winners that cleared the seed-noise floor, run them together, and run each ablation-of-the-combination so the report can say which parts survive contact with each other. Then the confirmation eval, 34 fonts at `n_samples 50`, on the baseline and on the combined model, with the paired Wilcoxon.

---

## 4. Schedule

Dated against a 15 August deadline. Day 1 is Monday 3 August.

| Date | Day | Work |
|---|---|---|
| Mon 3 Aug | 1 | Sync the cluster, GPU dry run (§7.4). Time 5 epochs, size the matrix (§3.3). Launch the 3-seed baseline (§3.2). Bin histogram (§2.3). Resolve the §1.5 eval-script question. |
| Mon 3 – Tue 4 | 1–2 | Extend `eval_reconstruction_error.py`: SSIM, renderability, per-font CSV. Quantization oracle. Rescore the epoch-100 and epoch-125 Chinese results. Answer §2.4. **Stage 1 closes.** |
| Wed 5 – Fri 7 | 3–5 | Tier 1: E9 sweep, E1, E7 sweep, E10 sweep. Three GPUs in parallel. |
| Fri 7 Aug | 5 | Compute the `val_metric` vs rendered-Error rank correlation (§3.2). Decide the screening metric for everything after this point. |
| Sat 8 – Mon 10 | 6–8 | Tier 2, subject to §8: E8, E13, E3, E14. Plus the test-time σ sweep on the E9 winner. |
| Tue 11 Aug | 9 | Tier 3, as many as fit. |

**Actual, as of 2026-08-04 (day 2).** Ahead of the table above, not behind it. Tier 1 launched on day 1 rather than day 3, and finished and re-ran under the `val_metric` fix by day 2 morning. Tier 2 is coded and staged on day 2 rather than day 6. Days 3 to 5 are therefore free, which is what buys back the report time §4 flags as tight, and it removes the §8 pressure to drop Tier 3.

What is *behind*: Stage 1 has not closed. SSIM and the quantization oracle are still unwritten, and §2.4 has not been answered in prose. Both are eval-only, neither needs a GPU day, and the oracle now has a second reason to exist — it explains the E13 result. Do them while Tier 2 trains.
| Wed 12 – Thu 13 | 10–11 | Combine winners (§3.7). Confirmation eval on baseline and combined. Paired Wilcoxon. |
| Thu 13 – Fri 14 | 11–12 | Report. |
| Sat 15 Aug | 13 | Buffer and submit. |

**Where the compression bites.** The original fourteen-day sketch had two clear days for the report and a full buffer day. Thirteen days leaves about a day and a half of writing with the buffer folded into submission day, and days 11 and 13 are double-booked. Two ways to buy that back, both in §8: drop Tier 3 entirely, or start writing the reconstruction section on day 2 when Stage 1 closes rather than at the end. The second is close to free, because §2 produces every number that section needs.

English is out of scope unless days 9 to 11 come in early. If it fits, run only the combined model against the existing English baseline, with the same confirmation protocol and `--n_samples 10` to match the paper.

---

## 5. Results table

One row per candidate, so the sweep itself is the evidence.

**Not filled in yet, on purpose.** This table is confirmation-budget numbers (`n_samples
50`, all 34 fonts) for finalists only, per §3.2's "screen cheap, confirm expensive" split.
Tier 1's screening-budget numbers (`n_samples 3`) are in §3.2's "Re-measured (2026-08-04)"
block instead, and none of E7/E9/E10/E1 cleared the noise floor there — so none of them
are finalists yet, and promoting one to a confirmation run isn't justified by what's
measured so far. Fill this table once §3.5-§3.8 produce a candidate that does.

| Row | Error ↓ | SSIM ↑ | s-IoU ↑ | Render % | Wilcoxon p |
|---|---|---|---|---|---|
| Paper, reported (CN) | 0.080 | — | — | — | — |
| Quantization oracle (floor) | | | | | — |
| **Reconstruction** (baseline, seed 1111) | 0.1668* | | 0.2550* | | — |
| Baseline, seed 2222 | | | | | — |
| Baseline, seed 3333 | | | | | — |
| E1 final encoder LayerNorm | | | | | |
| E7 `loss_w_aux = 0.3` | | | | | |
| E9 train σ = 0.25 | | | | | |
| E10 dropout = 0.1 | | | | | |
| … one row per candidate … | | | | | |
| **Improved model** (combined winners) | | | | | |

\* epoch 125, pending the §1.5 check and the §2.4 explanation.

The three baseline seed rows are what license every claim below them. Put them in the table, not in a footnote.

---

## 6. Report structure

1. **Original architecture.** §1.1. Explain the relaxation representation properly, since it is the paper's headline contribution and it is what makes each command carry eight coordinate arguments.
2. **Paper results.** §1.3.
3. **Reconstruction results.** Your Chinese numbers against 0.080, with SSIM, s-IoU and renderability, plus the §2.4 account of the gap. Include the paper-versus-code deviations from §1.4; they are legitimate reconstruction findings and three of them became experiments.
4. **Improved architecture.** Present the method as a controlled sweep. State the assignment categories, the one-factor-at-a-time rule, the seed-noise floor, and the screening/confirmation split, then describe the winners and why each was expected to help.
5. **Improved results.** §5, with the Wilcoxon column and the oracle floor row.
6. **Discussion.** What the metric can and cannot see (§1.2). Which candidates fell inside the seed noise and therefore prove nothing. Whether `val_metric` turned out to predict rendered Error. What the quantization floor implies about the ceiling on any coordinate-level change. One paragraph on the continuous-coordinate head that was scoped and rejected on schedule grounds, citing `archive/FLOW_MATCHING_PLAN.md`.
7. **References.** §9.

---

## 7. Infrastructure

### 7.1 What changed

Full change log in `docs/infra-upgrade.md`. In summary: `--seed` replaces the hardcoded `setup_seed(1111)`; wandb mirrors every TensorboardX scalar, metrics only; `--max_ckpt_keep` defaults to 1; and `--enc_noise_std_train`, `--enc_noise_std_test`, `--dropout` are declared with behaviour-preserving defaults, ready for E9 and E10 to wire.

### 7.2 Why the flags are declared before they are wired

Every run dumps `vars(opts)` to `opts.txt` and now to the wandb config. Declaring the Stage 2 flags on day 0, with defaults that reproduce current behaviour exactly, means the baseline runs and the candidate runs carry the same config schema. Comparing runs in the wandb UI then works without special-casing, and there is no gap in the record where a flag existed for some runs but not others.

### 7.3 Storage budget

`data/` and `experiments/` are symlinked to `/data/bens/deepvecfont-v2` (2026-08-03). **Quota: 200 GB.**

| Item | Size |
|---|---|
| Dataset, unzipped | ~30 GB |
| One checkpoint | 1.2 GB |
| Confirmation eval (34 fonts, `n_samples 50`) | ~1 GB of SVGs and PNGs |
| Screening eval (8 fonts, `n_samples 3`) | ~15 MB |

That leaves roughly 170 GB. The sweep is about 33 training runs: 3 seed-floor, 11 in Tier 1 after removing the sweep points that are just the baseline, 7 in Tier 2, 7 in Tier 3, 5 for the combination and its ablations.

| `--max_ckpt_keep` | Files per run | GB per run | 33 runs |
|---|---|---|---|
| 1 | best + latest = 2 | 2.4 | 79 GB |
| 2 | 3 | 3.6 | 119 GB |
| 3 | 4 | 4.8 | 158 GB |

So `1` fits comfortably, `2` fits, and `3` does not once the dataset and results are counted. That is the constraint as things stand.

**Most of a checkpoint is optimizer state, and most of that is dead weight.** `train.py` saves `{'model': ..., 'opt': optimizer.state_dict(), ...}`. Adam keeps two moment tensors per trainable parameter, so the optimizer entry is roughly twice the size of the model entry. Working back from 1.2 GB, the trainable parameters are around 380 MB and Adam's state is around 760 MB.

`test_few_shot.py:22` reads only `torch.load(path_ckpt)['model']`. Every checkpoint retained for *evaluation* therefore carries about 760 MB that nothing will ever read. Only the newest checkpoint needs the optimizer, and only for `--resume`.

Saving the optimizer state for the latest checkpoint alone and writing the retained best ones model-only changes the arithmetic:

| `--max_ckpt_keep` | GB per run | 33 runs |
|---|---|---|
| 1 | 1.64 | 54 GB |
| 2 | 2.08 | 69 GB |
| 3 | 2.52 | 83 GB |

Keeping *three* checkpoints per run would then cost less than keeping one does today. That matters beyond disk: §3.3 wants one candidate scored at both 60 and 150 epochs to validate the screening budget, and plotting the rendered metric across a training curve needs several snapshots per run. Right now those are expensive; they would stop being so.

**Decision 2026-08-03: deferred.** 79 GB at `--max_ckpt_keep 1` fits, so this is an optimization rather than a blocker, and it adds moving parts to the retention logic at the point where the seed-floor runs are about to launch. Revisit if the matrix grows past Tier 2, if Tier 3 expands, or if epoch-wise metric curves turn out to be wanted for the report. Until then, run at `1` and reap losers as you go.

Verify the decomposition on the cluster before relying on it:

```bash
python -c "
import torch; c = torch.load('experiments/dvf_base_exp_chn_main_model/checkpoints/125_5040_valloss3.8273.ckpt', map_location='cpu')
for k in c: print(k, sum(v.numel()*v.element_size() for v in c[k].values())/2**30 if hasattr(c[k],'values') else c[k])
"
```

Housekeeping either way: delete the checkpoints of any candidate that has been screened and dropped, and delete screening eval trees once their numbers are in the tracker. Neither is needed again.

### 7.4 The dry-run ladder

Three rungs, cheapest first. Do not skip a rung.

1. **Local, no GPU.** `python scripts/check_infra.py`. Verifies flag names, types and defaults; verifies `--wandb False` actually disables; statically verifies the `train.py` wiring, the checkpoint-metric manifest, and that every Tier 1 and Tier 2 flag reaches the model or the optimizer; runs a live wandb init/log/finish in offline mode. **127 checks**, seconds. Passing as of 2026-08-04.
2. **Cluster, GPU, 2 epochs.** The smoke test in `docs/infra-upgrade.md`, and for Tier 2 the four in `docs/tier2-launch.md` §1. Confirms a run appears in wandb with config and scalars, and that a checkpoint is written, logged to the manifest and pruned. Candidates that change tensor shapes or `state_dict` keys (E13, E3) need this rung; loss-only and optimizer-only ones (E8, E14) cannot fail on load.
3. **Cluster, full.** The batch itself.

---

## 8. Decisions still open

These are yours. The plan does not commit to them.

1. ~~**Tier 2 scope.**~~ **Resolved 2026-08-04: all four, one seed each, eight runs.** ~1 h per run puts this in §3.3's ≤3 h band, so nothing needed cutting. Breadth over depth for the first pass — deepen whichever candidate leads rather than guessing which one deserves three seeds up front. E13 runs without waiting on its oracle gate; see §3.5.
2. ~~**Tier 3 at all.**~~ **Resolved 2026-08-04: both, in one batch.** The item framed Tier 3 as competing with multi-seed replication for the same slot. At two GPUs and ~1 h per run, sixteen runs is one overnight, so the tier runs as Batch A (breadth, 10 runs) plus Batch B (the three largest deltas at two more seeds each, 6 runs). See §3.6. Batch B is the half to protect if the night is cut short, because it is the one that yields an interpretable number under a null.
3. ~~**The screening bar itself.**~~ **Resolved and acted on 2026-08-04.** `eval_noise.sh` puts decode noise at 0.0011 and seed noise at ~0.0082 of the 0.0093 floor, so raising `n_samples` will not shrink the bar and multiple seeds is the only lever. Acted on as Batch B in §3.6: three candidates × two additional seeds, chosen as the three largest measured deltas rather than one arbitrary leader, since at this resolution they are indistinguishable from each other. Separately, the **s-IoU** floor is now measured at 0.0401 from the existing seed-floor runs (§3.5) — s-IoU is roughly three times noisier than L1 in relative terms, and the §5 table needs both floors, not just the L1 one.
4. **English. Back in scope 2026-08-04, sized rather than committed.** A second language column and a stronger reconstruction section, against roughly a day. The cost of one English run has never been measured and the two relevant factors pull opposite ways — shorter sequences and half the reference shots make an epoch cheaper, while the budget is 4× longer — so this follows §3.3: time it, then pick the matrix from the measured cost. Plan and cut-off rule in `docs/english-arm.md`. Two things it resolves that are open right now: the `--n_samples` discrepancy for English confirmation (§4 says 10, `COMMANDS.md` uses 20, and the ladder in §3.2 showed that gap is larger than most candidate deltas), and the fact that the English epoch budget must be **frozen from the baseline curve before any candidate is looked at**, since `warmup_cosine` derives its schedule from `--n_epochs` and cannot be compared across budgets at all.

   Scope it as a generalization test, not a second sweep: carry across whatever Batch B promotes plus the §3.7 combination, two or three candidates, not eighteen. Note also that the paper's entire published English ablation spans 0.0069 absolute with individual steps of 0.003 or less, so if the English noise floor lands anywhere near the Chinese 0.0093 it will be wider than the whole effect range being chased.
5. ~~**Screening budget.**~~ **Resolved 2026-08-03: 150 epochs everywhere.** Superseded in part by item 3 — the open question is no longer epochs but `n_samples`, and §3.2's decomposition answers it.
6. ~~**Fix the §1.5 eval-script issue.**~~ **Resolved 2026-08-03**, layout is per-checkpoint and the script handles both.
7. **What `--max_ckpt_keep` means for the sweep.** Keeping one checkpoint per run is right for disk, but if you later want to score a candidate at both 60 and 150 epochs you need both. Consider 2 for the runs that feed §3.3's validation.

8. **Does `val_metric` omitting the KL term stay a choice or become a fix?** New 2026-08-04. The training loss is `... + kl_beta * kl`; `val_metric` is the same sum without it. Defensible, since KL does not affect rendered output directly, but it means `val_metric` is not the training objective and that E12's checkpoint selection is blind to exactly what E12 varies. The `svg_para` omission in the same function was treated as a bug and fixed; this one has never been decided either way, which is how that one survived. Decide it, in one sentence, and put the sentence in §1.5.

9. **What to do with the σ_test result.** New 2026-08-04. `scripts/sigma_test_sweep.sh` is eval-only and answers a question §3.4 raised and §4 scheduled. If the optimum σ_test differs from the released 1.0 by more than the 0.0011 decode noise, every number in the results table was taken at an arbitrary point on that curve and the screening comparisons need re-running at the better value. That is cheap for the eval but it invalidates the table as a *set*, so decide before running whether a shifted optimum triggers a re-screen or gets reported as a caveat.

---

## 9. Do this first

1. Sync the cluster and run rung 1 then rung 2 of §7.4.
2. Resolve the §1.5 eval-script question. Nothing downstream is trustworthy until you know what the 0.1668 was measured on.
3. Time five epochs. Pick the matrix size from §3.3.
4. Launch the 3-seed baseline. Nothing in §3 is interpretable until it finishes.
5. **Run the bin histogram first**, before the oracle. It costs minutes, it answers the two-grid question in §1.5, and it decides whether E13 stays on the list. Then run the oracle on whichever grid it says is in force.
6. Extend `eval_reconstruction_error.py` with SSIM, renderability and per-font CSV, rescore the existing Chinese results, and answer §2.4. Stage 1 is then done and can be written up while Stage 2 runs.
7. Wire `--enc_noise_std_*` and `--dropout` into the model, and add the two `nn.LayerNorm(512)` lines for E1. That is the entire code diff for all of Tier 1 except E7, which needs none.

---

## 10. References

- Wang, Wang, Yu, Zhu, Lian. *DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality.* CVPR 2023.
- Wang, Lian. *DeepVecFont: Synthesizing High-Quality Vector Fonts via Dual-Modality Learning.* ACM TOG 2021.
- Carlier, Danelljan, Alahi, Timofte. *DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation.* NeurIPS 2020.
- Lopes, Ha, Eck, Shlens. *A Learned Representation for Scalable Vector Graphics (SVG-VAE).* ICCV 2019.
- Liu, Guo, Wang, Zhang. *DualVector: Unsupervised Vector Font Synthesis with Dual-Part Representation.* CVPR 2023. Source of the SSIM / L1 / s-IoU reporting convention.
- Wang, Bovik, Sheikh, Simoncelli. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP 2004. SSIM, for §2.2.
- Szegedy, Vanhoucke, Ioffe, Shlens, Wojna. *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. Label smoothing, for E8.
- Xiong et al. *On Layer Normalization in the Transformer Architecture.* ICML 2020. Pre-norm and the terminal LayerNorm, for E1.
- Loshchilov, Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. For E11.
- Higgins et al. *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework.* ICLR 2017. For E12.
- Lipman, Havasi, Holderrieth, Shaul, Le, Karrer, Chen, Lopez-Paz, Ben-Hamu, Gat. *Flow Matching Guide and Code.* arXiv:2412.06264. On the course reading list; cited in the future-work paragraph.
- Li, Tian, Li, Deng, He. *Autoregressive Image Generation without Vector Quantization.* NeurIPS 2024. The per-token generative head the rejected design mirrored.
