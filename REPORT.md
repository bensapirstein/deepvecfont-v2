# Reproducing DeepVecFont-v2, and measuring what a sweep over it can resolve

Ben Sapirstein · Generative Models for Text and Images, Reichman University · August 2026

> **Drafting note, delete before submission.** Sections 1 to 3 are written. Sections 4 to 6
> carry real numbers throughout but the connecting prose is still an outline in most places
> (backtick-bracketed notes to self) — that write-up is scheduled for later in §4's revised
> schedule, not this session. The two gaps that *were* today's blockers, the SSIM column and
> the quantization oracle row, are filled: both landed 2026-08-05 via `docs/stage1-closeout.md`
> on the cluster (see `PROJECT_PLAN.md` §2.3 for a rendering bug caught and fixed in the new
> oracle script along the way — confined to that script's first run, it does not touch any
> Tier 1–3, seed-floor, or confirmation number). Every number here is traceable to
> `RESULTS.csv` or to a dated **Measured** paragraph in `PROJECT_PLAN.md`; nothing is typed
> twice by hand.
>
> **Updated 2026-08-06.** Section 2.4 was rewritten outright. The previous version attributed
> the gap to 0.080 primarily to training budget; scoring the authors' released checkpoints
> through this harness falsified that, and the section now argues from the measurement. The
> abstract, §2.2's metric properties, §2.3's table and §6's outline all moved with it. The
> superseded text is in git history at `661b556`. Two findings were added to §6 and they are
> the strongest material in the report, so the discussion write-up should start there rather
> than at finding 1.

---

## Abstract

We reproduce DeepVecFont-v2 (Wang et al., CVPR 2023) on Chinese vector font synthesis and
then run a controlled sweep of twenty-one single-factor architectural changes against it.
The reproduction reaches a rendered reconstruction error of 0.1621 (three-seed mean, 34 test
fonts, best-of-50 decoding) against the paper's reported 0.080. To locate that gap we scored
the authors' own released checkpoints through the same harness: they return 0.1629, which is
0.0008 from our figure and well inside the seed-noise floor. The reproduction is therefore
faithful to the released model, training budget is eliminated as an explanation, and the
distance to the published number is a property of the evaluation. We measure one term of it
directly, a cross-rasterizer disagreement worth 0.0455 on Chinese against 0.0074 on English,
and report the remaining Chinese-specific 0.037 as open.

The sweep's headline is a measurement of its own resolution. Before running any candidate we
measured the spread of the reported metric across three re-seedings of the released model,
and it is 0.0097 in L1. Twenty-six single-factor changes, spanning normalization, latent
width, optimizer, quantization grid, loss weighting, learning-rate schedule, regularization
and capacity, together span 0.0101. Architectural change moves this metric about as much as
the random seed does. One candidate survives: additive Gaussian noise on the sequence
encoder at train-time σ = 0.5, which improves L1 by 0.0040 and s-IoU by 0.0271 against the
baseline mean, with the same sign at all three seeds on both metrics. Neither of those two
means clears its own pre-committed floor, so the candidate rests on the sign test rather than
on an effect size, and we report it that way. One candidate is a reproducible defect: a
terminal LayerNorm on the sequence encoder degrades s-IoU at all three seeds by a mean of
0.0376 at matched checkpoint epoch, 1.2 times that metric's own floor, after an
epoch-selection audit showed the originally reported 0.0760 mixed baseline and candidate
checkpoints from different training epochs.

Carrying the sweep to English changes the conclusion rather than extending it. The encoder-noise
candidate does not replicate there, with mixed sign across three seeds. Two changes that are
null on Chinese clear both English floors in the degrading direction, one of them replicated at
three seeds, so the same one-factor edit points in opposite directions on the two scripts the
paper reports. Underneath both arms sits a defect in the apparatus rather than in any model:
checkpoints are selected by a validation criterion containing no term that is the quantity
being reported, and it ranks candidates at Spearman ρ = 0.125 against the rendered metric,
mis-orders the authors' own released checkpoints, and when adopted as the selection rule never
improves a candidate's rendered score while flipping two of five from null to degrading. We
report it as measured and name the fix as future work rather than retrofitting it onto 113
already-scored checkpoints.

---

## 1. The original architecture

DeepVecFont-v2 is a few-shot vector font generator. Given a small set of reference glyphs
from an unseen font, it synthesizes the remaining glyphs as outlines rather than as pixels.

**Two encoders, one latent, two decoders.** A CNN reads the reference glyphs as 64×64
rasters. A twelve-block self-attention stack reads the same glyphs as SVG command sequences.
`ModalityFusion` merges the two into a VAE latent, which feeds an image decoder producing a
raster and an autoregressive Transformer decoder producing drawing commands. A second,
shallower decoder then refines the emitted sequence in a single parallel pass with full
bidirectional context.

**The relaxation representation is the paper's headline contribution**, and it is what makes
the rest of the design possible. SVG paths mix command types with different arity: a move and
a line need two coordinates, a cubic curve needs eight. A decoder emitting a heterogeneous
argument list per command type needs either a switch in the output head or a separate head
per type. The relaxation sidesteps both. Every command carries all eight coordinate arguments
regardless of type, and a fixed `cmd_args_mask` selects the live slots when the loss is
computed: all eight for `CurveFromTo`, indices 0, 1, 6 and 7 for `MoveFromTo` and
`LineFromTo`, none for `EOS`. A single uniform head then serves every command type, and the
dead slots are free to take whatever values make the sequence smooth. The paper's Eq. 10
supplies a consistency term that keeps those slack coordinates from drifting arbitrarily.

**The loss has six terms**: image L1 and a VGG perceptual loss on the raster branch, a KL
term on the latent, cross-entropy on the command type, cross-entropy on the quantized
coordinate arguments, a Bézier alignment term (Eq. 9), and the relaxation consistency term
(Eq. 10). Both decoders contribute their own command and argument terms, appearing in the
code as `loss_dict['svg']` for the autoregressive pass and `loss_dict['svg_para']` for the
refinement pass.

**Coordinates are classified, not regressed.** `numericalize` maps each coordinate onto a
discrete grid and the head predicts a class over it, with `denumericalize` mapping the
prediction back to a coordinate at inference. The released grid has 128 bins. This choice is
load-bearing for two later sections: it puts a hard floor under the achievable
reconstruction error (§2.3), and it is what candidate E13 varies.

Two properties of the released code shape everything downstream and are not obvious from
the paper.

**The argument head is shared between both decoders.** `self.args_fcn = nn.Linear(512, 8 *
128)` is instantiated once and called from both `Transformer_decoder.forward` and
`Transformer_decoder.parallel_decoder`. `command_fcn` is shared the same way. Any change to
the argument head therefore reaches the refinement pass as well, whether or not that was
intended.

**The refinement decoder produces what actually gets scored.** `test_few_shot.py` selects
its best-of-N candidate on `syn_{i}_{sample}_refined.svg`, which comes from `sampled_svg_2`,
the refinement decoder's output. The merge HTML that the evaluation reads is built from those
refined files. A modification that improves the autoregressive decoder but stops short of the
refinement pass does not reach the reported number at all. This constraint eliminated several
candidate changes during Stage 2 scoping before any of them cost a training run.

### 1.1 Paper versus released code

Four places where the released implementation departs from the paper text. All four are
legitimate reproduction findings, and three of them became experiments.

| Where | Paper says | Code does | Became |
|---|---|---|---|
| Sec. 3.1 | one-hot arguments of **256** dimensions | quantizes to **128** bins; the stale comment on `args_fcn` still reads `# shape: bs, max_len, 8, 256` | E13 |
| Sec. 3.3 | self-refinement is "a **2-layer** Transformer decoder" | `clones(DecoderLayer(...), 1)`, one layer | E3 |
| Eq. 11 | weights the Bézier term at **1.0** | `loss_w_aux = 0.01`, a factor of 100 below | E7 |
| Sec. 4.1 | the `N(0,I)` perturbation is an **inference-time** device simulating "the feature distortion caused by the human-designing uncertainty" | applied unconditionally in training, validation and test | E9 |

The fourth is the one that mattered. The paper motivates the encoder perturbation as a
test-time robustness device; the code adds it in every mode with no flag and no scale
parameter, so its magnitude had never been examined at all. Making that scale a swept
variable is the single change that survived this project's screening.

A fifth item is not a discrepancy with the paper but belongs in the same discussion. In
`Transformer.forward` the loop unpacks `cross_attn, cross_ff, self_attns` and uses only
`self_attns`. The cross-attention modules and `self.latents = nn.Parameter(torch.randn(256,
512))` are constructed, handed to the optimizer, and never called. With `depth=6` and
`self_per_cross_attn=2` the encoder is twelve self-attention blocks and zero cross-attention
blocks, so the model is not the Perceiver its configuration block advertises. We counted the
cost: **23,204,402 of the model's 136,055,207 parameters are dead, 17.06%**, carrying 185.6 MB
of Adam moment state that trains nothing (`scripts/dead_params.py`).

We did not run an ablation removing them, and the reason is instructive. The dead modules are
constructed *before* several live ones, and every `nn.Linear` and `nn.Parameter` draws from
the global RNG stream, so deleting them shifts the initialization of everything built
afterwards. A "dead parameters removed" run differs from the baseline by an effective seed
change, and §3.1 shows the seed floor exceeds any effect this setup can resolve. The
ablation could not have been read in either direction, so it is reported as a count.

---

## 2. Reproduction

### 2.1 Setup

Chinese, 34 test fonts, 52 glyphs each, `max_seq_len` 71, eight reference glyphs at fixed
`ref_char_ids` 0, 1, 2, 3, 26, 27, 28, 29. Trained from scratch on a single RTX 3090 at
roughly one hour per 150-epoch run. Checkpoints are selected on validation metric from a
per-run manifest rather than from a filename, for reasons given in §2.4.

The reported metric is the paper's "Error" (Sec. 4.1): mean L1 distance between the
rasterized synthesized glyph and the ground-truth glyph image at 64×64. Both sides are
binary masks, so it is the fraction of the 4096 pixels that disagree, and `Error = 0.080`
reads as "8.0% of pixels are wrong". We report it alongside s-IoU and SSIM, which is the
triple DualVector (CVPR 2023) uses on this task.

### 2.2 What the metric cannot see

This deserves its own subsection because three later results turn on it.

The metric is a rasterized proxy for vector fidelity. It says nothing about command count,
self-intersection, or control-point placement, only whether the inked region lands in the
right place. Three further properties matter:

1. **It is scored after best-of-N selection**, so it measures the best of N decoded
   candidates rather than the average sample. N is therefore a parameter of the metric, not
   of the model. We measured its effect: L1 reads 0.1722 at N = 3, 0.1691 at N = 10, 0.1678
   at N = 20 and 0.1662 at N = 50 on one fixed checkpoint. That 0.006 spread across the
   budget range is larger than any candidate effect in this project, which is why screening
   and confirmation numbers are never compared to each other anywhere in this report.
2. **Sub-pixel coordinate accuracy is largely invisible to it.** At the released 128-bin
   grid one quantization step is 0.625 rendered pixels, and rounding error per coordinate is
   uniform on ±0.3125 px. A coordinate-level improvement can therefore be real and still
   move this metric very little, which caps what any change to the argument head could ever
   have shown.
3. **It is undefined until the ground-truth side is specified**, and on Chinese that choice
   is worth 0.0455, more than seven times the paper's own margin over its predecessor. The
   candidate glyph is always rendered by our rasterizer; the ground truth may be the
   dataset's stored bitmap, produced by a different rasterizer at dataset-build time, or the
   ground-truth outline pushed through the rasterizer the candidate uses. Section 2.4
   measures the gap between the two and shows it tracks the pipeline floor rather than the
   model. Every Stage 2 comparison in this report holds one convention fixed on both arms,
   so the term cancels in the paired differences and no result in §4 or §5 depends on it; it
   matters only where an absolute value meets a published one.

### 2.3 Reconstruction results

Every row below is the raster ground-truth convention unless marked otherwise, which per
§2.2 is a required label rather than a detail.

| Row | n_samples | Error (L1) ↓ | SSIM ↑ | s-IoU ↑ | Render % |
|---|---|---|---|---|---|
| Paper, DeepVecFont-v2 (CN) | 50 | 0.080 | | | |
| Paper, DeepVecFont (CN) | | 0.086 | | | |
| Paper, DeepSVG (CN) | | 0.167 | | | |
| Quantization oracle, pipeline floor (n = ∞) | | 0.1422 | 0.4916 | 0.3713 | 34/34 |
| Quantization oracle, 128-bin grid | | 0.1443 | 0.4884 | 0.3664 | 34/34 |
| **This reproduction, seed 1111** | 50 | 0.1662 | 0.4375 | 0.2545 | 34/34 |
| **This reproduction, seed 2222** | 50 | 0.1569 | 0.4487 | 0.2716 | 34/34 |
| **This reproduction, seed 3333** | 50 | 0.1632 | 0.4413 | 0.2781 | 34/34 |
| **Three-seed mean** | 50 | **0.1621** | **0.4425** | **0.2681** | 34/34 |
| Official checkpoint, 600 ep | 50 | 0.1629 | 0.4373 | 0.3225 | 34/34 |
| Official checkpoint, 600 ep, *svg GT* | 50 | 0.1174 | 0.5450 | 0.4627 | 34/34 |

The two official rows are the authors' released weights scored through this harness, not runs
of ours. Section 2.4 reads them; the short version is that our three-seed mean sits 0.0008
from the released model under the identical convention, and that switching the ground-truth
convention moves the same fixed checkpoint by 0.0455.

Renderability is reported because it is silently rewarded otherwise. `test_few_shot.py`
wraps its rasterizer in a bare `except: continue`, and a model that fails on hard glyphs gets
those glyphs dropped from its average rather than penalized. Every row in this report renders
1768 of 1768 glyphs, so no comparison here is confounded by it. One earlier Stage 1 checkpoint
did not: `dvf_base_exp_chn` at epoch 135 scored 0.1641 over 33 of 34 fonts, with font 0028
failing to render entirely, which is why the quantity is now reported per run.

### 2.4 The gap to 0.080

The reproduction lands at roughly double the paper's Chinese number. The question is whether
that is a failure of our training run or a property of the evaluation, and it is answerable
without arguing about it: the authors released their trained checkpoints, so we scored those
through our own harness. Nothing was trained on our side for this measurement, which is what
makes it decisive. Any shortfall the released weights show here is attributable to the
evaluation rather than to us.

| | paper | official 600 ep, raster GT | official 600 ep, svg GT | ours, 150 ep, 3-seed mean |
|---|---|---|---|---|
| Chinese, full 34 fonts | 0.080 | 0.1629 | 0.1174 | **0.1621** |
| English, 34-font subset † | 0.052 | 0.0658 | 0.0584 | **0.0597** ‡ |

Checkpoints load strictly against all 136,055,207 parameters, and every row is best-of-50
decoding. The two ground-truth conventions are defined below.

**The reproduction is faithful.** Our 150-epoch Chinese baseline scores 0.1621 against the
released 600-epoch checkpoint's 0.1629 under the identical convention, font set and sample
budget: a difference of 0.0008, roughly a twelfth of the 0.0093 seed-noise floor §3.2
measures, which is to say indistinguishable from re-seeding the same model. Whatever
separates this work from the published number, it is not the training run reported here.

**Training budget does not explain the gap.** The released checkpoint carries four times our
epoch budget and scores no better through this evaluation. The 500, 550 and 600 epoch
checkpoints also agree within 0.0013 of each other in both languages, so the authors' own
optimization had saturated well before the budget ended. This eliminates the explanation
that a longer Chinese run would close the distance to 0.080, and it eliminates it on the
strongest available evidence, which is the paper's own weights rather than an extrapolation
from ours.

What remains divides into one term we can measure and one we can only bound.

**The ground-truth convention, worth roughly half the Chinese gap.** "Error" names a
comparison, and it is undefined until both operands are fixed. The synthesized glyph is
always rendered by our rasterizer. The ground truth can be the dataset's stored bitmap,
produced by a *different* rasterizer when the dataset was built, or the ground-truth outline
pushed through the same rasterizer as the candidate. We implemented both (`--gt_source
raster` and `--gt_source svg`) and the difference between them is pure cross-rasterizer
disagreement, carrying no information about any model. It is not a rounding term and it is
not symmetric across languages: **0.0455 on Chinese** against **0.0074 on English**.

The pipeline floor confirms the reading independently. Scoring ground truth against ground
truth through our rasterizer gives 0.1422 on Chinese and 0.0253 on English. **The paper's
reported 0.080 sits 0.062 below the Chinese floor**, so under the raster convention it is
unreachable by any model whatsoever, the authors' included. English's floor at 0.0253 sits
comfortably below its reported 0.052. The convention is load-bearing for Chinese and close to
irrelevant for English, and the difference tracks the floors rather than the models.

The scale matters here. The paper's entire margin over its own predecessor on Chinese is
0.006 absolute (§1.3's table). The convention choice moves the number by 0.0455, more than
seven times that margin, so a Chinese Error quoted without its ground-truth convention cannot
be compared against published values at all.

**A Chinese-specific residual of about 0.037, which we report open.** Under the svg
convention the rasterizer term is gone by construction, and the two languages still diverge.
English reaches 0.0584 against a reported 0.052, a 12% shortfall on a scoped subset. Chinese
reaches 0.1174 against a reported 0.080, still 47% high. Same harness, same evaluation code,
same released weights, opposite outcomes. What is left points at the Chinese data or test
protocol rather than at the model: the test font list, the character subset, `ref_char_ids`,
or the dataset build itself. We did not measure it, and we state its size rather than
speculate about its cause.

**Training our own English baseline sharpens that residual rather than softening it.** Three
English seeds trained to a budget frozen from their own convergence curves score **0.0597**
mean L1 on the subset, which is *better* than the released checkpoint's 0.0658 at the same
epoch range and closer to the reported 0.052. The margin over the release, 0.0048 to 0.0061
across the three released checkpoints, is 1.3 to 1.6 times the English seed-noise floor of
0.0038, so it clears the floor while staying the same order of magnitude as it. The pattern
across the two scripts is the informative part: on Chinese our training matches the authors'
released weights and both fall well short of the published figure, while on English our
training slightly exceeds their released weights and lands near the published figure. A
training deficiency on our side would have to show up in both columns. It shows up in
neither, which leaves the Chinese-specific residual exactly where the previous paragraph
puts it.

**Three earlier candidates, eliminated before the checkpoints were scored.** Each was checked
rather than argued away, and each would have invalidated the reproduction had it held.

*An evaluation-harness fault.* The evaluation script's results glob sat one directory level
above the layout `test_few_shot.py` writes, so against the current tree every font was
skipped silently. This was found and fixed, and the fixed script prints the layout it
detected on its first line. It reports `per-checkpoint`, so every figure was scored against
the tree its named checkpoint actually produced.

*A coarser quantization grid than assumed.* The codebase carries two `numericalize`
definitions on different grids, 128 bins in the model path and 64 in `data_utils/relax_rep.py`,
and the preprocessing copy round-trips the sequence array in place through a numpy view. Had
that mutation reached the persisted training data, the head would predict over 128 bins that
the data populates only at 64 and half the vocabulary would be structurally unreachable. It
does not: `relax_rep.process` writes `sequence_relaxed.npy` *before* calling the mutating
function and never saves the array again, and the dataloader reads that file. Quantization at
the released grid costs 0.0021 on Chinese and 0.0013 on English, measured directly against
the n = ∞ floor.

*A different test protocol.* The paper does not fully specify `ref_char_ids`, `n_samples` or
the font list, and all three move the number. Section 2.2 measured the largest of them: the
whole best-of-N budget range is worth about 0.006, and our figures are taken at N = 50, the
top of it. Granting the paper an unstated advantage of similar size on each of the other two
puts the entire protocol surface near 0.01. It is a real term and it is a small one.

**What we do not claim.** We have not identified the definition under which 0.080 is
obtained, and we do not assert that the paper's number is wrong. We assert something
narrower and better supported: under a single evaluation applied identically to their weights
and ours, the two models are indistinguishable, and the distance from both to the published
figure is a property of that evaluation. The Chinese residual under the matched-rasterizer
convention remains unexplained and is reported as such.

We also drop an observation earlier drafts leaned on, that the reproduction lands near
DeepSVG's published Chinese 0.167. With the raster-convention floor measured at 0.1422, any
model scored this way inherits the same offset, so agreement with another paper's number
under an unknown pipeline is arithmetic rather than corroboration.

† **The English rows are scoped, not full-set.** English's test split is 1,386 fonts against
Chinese's 34, and a full three-checkpoint sweep projected to roughly 40 GPU-hours and over
100 GB against a 200 GB quota near its cap. We cut to a deterministic 34-font prefix of the
unshuffled split for the 500/550/600 comparison, plus an 862-font partial decode of
checkpoint 500. The subset reads **optimistic** against the larger sample by 0.0074 (raster)
and 0.0106 (svg), so a full-set English figure would sit nearer 0.073 and 0.069. That widens
English's shortfall without disturbing the Chinese-versus-English contrast this section rests
on, since the contrast is an order of magnitude larger than the bias.

‡ **Our English figure carries the same subset bias, and it is the same subset.** The three
English baselines are scored on the identical 34-font prefix, at the identical sample budget,
under the identical convention, so the comparison against the released checkpoint in the same
row is exact even though the absolute value is optimistic. Epochs differ slightly across the
three seeds (640, 580, 640) because checkpoint retention keeps the lowest-validation
checkpoints rather than a fixed stride; these runs use a fixed per-epoch exponential learning
rate schedule that does not depend on the declared budget, so a checkpoint drawn at a given
epoch is not distinguishable from one a shorter run would have produced there.

---

## 3. Method: a controlled sweep, and the instrument it needed

### 3.1 Why a sweep

The assignment asks for one meaningful architectural change and for an account of what
worked, what did not, and what was learned. A single large bet answers the first and risks
having nothing for the second. We ran a controlled sweep instead: single-factor
modifications, each drawn from a category the assignment lists, each measured against a
measured noise floor, with any winner confirmed at a larger evaluation budget.

The alternative was scoped and rejected. A conditional flow-matching head replacing the
quantized classification head with a continuous coordinate model was costed and dropped on
schedule grounds, not on merit: it touches the loss, the sampler, the autoregressive feedback
path and the refinement decoder simultaneously, with one attempt at getting it right. It
returns as future work in §6.

### 3.2 The floor, measured before any candidate ran

Until this project `train.py` called `setup_seed(1111)` with nothing varying it. We added a
`--seed` flag and trained the unmodified baseline three times, at seeds 1111, 2222 and 3333.
The spread of the reported metric across those three runs is the resolution limit of every
comparison that follows.

| Seed | Error (L1) | s-IoU |
|---|---|---|
| 1111 | 0.1728 | 0.2240 |
| 2222 | 0.1631 | 0.2410 |
| 3333 | 0.1680 | 0.2555 |
| **Spread (the floor)** | **0.0097** | **0.0315** |

Screening budget, N = 3, all 34 fonts. Two things follow immediately.

**The floor is larger than the paper's entire published margin.** DeepVecFont-v2 beats its
own predecessor on Chinese by 0.006 absolute. Our resolution limit is 0.0097. A setup that
cannot resolve the effect the original paper reports is a fact about the setup, and stating
it up front is more useful than discovering it one candidate at a time.

**The floor is per metric, and s-IoU is far noisier.** 0.0315 on a baseline near 0.24 is a
13% relative spread, against 5.8% for L1. Judging a candidate on one metric against the
other's floor is a specific mistake we made and caught, described in §6.

### 3.3 What the floor is made of

The two components have different remedies, so we separated them before spending training
budget on either. Re-evaluating one fixed checkpoint three times varies the decode and holds
the weights, isolating decode noise: L1 came back 0.1722, 0.1717, 0.1711, a spread of 0.0011.

**Decode noise is 12% of the floor. The remaining 0.0082 is training-seed variance.** Raising
the evaluation budget therefore buys almost nothing, and running each candidate at multiple
seeds is the only lever, at three times the training matrix. That measurement redirected the
entire second half of the sweep from breadth to depth.

### 3.4 Screening and confirmation

Screening runs at N = 3 over all 34 fonts. Confirmation runs at N = 50. §2.2 shows the two
budgets differ systematically by more than any candidate effect, so no number from one is
ever compared against a number from the other in this report.

A cheaper screen was available in principle. The validation metric is deterministic given a
checkpoint, computed over the whole validation set, and logged at every checkpoint, where the
rendered metric carries decode noise and is measured once. If the two agreed on ranking,
every candidate after the first tier could have been screened for free. **They do not agree.**
Across the 26 candidates, Spearman ρ between validation metric and rendered L1 is **0.125**,
Kendall τ is 0.083, and excluding the candidates whose learning-rate schedule or weight
averaging confounds late-training validation loss raises it only to 0.179. The threshold set
in advance was 0.4.

The teacher-forced validation loss and the autoregressive best-of-N rollout are measuring
different things, and exposure bias is invisible to the first. Every screening decision in
this project was therefore paid for at evaluation cost. This is reported as a finding rather
than as an inconvenience: a cheap proxy that correlates at ρ = 0.125 with the reported metric
would have produced a confidently wrong candidate ranking had we trusted it.

### 3.5 Rules fixed before the numbers

Three decision rules were written down and committed before the measurements they govern
existed, so they could not be chosen to suit an outcome.

- **What a shifted test-time σ licenses.** The encoder perturbation applies at evaluation
  regardless of its training value, and no one had checked whether the released σ = 1.0 is a
  sensible point on that curve. If tuning it helped, every number in the results table would
  have been taken at an arbitrary point. The rule distinguishes three cases and specifies the
  response to each, including the trap case where only the candidate's own curve shifts. It
  is in `docs/confirmation-launch.md` under "Rule 1", dated before the sweep ran.
- **What the confirmation evaluation can conclude.** Not "the candidate beats the noise
  floor", since the floor is the baseline's own seed spread and is the quantity a paired
  design replaces. The permitted claim is that a paired per-font test favours the candidate
  at every seed.
- **The English epoch budget** is frozen from the baseline curves before any candidate is
  looked at, in `docs/english-arm.md`, because the cosine schedule derives its shape from the
  epoch count and cannot be compared across budgets at all. The English training arm was
  descoped on cost grounds before it ran (§2.4 †), so this rule was never exercised; it is
  listed because it was pre-registered, not because it produced a number.

---

## 4. Candidates

Twenty-one distinct one-factor changes were run across four tiers, plus a replication
batch and a category-coverage batch, for 113 scored checkpoints in total. Every candidate
obeys the same construction rule: **one factor moves, every other flag holds at the value
`COMMON_ARGS` carries for the baseline.** Each new flag was added defaulting to the
released behaviour and asserted by `scripts/check_infra.py`, which grew to 196 checks, so
a candidate's own default reproduces the baseline exactly and any difference in a result
is attributable to the flag rather than to the diff that introduced it.

The candidates were not chosen for expected gain. Two sources drove the list. The first is
§1.1's paper-versus-code audit: four places where the released implementation departs from
what the paper describes, three of which became experiments (E9, E3, E7) and are the most
defensible candidates in the set, because each one asks whether the paper's stated design
or the shipped code is better. The second is the assignment's own list of seven example
categories, which the coverage batch of §5.4 exists to satisfy.

### 4.1 Coverage of the assignment's categories

| Category from the brief | Candidates | Replicated at 3 seeds |
|---|---|---|
| Add normalization layers | E1 terminal LayerNorm, E2 image-branch norm | E1, E2 |
| Change the encoder or decoder | E4 encoder width, E16 encoder depth, E17 decoder feed-forward width | E4, E16, E17 |
| Change the latent dimension | E5 bottleneck 256 | E5 |
| Add residual or attention layers | E3 refinement depth, E16 encoder depth | E3, E16 |
| Modify the loss function | E7 Bézier weight, E8 ordinal label smoothing, E12 KL weight | E7 |
| Change the noise schedule | E9 encoder noise σ, E14 warmup-cosine learning rate | E9 |
| Add regularization | E10 dropout, E11 AdamW, E15 weight EMA | E11 |

All seven are covered, and one representative of each carries three seeds. That last
column is the point of the coverage batch rather than an afterthought: §3.5 measured that
a single-seed leader in this setup survives replication one time in three, so a coverage
table built from single points would have documented breadth and evidenced nothing.

E13, which doubles the argument quantization grid from the code's 128 bins to the 256 the
paper's Sec. 3.1 specifies, sits outside the seven categories. It changes the
representation rather than the architecture, and it is included because §5.6 can bound its
best possible outcome before it runs.

### 4.2 Tier 1: the paper-versus-code candidates

Run first because they are the ones a reader can argue with on grounds other than a number.

**E9, encoder noise.** `models/transformers.py:450` applies `x = x + torch.randn_like(x)`
unconditionally in training, validation and test, while Sec. 4.1 describes the
perturbation as an inference-time device simulating human design uncertainty. Two flags
were added, `--enc_noise_std_train` and `--enc_noise_std_test`, with σ selected by
`self.training`. The noise is always drawn even when scaled to zero, so the RNG stream
stays aligned across the sweep and a σ change is not silently also an initialization
change. Swept at σ_train ∈ {0, 0.25, 0.5}.

**E1, terminal LayerNorm.** Two `nn.LayerNorm(512)` lines behind `--enc_final_norm`, off by
default so existing checkpoints still load strictly. Standard pre-norm practice and absent
from this encoder.

**E7, Bézier weight.** Eq. 11 weights the Bézier alignment loss at 1.0; the code sets
`loss_w_aux = 0.01`, a factor of 100 below. Swept upward. Needs no code change.

**E10, dropout.** Wired into the attention and feed-forward sublayers of both stacks,
passed from `ModelMain` rather than read off the module-level `opts` global. Swept at 0.1
and 0.2.

### 4.3 Tier 2: small structural diffs

**E8**, ordinal label smoothing over the quantized coordinate bins, on the reasoning that
adjacent bins are metrically close and one-hot cross-entropy treats them as unrelated.
Swept at σ ∈ {0.5, 1.0, 2.0}. **E13**, split into two flags, the 256-bin grid and the
removal of `padding_idx`. **E3**, refinement depth, which restores the "2-layer Transformer
decoder" of Sec. 3.3 against the code's `clones(DecoderLayer(...), 1)`. **E14**, a
warmup-cosine learning-rate schedule replacing the plain exponential decay.

E8, E13 and E14 carry a caveat that is stated wherever their numbers appear: all three
alter terms inside the training loss, so their `val_metric` is on a different scale by
construction and is not comparable across candidates. They screen on the rendered metric
regardless, which by §3.4 is what every candidate ended up doing.

### 4.4 Tier 3: breadth, then depth

Batch A ran six further candidates at one seed for coverage: **E12** KL weight, **E11**
AdamW, **E2** image-branch normalization, **E4** encoder width, **E15** weight EMA, **E5**
latent width. Batch B then took the three largest deltas from Tiers 1 and 2 (E9 at
−0.0059, E1 at −0.0054, E13 at −0.0050, all roughly half the floor and indistinguishable
from each other) and ran two further seeds on each. Batch B is what separated E9 from the
other two and is the only reason this report has a finalist rather than a ranking.

Two candidates did not survive contact with the code, and both are reported rather than
dropped. **E2 with `instancenorm` failed at the smoke test**: `InstanceNorm2d` cannot
compute a per-instance variance at the `[32, 1024, 1, 1]` bottleneck, which is an
architectural incompatibility rather than a wiring fault, so E2 ran with batch
normalization only. **E6**, an ablation of the dead cross-attention path, was audited but
never trained. Section 1.1 reports the audit: 23,204,402 of 136,055,207 parameters, 17.06%
of the model, are constructed, handed to the optimizer and never called. Removing them
changes the parameter count and therefore shifts the global RNG stream for every module
built afterwards, so the ablation would have been part initialization change and could not
be read against the seed floor.

### 4.5 Tier 4: the transformer's own capacity

Added late, and for a stated reason. Tiers 1 to 3 cover six of the seven categories
cleanly, but *add residual or attention layers* rested on E3's refinement depth alone. E4
widened the image stacks and E5 the latent, and neither touches the sequence transformer
that does the actual modelling. Two flags close that: **E16 `--enc_depth`** 6 → 8, taking
the encoder from 12 to 16 self-attention blocks, and **E17 `--dec_d_ff`** 1024 → 2048, a
2× feed-forward expansion on `d_model` 512 where the literature default is 4×. The `ff`
module is deep-copied into both decoder stacks, so E17 widens the autoregressive decoder
and the refinement decoder together.

Both change the parameter count and inherit the same RNG-shift objection that kept E6 off
the list, which is why three seeds was a precondition rather than an option here:
averaging over three draws is what converts an initialization shift into a seed draw and
makes a capacity change readable at all. Both were pre-registered as expected-null before
they ran.


## 5. Results

Every number in this section is Error (L1), s-IoU and SSIM over the same 34-font test
subset, scored through `eval_reconstruction_error.py` under the **raster** ground-truth
convention. Section 2.2 showed that the convention is worth 0.0455 on Chinese, which is
larger than anything the sweep set out to measure, so it is fixed here and stated once
rather than carried on each row. Screening figures use `n_samples 3` on Chinese and
`n_samples 10` on English; confirmation figures use `n_samples 50`. The two budgets are
never compared against each other, because §3.4 measured that gap to be wider than most
candidate deltas.

The bars a result has to clear are the seed floors of §3.2, and there are four of them
rather than one:

| Arm | Budget | Error (L1) | s-IoU | SSIM |
|---|---|---|---|---|
| **Chinese, the pre-committed bar** | screening, `n=3` | **0.0097** | **0.0315** | — |
| Chinese, observed at confirmation | `n=50` | 0.0093 | 0.0236 | 0.0112 |
| **English, the pre-committed bar** | `n=50` | **0.0038** | **0.0129** | 0.0140 |

The Chinese L1 floor does not shrink between screening and confirmation, which is what the
noise decomposition in §3.3 predicted: 88% of it is training-seed variance and only 12% is
decode noise, so spending more samples buys almost nothing. The s-IoU spread does narrow,
from 0.0315 to 0.0236, on three seeds either way.

**Every Chinese candidate in this report is judged against the screening row, including at
confirmation budget.** The narrower confirmation spread is reported because it was measured
and is not used as a bar. Both figures come from three seeds, the wider one was fixed before
any candidate ran, and swapping to whichever of two available floors a result happens to
clear is the failure mode §3.5 exists to prevent. The consequence is stated where it bites,
in §5.2.

### 5.1 The three-way comparison

The assignment asks for the published numbers, our reconstruction and our improved model
side by side. A fourth column is added because it is available and because it carries
most of §2.4's argument: the authors' own released weights, run through our harness under
our convention.

| | Paper (reported) | Released weights, our harness | Our reproduction, 3-seed mean | Our improved model (E9), 3-seed mean |
|---|---|---|---|---|
| **Chinese**, Error (L1) ↓ | 0.080 | 0.1629 | 0.1621 | **0.1581** |
| **Chinese**, s-IoU ↑ | not reported | 0.3225 | 0.2681 | **0.2952** |
| **Chinese**, SSIM ↑ | not reported | 0.4373 | 0.4425 | **0.4479** |
| **English**, Error (L1) ↓ | 0.052 | 0.0658 | 0.0597 | 0.0601 |
| **English**, s-IoU ↑ | not reported | 0.7029 | 0.7309 | 0.7305 |
| **English**, SSIM ↑ | not reported | 0.7181 | 0.7374 | 0.7358 |

Released weights are epoch 600 on both languages, the checkpoint the authors shipped.
Our Chinese reproduction is 150 epochs, our English 630. SSIM is the metric required from
the course material; s-IoU comes from the vector-font literature and is reported as a
second instrument rather than as a requirement.

Three readings, and the first is the one the assignment's "compare to within negligible
differences" clause asks for.

**The reproduction is faithful, and the published figure is not the test of that.** Our
Chinese baseline sits 0.0008 from the released checkpoint, roughly a twelfth of the seed
floor, on a quarter of the training budget. On English our own baseline is *ahead* of all
three released checkpoints by 0.0048 to 0.0061, which is 1.3× to 1.6× the English floor.
What neither column reaches is the paper's own 0.080 and 0.052. Section 2.4 separates
those two claims, and the separation is only possible because the released weights exist.

**The English column is optimistic against the paper's own protocol.** Sec. 4.1 sets
`Ns` = 10 for English, and every English row above used 50. Best-of-N selection can only
lower L1 as N grows, so the true distance from 0.052 is wider than the table shows. The
effect is measured directly in §5.4: the same anchor checkpoint scores 0.0583 at N = 50
and 0.0610 at N = 10, a difference of 0.0027 that is entirely sampling budget and
contains no model at all. It is quoted rather than corrected because every paired
comparison in this report holds N fixed across both arms.

**E9 improves Chinese and does not transfer.** The Chinese column moves in the right
direction on all three metrics; the English column moves by less than its own floor in
both directions. That asymmetry is the subject of §5.3.

### 5.2 E9, the finalist, on Chinese

E9 sets `enc_noise_std_train = 0.5`, halving the `N(0,I)` perturbation the released code
applies during training. Section 1.1 records why this is a paper-versus-code candidate
rather than a hyperparameter guess: Sec. 4.1 describes the perturbation as an
inference-time device, and the code applies it unconditionally in all three phases.

Confirmation budget, `n_samples 50`, each candidate seed paired against the baseline seed
of the same number at the same epoch.

| Seed | Error (L1) ↓ | Δ vs paired baseline | Wilcoxon p (HL shift) | s-IoU ↑ | Δ | Wilcoxon p (HL shift) |
|---|---|---|---|---|---|---|
| 1111 | 0.1588 | −0.0074 | 0.0012 (−0.0074) | 0.2870 | +0.0325 | 0.0001 (+0.0317) |
| 2222 | 0.1531 | −0.0038 | 0.0437 (−0.0039) | 0.3170 | +0.0454 | 0.0000 (+0.0453) |
| 3333 | 0.1623 | −0.0009 | 0.3050 (−0.0011) | 0.2816 | +0.0035 | 0.4417 (+0.0045) |
| **Mean** | **0.1581** | **−0.0040** | | **0.2952** | **+0.0271** | |
| Baseline mean | 0.1621 | | | 0.2681 | | |

Wilcoxon is per-font paired, per seed, with the Hodges–Lehmann shift in brackets. Seeds
are never pooled, since a pooled test would treat three training runs as one sample.

**What this does and does not establish, stated precisely. Neither of E9's two means clears
its pre-committed floor.** The L1 improvement of 0.0040 sits inside the 0.0093 bar, and the
s-IoU improvement of 0.0271 sits inside the 0.0315 bar. E9 would clear the s-IoU spread
observed at confirmation budget, 0.0236, and that comparison is not made: the bar was fixed
at 0.0315 before any candidate ran, and moving it afterwards to the value a result happens
to clear would forfeit the only thing that makes this sweep worth reading.

What makes E9 the finalist is the other pre-committed criterion of §3.5: same-sign paired
improvement at every seed, on both metrics, six of six, on two instruments that correlate at
only r = −0.335 across the full table. Under a floor test alone E9 is null. Under a sign
test across three seeds and two near-independent metrics it is not, and the probability of
six of six agreeing by chance is 1 in 64 if the metrics were independent, which they nearly
are. Both readings are reported because the second is the weaker kind of evidence and the
report should not present it as the stronger one. Seed 3333 is the weak leg on both metrics
and was the weak leg at screening too, before any of this was looked at.

The practical consequence: **E9 is a direction with consistent sign, not a demonstrated
improvement of a stated size.** Section 5.3 then shows the direction does not survive a
change of script, which is the more informative half of the result.

The honest summary is that E9 is a real direction and a small one. It is roughly the size
of the paper's own margin over its predecessor on this language, which is 0.006 absolute,
and §6 argues that this is a fact about the benchmark rather than about E9.

### 5.3 E9 on English: the generalization test, and it fails

The English arm carried one candidate at three seeds rather than three candidates at one
seed, for the reason §3.5 gives and §4 sizes. Each E9 seed is paired to the English
baseline of the same seed at the same epoch, so the pruning-survivor and epoch confounds
never enter.

| Seed | Epoch | E9 L1 | Baseline L1 | Δ | E9 s-IoU | Baseline s-IoU | Δ |
|---|---|---|---|---|---|---|---|
| 1111 | 640 | 0.0602 | 0.0583 | +0.0019 | 0.7308 | 0.7371 | −0.0063 |
| 2222 | 580 | 0.0594 | 0.0621 | −0.0027 | 0.7344 | 0.7242 | +0.0102 |
| 3333 | 640 | 0.0608 | 0.0587 | +0.0021 | 0.7263 | 0.7314 | −0.0051 |
| Mean | | 0.0601 | 0.0597 | +0.0004 | 0.7305 | 0.7309 | −0.0004 |

One seed of three favours E9 on both metrics; two disfavour it on both. All six deltas
sit at or inside the English floor of 0.0038 and 0.0129. This is outcome 3 of the reading
rule fixed in `docs/english-candidate.md` §4 before the runs launched, and it is reported
as a result rather than as a failed run: **the project's one confirmed Chinese improvement
does not replicate on a second script.**

Mixed sign is floor-independent, which is why no amount of extra baseline seeding could
change this row. One caveat is recorded and does not move the reading: the three E9
English runs rendered 33 of 34 fonts against the baseline's 34, so their means are taken
over one fewer font.

### 5.4 Category coverage, and what a controlled sweep returns

The brief lists seven example categories of architectural change. Section 4 maps the
candidate set onto them. Because §3.5 rules out reading single-seed points at this
resolution, one representative per category was replicated at three seeds on Chinese
(Job C) and screened on English (Job C-EN), and the transformer's own capacity was added
as a fourth tier (Job D). Every run below was trained with all checkpoints kept and
scored at matched epoch 150, so the epoch confound of §5.5 never enters the table.

**Chinese, screening budget `n_samples 3`, matched epoch 150, per-seed pairing.** Baseline
three-seed screening mean: L1 0.1680, s-IoU 0.2402. Floors 0.0097 and 0.0315.

| Candidate | Category | ΔL1 per seed | Mean ΔL1 | Δs-IoU per seed | Mean Δs-IoU | Reading |
|---|---|---|---|---|---|---|
| E2 `img_norm batch` | normalization | −0.0080, −0.0022, −0.0095 | −0.0066 | +0.0241, +0.0185, +0.0238 | +0.0221 | same sign, sub-floor |
| E4 `ngf 32` | encoder width | −0.0016, +0.0144, +0.0048 | +0.0059 | +0.0338, −0.0013, +0.0194 | +0.0173 | mixed |
| E5 `bottleneck 256` | latent dimension | −0.0052, 0.0000, +0.0023 | −0.0010 | +0.0375, +0.0312, −0.0077 | +0.0203 | mixed |
| E7 `loss_w_aux 0.1` | loss function | −0.0011, +0.0025, −0.0043 | −0.0010 | +0.0050, −0.0210, +0.0002 | −0.0053 | mixed |
| E11 AdamW | regularization | −0.0044, +0.0036, −0.0025 | −0.0011 | +0.0019, −0.0046, +0.0078 | +0.0017 | mixed |
| E3 `n_layers_refine 2` | decoder depth | −0.0030, −0.0001, +0.0019 | −0.0004 | +0.0078, +0.0111, −0.0167 | +0.0007 | mixed |
| E16 `enc_depth 8` | encoder depth | −0.0008, +0.0067, +0.0017 | +0.0025 | +0.0058, −0.0243, +0.0049 | −0.0045 | mixed |
| E17 `dec_d_ff 2048` | decoder width | −0.0058, +0.0041, +0.0032 | +0.0005 | +0.0284, +0.0103, −0.0235 | +0.0051 | mixed |

**All eight are null.** E2 is the only one with a consistent sign on both metrics, and
both of its means sit under their floor, so it does not clear the pre-committed bar
either. The two capacity rows were pre-registered as expected-null in §8 of the plan and
came back that way; they are in the table for category coverage rather than because a win
was expected.

**English, screening budget `n_samples 10`, checkpoint 620, against the seed-1111 anchor
at the same budget** (L1 0.0610, s-IoU 0.7270). Floors 0.0038 and 0.0129.

| Candidate | ΔL1 | Δs-IoU | Reading |
|---|---|---|---|
| E11 AdamW | +0.0013 | −0.0048 | null |
| E4 `ngf 32` | +0.0022 | −0.0126 | null |
| E2 `img_norm batch` | +0.0026 | −0.0126 | null |
| E5 `bottleneck 256` | +0.0058 | −0.0242 | clears, **degrading** |
| E3 `n_layers_refine 2` (3 seeds) | +0.0047 | −0.0208 | clears, **degrading** |

E3 was replicated to three seeds because its first reading cleared the floor. All three
seeds clear both floors with the same sign: ΔL1 +0.0048, +0.0053, +0.0040 and Δs-IoU
−0.0254, −0.0218, −0.0153. That makes `--n_layers_refine 2` a **confirmed degrading result
on English**, replicated at the same bar E9 cleared to become the finalist.

The result worth carrying into §6 is that E3 and E5 are null and mixed-sign on Chinese and
confirmed degrading on English. The two scripts disagree in magnitude and in direction on
the same one-factor change, which is a stronger statement about the generality of any
architectural claim in this literature than either arm makes alone. E7 has no English row:
its run was stopped at epoch 294 of 631 on a read of the training curve, a decision §6
returns to.

### 5.5 The one effect that exceeds its floor, and it is negative

E1 adds a terminal `nn.LayerNorm(512)` to both encoder stacks. Its first reading was a
mean s-IoU deficit of −0.0760, 2.4× the screening floor, and it was the only effect
anywhere in this project to clear its own bar in either direction.

That reading was confounded. Every Chinese baseline was scored at epoch 150, but
`val_metric` had auto-selected epoch 125 for E1's seed 2222 and epoch 100 for its seed
3333, so E1's two worst legs were also its two earliest checkpoints. Both legs were
retrained with every checkpoint kept and re-read at matched epoch 150, under a rule fixed
before the retrain launched.

| Seed | Selected epoch, original | Δs-IoU, original | Δs-IoU at matched 150 |
|---|---|---|---|
| 1111 | 150 (already matched) | −0.0139 | −0.0139 |
| 2222 | 125 | −0.0681 | −0.0449 |
| 3333 | 100 | −0.1460 | −0.0539 |
| **Mean** | | **−0.0760** (2.4× floor) | **−0.0376** (1.2× floor) |

The confound was real and partial. The effect halves and survives: the sign holds at all
three seeds and the mean still clears the 0.0315 floor. L1 is unaffected throughout, with
every delta at or under 0.0058 and inside its own floor. The retrained baseline seed
reproduces the original within 0.0012 L1, well inside the floor, so the retrain itself is
trustworthy.

E1 is therefore the project's one clearing effect, and it is a degradation. Section 6
treats the audit that produced this table as a finding in its own right.

### 5.6 Two figures that bound what any of this could have achieved

The quantization oracle of §2.3 encodes each ground-truth outline into the model's own
representation and decodes it again, with no model involved. On Chinese it returns L1
0.1422 at infinite precision and 0.1443 at the released 128-bin grid. That first number is
below the baseline's 0.1621 and above the paper's 0.080, so no coordinate-level change
inside this representation could have reached the published figure through this
evaluation. The quantization cost itself is 0.0021, and the E13 candidate that doubled the
grid to 256 bins had an upper bound of 0.0016 before it ran, which is under the seed
floor. E13 could not have cleared its bar whatever it did, and it screened null.


## 6. Discussion

Stage 2 set out to find an architectural change that improves DeepVecFont-v2 and returned
one small positive result on one language, one confirmed degradation, and twenty nulls.
The more useful output was the instrument built to judge them. This section argues that
the measurements which fell out of building it are the substantive contribution, and that
several of them apply to the published work as directly as they apply to ours.

### 6.1 What the sweep found

E9, halving the encoder perturbation during training, improves Chinese on both metrics at
all three seeds, by 0.0040 in L1 and 0.0271 in s-IoU. It does not replicate on English.
E1, a terminal LayerNorm on the encoder stacks, degrades Chinese s-IoU by 0.0376 at all
three seeds, which is 1.2× the floor and the largest effect measured anywhere in this
project. E3, restoring the paper's stated 2-layer refinement decoder, degrades English on
both metrics at all three seeds and is null on Chinese. Everything else, across seven
categories of change and four tiers, sits inside its own noise.

Two of those three are paper-versus-code candidates, and the direction is worth noting
without overreading a sample of two: on E9 the code's choice is worse than the paper's
description implies, and on E3 the paper's stated design is worse than what the code
ships. The released implementation is not a degraded copy of the paper. It is a different
system, and the differences run in both directions.

### 6.2 The instrument is wider than the effects it was built to measure

Three baseline runs differing only in seed span 0.0093 in Chinese L1 and 0.0038 in
English. The decomposition in §3.3 puts 88% of the Chinese figure in training-seed
variance and 12% in decode sampling, which is why raising `n_samples` from 3 to 50 did not
narrow it and only more seeds could.

Set that against the published numbers. The paper's entire margin over its own predecessor
on Chinese is 0.006 absolute, below our Chinese floor. Its English ablation in Tab. 1 spans
0.0069 in total across four rows, with individual steps of 0.0031, 0.0028 and 0.0010
against our English floor of 0.0038. Its sampling-point study in Tab. 3 spans 0.0037 across
six settings with individual steps between 0.0002 and 0.0014, so every row of it sits under
that floor as well. Two of the paper's three quantitative tables report per-row differences
this instrument could not have resolved, and the instrument in question is the paper's own
metric on the paper's own data.

This is not an accusation that those results are wrong, and it is not an excuse for our
nulls. It is a statement about what the benchmark can support: at this scale, a single
training run per configuration cannot distinguish a 0.003 architectural effect from a
0.0038 seed draw, and neither the paper nor any of the prior work it compares against
reports a seed floor. The reason we can say this at all is that measuring the floor was
the first thing done, on day one, before any candidate existed.

### 6.3 A single-seed baseline reports the draw, not the change

We trained the baseline three times before running any candidate, which makes it possible
to show directly what single-seed anchoring would have done to our own results. Referenced
to seed 1111, **22 of 26 candidates beat the baseline**. Referenced to the three-seed mean,
**6 of 26** do. The whole difference is that seed 1111 landed 0.0048 above the mean, well
inside the seed spread and entirely unremarkable as a draw.

Twenty-two out of twenty-six reads as *almost any arbitrary change helps*, which is not
credible on its face and would have been the headline of this report had the floor not been
measured first. The cost is quantifiable a second way. Ranking candidates against the
single-seed anchor and replicating the top three at two further seeds, **one of three
survived**. That is a measured base rate for "the leading single-seed candidate is real" in
this setup, and it is the number to hand a reader who is weighing a single-seed ablation
table anywhere in this literature.

A related caution applies to the floor itself. The Chinese s-IoU floor moved from 0.0401 to
0.0315 on re-measurement. A reference computed from three points is itself a sample, which
is the same lesson one level up, and every bar quoted in §5 carries that uncertainty.

Against all of this, the sweep's own dispersion is the quiet result. Twenty-six candidates
span 0.0101 in L1; three re-seedings of the identical configuration span 0.0097. Twenty-six
draws from one distribution should span roughly 2.3 times the range of three, and the
observed ratio is 1.07. **Changing the architecture moved this metric about as much as
changing the random seed.**

### 6.4 The selection criterion is not the reported quantity

Checkpoints in this project, and in the released code, are selected by `val_metric`.
Nothing scored is `val_metric`. Its dominant term `img_l1` belongs to the image decoder
branch, while the reported number comes from the refinement decoder's SVG output
rasterized and compared to a ground-truth image, and `svg_para_total` is cross-entropy over
command types and quantized bins rather than pixels. No term in the criterion is the
quantity being reported, and four separate measurements say the gap is not academic.

**Across candidates it does not rank.** Spearman ρ between `val_metric` and rendered L1 is
0.125, far under any threshold that would license screening on it. Every candidate after
that point was screened on the rendered metric.

**Within a run it mis-orders.** In three retrains with every checkpoint kept, `val_metric`
agreed on the best checkpoint (150, all three) but ranked epoch 100 above 125 in all three,
while every rendered L1 and s-IoU number ranked 125 above 100.

**It mis-ranks the authors' own released weights.** Computing `val_metric` directly on the
three official English checkpoints, it prefers 600, the one they shipped, while every
rendered metric prefers 500 and ranks 600 worst (0.0645 against 0.0658). Chinese disagrees
in a different direction again. Selection was never validated against the metric it exists
to serve, on either side of this project.

**And switching to it wholesale makes things worse.** The obvious repair is to score each
run at its own lowest-`val_metric` checkpoint rather than at a fixed matched epoch. Tested
against all five scored English category candidates, three of which have a lower-`val_metric`
checkpoint available below the matched epoch: re-scoring there **never improved a single
candidate's rendered score, flipped two of five from null to confirmed degrading, and left
three unchanged.** A one-sided result across five candidates is not noise in either
direction. Checking whether some other already-logged term does better, pooled Spearman
against rendered L1 gives `val_metric` +0.283, `val_l1` +0.350, `val_svg_total` +0.450 and
`val_svg_para_total` −0.250. The refinement decoder's own loss is the best of the four and
still recovers little, on nine points and with none reaching significance.

This has a direct consequence for §5.5. E1's original s-IoU deficit of −0.0760 was measured
on checkpoints `val_metric` had selected at epochs 100 and 125 against baselines at 150, so
E1's two worst legs were also its two earliest. Retraining and re-reading at matched epoch
150 halves the effect to −0.0376. The finding survived; had it not, the project's one
clearing effect would have been an artifact of a criterion nobody had audited.

The failure mode is not confined to the automatic criterion. Two English runs were judged
by eye from their wandb training curves. One, E7, was stopped at epoch 294 of 631 on the
shape of its validation curve and has no result. Another, E3, was read from the same kind
of curve as "doing great" before scoring and came back as the project's confirmed degrading
result on English. Both readings were made in good faith by someone who had spent a week
with these curves, and both point the same way as ρ = 0.125.

**None of this was retrofitted.** Switching the whole project to rendered-L1 checkpoint
selection would mean decoding and rasterizing the validation split at every checkpoint, and
it would invalidate all 113 rows. The protocol stayed at matched-epoch selection, and the
defect is reported as measured rather than patched. The repair belongs in future work: a
validation-time hook that decodes at `n_samples 1` and rasterizes a small held-out subset
periodically would cost a fraction of a full evaluation and would select on something that
is at least the same kind of quantity as the score.

### 6.5 A metric is a pipeline, not a formula

"Reconstruction error" is an L1 distance between two 64×64 rasters, and that description is
not enough to reproduce a number. Two choices sit underneath it, neither stated in the
paper, and each moves the result by more than anything the sweep measured.

**Which rasterizer draws the ground truth.** The candidate SVG is rendered through
cairosvg; the ground truth is the dataset's pre-rendered raster from a different
rasterizer. Two rasterizers, one comparison. Rendering the ground-truth outline through the
candidate's own rasterizer removes the term, and on one fixed released checkpoint that
single change moves Chinese from 0.1629 to 0.1174. **That is 0.0455, against a paper whose
entire margin over its predecessor on this language is 0.006.** The effect is 0.0455 on
Chinese and 0.0074 on English, tracking the two pipeline floors of 0.1422 and 0.0253 rather
than the two models, so it is a property of how a particular dataset was built and not a
constant anyone can correct for once.

**How many candidates the best-of-N selection draws from.** Error is scored after selecting
the highest-IoU sample, and N is not free: a larger N can only lower L1. Sec. 4.1 sets N to
10 for English and 50 for Chinese. Every English row in this project used 50. Scoring one
fixed checkpoint at both budgets puts the difference at 0.0027 in L1, which is 0.7× the
English floor and larger in absolute size than six of the eight category deltas in §5.4.
That quantity is pure sampling budget and contains no model at all.

Both terms cancel exactly in the paired comparisons of §5, because both arms of every delta
hold the convention and N fixed. They do not cancel when a number is compared to a
published one, which is the whole of §2.4 and the next subsection.

### 6.6 Reproducing a model and reproducing a number are different claims

Our 150-epoch Chinese baseline scores 0.1621. The authors' released 600-epoch checkpoint,
through the same harness under the same convention on the same fonts, scores 0.1629. The
difference is 0.0008, about a twelfth of the seed floor, at a quarter of the training
budget. On English our own baseline is ahead of all three released checkpoints by 1.3× to
1.6× the English floor. Training our own runs to 600 epochs adds nothing detectable, and
their own best-`val_metric` checkpoints land at epochs 150 to 200.

Training budget is therefore eliminated as an explanation for the distance to 0.080, and it
was the leading explanation for the first three days of this project. What remains is
measured in two parts and one part is still open: the rasterizer convention accounts for
roughly half the Chinese distance, and a Chinese-specific residual of about 0.037 does not
have an explanation yet. The live lead is the training data. Sec. 4.1 says the Chinese
training set is "enlarged ten times" by affine augmentation; counting the built dataset
gives 1,272 entries over 212 base fonts, which is six times, not ten. That is exactly the
shape of cause that produces a Chinese-specific gap while English, whose 8,035 training
fonts need no augmentation, reproduces cleanly on the same code path. Verifying it needs a
retrain on a correctly augmented set, which is affordable on Chinese at about an hour per
run and is the single most interesting experiment still available here.

Two things are worth stating plainly. A reproduction can be exact at the level of the
artifact and still not land on the reported figure, which means the reported figure was
carrying information about the evaluation that the paper does not state. And this is only
visible because the released weights exist: every diagnostic above required a checkpoint to
compare against, and without one the strongest conclusion available would have been the
much weaker "we did not reach 0.080, probably undertrained" that earlier drafts of this
report in fact carried. Releasing weights is what made the difference between an excuse and
a measurement.

### 6.7 The same change points in opposite directions on two scripts

Running the category batch on both languages was intended as coverage and produced
something else. E3, the refinement depth, is null and mixed-sign on Chinese and clears both
floors in the degrading direction on English at all three seeds. E5, the latent width, is
mixed-sign on Chinese and clears both floors degrading on English. E9 improves Chinese at
all three seeds and does not replicate on English at any of them. Three candidates, three
disagreements, and the disagreements are in direction rather than only in magnitude.

The two arms differ in script, in sequence length, in reference-shot count, in training set
size, and in the pipeline floor their metric sits on, so no single cause is isolated here.
What the pattern does support is a limit on how far a single-language ablation licenses a
claim about an architecture. The paper reports its ablation on English only and its
headline on both, which is standard practice, and these results suggest that the ablation
generalizes across scripts less readily than that arrangement implies.

### 6.8 Limitations, and what we would do next

The Chinese floor and both English floors rest on three seeds each, so every bar in §5 has
its own uncertainty, demonstrated when the s-IoU floor moved 21% on re-measurement.
Checkpoint selection stayed on a criterion §6.4 shows to be defective, deliberately, because
fixing it properly invalidates the table. The English arm was scored on a 34-font
deterministic subset whose optimistic bias is measured at 0.0074 rather than assumed, and
at N = 50 rather than the paper's 10. E6's dead-parameter ablation was audited but never
run. E7 has no English result. Nothing here was run at more than three seeds, which is the
minimum that makes a sign test meaningful and well short of what would put a 0.004 effect
beyond argument.

Three things follow, in the order we would do them. Rebuild the Chinese training set at the
paper's stated 10× augmentation and retrain, which is cheap and is the only open lead on
the residual in §6.6. Add the validation-time rasterized hook of §6.4, which would let
checkpoint selection and scoring finally measure the same kind of thing. Then reconsider
the generative head: the current decoder is autoregressive over quantized coordinate bins,
and the quantization oracle of §5.6 shows the grid itself costs only 0.0021, so the bins
are not the bottleneck but the autoregressive factorization over them may be. A
flow-matching head operating on continuous coordinates, designed and costed in
`archive/FLOW_MATCHING_PLAN.md` and rejected for this project on schedule grounds rather
than on merit, would remove the quantization step and the sequential decode together.

The result we would most want checked by someone else is the simplest one. Take any
published vector-font ablation, retrain two rows of it at three seeds, and see whether the
ordering holds.


## 7. References

1. Wang, Y., Wang, Y., Yu, L., Zhu, Y., Lian, Z. *DeepVecFont-v2: Exploiting Transformers to
   Synthesize Vector Fonts with Higher Quality.* CVPR 2023. arXiv:2303.14585. The paper
   reproduced here; Sec. 3.1, 3.3 and 4.1 and Tab. 1–3 are cited throughout §1, §4 and §6.
2. Wang, Y., Lian, Z. *DeepVecFont: Synthesizing High-Quality Vector Fonts via Dual-Modality
   Learning.* ACM TOG 2021. The predecessor whose 0.086 Chinese figure sets the 0.006 margin
   §6.2 measures the seed floor against.
3. Carlier, A., Danelljan, M., Alahi, A., Timofte, R. *DeepSVG: A Hierarchical Generative
   Network for Vector Graphics Animation.* NeurIPS 2020. Baseline in the paper's Tab. 2.
4. Lopes, R. G., Ha, D., Eck, D., Shlens, J. *A Learned Representation for Scalable Vector
   Graphics.* ICCV 2019. Baseline in the paper's Tab. 2.
5. Liu, Y., Guo, F., Wang, Z., Zhang, D. *DualVector: Unsupervised Vector Font Synthesis with
   Dual-Part Representation.* CVPR 2023. Source of the SSIM / L1 / s-IoU reporting convention
   adopted in §2.2 and §5.
6. Wang, Z., Bovik, A. C., Sheikh, H. R., Simoncelli, E. P. *Image Quality Assessment: From
   Error Visibility to Structural Similarity.* IEEE TIP 2004. SSIM, the metric from the course
   material required by the brief; implementation and validation in §2.2.
7. Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., Zhang, H., Lan, Y., Wang, L.,
   Liu, T.-Y. *On Layer Normalization in the Transformer Architecture.* ICML 2020. Motivates
   the terminal LayerNorm of candidate E1 (§4.2).
8. Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., Wojna, Z. *Rethinking the Inception
   Architecture for Computer Vision.* CVPR 2016. Label smoothing, adapted to the ordinal
   coordinate bins in candidate E8 (§4.3).
9. Loshchilov, I., Hutter, F. *Decoupled Weight Decay Regularization.* ICLR 2019. AdamW,
   candidate E11 (§4.4).
10. Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S.,
    Lerchner, A. *β-VAE: Learning Basic Visual Concepts with a Constrained Variational
    Framework.* ICLR 2017. KL weighting, candidate E12 (§4.4).
11. Jaegle, A., Gimeno, F., Brock, A., Zisserman, A., Vinyals, O., Carreira, J. *Perceiver:
    General Perception with Iterative Attention.* ICML 2021. The architecture the released
    encoder's configuration block advertises and, per the dead-parameter audit in §1.1, does
    not instantiate.
12. Wilcoxon, F. *Individual Comparisons by Ranking Methods.* Biometrics Bulletin 1945. The
    paired signed-rank test used per seed in §5.2, with the Hodges–Lehmann shift as the
    accompanying effect size.
13. Hodges, J. L., Lehmann, E. L. *Estimates of Location Based on Rank Tests.* Annals of
    Mathematical Statistics 1963. The shift estimator reported beside each p-value in §5.2.
14. Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R. T. Q.,
    Lopez-Paz, D., Ben-Hamu, H., Gat, I. *Flow Matching Guide and Code.* arXiv:2412.06264.
    On the course reading list; basis for the generative head proposed as future work in §6.8
    and designed in `archive/FLOW_MATCHING_PLAN.md`.
15. Li, T., Tian, Y., Li, H., Deng, M., He, K. *Autoregressive Image Generation without Vector
    Quantization.* NeurIPS 2024. The per-token continuous generative head that design mirrors
    (§6.8).

### Code and artifacts

- Upstream implementation: `github.com/yizhiwang96/deepvecfont-v2`, mirrored on branch `main`.
  All work in this report is the `main..repro` diff.
- Released checkpoints: the authors' Chinese and English weights at epochs 500, 550 and 600,
  evaluated in §5.1 and §6.6.
- `RESULTS.csv`, 113 scored checkpoints, one row each, rebuilt by
  `scripts/build_results_table.py`. Every figure in §5 is derived from it.
- `PROJECT_PLAN.md`, the dated working record, including every decision reopened or overruled
  and the reasons, in §8.
