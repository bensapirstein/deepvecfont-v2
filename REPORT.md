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
baseline mean, with the same sign at all three seeds on both metrics. One candidate is a
reproducible defect: a terminal LayerNorm on the sequence encoder degrades s-IoU at all three
seeds by a mean of 0.0760, which is 2.4 times that metric's own floor.

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

`[TO WRITE from PROJECT_PLAN.md §3.4–§3.6. One paragraph per tier, the category each
candidate answers from the assignment list, and the one-factor rule. Tier 1: E9 encoder
noise, E1 terminal LayerNorm, E7 Bézier weight, E10 dropout. Tier 2: E8 ordinal label
smoothing, E13 bins and padding_idx, E3 refinement depth, E14 warmup-cosine. Tier 3A: E12 KL
weight, E11 AdamW, E2 image-branch norm, E4 width, E15 weight EMA, E5 latent width. Tier 3B:
the three largest deltas replicated at two further seeds. Note E2 instancenorm dropped at
smoke test: InstanceNorm2d cannot compute a per-instance variance at the [32,1024,1,1]
bottleneck, an architectural incompatibility rather than a wiring fault.]`

## 5. Results

`[TO WRITE from PROJECT_PLAN.md §5 and scripts/recompute_deltas.py output. Headline table is
the six confirmation rows plus the Wilcoxon column, already measured:`

| Row | Error (L1) ↓ | s-IoU ↑ | Wilcoxon p (L1, HL shift) |
|---|---|---|---|
| Baseline mean of three seeds | 0.1621 | 0.2681 | |
| **E9 σ_train = 0.5**, seed 1111 | 0.1588 | 0.2870 | 0.0012 (−0.0074) |
| **E9 σ_train = 0.5**, seed 2222 | 0.1531 | 0.3170 | 0.0437 (−0.0039) |
| **E9 σ_train = 0.5**, seed 3333 | 0.1623 | 0.2816 | 0.3050 (−0.0011) |
| **E9 mean of three seeds** | **0.1581** | **0.2952** | all same sign |

`Delta against the baseline mean: L1 −0.0040, s-IoU +0.0271. s-IoU Wilcoxon: p = 0.0001
(+0.0317), p = 0.0000 (+0.0453), p = 0.4417 (+0.0045). Test-time σ stays at the released 1.0:
both six-point ladders land inside the 0.0011 decode-noise band. Add the full screening table
from` scripts/recompute_deltas.py `and the E1 negative result: mean s-IoU −0.0760 across the
three seeds, 2.4× the 0.0315 floor, with its seed-3333 leg at s-IoU 0.1095, the lowest value
in the whole table by a wide margin.]`

## 6. Discussion

`[TO WRITE. Seven findings, each already measured. Findings 6 and 7 arrived last (2026-08-06)
and are the strongest in the report, so write them first and let the order below become the
order of presentation only if that still reads well afterwards:`

1. **Why every delta in this report is quoted against a three-seed mean rather than against a
   single baseline run**, and it is the one to lead with. Write it as method, not as
   confession: we trained the baseline three times before running any candidate (day 1,
   dated), so we can show directly what single-seed anchoring would have done to the same 26
   results. Referenced to seed 1111, **22 of 26 candidates "beat" the baseline**, which reads
   as *almost any arbitrary change helps* and is not credible. Referenced to the three-seed
   mean, **6 of 26** do. The difference is that seed 1111 sits 0.0048 above the mean, well
   inside the seed spread. Anyone reporting a single-seed baseline in this size class is
   reporting the draw, not the change. We can also quantify the cost: ranking candidates on
   the single-seed reference and replicating the top three at two further seeds, **1 of 3**
   survived. That is a measured base rate for "the leading single-seed candidate is real",
   and it is the number to hand a reader.
2. **Architecture moves the metric about as much as the seed does**, 0.0101 against 0.0097
   across 26 changes and 3 re-seedings. Twenty-six draws from one distribution should span
   roughly 2.3 times the range of three; the observed ratio is 1.07.
3. **L1 and s-IoU are near-orthogonal here**, r = −0.335, r² = 0.11, permutation p = 0.07.
   Reporting one metric would have missed both E9's case and E1's defect.
4. **The cheap proxy does not work**, ρ = 0.125. §3.4.
5. **E9 replicates across budgets** at reduced magnitude, and the seed spread does *not*
   shrink at confirmation budget, exactly as the noise decomposition predicted.

`Then: what the metric cannot see (§2.2) and what the oracle floor implies about the ceiling
on any coordinate-level change, including the E13 upper bound. Numbers, measured 2026-08-05
(PROJECT_PLAN.md §2.3): pipeline floor (n=inf) L1 0.1422, below the three-seed baseline mean
of 0.1621. **Do not read that headroom as "close it by training more" -- earlier drafts did,
and the 2026-08-06 official-checkpoint measurement kills it**: the authors' own 600-epoch
weights sit at 0.1629, so four times the budget does not consume the 0.02 of headroom either.
The floor is better read as the raster convention's own offset, which the sixth finding below
makes explicit. Quantization cost at the released 128-bin grid is only +0.0021 over that
floor. The E13 upper bound (128 to 256 bins) is +0.0016, under the 0.0097 seed floor -- E13
could not have cleared it no matter how it landed, which is why it screened null. Then the
floor-estimation point: a floor computed from three points is itself noisy, and s-IoU's moved
from 0.0401 to 0.0315 on re-measurement, which is the same point one level up: a reference
computed from few samples is itself a sample.`

`Sixth finding, MEASURED 2026-08-06 (`docs/official-checkpoints-and-600.md`), and it is the
strongest single result in the report: **a metric is a pipeline, not a formula, and here two
implementations of the same formula differ by more than any effect the sweep set out to
measure.** "Reconstruction error" turns out to depend on a choice nobody states: the
candidate is rendered through cairosvg, while the ground truth is the dataset's own
pre-rendered raster from a different rasterizer. Two rasterizers, one comparison. Scoring the
ground-truth outline through the candidate's rasterizer removes the term, and on one fixed
released checkpoint that single change moves Chinese from 0.1629 to 0.1174. **0.0455, against
a paper whose entire margin over its predecessor on this language is 0.006.** The general
lesson for section 6 writes itself, and the specific lesson is sharper than the general one:
the effect is 0.0455 on Chinese and 0.0074 on English, tracking the two pipeline floors
(0.1422 and 0.0253) rather than the two models, so it is a property of a dataset's build and
not a constant one can correct for once.`

`Seventh finding, same session, and the one to close section 6 on: **we reproduced the
released model without reproducing the published number, and those are different claims.**
Our 150-epoch baseline and the authors' released 600-epoch checkpoint are indistinguishable
through this evaluation, 0.1621 against 0.1629, inside the seed floor. Four times the epochs
buys nothing, which eliminates training budget. The convention accounts for about half the
remaining Chinese distance and a Chinese-specific 0.037 stays open, while English under the
same code reaches 0.0584 against a reported 0.052. Two things worth saying plainly: a
reproduction can be exact at the level of the artifact and still not land on the reported
figure, which means the reported figure was carrying information about the evaluation that
the paper does not state; and this is only visible because the released weights exist. Every
diagnostic in section 2.4 that separated our error from theirs required a checkpoint to
compare against, and without one the honest conclusion available to us would have been the
much weaker "we did not reach 0.080, probably undertrained" that earlier drafts of this
report in fact carried. Releasing weights is what made the difference between an excuse and
a measurement. Close with the flow-matching head as future work, citing
`archive/FLOW_MATCHING_PLAN.md`.]`

## 7. References

`[TO WRITE from PROJECT_PLAN.md §10.]`
