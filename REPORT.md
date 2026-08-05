# Reproducing DeepVecFont-v2, and measuring what a sweep over it can resolve

Ben Sapirstein · Generative Models for Text and Images, Reichman University · August 2026

> **Drafting note, delete before submission.** Sections 1 to 3 are written. Sections 4 to 6
> are laid in against real numbers with the two gaps marked `[PENDING]`, both of which are
> filled by `docs/stage1-closeout.md` on the cluster: the SSIM column and the quantization
> oracle row. Drafted back to front from §5 and §6 so the early sections set up exactly what
> the discussion needs. Every number here is traceable to `RESULTS.csv` or to a dated
> **Measured** paragraph in `PROJECT_PLAN.md`; nothing is typed twice by hand.

---

## Abstract

We reproduce DeepVecFont-v2 (Wang et al., CVPR 2023) on Chinese vector font synthesis and
then run a controlled sweep of twenty-one single-factor architectural changes against it.
The reproduction reaches a rendered reconstruction error of 0.1621 (three-seed mean, 34 test
fonts, best-of-50 decoding) against the paper's reported 0.080. We attribute the gap
primarily to training budget, and we rule out three of the four candidate explanations by
measurement rather than by argument.

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
right place. Two further properties matter:

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

### 2.3 Reconstruction results

| Row | n_samples | Error (L1) ↓ | SSIM ↑ | s-IoU ↑ | Render % |
|---|---|---|---|---|---|
| Paper, DeepVecFont-v2 (CN) | 50 | 0.080 | | | |
| Paper, DeepVecFont (CN) | | 0.086 | | | |
| Paper, DeepSVG (CN) | | 0.167 | | | |
| Quantization oracle, pipeline floor (n = ∞) | | `[PENDING]` | `[PENDING]` | `[PENDING]` | |
| Quantization oracle, 128-bin grid | | `[PENDING]` | `[PENDING]` | `[PENDING]` | |
| **This reproduction, seed 1111** | 50 | 0.1662 | `[PENDING]` | 0.2545 | 34/34 |
| **This reproduction, seed 2222** | 50 | 0.1569 | `[PENDING]` | 0.2716 | 34/34 |
| **This reproduction, seed 3333** | 50 | 0.1632 | `[PENDING]` | 0.2781 | 34/34 |
| **Three-seed mean** | 50 | **0.1621** | `[PENDING]` | **0.2681** | 34/34 |

Renderability is reported because it is silently rewarded otherwise. `test_few_shot.py`
wraps its rasterizer in a bare `except: continue`, and a model that fails on hard glyphs gets
those glyphs dropped from its average rather than penalized. Every row in this report renders
1768 of 1768 glyphs, so no comparison here is confounded by it. One earlier Stage 1 checkpoint
did not: `dvf_base_exp_chn` at epoch 135 scored 0.1641 over 33 of 34 fonts, with font 0028
failing to render entirely, which is why the quantity is now reported per run.

### 2.4 The gap to 0.080

The reproduction lands at roughly double the paper's Chinese number. Four explanations were
identified in advance; three are now eliminated and the fourth accounts for the gap on its
own.

**Eliminated by measurement: an evaluation-harness fault.** The evaluation script's results
glob sat one directory level shallower than the layout `test_few_shot.py` writes, so against
the current tree every font was skipped silently. This was found and fixed, and the fixed
script reports the layout it detected on its first line. It reports `per-checkpoint`, so the
figure was scored against the tree its named checkpoint actually produced.

**Eliminated by reading the preprocessing driver: coarser quantization than assumed.** The
codebase carries two `numericalize` definitions on different grids, 128 bins in the model
path and 64 in `data_utils/relax_rep.py`, and the preprocessing copy round-trips the sequence
array in place through a numpy view. Had that mutation reached the persisted training data,
the head would predict over 128 bins that the data only populates at 64, and half the
vocabulary would be structurally unreachable. It does not. `relax_rep.process` writes
`sequence_relaxed.npy` *before* calling the mutating function and never saves the array
again, and the dataloader reads that file. The training sequences are at full float
resolution.

**Eliminated by arithmetic: a different test protocol.** This one looked live. The paper does
not fully specify `ref_char_ids`, `n_samples` or the font list, and all three move the number.
But §2.2 measured the largest of them: the whole best-of-N budget range is worth about 0.006,
and the reported figure is already taken at N = 50, the top of it. Granting the paper an
unstated advantage of similar size on the other two puts the entire protocol surface at
roughly 0.01 against a gap of 0.087. **Protocol accounts for a tenth of the gap at most.**

**Not eliminated, and sufficient alone: training budget.** The baseline ran 125 epochs and
the seed-floor runs 150. Three observations point the same way and none required an
experiment.

First, the validation loss was still falling when every run ended. Epoch 135 scores 0.1641
against epoch 125's 0.1668, and epoch 150 is the best-validation checkpoint for five of the
six confirmation-budget runs. A model still improving when its budget expires is
undertrained by definition.

Second, the English arm is an internal control. `dvf_base_exp_eng` reached epoch 600 on
51-step sequences, where the Chinese runs stopped at 150 on 71-step sequences. The one
configuration here that resembles a converged budget ran four times as long on an easier
task.

Third, and most persuasively, this comes free from Stage 2. **Twenty-six deliberate
single-factor changes to the architecture, optimizer, loss weighting, quantization grid,
latent width and regularization together span 0.0101 in L1** (§3.1). The gap to the paper is
0.087, roughly nine times that entire span. No architectural difference between this
reproduction and the paper's could plausibly be worth nine times the range of twenty-six
deliberate ones. Training budget can be, because it is the one axis the sweep never varied.

That the reproduction lands on DeepSVG's published Chinese result of 0.167 is a coincidence
worth stating and not worth leaning on. DeepSVG is a different model and the agreement
carries no information about this one.

**What this does not claim.** No run was made to test whether training longer reaches 0.080.
Confirming it costs a 600-epoch Chinese run, roughly four hours, and it would answer a
reproduction question with budget the sweep had better uses for. It is reported as the
leading explanation with its evidence and its status stated.

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
  epoch count and cannot be compared across budgets at all.

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

`[TO WRITE. The five findings, in this order, each already measured:`

1. **The anchor bias**, and it is the one to lead with. Every delta in the project was quoted
   against seed 1111, which turned out to be the worst of the three baseline draws and sits
   0.0048 above their mean. 22 of 26 candidates beat that anchor, which reads as "almost any
   arbitrary change helps" and is not credible; only 6 of 26 beat the mean. The replication
   batch was selected on that ranking and its selection survived **1 time in 3**. Nothing
   recorded was wrong, only the reading. The lesson generalizes past this project.
2. **Architecture moves the metric about as much as the seed does**, 0.0101 against 0.0097
   across 26 changes and 3 re-seedings. Twenty-six draws from one distribution should span
   roughly 2.3 times the range of three; the observed ratio is 1.07.
3. **L1 and s-IoU are near-orthogonal here**, r = −0.335, r² = 0.11, permutation p = 0.07.
   Reporting one metric would have missed both E9's case and E1's defect.
4. **The cheap proxy does not work**, ρ = 0.125. §3.4.
5. **E9 replicates across budgets** at reduced magnitude, and the seed spread does *not*
   shrink at confirmation budget, exactly as the noise decomposition predicted.

`Then: what the metric cannot see (§2.2) and what the oracle floor implies about the ceiling
on any coordinate-level change, including the E13 upper bound. Then the floor-estimation
point: a floor computed from three points is itself noisy, and s-IoU's moved from 0.0401 to
0.0315 on re-measurement, which is the anchor bias one level up. Close with the flow-matching
head as future work, citing` archive/FLOW_MATCHING_PLAN.md`.]`

## 7. References

`[TO WRITE from PROJECT_PLAN.md §10.]`
