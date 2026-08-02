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
| Missing for Stage 1 | SSIM, renderability rate, per-font CSV, quantization oracle, bin histogram |
| Missing for Stage 2 | Everything below §3, plus the seed-noise floor |

0.1668 sits essentially on DeepSVG's published Chinese number (0.167), so the baseline is inside the benchmark's range even though it does not reach the paper. §2.4 says how to write that up. It is not a blocker for Stage 2, which is measured against your own baseline rather than against 0.080.

Infrastructure landed today: `--seed`, wandb mirroring alongside TensorboardX, `--max_ckpt_keep` defaulting to 1, and three Stage 2 flags declared but not yet wired. Details in `docs/infra-upgrade.md`.

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

**`val_metric` is not comparable across candidates that change the loss.** `compute_val_loss` builds it as `loss_w_l1 · img_l1 + loss_w_pt_c · vggpt + svg_total`, and the checkpoint filename embeds it. Any candidate that alters the cross-entropy itself changes what the number means, and `prune_checkpoints` will then be selecting on a different quantity. E8 and E13 are the two affected candidates; §3.2 handles them.

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

**Bin histogram.** Histogram `numericalize` output over the ground-truth training data and count the mass in bins 0 and 127, for Chinese and for English. `numericalize` does `.clip(min=0, max=n-1)`, so any coordinate outside `[0, 30]` is destroyed. Separately, `SVGEmbedding.arg_embed = nn.Embedding(128, 128, padding_idx=0)` maps bin 0 to a frozen zero vector, so a coordinate at the lower boundary is truncated on the way out and unrepresented on the way in. One script, trivially checkable, and it makes a good figure.

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

**Screen cheap, confirm expensive.** Do not run `test_few_shot.py --n_samples 50` over all 34 test fonts for every candidate.

- *Screening eval*: 8 fixed test fonts, `--n_samples 3`, at a fixed epoch. Minutes.
- *Confirmation eval*: all 34 fonts, `--n_samples 50`, matching the paper. Finalists only.

Freeze the screening font list and `ref_char_ids` before the first run and never change them.

**Check whether `val_metric` predicts the test metric.** `compute_val_loss` produces `val_metric` free at every checkpoint, and it now also goes to wandb as `CKPT/val_metric`. It is a teacher-forced loss; the test metric is a rendered, autoregressively decoded, best-of-N L1. How well they correlate on this model is unknown. After the first six experiments, compute the **rank correlation between `val_metric` and screening Error** across those six. High correlation means you screen everything else for free and spend the saved time on more candidates. Low correlation means the validation loss is not a usable proxy, which is a reportable finding in itself, and you screen on the rendered metric from then on.

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

**E8 — Ordinal label smoothing on the argument head.** *Maps to: modify the loss function, add regularization.*

The 128-way cross-entropy over quantized coordinates is permutation-invariant in the bin index: predicting bin 5 when the target is 60 costs exactly what predicting bin 61 costs. The head has no notion that coordinates live on a line, which is why the Bézier and smoothness losses have to reach back through a temperature-0.1 softmax and a straight-through estimator to recover geometry.

Replace the one-hot target `F.one_hot(tgt_args, 128)` in `Transformer.loss` with a discretized Gaussian centred on the true bin, σ ≈ 1–2 bins, renormalized. Five lines. No architecture change, no inference change, no extra cost.

This is the cheap version of the argument the flow-matching plan was built on, and it is the most interesting idea here from a modelling standpoint. Sweep σ ∈ {0.5, 1.0, 2.0} bins. Screen on the rendered metric, since it changes the loss scale.

**E13 — 256 quantization bins, and drop `padding_idx`.** *Maps to: change the encoder or decoder.*

Sec. 3.1 specifies 256; the model path uses 128. Doubling the bins halves the ±0.3125 px rounding error from §2.3.

Touch points: `numericalize(n=128)` and `denumericalize(n=128)` in `models/transformers.py`, `SVGEmbedding.arg_embed`, `args_fcn` output width, the `reshape(N, S, 8, 128)` calls, and `F.one_hot(tgt_args, 128)`. Five constants, all findable by grepping `128`. Careful: `arg_embed` is `nn.Embedding(128, 128)`, where the two 128s mean different things, vocabulary and embedding width. Only the first changes.

While there, drop `padding_idx=0` from `arg_embed`, or shift arguments by +1. Bin 0 is a legitimate coordinate and currently gets a frozen zero embedding.

**Gated twice, and the gate is cheap.** Run this only if the §2.3 bin histogram shows a filled distribution rather than a comb, and only if the oracle then shows the quantization floor is a material fraction of your gap. Per §1.5, there is a second quantization grid at n=64 in the preprocessing path, and if the training sequences went through it then adding bins to the head adds resolution the data does not have. Both checks are minutes of work and they run on day 1. Screen on the rendered metric.

**E3 — Self-refinement decoder, 1 layer → 2.** *Maps to: add residual or attention layers, change the decoder.*

Sec. 3.3 states the refinement module "is actually a 2-layer Transformer decoder". The code has `clones(DecoderLayer(...), 1)`. Change `1` to `2`. One character.

It earns its slot because, per §1.1, the refinement decoder produces the output that is actually scored. Try 3 layers while you are there.

**E14 — Warmup and cosine LR.** *Maps to: etc.*

`ExponentialLR(gamma=0.997)` stepped per epoch. Over 150 epochs that is `0.997^150 = 0.64`, so the learning rate barely moves and the schedule is effectively constant at the budget you are training to. There is no warmup, on a 6-layer transformer decoder with Adam at 2e-4.

Add linear warmup over the first ~500 steps and cosine decay to the epoch budget. Standard transformer practice, and it disproportionately helps short runs, which is what your entire matrix consists of. If it wins, apply it to every subsequent run and say so in the protocol section.

### 3.6 Tier 3 — if the budget allows

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
| Wed 12 – Thu 13 | 10–11 | Combine winners (§3.7). Confirmation eval on baseline and combined. Paired Wilcoxon. |
| Thu 13 – Fri 14 | 11–12 | Report. |
| Sat 15 Aug | 13 | Buffer and submit. |

**Where the compression bites.** The original fourteen-day sketch had two clear days for the report and a full buffer day. Thirteen days leaves about a day and a half of writing with the buffer folded into submission day, and days 11 and 13 are double-booked. Two ways to buy that back, both in §8: drop Tier 3 entirely, or start writing the reconstruction section on day 2 when Stage 1 closes rather than at the end. The second is close to free, because §2 produces every number that section needs.

English is out of scope unless days 9 to 11 come in early. If it fits, run only the combined model against the existing English baseline, with the same confirmation protocol and `--n_samples 10` to match the paper.

---

## 5. Results table

One row per candidate, so the sweep itself is the evidence.

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

1. **Local, no GPU.** `python scripts/check_infra.py`. Verifies flag names, types and defaults; verifies `--wandb False` actually disables; statically verifies the `train.py` wiring; runs a live wandb init/log/finish in offline mode. 59 checks, seconds. Passing as of today.
2. **Cluster, GPU, 2 epochs.** The smoke test in `docs/infra-upgrade.md`. Confirms a run appears in wandb with config and scalars, and that a checkpoint is written and pruned.
3. **Cluster, full.** The 3-seed baseline from §3.2.

---

## 8. Decisions still open

These are yours. The plan does not commit to them.

1. **Tier 2 scope.** Four candidates (E8, E13, E3, E14) in three days is realistic only if the budget calibration in §3.3 lands in the ≤3h band. Which do you keep if it does not? E3 is one character and E14 helps short runs generally, so those two are the cheap keeps. E8 is the most interesting modelling idea. E13 is gated on the oracle result anyway.
2. **Tier 3 at all.** Day 9 is the only slot, and dropping it buys back the report time §4 flags as tight.
3. **English.** Currently out of scope. It is a second language column in the results table and a stronger reconstruction section, against roughly a day.
4. **Screening budget.** 60 epochs doubles the candidate count and needs the one validation run described in §3.3. 150 epochs everywhere is safer and roughly halves the matrix.
5. **Fix the §1.5 eval-script issue, or work around it.** If the per-checkpoint layout is genuinely broken, fixing it properly is 20 minutes and touches a graded diff. Worth doing.
6. **What `--max_ckpt_keep 1` means for the sweep.** Keeping one checkpoint per run is right for disk, but if you later want to score a candidate at both 60 and 150 epochs you need both. Consider 2 for the runs that feed §3.3's validation.

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
