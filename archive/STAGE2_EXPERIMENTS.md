> **ARCHIVED, 2026-08-02. Superseded by `../PROJECT_PLAN.md`.**
>
> Not superseded on merit: the sweep strategy, the protocol and the experiment list
> in this file were carried into `PROJECT_PLAN.md` essentially intact. What changed is
> scope. Stage 1 was orphaned across two documents (this one referred its metric work
> back to the archived flow-matching plan), the schedule had no anchor date, and the
> paper-versus-code discrepancy list was duplicated in both files.
>
> `PROJECT_PLAN.md` now carries both stages, the shared context they both depend on,
> the dated schedule against the 15 August deadline, and an explicit list of decisions
> still open. Edit that file, not this one.
>
> Kept for provenance and because the §1 argument for a sweep over a single large bet
> is worth having in its original form.

# Final Project Part 2 — Stage 2 Experiment Plan

Paper: Wang et al., *DeepVecFont-v2*, CVPR 2023. Language: Chinese, from scratch, matched budget.

Strategy: instead of one large architectural bet, run a controlled sweep of single-factor modifications drawn from the categories the assignment lists, measure each against a seed-noise floor, and combine the ones that clear it into the submitted improved model.

This is a better report than one big idea. The rubric asks for "what worked, what did not, and what you learned," and a sweep answers that by construction. It also removes the failure mode where the single idea does not land and there is nothing to write.

Superseded plan: `archive/FLOW_MATCHING_PLAN.md`. Its §2 (metrics) is carried forward here unchanged and is still mandatory.

---

## 1. The protocol matters more than the experiment list

The paper's entire margin over its predecessor on Chinese is **0.006** absolute (0.086 → 0.080). Every candidate below will produce a delta in that neighbourhood or smaller. Three consequences drive the whole schedule.

### 1.1 Measure the seed-noise floor before anything else

`train.py` calls `setup_seed(1111)` and nothing varies it. Run the **baseline three times, seeds 1111 / 2222 / 3333**, at the screening budget. The spread of the screening metric across those three runs is your resolution limit. Any candidate whose improvement falls inside it is not a result, and reporting it as one is the single most likely way to lose marks in the discussion section.

This costs three screening runs and it is the highest-value thing in this document. Do it first, on day one, in parallel with the metric work in §2.

### 1.2 Screen cheap, confirm expensive

Do not run `test_few_shot.py --n_samples 50` over all 34 test fonts for every candidate. Two tiers:

- **Screening eval**: 8 fixed test fonts, `--n_samples 3`, at a fixed epoch. Minutes, not hours.
- **Confirmation eval**: all 34 fonts, `--n_samples 50`, matching the paper. Finalists only.

Freeze the screening font list and `ref_char_ids` before the first run and never change them.

### 1.3 Check whether `val_metric` predicts the test metric

`compute_val_loss` already produces `val_metric` for free every checkpoint. It is a teacher-forced loss; the test metric is a rendered, autoregressively-decoded, best-of-N L1. Nobody knows how well they correlate on this model.

After the first six experiments, compute the **rank correlation between `val_metric` and screening Error** across those six. If it is high, screen everything else on `val_metric` at zero cost and spend the saved time on more candidates. If it is low, you have learned that the validation loss is not a usable proxy, which is itself a reportable finding, and you screen on the rendered metric from then on.

Caveat: `val_metric` is comparable across candidates only when the loss definition is unchanged. **E8 (label smoothing) and E13 (256 bins) change the cross-entropy itself**, so those two must always be screened on the rendered metric regardless.

### 1.4 One factor at a time, then combine

Every run differs from the baseline in exactly one flag. Combine winners only at the end, and verify the combination beats each part alone. Interactions are real here: dropout, KL weight and encoder noise are all regularizers pulling on the same slack, and stacking three of them will likely underperform the best one.

---

## 2. Calibrate the budget before sizing the matrix (day 1)

Time five epochs of the Chinese baseline and extrapolate. From the checkpoint names, `125_5040` implies ~40 steps per epoch, so a 150-epoch Chinese run is around 6,000 steps. What that costs in wall-clock depends on throughput you have not measured.

| Measured cost of one 150-epoch Chinese run | Matrix to run |
|---|---|
| ≤ 3 h | Full: everything in §4, all sweep points |
| 3–8 h | Tiers 1 and 2, single sweep points, seeds on baseline only |
| > 8 h | Tier 1 only, and drop the screening budget to 60 epochs |

Relative ordering between variants usually shows up well before convergence, so a **60-epoch screening budget with a 150-epoch confirmation for finalists** is a legitimate way to double the number of candidates. Validate that assumption once: take one Tier-1 candidate, score it at 60 and at 150, and check the sign of the delta agrees.

You have at least three GPUs in play (`CUDA_VISIBLE_DEVICES` 1 and 2 in `COMMANDS.md`). Run three candidates concurrently.

---

## 3. Stage 1 metrics, carried forward and still required (day 1–2)

Unchanged from the archived plan. This is not Stage 2 work, it is the measuring instrument that every Stage 2 comparison depends on, and it is also a graded deliverable.

### 3.1 The paper's metric

DeepVecFont-v2 reports one number, "Error" (Sec. 4.1): the average L1 distance between the rasterized synthesized vector glyph and the ground-truth glyph image at 64×64. Both sides are binary masks, so the mean absolute difference is the fraction of the 4096 pixels that disagree. `Error = 0.080` reads as "8.0% of pixels are wrong". It is a rasterized proxy for vector fidelity: it says nothing about command count, self-intersection, or control-point placement, only whether the inked region lands in the right place. It is also scored after best-of-N_s selection, so it measures the best candidate rather than the average sample.

| Model | Error-EN ↓ | Error-CN ↓ |
|---|---|---|
| DeepSVG | 0.125 | 0.167 |
| DeepVecFont | 0.056 | 0.086 |
| DeepVecFont-v2 | 0.052 | 0.080 |

Ablation (Tab. 1, English): base 0.0588 → +relaxation 0.0557 → +Bézier alignment 0.0529 → +self-refinement 0.0519.

Already implemented correctly in `eval_reconstruction_error.py`.

### 3.2 What to add

**SSIM.** The metric from class. Gaussian window, `data_range=1.0`. It complements L1: L1 counts disagreeing pixels wherever they fall, SSIM penalizes disagreement that breaks local structure. A glyph with a uniformly slightly-too-thick stem and a glyph with one mangled stroke can share an L1 score and separate cleanly on SSIM.

**IoU.** Already computed. Label it s-IoU when comparing to the vector-font literature; DualVector (CVPR 2023) reports SSIM / L1 / s-IoU on this task, so the triple is the field's convention.

**Renderability rate.** `test_few_shot.py` wraps `render()` in a bare `except: continue`, and `eval_reconstruction_error.py` then skips any font whose merge HTML lacks exactly `2 × char_num` SVGs. A model that fails on hard glyphs is currently rewarded by having those fonts dropped from the average. Log the fraction of glyphs that fail to render and the fraction of fonts skipped, **per candidate**. If two candidates differ here their Error values are not comparable. With a sweep of fifteen variants this will bite at some point.

**Quantization oracle.** Push ground-truth sequences through `numericalize` then `denumericalize`, render, score identically. This is the floor no model with the 128-bin head can beat. One pass over the test set, and it is the reference row that tells you whether E13 is worth running.

**Per-font CSV.** `eval_reconstruction_error.py` already accumulates per-font means. Write them out. With 34 test fonts, run a **paired Wilcoxon signed-rank** on per-font Error, candidate against baseline. Given expected deltas around 0.005, comparing two grand means is not evidence.

**Bin histogram.** Histogram `numericalize` output over the ground-truth training data and count the mass in bins 0 and 127. `numericalize` does `.clip(min=0, max=n-1)`, so any coordinate outside `[0, 30]` is destroyed. Separately, `SVGEmbedding.arg_embed = nn.Embedding(128, 128, padding_idx=0)` maps bin 0 to a frozen zero vector, so a coordinate at the lower boundary is truncated on the way out and unrepresented on the way in. This one script decides whether E13 goes in Tier 2 or gets dropped.

---

## 4. The experiment list

Every entry maps to a bullet on the assignment sheet. The mapping column is there so the report can say so explicitly.

### Tier 1 — run first, highest expected value per GPU-hour

**E9 — Encoder noise scale.** *Maps to: change the noise schedule.*

`models/transformers.py:450`, at the end of the sequence encoder's forward pass:

```python
x = x + torch.randn_like(x)   # add a perturbation
```

Unit-variance Gaussian noise, added unconditionally, in training and validation and test. This is the paper's Sec. 4.1 noise that "simulates the feature distortion caused by the human-designing uncertainty," except the paper describes it as an *inference-time* device and the code applies it during training too. The scale σ=1 appears to be arbitrary and has never been tuned. It is applied to a 512-dimensional residual stream whose scale is unknown (see E1), so the effective perturbation could be 20% or 200%.

Add `--enc_noise_std_train` and `--enc_noise_std_test` as separate flags. Sweep train σ ∈ {0, 0.1, 0.25, 0.5, 1.0} with test σ held at 1.0.

Keep test-time σ > 0: it is the *only* source of stochasticity at inference, and therefore the only reason the N_s candidates differ from each other. Setting it to zero makes `--n_samples 50` produce fifty identical glyphs. Sweep it separately, second, on the winning train σ.

Optional extra: anneal train σ from 1.0 to 0.1 over training. That is a noise schedule in the literal sense the assignment means.

This is the biggest untuned knob in the model and it is a one-line change.

**E1 — Final LayerNorm on the sequence encoder.** *Maps to: add normalization layers.*

Both transformer stacks are pre-norm (`PreNorm` in the encoder, `SublayerConnection` applying `norm` before the sublayer in the decoder). A pre-norm stack requires a terminal LayerNorm after the last block, otherwise the residual stream leaves the encoder unnormalized and its scale grows with depth.

The decoder has this: `decoder_norm` and `decoder_norm_parallel`. **The encoder does not.** `Transformer.forward` runs its 12 self-attention blocks and goes straight to the noise injection with no final norm, and `att_residual` has the same gap.

Add `nn.LayerNorm(512)` at the end of both. One line each. It is a genuine architectural omission, it is the assignment's first bullet, and it interacts with E9 in a principled way: normalizing the stream is what makes σ a meaningful quantity rather than an arbitrary one. Run E1 alone, E9 alone, and E1+E9 together.

**E7 — Bézier alignment loss weight.** *Maps to: modify the loss function.*

Eq. 11 of the paper sets the weight of `L_bézier` to **1.0**. `options.py` sets `loss_w_aux = 0.01`, a factor of 100 below the paper. The paper's own ablation (Tab. 1) credits this loss with 0.0028 of its total 0.0069 gain over the base model, the second-largest single contribution.

Sweep `loss_w_aux` ∈ {0.01, 0.1, 0.3, 1.0}. One flag, already exists.

Frame this honestly in the report: it is simultaneously a reconstruction-fidelity correction and a Stage 2 loss modification. Saying so is better than pretending it is only one of them.

**E10 — Dropout.** *Maps to: add regularization.*

Every dropout in the model is zero: `MultiHeadedAttention(dropout=0.0)`, `PositionwiseFeedForward(dropout=0.0)`, `attn_dropout=0.`, `ff_dropout=0.`. The Chinese training set is 212 fonts expanded 10× by affine augmentation, which is small for a 512-wide, 12-block encoder plus a 6-layer decoder.

Plumb one `--dropout` flag through and sweep {0, 0.1, 0.2}. Note the interaction with E9: both are regularizers, so run E10 at the baseline noise σ, and re-check the winner jointly at the end.

### Tier 2 — small structural diffs, good ideas, slightly more code

**E8 — Ordinal label smoothing on the argument head.** *Maps to: modify the loss function, add regularization.*

The 128-way cross-entropy over quantized coordinates is permutation-invariant in the bin index: predicting bin 5 when the target is 60 costs exactly what predicting bin 61 costs. The head has no notion that coordinates live on a line, which is why the Bézier and smoothness losses have to reach back through a temperature-0.1 softmax and a straight-through estimator to recover geometry.

Replace the one-hot target `F.one_hot(tgt_args, 128)` in `Transformer.loss` with a discretized Gaussian centred on the true bin, σ ≈ 1–2 bins, renormalized. Five lines. No architecture change, no inference change, no extra cost.

This is the cheap version of the argument the flow-matching plan was built on, and it is the most interesting idea here from a modelling standpoint. Sweep σ ∈ {0.5, 1.0, 2.0} bins. Must be screened on the rendered metric, since it changes the loss scale.

**E13 — 256 quantization bins, and drop `padding_idx`.** *Maps to: change the encoder or decoder.*

Sec. 3.1 of the paper describes one-hot arguments of **256** dimensions (`δp ∈ R^256×8`, `W_args^b ∈ R^dE×256`). The code uses **128**, and the stale comment on `args_fcn` still reads `# shape: bs, max_len, 8, 256`. Bin width is currently `30/128 = 0.2344` viewBox units, which at a 24-unit viewBox rendered to 64×64 is `0.625` px, with per-coordinate rounding error uniform on ±0.3125 px. Doubling the bins halves that.

Touch points: `numericalize(n=128)`, `denumericalize(n=128)`, `SVGEmbedding.arg_embed`, `args_fcn` output width, the `reshape(N, S, 8, 128)` calls, and `F.one_hot(tgt_args, 128)`. Five constants, all findable by grepping `128`. Careful: `arg_embed` is `nn.Embedding(128, 128)`, where the two 128s mean different things (vocabulary and embedding width). Only the first changes.

While there, drop `padding_idx=0` from `arg_embed`, or shift arguments by +1. Bin 0 is a legitimate coordinate and currently gets a frozen zero embedding.

Run only if the §3.2 oracle shows the quantization floor is a material fraction of your gap to 0.080. Must be screened on the rendered metric.

**E3 — Self-refinement decoder, 1 layer → 2.** *Maps to: add residual or attention layers, change the decoder.*

Sec. 3.3 states the refinement module "is actually a 2-layer Transformer decoder." The code has `decoder_layers_parallel = clones(DecoderLayer(...), 1)`. Change `1` to `2`. One character.

Worth its Tier-2 slot because the refinement decoder produces the output that is actually scored: `test_few_shot.py` selects on `syn_{i}_{sample}_refined.svg`, from `sampled_svg_2`, and the merge HTML that `eval_reconstruction_error.py` reads is built from those files. Also try 3 layers while you are there.

**E14 — Warmup and cosine LR.** *Maps to: etc.*

`ExponentialLR(gamma=0.997)` stepped per epoch. Over 150 epochs that is `0.997^150 = 0.64`, so the learning rate barely moves and the schedule is effectively constant at the budget you are training to. There is no warmup, on a 6-layer transformer decoder with Adam at 2e-4.

Add linear warmup over the first ~500 steps and cosine decay to the epoch budget. This is standard transformer practice and it disproportionately helps short runs, which is what your entire matrix consists of. If it wins, apply it to every subsequent run and say so in the protocol section.

### Tier 3 — run if the budget allows

| ID | Change | Maps to | Note |
|---|---|---|---|
| E5 | `bottleneck_bits` ∈ {128, 256, 512, 1024} | change the latent dimension | **Gotcha below** |
| E2 | Image encoder/decoder norm: spatial `LayerNorm([C,H,W])` → `GroupNorm` / `BatchNorm2d` / `InstanceNorm2d` | add normalization layers | Current norm couples channel and spatial statistics, which discards per-channel scale |
| E12 | `kl_beta` ∈ {0, 0.003, 0.01, 0.03, 0.1} | add regularization | Note the interaction with E9: with σ=1 additive noise on the encoder output, the reparameterization is nearly decorative |
| E11 | `Adam` → `AdamW`, `weight_decay` 0 → 0.01 | add regularization | `AdamW` is already imported in `train.py` and unused |
| E4 | `ngf` 16 → 32 | change the encoder or decoder | Widens both image encoder and decoder; check `fc_fusion` still matches |
| E15 | EMA of weights for evaluation | etc. | Reliable small gain, ~15 lines, no effect on training dynamics |
| E6 | The Perceiver cross-attention path is dead code | change the encoder | See below |

**E5 gotcha.** `--bottleneck_bits` looks like a free flag but is not. `ModalityFusion` does `seq_feat_[:, 0] = z`, and `seq_feat_` has last dimension 512 fixed by `fc_merge = nn.Linear(seq_latent_dim * ref_nshot, 512)`. Any `bottleneck_bits ≠ 512` fails there. Add a `nn.Linear(bottleneck_bits, 512)` projection before the assignment. Small, but it is a code change rather than a flag flip, so budget for it.

**E6 note.** In `Transformer.forward` the loop unpacks `cross_attn, cross_ff, self_attns` and uses only `self_attns`. The cross-attention modules and `self.latents = nn.Parameter(torch.randn(256, 512))` are constructed, handed to the optimizer, and never called. With `depth=6` and `self_per_cross_attn=2` the encoder is 12 self-attention blocks and zero cross-attention blocks, so the model is not the Perceiver its configuration block advertises. The authors left a comment on that loop marking it as a known problem. Either delete the dead parameters, which is a clean finding for the reconstruction section and slightly reduces optimizer state, or wire the cross-attention to the image features, which is a real architecture change and belongs here only if everything else is done.

---

## 5. Two-week schedule

| Days | Work |
|---|---|
| 1 | Time 5 epochs, size the matrix (§2). Launch 3-seed baseline (§1.1). Bin histogram (§3.2). |
| 1–2 | Extend `eval_reconstruction_error.py`: SSIM, renderability, per-font CSV. Quantization oracle. Score the existing epoch-100 and epoch-125 Chinese results. **Stage 1 closes.** |
| 3–5 | Tier 1: E9 sweep, E1, E7 sweep, E10 sweep. Three GPUs in parallel. |
| 5 | Compute the `val_metric` vs rendered-Error rank correlation (§1.3). Decide the screening metric for the rest. |
| 6–8 | Tier 2: E8, E13, E3, E14. Plus test-time σ sweep on the E9 winner. |
| 9 | Tier 3, as many as fit. |
| 10–11 | Combine winners. Run the combined model and each ablation-of-the-combination. Confirmation eval (34 fonts, `n_samples 50`) on baseline and combined. Paired Wilcoxon. |
| 12–13 | Report. |
| 14 | Buffer. Something will break. |

English is out of scope unless days 9–11 come in early. If it does fit, run only the combined model against the existing English baseline, with the same confirmation protocol and `--n_samples 10` to match the paper.

---

## 6. Results table for the report

One row per candidate, so the sweep itself is the evidence.

| Row | Error ↓ | SSIM ↑ | s-IoU ↑ | Render % | Wilcoxon p |
|---|---|---|---|---|---|
| Paper, reported (CN) | 0.080 | — | — | — | — |
| Quantization oracle (floor) | | | | | — |
| **Reconstruction** (baseline, seed 1111) | | | | | — |
| Baseline, seed 2222 | | | | | — |
| Baseline, seed 3333 | | | | | — |
| E1 final encoder LayerNorm | | | | | |
| E7 `loss_w_aux = 0.3` | | | | | |
| E9 train σ = 0.25 | | | | | |
| E10 dropout = 0.1 | | | | | |
| ... one row per candidate ... | | | | | |
| **Improved model** (combined winners) | | | | | |

The three baseline seed rows are what license every claim below them. Put them in the table, not in a footnote.

---

## 7. Report structure, mapped to the rubric

1. **Original architecture.** Dual-branch encoder (CNN over reference glyph images, transformer over reference SVG sequences), modality fusion into a VAE latent, image decoder and autoregressive sequence decoder, context-based self-refinement. Losses: image L1 and perceptual, KL, command and argument cross-entropy, Bézier alignment (Eq. 9), relaxation consistency (Eq. 10). Explain the relaxation representation, since it is the paper's headline contribution and it is what makes each command carry 8 coordinate arguments.
2. **Paper results.** §3.1 above.
3. **Reconstruction results.** Your Chinese numbers against 0.080, with SSIM, s-IoU and renderability. Include the deviations you found between released code and paper text; these are legitimate reconstruction findings and three of them became experiments:
   - Sec. 3.1 specifies 256-dimensional one-hot arguments; the code quantizes to 128 (→ E13).
   - Sec. 3.3 specifies a 2-layer self-refinement decoder; the code has 1 (→ E3).
   - Eq. 11 weights `L_bézier` at 1.0; `options.py` uses 0.01, and splits `L_img` into `loss_w_l1 = 10` and `loss_w_pt_c = 0.01` rather than a single 1.0 (→ E7).
   - Sec. 4.1's inference-time `N(0,I)` noise is implemented as a perturbation of the sequence-encoder output applied during training as well (→ E9).
4. **Improved architecture.** Present the method as a controlled sweep, not a list of tweaks. State the categories from the assignment, the one-factor-at-a-time rule, the seed-noise floor, and the screening/confirmation split. Then describe the winners and why each was expected to help.
5. **Improved results.** The table from §6, with the paired Wilcoxon column and the oracle floor row.
6. **Discussion.** What the metric can and cannot see (sub-pixel accuracy is largely invisible to a rasterized 64×64 binary L1). Which candidates fell inside the seed noise and therefore prove nothing. Whether `val_metric` turned out to predict rendered Error. What the quantization floor implies about the ceiling on any coordinate-level change. One paragraph on the continuous-coordinate head that was scoped and rejected on schedule grounds, citing `archive/FLOW_MATCHING_PLAN.md`.
7. **References.** §8.

---

## 8. References

- Wang, Wang, Yu, Zhu, Lian. *DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality.* CVPR 2023.
- Wang, Lian. *DeepVecFont: Synthesizing High-Quality Vector Fonts via Dual-Modality Learning.* ACM TOG 2021.
- Carlier, Danelljan, Alahi, Timofte. *DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation.* NeurIPS 2020.
- Lopes, Ha, Eck, Shlens. *A Learned Representation for Scalable Vector Graphics (SVG-VAE).* ICCV 2019.
- Liu, Guo, Wang, Zhang. *DualVector: Unsupervised Vector Font Synthesis with Dual-Part Representation.* CVPR 2023. Source of the SSIM / L1 / s-IoU reporting convention.
- Wang, Xu, et al. *SSIM: Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP 2004. For the metric definition in §3.2.
- Szegedy et al. *Rethinking the Inception Architecture for Computer Vision.* CVPR 2016. Label smoothing, for E8.
- Xiong et al. *On Layer Normalization in the Transformer Architecture.* ICML 2020. Pre-norm and the terminal LayerNorm, for E1.
- Loshchilov, Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. For E11.
- Higgins et al. *β-VAE.* ICLR 2017. For E12.

---

## 9. Do this first

1. Time 5 epochs. Pick the matrix size from the table in §2.
2. Launch the 3-seed baseline. Nothing else is interpretable until it finishes.
3. Run the bin histogram and the quantization oracle while it trains.
4. Extend `eval_reconstruction_error.py` with SSIM, renderability and per-font CSV, and score your existing epoch-100 and epoch-125 results. Stage 1 is then done and can be written up while Stage 2 runs.
5. Add `--enc_noise_std_train`, `--enc_noise_std_test`, `--dropout` to `options.py`, and the two `nn.LayerNorm(512)` lines for E1. That is the entire code diff for all of Tier 1 except E7, which needs no code at all.
