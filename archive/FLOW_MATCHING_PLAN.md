> **ARCHIVED, 2026-08-01. Superseded by `STAGE2_EXPERIMENTS.md`, which was in turn absorbed into `../PROJECT_PLAN.md` on 2026-08-02.**
>
> Shelved on schedule risk, not on merit. Two weeks to submission leaves no room for a single change that touches the loss, the sampler, the AR feedback path and the refinement decoder at once, with one shot at getting it right.
>
> What was carried forward into the new plan and should not be re-derived:
> - §2, the Stage 1 metric work (SSIM, renderability rate, quantization oracle, per-font CSV, paired Wilcoxon). Required regardless of what Stage 2 turns out to be.
> - §3.1, the four properties of the quantized head. Now motivates two cheap experiments instead of one expensive one: 256 bins, and ordinal label smoothing over neighbouring bins.
> - The paper-versus-code discrepancy list. Feeds three experiments directly.
> - The checkpoint-selection warning about `val_metric`.
>
> Keep this file. The report's future-work paragraph writes itself from §3 and §4, and "we scoped a continuous-coordinate head, costed it, and rejected it on schedule" is a better sentence than silence.

# Final Project Part 2 — Plan (ARCHIVED)

Paper: Wang et al., *DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality*, CVPR 2023.

Stage 2 modification: replace the quantized-coordinate classification head of the autoregressive sequence decoder with a per-token **conditional flow-matching** head.

Decisions already fixed: Chinese trained from scratch, matched epoch budget against the existing baseline; command-type head stays cross-entropy.

> **Scope correction, read §3.4 first.** The intended scope was "AR coordinate head only". The code makes that scope the *more* expensive option, not the cheaper one, and it also removes most of the benefit. `self.args_fcn` is a single shared module called from both `Transformer_decoder.forward` (line 216) and `Transformer_decoder.parallel_decoder` (line 267), and the output that actually gets scored is the refined one. Swapping the shared module covers both decoders in one edit. §3.5 lays out the choice.

---

## 0. Why flow matching is the right Stage 2 for this course

The syllabus reading list (`1621393.pdf`, entries 85 and 87) contains exactly two flow-matching references, both of which are already sitting in the course folder:

- Lipman et al., *Flow Matching Guide and Code* (`2412.06264v1.pdf`)
- Holderrieth & Erives, *An Introduction to Flow Matching and Diffusion Models* (`2506.02070v2.pdf`)

Lectures 07 and 08 cover normalizing flows. Flow matching is the simulation-free training objective for continuous normalizing flows, so the modification sits inside a topic the course has already built up, and the report can lean on notation the grader has seen.

The assignment lists "modify the loss function" and "improve the flow coupling layer" as sanctioned examples. Swapping a categorical head for a learned transport map is both.

---

## 1. Where the project stands

| | |
|---|---|
| Baseline, Chinese | `dvf_base_exp_chn_main_model`, from scratch, checkpoints at epochs 100 and 125 (`125_5040_valloss3.8273.ckpt`) |
| Baseline, English | `600_192921_valloss2.0824.ckpt` available |
| Test runs | Chinese few-shot at epochs 100 and 125, `n_samples 50`, `ref_nshot 8` |
| Paper metric | Implemented in `eval_reconstruction_error.py` (rendered L1 + IoU) |
| Missing | A second metric for Stage 1, the quantization floor measurement, per-font statistics |

Stage 1 is close to done. The gap is metrics, which is also what makes Stage 2 legible, because the effect size here will be small and needs a measurement setup that can resolve it.

---

## 2. Stage 1: metrics

### 2.1 The paper's metric, and what it means

DeepVecFont-v2 reports a single number, "Error" (Sec. 4.1):

> the average L1 distance between the rasterized image of each synthesized vector glyph and its corresponding ground-truth glyph image (at the resolution of 64×64) in the testing set.

Because both sides are binarized masks, the mean absolute difference equals the fraction of the 4096 pixels where the two masks disagree. `Error = 0.080` reads as "8.0% of pixels are wrong". The metric is a rasterized proxy for vector fidelity: it says nothing about how many drawing commands were used, whether the outline self-intersects, or whether control points are placed sensibly, only whether the inked region lands in the right place.

Two consequences worth stating in the report:

1. The metric is scored at 64×64 after a best-of-N_s selection, so it measures the best candidate, not the model's average sample.
2. Sub-pixel coordinate accuracy is partly invisible to it. This caps how much any coordinate-level improvement can show up, and the discussion section should say so.

**Reported numbers (Tab. 2).**

| Model | Error-EN ↓ | Error-CN ↓ |
|---|---|---|
| DeepSVG | 0.125 | 0.167 |
| DeepVecFont | 0.056 | 0.086 |
| DeepVecFont-v2 | 0.052 | 0.080 |

Ablation (Tab. 1, English): base 0.0588 → +relaxation 0.0557 → +Bézier alignment 0.0529 → +self-refinement 0.0519.

Note the scale. The paper's entire advantage over DeepVecFont on Chinese is 0.006 absolute. Any Stage 2 result will live in that range.

### 2.2 What to add

Extend `eval_reconstruction_error.py`. All of these run on the same rendered 64×64 mask pair it already produces, so this is one function each.

**SSIM.** The metric from class. Use a Gaussian window, `data_range=1.0`, computed on the mask pair before binarization if you keep the anti-aliased render, otherwise on the binary masks. It complements L1: L1 counts disagreeing pixels wherever they are, SSIM penalizes disagreement that breaks local structure. A glyph with a uniformly slightly-too-thick stem and a glyph with one mangled stroke can share an L1 score and differ sharply in SSIM. Report both.

**IoU.** Already computed. Keep it, and label it s-IoU when comparing to the vector-font literature (DualVector, CVPR 2023, reports SSIM / L1 / s-IoU on this task, so the triple is the field's convention).

**Renderability rate.** `test_few_shot.py` wraps `render()` in a bare `except: continue`, and `eval_reconstruction_error.py` then skips any font whose merge HTML does not contain exactly `2 × char_num` SVGs. A model that fails to produce parseable output on hard glyphs is currently rewarded by silently dropping those fonts from the average. Log the fraction of glyphs that fail to render and the fraction of fonts skipped, per model. If the two models differ here, the Error comparison is not valid as-is.

**Quantization oracle.** Take the ground-truth sequences, push them through `numericalize` then `denumericalize`, render, and score with the identical pipeline. This is the floor that no model with the current 128-bin head can beat. It costs one pass over the test set and it is the single most useful number for motivating Stage 2. Put it in the results table as a row.

**Per-font statistics.** `eval_reconstruction_error.py` already accumulates per-font means. Save them to CSV. With 34 Chinese test fonts you can run a Wilcoxon signed-rank test on paired per-font Error between baseline and flow variant. Given an expected delta on the order of 0.005, an unpaired eyeball comparison of two grand means is not enough to claim an improvement, and the report will be stronger for saying so.

Optional, only if time allows: FID on rendered glyph images. Legitimate for English (1425 test fonts × 52 glyphs ≈ 74k images). For Chinese it is 34 × 52 = 1768 images, far below where FID is stable, so either skip it or report it with an explicit caveat about the sample size bias.

---

## 3. Stage 2: the flow-matching coordinate head

### 3.1 What the current head does

`models/transformers.py`, `Transformer_decoder`:

```python
self.args_fcn = nn.Linear(512, 8 * 128)
...
args_logits = self.args_fcn(out).reshape(N, S, 8, 128)
```

Each drawing command carries 8 coordinate arguments (4 control points × 2 axes, per the relaxation representation). Each argument is quantized to one of 128 bins by

```python
def numericalize(cmd, n=128):
    return (cmd / 30 * n).round().clip(min=0, max=n-1).int()
```

and supervised with cross-entropy against a 128-way one-hot. At inference the model takes `argmax` per argument and calls `denumericalize` to get back to viewBox units.

Four properties of this head matter:

1. **It is factorized.** Given the decoder hidden state `h ∈ R^512`, the 8 arguments are 8 independent softmaxes. The joint `p(a₁..a₈ | h)` is modelled as a product of marginals. The four control points of a cubic Bézier are strongly dependent on each other; a factorized categorical cannot represent that dependence, so an argmax over independent marginals can select a combination that occurs in no real glyph.
2. **It has no metric structure.** Cross-entropy over bin indices is invariant to permuting the bins. Predicting bin 5 when the target is bin 60 costs exactly what predicting bin 61 costs. Every geometric prior in the model has to be reintroduced by hand on top of this.
3. **It quantizes.** Bin width is `30/128 = 0.2344` viewBox units. The canvas is `viewBox="0 0 24 24"` and rendering is at 64×64, so one bin is `0.2344 × 64/24 = 0.625` px, with per-coordinate rounding error uniform on ±0.3125 px.
4. **It clips.** `.clip(min=0, max=n-1)` destroys any coordinate outside `[0, 30]`. Separately, `SVGEmbedding.arg_embed = nn.Embedding(128, 128, padding_idx=0)` maps bin 0 to a zero vector, so a coordinate at the clip boundary is both truncated on the way out and unrepresented on the way in.

Point 4 is worth measuring before you write a line of model code: histogram the ground-truth `numericalize` output and count the mass in bins 0 and 127, for Chinese and for English. If it is non-trivial (English descenders and overshoots are the likely culprits) that is a hard ceiling on the baseline, it is trivially checkable, and it makes a good figure in the report.

### 3.2 The replacement

Model `p(a | h, z)` directly on `R^8`, where `z` is the command type, using conditional flow matching with the linear (rectified-flow) probability path.

**Normalization.** Map viewBox coordinates to roughly unit scale: `x = (a / 30) * 2 - 1`. Better still, compute the empirical per-dimension mean and standard deviation of the training coordinates once and standardize, so the velocity target is not dominated by one axis. Store the stats alongside the checkpoint.

**Training objective.** For each supervised token position:

```
x₀ ~ N(0, I₈)
x₁ = normalized ground-truth coordinates
t  ~ U(0,1)
x_t = (1 - t)·x₀ + t·x₁
u   = x₁ - x₀
L_fm = ‖ v_θ(x_t, t, h, z) - u ‖²   masked by cmd_args_mask and seqlen_mask
```

The masks are the ones already in `Transformer.loss`: `cmd_args_mask` selects which of the 8 arguments a given command type actually uses (all 8 for `CurveFromTo`, indices 0,1,6,7 for `MoveFromTo` and `LineFromTo`, none for `EOS`), and `seqlen_mask` handles padding. Reuse them unchanged.

**Timestep distribution.** `U(0,1)` for the main run. A logit-normal schedule (Esser et al., SD3) concentrates samples near `t = 0.5` where the regression is hardest; run it as an ablation if time permits.

**Head architecture.** Follow the per-token diffusion head of Li et al. (MAR, NeurIPS 2024), which does exactly this for image tokens, but with the flow-matching objective instead of DDPM. A small MLP suffices for 8-dimensional data:

```
t   → sinusoidal embedding (128) → MLP → t_emb (256)
cond = Linear([h ; command_embed]) → 256
c   = t_emb + cond
x_t (8) → Linear → 256 → 3 × ResBlock with AdaLN(c) → Linear → 8
```

Roughly 0.5–1M parameters against a 6-layer, 512-wide transformer decoder, so training cost per step barely moves.

**Conditioning on the command type.** At training, condition on the ground-truth command. At inference, sample the command from `cmd_logits` first, then the coordinates, preserving the factorization `p(z, a) = p(z)·p(a | z)`. This also tells the head which argument slots are live.

**Sampling.** Euler integration from `t=0` to `t=1` in K steps. `K = 10` for the main run. `K ∈ {1, 5, 25}` are inference-only sweeps off the same checkpoint, so they are free. `K = 1` collapses to the conditional mean, which is the closest analogue of the baseline's argmax and therefore the cleanest ablation for isolating "continuous vs quantized" from "stochastic vs mode-seeking".

### 3.3 Wiring, file by file

**`models/transformers.py`**

- New `FlowMatchingArgsHead(nn.Module)` with `loss(h, z, x₁, mask)` and `sample(h, z, K)`.
- `Transformer_decoder.__init__`: replace `self.args_fcn` with the head. Keep `self.command_fcn`.
- `Transformer_decoder.forward`: return `out` (the hidden states) alongside `cmd_logits` instead of `args_logits`, so the head can be called from the loss and from the sampler.
- `Transformer.loss`: replace the `loss_args` cross-entropy block with the flow-matching term. Two further edits in the same function:
  - `smooth_constrained` (the paper's `L_cons`, Eq. 10) currently computes an L2 between *softmax probability vectors* over bins for adjacent start and end arguments. Eq. 10 specifies an L2 between predicted coordinates. With a flow head you compute it on coordinates directly, so this reimplementation becomes closer to the paper than the released code is.
  - `loss_aux` (the Bézier alignment loss, Eq. 9) currently recovers coordinates through a straight-through estimator: a temperature-0.1 softmax plus `p_argmax = args_prob2 + (argmax - args_prob2).detach()`. That hack exists only because the head is categorical. Replace it with the one-step estimate `x̂₁ = x_t + (1-t)·v_θ(x_t, t, h, z)`, which is differentiable end to end with no straight-through path. This is the single largest practical benefit of the change: the geometric losses stop fighting the parameterization.

**`models/model_main.py`**

- Training and validation branch: pass `out` to the head, accumulate `loss_fm` in place of `loss_args`.
- Test branch: inside the autoregressive loop, after `next_command = argmax(cmd_logits[:, -1])`, call `head.sample(h[:, -1], next_command, K)` and denormalize. Then re-quantize to bins before concatenating into `sampled_svg`, because `SVGEmbedding` consumes integers.
- `parallel_decoder` currently receives `args_logits` and takes `argmax` in train mode. Feed it the one-step estimate, re-quantized and detached, so the refinement decoder is untouched.

**Design decision to state explicitly in the report:** only the *output* head becomes continuous. The *input* embedding stays quantized, so the sampled coordinates are re-binned before being fed back into the AR loop. This keeps the diff small, keeps the input pathway byte-identical to the baseline, and isolates the measured effect to the output distribution. A continuous input embedding (Fourier features into a linear projection) is the obvious follow-up and belongs in the future-work paragraph, not in this run.

**`options.py`**

Add `--args_head` (`ce` | `fm`), `--fm_steps`, `--fm_width`, `--fm_blocks`, `--fm_time_dist` (`uniform` | `logitnormal`), `--loss_w_fm`. Keeping both heads behind a flag means one codebase produces both submitted models, which is what the assignment asks for.

**`train.py`**

`compute_val_loss` builds `val_metric` from `loss_w_l1 · img_l1 + loss_w_pt_c · vggpt + svg_total`, and the checkpoint filename embeds it. `svg_total` is in nats for the CE head and in squared normalized coordinate units for the flow head. These are not comparable, so "best checkpoint" would mean different things for the two models and `prune_checkpoints` would keep different things. Fix this before training: either select on `img_l1 + vggpt` alone, which is head-agnostic, or run a small rendering-based validation every `freq_ckpt` epochs and select on that. The second is more faithful to the test-time metric and worth the extra minutes.

### 3.4 The argument head is shared between both decoders

`self.args_fcn = nn.Linear(512, 8 * 128)` is instantiated once and called twice:

```python
# Transformer_decoder.forward, line 216
args_logits = self.args_fcn(out)          # autoregressive decoder

# Transformer_decoder.parallel_decoder, line 267
args_logits = self.args_fcn(out)          # context-based self-refinement
```

The two decoders have separate transformer stacks (`decoder_layers`, 6 layers; `decoder_layers_parallel`, 1 layer) and separate norms, but one shared argument projection. `command_fcn` is shared the same way.

This matters more than it looks, because of what gets scored. In `test_few_shot.py` the IoU-based candidate selection runs on `syn_{i}_{sample}_refined.svg`, which comes from `sampled_svg_2`, the **refinement** decoder's output, and the merge HTML that `eval_reconstruction_error.py` reads is built from those refined files. The final number in the results table is produced by `parallel_decoder`.

So a literal "AR head only" change leaves the last operation in the pipeline as an `argmax` over 128 bins, and the quantization floor survives intact into the metric. It would also cost *more* code, because you would have to un-share `args_fcn` and maintain two differently-parameterized argument heads.

**Recommendation: swap the shared module.** One edit covers both decoders, matches how the code is already factored, and the extra inference cost is K head evaluations on a single refinement pass rather than K per token. Concretely:

- `forward` returns `out` instead of `args_logits`; the head is called for the flow loss (training) or for `sample` (inference).
- `parallel_decoder` takes the AR decoder's sampled or one-step-estimated coordinates, re-quantizes them for `SVG_embedding`, and calls the same head on its own `out` to produce the refined coordinates.
- `Linit_CE` and `Lrefine_CE` (Eq. 5 and Eq. 8) each keep their command cross-entropy term and each get a flow-matching term in place of their argument cross-entropy term. `loss_dict['svg']` and `loss_dict['svg_para']` keep their existing structure, so `train.py` logging needs only a rename.

Fall back to un-shared heads only if the refinement pass turns out to dominate inference time, and if you do, say in the report that the reported Error still carries the 128-bin floor.

### 3.5 Loss weight calibration

`loss_w_args = 1.0` was tuned for a cross-entropy term. The flow-matching MSE has a different scale entirely. Do not guess:

1. Log the magnitude of `loss_args` on the existing baseline over the first few hundred steps.
2. Log the magnitude of `L_fm` on an untrained flow head over the same steps.
3. Set `loss_w_fm` so the two terms contribute comparably at initialization.
4. Sweep ×0.3 and ×3 around that value on short runs before committing to the full training run.

Skipping this is the most likely way to get a flat or negative result for reasons that have nothing to do with flow matching.

---

## 4. What improvement to expect, and why

Ordered by how confident the mechanism is, not by expected effect size.

**Joint modelling of the eight coordinates.** The strongest argument, and the one that maps most directly onto course material. The baseline models `p(a|h) = Π_k p(a_k|h)`, a factorized conditional. The flow head learns a single velocity field on `R^8`, so correlations among the four control points of a command are representable. This is the same move the field made from vector-quantized to continuous tokens: a factorized categorical over a discretized space, replaced by a learned transport from a simple base density.

**Clean gradients for the geometric losses.** In the paper's own ablation the Bézier alignment loss is worth 0.0028 (0.0557 → 0.0529), close to the relaxation representation's 0.0031 and nearly triple the self-refinement module's 0.0010. It is one of the two things carrying the paper. It currently reaches the coordinates through a temperature-0.1 softmax and a straight-through estimator. Removing that indirection should let a loss that already works well work better.

**Quantization floor removed.** 0.625 px per bin at 64×64, rounding error uniform on ±0.3125 px per coordinate. Measure the actual floor with the oracle run in §2.2 rather than predicting it. Likely a few units in the third decimal of Error.

**Clipping and boundary-bin pathology removed.** Size this empirically with the bin histogram in §3.1. Zero-ish for Chinese, plausibly not for English, which is one reason the English run is worth doing if the Chinese result holds.

**Best-of-N_s selection gets something to select from.** The only stochasticity at inference today is `x = x + torch.randn_like(x)` at `models/transformers.py:450`, a perturbation of the sequence-encoder output. This is the paper's `N(0,I)` noise from Sec. 4.1. Everything downstream is argmax. A stochastic coordinate head adds variation in output space, so the IoU-based selection over 50 Chinese candidates should pay off more. Sweep `n_samples ∈ {1, 10, 50}` for both models and plot Error against N_s. If the flow curve falls faster, that isolates the mechanism cleanly, and it is a good figure.

### Counterweights, to state up front

- The metric is a rasterized 64×64 binary L1. Sub-pixel gains are partly invisible to it. The measured improvement will understate the modelling improvement.
- The paper's own margin over its predecessor on Chinese is 0.006. Expect deltas of that order. Paired statistics are not optional.
- Inference costs K extra head evaluations per token per candidate. The head is small, but report wall-clock test time as a row in the results table rather than hiding it.
- Convergence may be slower. Regressing a velocity field is a harder early-training signal than classifying into bins, and the AR decoder is teacher-forced on discrete inputs regardless. At matched epochs the flow variant might not have caught up. Mitigate by logging the rendering metric across the whole training curve, not just at the end, so a "slower but still climbing" outcome is visible and reportable rather than looking like a failure.

A negative or neutral result, reported with the oracle floor, the bin histogram, and a paired test, is a good Stage 2. The rubric asks for "what worked, what did not, and what you learned."

---

## 5. Experiment matrix

Chinese, from scratch, matched epoch budget against the existing baseline. You reported good checkpoints around epoch 100–150, so target 150 and compare at 100, 125 and 150.

| # | Run | Trains? | Purpose |
|---|---|---|---|
| 0 | GT quantization oracle | no | The floor imposed by the 128-bin head |
| 1 | Baseline, existing `dvf_base_exp_chn` | done | Stage 1 reconstruction number |
| 2 | Flow head, `U(0,1)`, K=10 | yes | Main Stage 2 result |
| 3 | Run 2 checkpoint, K ∈ {1, 5, 25} | no | ODE step budget, and mean-vs-sample ablation |
| 4 | Flow head, logit-normal timesteps | yes | Timestep schedule ablation |
| 5 | Runs 1 and 2, N_s ∈ {1, 10, 50} | no | Does a stochastic head make best-of-N pay off |

Only 2 and 4 cost training time. If the budget tightens, drop 4. Runs 3 and 5 are inference sweeps and cost hours, not days.

English only if Chinese shows a gain. Same protocol, `max_seq_len 51`, `ref_nshot 4`, `n_samples 10` to match the paper.

Results table for the report, one row per model, columns: Error ↓, SSIM ↑, IoU ↑, renderability %, test wall-clock. Plus the paper's reported number and the oracle floor as reference rows.

---

## 6. Report skeleton, mapped to the rubric

1. **Original architecture.** Dual-branch encoder (CNN over reference glyph images, 6-layer transformer over reference SVG sequences), modality fusion into a VAE latent, image decoder and autoregressive sequence decoder, context-based self-refinement. Losses: image L1 and perceptual, KL, command cross-entropy, argument cross-entropy, Bézier alignment (Eq. 9), relaxation consistency (Eq. 10). Note the relaxation representation and why it exists, since it is the paper's headline contribution and it is what makes the coordinate head 8-dimensional rather than 6.
2. **Paper results.** Tab. 2 and Tab. 1 above, plus the definition and meaning of Error from §2.1.
3. **Reconstruction results.** Your Chinese numbers against 0.080, with SSIM, IoU and renderability. Note the deviations between the released code and the paper text, which are legitimate reconstruction findings and belong in the report:
   - Sec. 3.1 describes one-hot arguments of **256** dimensions (`δp ∈ R^256×8`, `W_args^b ∈ R^dE×256`). The code quantizes to **128** bins, and the stale comment on `args_fcn` still reads `# shape: bs, max_len, 8, 256`.
   - Sec. 3.3 describes the self-refinement module as a **2-layer** transformer decoder. `Transformer_decoder.decoder_layers_parallel` is `clones(DecoderLayer(...), 1)`.
   - Eq. 11 sets the `L_cons` weight to 10 and the rest to 1.0 except `L_kl` at 0.01. `options.py` additionally carries `loss_w_aux = 0.01` and `loss_w_l1 = 10`, so the effective weighting differs from the text.
   - Sec. 4.1's inference-time `N(0,I)` noise is implemented as a full-rank perturbation of the sequence-encoder output (`transformers.py:450`), applied in training as well as at test.
4. **Improved architecture.** §3 above. Lead with the factorization argument, then the geometric-loss argument, then quantization.
5. **Improved results.** The table from §5, with the paired Wilcoxon result and the oracle floor row.
6. **Discussion.** What the metric can and cannot see, whether the effect exceeded the quantization floor, the inference cost, and the convergence behaviour.
7. **References.** §7 below.

---

## 7. References

- Wang, Wang, Yu, Zhu, Lian. *DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality.* CVPR 2023.
- Wang, Lian. *DeepVecFont: Synthesizing High-Quality Vector Fonts via Dual-Modality Learning.* ACM TOG 2021.
- Lipman, Havasi, Holderrieth, Shaul, Le, Karrer, Chen, Lopez-Paz, Ben-Hamu, Gat. *Flow Matching Guide and Code.* arXiv:2412.06264. **On the course reading list**, already in the course folder.
- Holderrieth, Erives. *An Introduction to Flow Matching and Diffusion Models.* arXiv:2506.02070, MIT 6.S184. **On the course reading list**, already in the course folder.
- Lipman, Chen, Ben-Hamu, Nickel, Le. *Flow Matching for Generative Modeling.* ICLR 2023. The original objective.
- Liu, Gong, Liu. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow.* ICLR 2023. The linear path used here.
- Li, Tian, Li, Deng, He. *Autoregressive Image Generation without Vector Quantization.* NeurIPS 2024. The per-token generative head this design mirrors, with a diffusion objective in place of flow matching.
- Esser et al. *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis.* ICML 2024. Logit-normal timestep sampling.
- Tong et al. *Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport.* TMLR 2024. Conditional flow matching variants.
- Lopes, Ha, Eck, Shlens. *A Learned Representation for Scalable Vector Graphics (SVG-VAE).* ICCV 2019. Predicts continuous coordinates with a Mixture Density Network. Useful framing: the field went continuous (MDN), then discrete (binned cross-entropy, DeepSVG and DeepVecFont), and this head returns to continuous with a far more expressive density model.
- Carlier, Danelljan, Alahi, Timofte. *DeepSVG: A Hierarchical Generative Network for Vector Graphics Animation.* NeurIPS 2020.
- Liu, Guo, Wang, Zhang. *DualVector: Unsupervised Vector Font Synthesis with Dual-Part Representation.* CVPR 2023. Source of the SSIM / L1 / s-IoU reporting convention on this task.

---

## 8. Order of work

1. Bin histogram of ground-truth coordinates, Chinese and English. One script, answers whether clipping matters. (§3.1)
2. Extend `eval_reconstruction_error.py` with SSIM, renderability, per-font CSV. Score the epoch-100 and epoch-125 Chinese results. Stage 1 closes. (§2.2)
3. Quantization oracle run. Gives the floor. (§2.2)
4. Fix `val_metric` so checkpoint selection is head-agnostic. (§3.3)
5. Implement `FlowMatchingArgsHead` behind `--args_head fm`. Verify on a single overfitted batch that it reproduces ground-truth coordinates before launching anything long.
6. Loss weight calibration, short runs. (§3.5)
7. Full Chinese training run to epoch 150.
8. Inference sweeps for K and N_s. Paired test against baseline.
9. English, conditional on the Chinese result.
10. Report.

Steps 1 through 4 are worth doing while the current Chinese test finishes, and they are useful whatever Stage 2 turns out to be.
