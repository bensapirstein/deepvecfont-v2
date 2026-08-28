# DeepVecFont-v2 — Project Plan

Final project, Generative Models for Text and Images, Reichman University.
Paper: Wang, Wang, Yu, Zhu, Lian, *DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality*, CVPR 2023.
Fork of `yizhiwang96/deepvecfont-v2`. All work on branch `repro`; `main` stays a pristine upstream mirror.

**Submission: ~~Saturday 15 August 2026~~ postponed to Tuesday 1 September 2026** (instructor notice received 2026-08-11). Day 1 was Monday 3 August; the original window was thirteen days, the actual window is now thirty. Everything dated against the 15 August deadline below (§4's schedule, the "N days out" framing in `docs/`) is a historical log of a compressed sprint that in fact had slack — read it as that, not as current pressure. The critical-path items in `_Open Tasks.md`'s "Day 9" block (push the commit, read the report end to end, figures, the `main..repro` diff review) are unaffected in kind, just no longer urgent in the same way.

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
| Stage 1 | **Closed 2026-08-05.** SSIM and the quantization oracle both landed (§2.3, §2.5); renderability, per-font CSV and the bin histogram were already in |
| Stage 2, Tier 1 | Run and screened 2026-08-04. Nothing cleared the noise floor; see §3.2 |
| Stage 2, Tier 2 | Run and screened 2026-08-04. Nothing cleared the noise floor either; see §3.5. E8's s-IoU shift was flagged as the one follow-up worth budget, then **closed 2026-08-04** without a GPU: it sits at the s-IoU seed floor, not above it |
| Stage 2, Tier 3 | Run and screened 2026-08-04. Fifteen runs (one dropped at rung 1). Batch A (breadth): null, like Tiers 1 and 2. Batch B (three seeds each on the three largest prior deltas): E9 σ=0.5 is the first candidate with same-sign L1 improvement at all three seeds; E1 and E13 are mixed-sign. See §3.6 |
| Seed-noise floor | L1 spread **0.0093** (re-measured 2026-08-04). Decomposed 2026-08-04: decode noise 0.0011 (12%), seed variance ~0.0082 (88%) dominates — see §3.2. **s-IoU spread 0.0401** (2026-08-04), roughly three times noisier than L1 in relative terms — see §3.5 |
| Results table | `RESULTS.csv` at the repo root, one row per scored checkpoint, rebuilt by `scripts/build_results_table.py`. 37 rows as of 2026-08-05 |
| Two readings of that table (2026-08-05) | **(a)** Every single-seed delta in this document is quoted against `seedfloor_1111_chn`, the *worst* of the three baseline draws and 0.0048 above their mean. **22 of 26** candidates beat that anchor; only **6 of 26** beat the mean. Every "largest delta" ranking here, including the one that chose Batch B, was computed against an unlucky draw. **(b)** The 26 candidates span 0.0101 in L1; three baseline seeds span 0.0097. Twenty-six draws should span ~2.3× the range of three, so the candidate-induced spread is roughly **half** the seed-induced one. Both in §3.2 |
| Deltas are generated, not typed (2026-08-05) | `scripts/recompute_deltas.py` regenerates every screening delta from `RESULTS.csv` against the corrected anchor (**0.1728** L1 / 0.2240 s-IoU), with a *vs anchor* and a *vs three-seed mean* column. **Floors: L1 0.0097, s-IoU 0.0315.** The hand-typed tables in §3.2, §3.5 and §3.6 are kept for provenance because they are what every ranking decision was made on; **the script wins on conflict.** Why the anchor moved is in §3.2 |
| Where Stage 2 landed | **E9 `enc_noise_std_train=0.5` is the single finalist.** Same-sign paired improvement at all three seeds on L1 *and* on s-IoU — six of six, on two metrics that correlate at only r = −0.335 across the table. **E1 is a same-sign negative result**, degrading s-IoU at all three seeds by 1.8× its own floor, the only effect anywhere in this project that exceeds its floor. §3.6 |
| Confirmation session | **Run and closed 2026-08-05** (`docs/confirmation-launch.md`). σ_test ladder null on both E9 and baseline (σ_test=1.0 stands, §8 item 9). `val_metric` doesn't predict the rendered metric (ρ=0.125, §3.2) — E14-deep cut to its two peak-lr rows. E9's confirmation eval, all three seeds, `n_samples 50`: **L1 −0.0040 vs baseline mean, s-IoU +0.0271**, paired Wilcoxon same-sign at all three seeds on both metrics (two of three strongly significant; seed 3333 the weak leg on both, as at screening). Table in §5 |
| Official checkpoints | **Scored 2026-08-06** (`docs/official-checkpoints-and-600.md`, Job A). The authors' released weights through our own harness, both GT conventions, `n_samples 50`. **Chinese 600 ep: 0.1629 raster / 0.1174 svg. English 600 ep: 0.0658 / 0.0584** (34-font subset, see §5.1's ‡). Rows in §5.1 |
| **The reproduction is faithful** | Our 150-epoch Chinese baseline (0.1621) sits **0.0008** from the released 600-epoch checkpoint (0.1629) under the identical convention — about a twelfth of the 0.0093 seed floor. The gap to the published 0.080 is a property of the evaluation, not of our training. §2.4, rewritten 2026-08-06 |
| **Training budget is eliminated** | It was §2.4's leading explanation from 3 to 5 August. Four times the epochs, in the authors' own weights, buys nothing through this harness. `seedfloor600_<seed>_chn` (three seeds, trained 2026-08-06) is now a confirmation with a known bound rather than the decisive test it was launched as. **Unscored** |
| Where the gap does live | Two measured terms. **Cross-rasterizer disagreement: 0.0455 on Chinese, 0.0074 on English** (§1.2, now a property of the metric). **A Chinese-specific residual of ~0.037** under the svg convention, unexplained, pointing at the Chinese data or test protocol. English nearly reproduces (0.0584 vs 0.052) on the same code path; Chinese does not. §2.4 |
| English arm | **Descoped 2026-08-06, deliberately.** The test split is 1,386 fonts, 40× Chinese; a full 3-checkpoint sweep projected to ~40 GPU-h and >100 GB against a near-cap quota. Cut to a 34-font deterministic subset plus an 862-font partial decode. The subset is optimistic by 0.0074/0.0106, measured. There is **no self-trained English baseline** — the epoch-600 row was the released checkpoint all along, and is retracted |
| Disk | 182 GB → 75 GB on 2026-08-06. Checkpoints and decode trees dropped for everything already in `RESULTS.csv`, `eval_*.csv` summaries kept |
| **English training arm, landed 2026-08-07** | The descoping above was of the official-checkpoint *evaluation*, not of training. Three seeds trained to 801 epochs; `E_conv` computed per seed (400 / 580 / 420) and the budget **frozen at 630** before any candidate was looked at, as §8 item 4 required. Scored at the nearest surviving checkpoint (640 / 580 / 640), `n_samples 50`, 34-font subset, raster |
| **Job A closed 2026-08-08**: `seedfloor600_*_chn` scored, both conventions | Three seeds, raster mean **0.1583**, 0.0046 below the official 600-epoch checkpoint's 0.1629 and inside the 0.0093 floor. Confirms §2.4's budget-elimination conclusion; gate did not fire, no rewrite needed. §5.1 |
| **English arm closed 2026-08-08: E9 does not replicate on English** | Three E9 seeds, paired same-seed same-epoch to the English baselines. Mixed sign on both L1 and s-IoU (1 of 3 seeds favourable), all six deltas at or inside the English floor. Outcome 3 of the pre-committed reading rule in `docs/english-candidate.md` §4 — a result, not a failed run. §5.2, §8 item 4 |
| **English seed floor, first measurement** | L1 spread **0.0038**, s-IoU **0.0129**, SSIM **0.0140** (2026-08-07). Relative to the mean that is 6.4% against Chinese's 5.7%, so the instrument is no sharper on English, it is measuring a smaller quantity. **The paper's published English ablation spans 0.0069 in total with individual steps of ≤0.003, so its individual steps sit below our floor** — a §6 finding about the benchmark, not an excuse. `docs/english-candidate.md` §3 |
| **English inverts the Chinese finding** | Our 3-seed English baseline (**0.0597** mean L1) scores *better* than the released checkpoints (0.0645 / 0.0649 / 0.0658 at epochs 500 / 550 / 600) on the same 34-font subset, and lands closer to the paper's 0.052. The margin, 0.0048 to 0.0061, is **1.3× to 1.6× the English floor**, so it clears but stays the same order. On Chinese our baseline and the official checkpoint are indistinguishable; on English ours is ahead. English training here is not undertrained relative to the release, and the residual gap on English looks like protocol or metric rather than budget |
| **English cut-off fired, and was overruled** | ~100 s/epoch × 630 epochs = **~17.5 GPU-h per run**, past `docs/english-arm.md` Step 1's own `> 6 h: drop English` line, reached independently by the timing run and the convergence rule. **Overruled deliberately 2026-08-07** on three free GPUs and six days of schedule slack. Logged as an overrule in `docs/english-arm.md` Step 3 and §8 item 4, and disclosed in `REPORT.md`'s methods rather than absorbed |
| Stage 2, English arm | **E9 `enc_noise_std_train=0.5`, three seeds, paired to the three baselines.** One candidate rather than three, because §0's own 2026-08-05 readings and Batch B's 1-in-3 replication rate rule out single-seed points at this resolution, and the 0.0038 floor sits just under E9's own Chinese confirmation delta of 0.0040. Reading rule pre-committed before launch in `docs/english-candidate.md` §4 |
| **Day-6 GPU push, staged 2026-08-08** | All GPUs available (3, 2, 1; 0 stays free). Five jobs in `docs/day6-gpu-push.md`, none on the critical path. **Job A is the one that matters: E1's s-IoU deficit is confounded with checkpoint epoch.** Every Chinese baseline was scored at epoch 150; `e1_norm_2222_chn` was selected at 125 and `e1_norm_3333_chn` at 100, so E1's two degrading legs are the two scored earliest, and E1 is the only effect in this project that clears its own floor. Selection is by `val_metric`, and no term in `val_metric` is the quantity being scored. ~4 GPU-h to retrain both legs plus a matched baseline with all checkpoints kept and read at matched 150. Also staged: Job B (`val_metric` on the released English checkpoints, ~15 min), **Job C** (one replicated candidate per assignment category, Chinese, 6 candidates x 3 seeds, ~18 GPU-h), **Job C-EN** (the same six at one seed on English, ~105 GPU-h, screening only, required for both-dataset coverage), Job D (Tier 4 capacity, E16 `--enc_depth` and E17 `--dec_d_ff`, three seeds each). Three more English baseline seeds were staged and then **demoted to optional**: mixed sign is floor-independent, so they cannot move §5.2. Scope reasoning in §8 item 12 |
| Next session | **Mac-side, 2026-08-07 (day 5):** day-4 work committed (`bb79e57`), `docs/english-candidate.md` written, §8 item 4 closed, §0 brought current, vault caught up. **Cluster, in order:** Job A score `seedfloor600_*_chn` both conventions (it gates §2.4, so it runs first), then Job B launch the three E9 English seeds, then Job C score them. All commands in `docs/english-candidate.md`. See §9 |
| **§8.2 augmentation check, run 2026-08-08, no GPU** | `ls data/vecfont_dataset/chn/train` shows 1272 entries, sorted-unique base ids give 212, and font `000` carries `000` + `000_0`..`000_4` (five augmented copies). **The Chinese training set was built at `n_aug=5` (6x), not the paper's stated 10x.** Confirmed independently through `get_loader` (`dataloader.py`): 1272 train entries, matching. Live lead on the ~0.037 Chinese residual (§2.4 term 2), not yet acted on — a 10x-augmented retrain is a scope change per §8's own gate and needs a decision logged before anything trains |
| **`cen_e11_adamw_1111_eng` and `cen_e3_refine2_1111_eng` scored 2026-08-10** | Both finished their 631-epoch budget naturally (no crash, no manual kill). Scored at checkpoint 620 (lowest-`val_metric` checkpoint on disk for both, and matches the Job C-EN anchor epoch), `n_samples 10` against `eng_seedfloor_1111`'s n=10 anchor (L1 0.0610, s-IoU 0.7270; floor L1 0.0038/s-IoU 0.0129). **`cen_e11_adamw`: null** — L1 0.0623 (Δ+0.0013), s-IoU 0.7222 (Δ−0.0048), both inside the floor, matching its Chinese Job C reading (also null there). **`cen_e3_refine2`: clears the floor on both metrics, in the *degrading* direction** — L1 0.0658 (Δ+0.0048), s-IoU 0.7016 (Δ−0.0254) — outcome 3 of the pre-committed reading rule, the same shape as `cen_e5_bneck256` in Job C-EN. This is the opposite of Ben's read of the run's wandb training-loss curve (noted 2026-08-10, before scoring, as "doing great" / "real improvement") — a live instance of Job B's finding that `val_metric`/the training curve doesn't track the rendered metric (ρ=0.125). Chinese Job C had this candidate as null/mixed-sign, not degrading, so English and Chinese disagree on direction here too |
| **`cen_e3_refine2`, 3-seed English reading closed 2026-08-11** | Seeds 2222 and 3333 finished their 631-epoch budget cleanly and scored at checkpoint 620, `n_samples 10`, same protocol as 1111. **All three seeds clear the English floor on both metrics, same sign, degrading:** L1 Δ +0.0048 / +0.0053 / +0.0040 (floor 0.0038), s-IoU Δ −0.0254 / −0.0218 / −0.0153 (floor 0.0129), mean L1 Δ +0.0047, mean s-IoU Δ −0.0208. This is the same reading-rule bar E9 cleared to become the project's improving finalist and E1 cleared to become its one confirmed degrading effect — `cen_e3_refine2` (`--n_layers_refine 2`) is now a **confirmed degrading result on English**, replicated at three seeds, not a single-seed artifact. It stands in direct contrast to the Chinese Job C reading for the same candidate (`c_e3_refine2`, mean L1 Δ −0.0004, mean s-IoU Δ +0.0007, both mixed-sign — null): **English and Chinese disagree in both magnitude and direction on this candidate**, same shape as the `cen_e5_bneck256` divergence. Caveat: seed 2222 rendered 33/34 fonts (1 skipped, renderability 0.9706); its L1/s-IoU are averaged over one fewer font than the other two seeds and the anchor, per `eval_reconstruction_error.py`'s own comparability warning — noted, doesn't change the reading since all three seeds already agree in sign independent of it. `RESULTS.csv` now 113 rows; `cen_e3_refine2_{2222,3333}_eng` added to `scripts/build_results_table.py`'s `BATCH` map (`job-c-en`) |
| **Best-val-loss vs. matched-epoch checkpoint selection, tested 2026-08-10** | Ben's proposed paradigm — train, then evaluate each experiment's lowest-`val_metric` checkpoint rather than a fixed matched epoch — tested directly against all five scored Job C-EN candidates, since three of them (`cen_e2_batchnorm`, `cen_e4_ngf32`, `cen_e5_bneck256`) have an on-disk checkpoint below 620 with lower `val_metric` than 620 itself. Re-scored at that checkpoint, `n_samples 10`, same protocol: `cen_e2_batchnorm` best-val ckpt **540** — L1 0.0652 (Δ+0.0042, was +0.0026 at 620), s-IoU 0.7074 (Δ−0.0196, was −0.0126) — **flips null → clears floor, degrading**. `cen_e4_ngf32` best-val ckpt **560** — L1 0.0650 (Δ+0.0040, was +0.0022), s-IoU 0.7071 (Δ−0.0199, was −0.0126) — **flips null → clears floor, degrading**. `cen_e5_bneck256` best-val ckpt **600** — L1 0.0670, s-IoU 0.7034, essentially unchanged from 620's already-degrading reading, no flip. `cen_e3_refine2` and `cen_e11_adamw`: best-val checkpoint *is* 620, nothing to compare. **Net: switching to best-val-loss selection never improved a single candidate's rendered score, flipped 2 of 5 from null to degrading, and left the other 3 unchanged** — a one-sided result, not noise in either direction, which is itself evidence against `val_metric` identifying a genuinely better checkpoint here. Consistent with Job B's ρ=0.125. Also checked whether a *different*, already-logged validation term correlates better: pooled Spearman ρ vs. rendered L1 across Job A's 9 fully-scored (checkpoint × run) points — `val_metric` +0.283, `val_l1` (image-decoder branch, `val_metric`'s dominant term) +0.350, `val_svg_total` (refinement decoder, the *right* head) +0.450, `val_svg_para_total` −0.250 (anti-correlated). `val_svg_total` is the best of the four but n=9 and none reach significance (p≥0.22); none of these logged terms is the rasterized, autoregressive, best-of-N quantity that's actually reported, so none is expected to fully recover it. **Decision: keep matched-epoch (620) as the protocol.** A real fix would need a cheap validation-time hook that decodes at `n_samples 1` and rasterizes a held-out subset periodically — not done, flagged as future work, not a fix to retrofit onto already-scored or already-pruned checkpoints. Bug caught and fixed in the same session: the five new eval CSVs were first written without the `_n10` filename suffix `build_results_table.py` needs, silently mislabeling them `n_samples=3` in `RESULTS.csv`; renamed and rebuilt (111 rows) |
| **Job C-EN, 3 of 6 scored 2026-08-09** | `cen_e2_batchnorm`, `cen_e4_ngf32`, `cen_e5_bneck256` all finished their 631-epoch budget naturally (GPUs 0/3 freed themselves before any manual intervention was needed). Scored at nearest-to-640 checkpoint (620, vs the anchor's 640 — the fixed 631-epoch budget never reaches a 640 checkpoint under `--freq_ckpt 20`), `n_samples 10` matching the paper's English protocol. `eng_seedfloor_1111` rescored at `n_samples 10` too: **L1 0.0610, s-IoU 0.7270** (vs its own n=50 figure of 0.0583/0.7371 — confirms §8's prediction that n=50 flatters English numbers; this is the first direct measurement of that gap). Reading against the n=10 anchor, floor L1 0.0038/s-IoU 0.0129: **cen_e2_batchnorm and cen_e4_ngf32 both null** (L1 Δ +0.0026/+0.0022, s-IoU Δ −0.0126/−0.0126, neither clears). **cen_e5_bneck256 clears the floor on both metrics, in the degrading direction** (L1 +0.0058, s-IoU −0.0242) — outcome 3 of the pre-committed rule (clears on English, was null/mixed-sign on Chinese in Job C), reported as a script-dependent result, not promoted to confirmation. `cen_e11_adamw_1111_eng` and `cen_e3_refine2_1111_eng` now training (GPUs 2 and 1 respectively, launched after Ben freed capacity) |
| **Job C-EN launched 2026-08-09, `cen_e7_aux01_1111_eng` stopped** | Four of six candidates started (`cen_e2_batchnorm`, `cen_e4_ngf32`, `cen_e5_bneck256`, `cen_e7_aux01`), one per GPU including GPU 0 for the extra headroom Ben made available overnight. `cen_e7_aux01_1111_eng` killed by Ben's own call on its wandb validation-loss curve at epoch 294/631 (last checkpoint 280, `val_metric` trend in `checkpoint_metrics.csv`: 3.749→3.388 across epochs 100-280, not obviously bad on the raw number, but the call was made from the curve shape, not this table). GPU 1 freed. `cen_e11_adamw_1111_eng` and `cen_e3_refine2_1111_eng` remain queued, not yet started. Two remaining candidates + GPU reduction expected 2026-08-10 per Ben's plan (3 or 2 GPUs from tomorrow) |
| **Job D closed 2026-08-09** | E16 (`--enc_depth 8`) and E17 (`--dec_d_ff 2048`), three seeds each, Chinese. Same epoch-selection confound Job A found for E1: `val_metric` auto-selected epoch 125 for 5 of the 6 runs, not the baseline's matched 150 (only `e17_dff2048_3333_chn` landed on 150). Every run's epoch-150 checkpoint was still on disk, so scored there directly rather than retraining. **Both null**, mixed sign on both metrics: E17 L1 deltas [−0.0058,+0.0041,+0.0032] mean +0.0005, s-IoU [+0.0284,+0.0103,−0.0235] mean +0.0051; E16 L1 [−0.0008,+0.0067,+0.0017] mean +0.0025, s-IoU [+0.0058,−0.0243,+0.0049] mean −0.0045. Matches the runbook's own expectation. Rows in `RESULTS.csv` (`job-d-capacity` batch) |
| **Job C closed 2026-08-08 ~16:00** | 18 runs (6 categories × 3 seeds), Chinese, `--max_ckpt_keep 10`, scored at matched epoch 150, `n_samples 3`. Baseline three-seed screening mean: L1 0.1680, s-IoU 0.2402. Reading rule (mean clears floor L1 0.0097/s-IoU 0.0315 **and** same sign at all three seeds): **all six categories null.** Per-seed L1 deltas vs matched baseline seed — E2 batchnorm: [−0.0080,−0.0022,−0.0095] mean −0.0066 (same sign, sub-floor); E4 ngf32: [−0.0016,+0.0144,+0.0048] mean +0.0059 (mixed); E5 bneck256: [−0.0052,0.0000,+0.0023] mean −0.0010 (mixed); E7 aux01: [−0.0011,+0.0025,−0.0043] mean −0.0010 (mixed); E11 adamw: [−0.0044,+0.0036,−0.0025] mean −0.0011 (mixed); E3 refine2: [−0.0030,−0.0001,+0.0019] mean −0.0004 (mixed). s-IoU deltas — E2: [+0.0241,+0.0185,+0.0238] mean +0.0221 (same sign, sub-floor); E4: [+0.0338,−0.0013,+0.0194] mean +0.0173 (mixed); E5: [+0.0375,+0.0312,−0.0077] mean +0.0203 (mixed); E7: [+0.0050,−0.0210,+0.0002] mean −0.0053 (mixed); E11: [+0.0019,−0.0046,+0.0078] mean +0.0017 (mixed); E3: [+0.0078,+0.0111,−0.0167] mean +0.0007 (mixed). **E2 is the only category with same-sign at all three seeds on both metrics** — consistent direction, but both means sit under their floor, so it does not clear the pre-committed bar. `c_e4_ngf32_2222_chn` rendered 32/34 fonts (2 skipped); doesn't change E4's reading since it was already mixed-sign. Full table in `docs/day6-gpu-push.md` §3, rows in `RESULTS.csv` (`job-c-category` batch) |
| **Job B closed 2026-08-08 ~14:35** | `scripts/val_on_checkpoint.py` had never actually been run before — it crashed on import, unconditionally, for any invocation that passes a CLI flag `models/transformers.py`/`models/modality_fusion.py` don't already know about, because both of those modules call `get_parser_main_model().parse_args()` against real `sys.argv` at *module import time*. Fixed by stripping the script's three custom flags out of `sys.argv` before the `ModelMain` import chain runs, restoring them after. **Result, `val_metric_audit.csv`:** English `val_metric` ranks the three official checkpoints 600 (3.0336, best) < 500 (3.0357) < 550 (3.0750, worst) — it prefers 600, the opposite of the rendered ranking (500 best at L1 0.0645, 600 worst at 0.0658). Chinese is milder but still disagrees: `val_metric` ranks 500 best / 600 worst, while rendered s-IoU ranks 600 best / 500 worst, and rendered L1 ties 500 and 600. **Outcome 2 of Job B's decision rule: `val_metric` does not track the rendered metric it's supposedly a proxy for, on either language — a defect in the evaluation apparatus, not in any model,** the same class as the two-rasterizer finding. Every checkpoint in this project, including all of `RESULTS.csv`, was selected the same way. Per the runbook's scope guard, **not** switching to rendered-L1 checkpoint selection — reported as a measured protocol defect, named as future work |
| **Job A closed 2026-08-08 ~14:20** | Training finished in ~18 min (this cluster is much faster than the runbook's ~1h estimate). Scoring hit one bug not in the runbook: `test_few_shot.py` needs `--enc_final_norm True` repeated for the two E1 legs or `load_state_dict` hard-crashes on the extra LayerNorm keys (COMMANDS.md already documented this requirement; the runbook's exact §1.4 command block omitted it). Fixed and rerun. **Reading, matched epoch 150, screening budget:** E1's s-IoU deficit holds at all three seeds beyond the 0.0315 floor, but shrinks from the original confounded mean of −0.0760 (2.4× floor) to **−0.0376 (1.2× floor)** — seed 1111 (already matched) −0.0139, seed 2222 (was ckpt125) −0.0449, seed 3333 (was ckpt100) −0.0539. Outcome 1 of §1.5's pre-committed rule: finding stands, confound was real but partial. L1 unaffected throughout (all deltas ≤0.0058, inside the 0.0097 floor). `a_seedfloor_3333_chn` retrain reproduces the original `seedfloor_3333_chn` within 0.0012 L1 — sanity check passed, retrain is trustworthy. Bonus from the same runs: within-run, `val_metric` agreed with the rendered metric on the best checkpoint (150, all three runs) but ranked epoch 100 above 125 in all three, while every rendered L1/s-IoU number ranked 125 above 100 — a small reproducible instance of §1.2's "no term in val_metric is the quantity being scored". `REPORT.md` abstract, §5 and §6 finding 3 updated with the corrected numbers. `RESULTS.csv` has the nine new rows (`job-a-deconfound` batch) |

| **`REPORT.md` back half drafted 2026-08-11 (day 9)** | §4, §5, §6 and §7 were `[TO WRITE]` outline stubs as of this morning and are now prose: 11.2k words, all seven sections the brief requires. **§5.1 is the required three-way paper / reconstruction / improved-model table**, with a fourth column for the released weights through our harness. §5 is subdivided into the three-way table, E9's Chinese confirmation, E9's English non-replication, category coverage across both languages, the E1 de-confound, and the oracle bounds. §6 runs eight subsections and absorbs everything after 2026-08-06 that the old seven-finding outline predated: Job A, Job B, the best-val-selection test, the English/Chinese direction disagreement, the `Ns` deviation and the `n_aug` lead. **Every figure re-derived programmatically from `RESULTS.csv` in a 68-check pass, 0 failures.** |
| **Correction made while drafting §5** | The draft initially quoted E9's s-IoU gain of +0.0271 as clearing a **confirmation-budget** s-IoU floor of 0.0236 (the n=50 baseline spread, measured but never used as a bar). That is post-hoc floor selection: the pre-committed bar is the screening 0.0315, and E9 does not clear it. Corrected before verification. **§5.2 now states plainly that neither of E9's means clears its pre-committed floor and that E9 rests on the same-sign criterion alone**, with the narrower spread reported and explicitly not used. The abstract carries the same qualifier. This is §3.5's own rule applied against the project's own finalist |

| **`REPORT.md` restructured and cut to the brief, 2026-08-11 (day 9)** | `FinalProjectPart2.pdf` re-read before acting, and its required structure is **not** what the report had: seven numbered sections, including a **Paper results** section we lacked, and no place for the standalone *Method* section we carried. Rewritten to match item for item: 1 Original architecture (model, loss, objective, hyperparameters), 2 Paper results, 3 Reconstruction results, 4 Improved architecture, 5 Improved results, 6 Discussion, 7 References. **Cut from 11,229 words to 4,396** at Ben's instruction to keep it minimal: the long gap-to-0.080 argument reduced to its conclusion plus one figure, the sweep protocol folded into §4, candidate-by-candidate prose replaced by one coverage table, references stripped of annotation. Language simplified throughout. **`REPORT.pdf` now builds**, 10 pages, via `report/build.sh` (regenerates figures, runs verification, then pandoc/xelatex) |
| **Figures, first ones in the project** | Four, all generated from `RESULTS.csv` by `report/make_figures.py`, none drawn by hand. **fig1** the 26-candidate spread against the 3-seed band, which is the headline finding and the one that reads instantly as a picture; **fig2** paper / released weights / reconstruction / improved, both languages, with seed spread as error bars; **fig3** E9's per-seed paired deltas side by side on Chinese and English, showing same-sign against mixed-sign; **fig4** one checkpoint scored under both GT conventions |
| **`report/verify_report.py` added** | Every figure quoted in the report is re-derived from `RESULTS.csv` and checked, **63 assertions, exit non-zero on any mismatch**, wired into `build_pdf.sh` so a stale number cannot reach a PDF. It immediately earned itself: it flagged the oracle numbers as wrong, which turned out to be the checker reading `oracle_chn.csv` row 0 (font `00`) instead of the 34-font column mean. Report was right, checker was wrong, both now correct |

| **External review, 2026-08-12: "not ready to submit"** | `docs/review-gemini.md`. Three critical issues — E9 promoted on a delta inside its own floor, checkpoint selection on a criterion measured at ρ = 0.125 against the reported metric, and a reproduction left incomplete at 6× Chinese augmentation against the paper's 10×. All three taken; §8 item 14 is the decision, `docs/review-response.md` the runbook. **Closed 2026-08-14 — see below and §8 item 14** |
| **`train.py` validates on the test split** | Found 2026-08-12 while implementing rendered-metric selection. Line 129 is `get_loader(..., 'test')`, so `val_metric` was always computed on the fonts the report scores, and a rendered metric computed the same way would be oracle selection rather than the fix the review asked for. Answered with a real held-out split carved out of train (`scripts/make_val_split.py`), not by moving that line. **Third instance of the same class as the two-rasterizer problem and Job B: an evaluation apparatus that does not measure what it is taken to measure.** §6 material |
| **`aug_rules` silently duplicates past index 4** | Found 2026-08-12. The released `data_utils/augment.py` ends its five rules in a bare `else`, so any `aug_idx ≥ 4` returns `rotate(-5)`: `--n_aug 9` would have written five identical copies and reported itself as 10× augmentation. Nine distinct rules now, 0–4 byte-identical to the released ones, unknown indices raise. An upstream data-pipeline bug, and the reason the 10× rebuild is trustworthy |
| **Rendered-metric selection, wired 2026-08-12** | `render_val.py` decodes the held-out val split at every checkpoint, rasterizes with cairosvg and binarizes at `eval_reconstruction_error.py`'s own threshold; `checkpoint_metrics.csv` gains `val_render_l1` / `val_render_siou` / `val_render_renderability`; `--ckpt_select val_render_l1` makes both `prune_checkpoints` and `scripts/best_checkpoint.py` rank on it. Defaults are off and `val_metric`, so every historical command reproduces unchanged. `check_infra.py` now 224 checks |
| **rv-review batch trained and scored 2026-08-12/13, floor read 2026-08-14: §4.5 gate fired** | 27 Chinese runs (`test_experiments_review_batch.out`), all 150 epochs, selected on `val_render_l1`, scored on the untouched test split at `n_samples 50`. Floor computation crashed on a script bug (`scripts/test_experiments.sh:321`, missing `rv_` prefix on the `L1S` lookup — fixed). **Re-measured Chinese floor 0.0151** (seed L1s 0.1534/0.1560/0.1685), **63% above the old 0.0093** — past `docs/review-response.md` §4.5's "materially larger" line. No candidate clears it on L1: E9 and E1 aren't even same-sign across seeds; E2/E3/E4 agree in sign but stay inside the floor. (The s-IoU floor came back narrower instead, 0.0223 vs the old 0.0315; on that metric alone E1 replicates its pre-review degrading result and E2 newly clears — disclosed, not promoted, since §4.5 gates on L1 before any candidate is read. §8 item 14.) **Ben's call, 2026-08-14: follow §4.5 as written.** The floor measurement is the finding, the pre-review 6× tables stand as the substantive results with their caveat, and **the English arm does not run** — no `rv_*_eng` runs exist. `scripts/selection_disagreement.py --floor 0.0151` (no GPU) independently confirms the review's question 2: 21/27 runs disagree between `val_metric` and rendered-metric selection, mean cost **+0.0186** rendered L1, above the new floor on 20/27 runs — the pre-review 26-candidate sweep *was* selection-confounded. All 27 runs in `RESULTS.csv` as `batch=rv-review`, tagged non-comparable to every earlier batch. Full account in §8 item 14 |
| **`main..repro` diff review, 2026-08-15 (day 13)** | Never read as one before this session, despite being a graded section on its own. 112 files, two focused passes: core model/train code (`train.py`, `models/*.py`, `options.py`, `checkpoint_log.py`) and eval pipeline plus all 18 `scripts/` files. No correctness bugs against any reported number. `eval_reconstruction_error.py`'s and `render_val.py`'s rasterize/binarize logic independently confirmed byte-for-byte consistent (same cairosvg call, same `255*3/4` threshold as the pre-existing `cal_iou`) — the thing most worth checking after finding the two-rasterizer property, and it checks out. The already-fixed `rv_` prefix bug's class (mismatched lookup-table keys built by string concatenation) does not recur anywhere in `test_experiments.sh` or `run_experiments.sh`. **Three doc-only fixes made, zero behavior change:** `WeightEMA`'s docstring claimed BatchNorm buffers are copied rather than EMA-averaged; the code averages them uniformly with the parameters and always has, so the docstring was wrong, not the code — corrected. `str2bool`'s docstring called `--resume` "pre-existing," but it's new to this project and simply never got moved onto the new helper (the actual `type=bool` bug on `--resume` is unchanged, per §8's existing deliberate deferral) — corrected to say so accurately. `--img_norm instance`'s help string didn't warn that it silently zeros the encoder's `[*,1024,1,1]` bottleneck layer, the reason E2 dropped it at rung 1 — a caveat is now in the help text itself, not just in this plan. **Reported, not acted on, four lower-severity items:** `train.py`'s `wandb.init` has no `mode=`/try-except guard against an unattended run hanging on an auth prompt (infra risk, not a report-correctness issue); `--wandb`/`--max_ckpt_keep` defaults deviate from released behavior, which is already documented policy in `docs/infra-upgrade.md`, not a new finding; `check_infra.py`'s run-name-consistency check only compares the Chinese `EXPERIMENTS` arrays, not the English ones (currently both match by hand-check, but the checker itself has a blind spot); `run_experiments.sh`'s entry parser lacks the no-space guard `test_experiments.sh` added, currently harmless since every entry carries `--seed N`. None of the four touches a number in `RESULTS.csv` or `REPORT.md` |

| **`REPORT.pdf` rebuilt 2026-08-15 (day 13), stale against the review-batch close** | `report/REPORT.pdf` was last built 2026-08-12, before the `cfbe8ab` commit that closed §8 item 14 and added §5.6 — the PDF submitters would have downloaded did not yet contain the review response. Rebuilt via `report/build.sh`. Caught one drift in the same pass: the References section's closing line still quoted "113 scored checkpoints," the count from before the 27 `rv-review` rows landed; `RESULTS.csv` is 140 rows now and the line is corrected. `verify_report.py` does not check that specific sentence (it asserts numeric claims, not the row-count prose), so this was a manual read-through catch, not a script catch — worth remembering next time a batch changes `RESULTS.csv`'s size. Rebuilt PDF passes `verify_report.py` clean, 140 rows, 0 failed claims |
| **Review batch staged, 27 + 9 runs** | Chinese: three baseline seeds plus E9, E1 and the six Job C categories, three seeds each, on a rebuilt 10×-augmented dataset with 20 base fonts held out. English: three baseline seeds plus E9 and `e3_refine2`, three seeds each, scored at `--n_samples 10` (the paper's own English protocol, closing the review's §2 point). **Nothing in this batch is comparable to any row in `RESULTS.csv`** — different training set, different size, different selection rule. Every comparison internal, floor re-measured from the batch's own three seeds. Arrays live in `scripts/run_experiments.sh` and `scripts/test_experiments.sh` |
| **Public release cut, 2026-08-28** | Branch **`submission`** = `repro` plus one cleanup commit, tagged **`v1.0-submission`**. Drops `PROJECT_PLAN.md`, the 13 `docs/` runbooks, `COMMANDS.md`, both PDFs, `report/assets/model_output/` and the review-batch log; keeps `RESULTS.csv`, the evidence CSVs, `data_splits/`, `report/`, and every upstream file untouched, so **`main..submission` is 69 files / 10,761 insertions / 0 upstream deletions** and reads as contribution rather than reorganization. New: `README.md` rewritten for the fork, `docs/REPRODUCE.md` (end-to-end), `docs/PROVENANCE.md` (the ~80 code comments citing this file resolve on `repro`, left unrewritten to keep the diff clean), `archive/README.md`. Fresh clone verified: `report/build.sh` → 0 failed claims, `check_infra.py` → 224 passed. Two stale counts corrected in the report while doing it: the verifier is **81** assertions not 63, and `check_infra.py` is **224+** not 196, so the report no longer states a preflight count at all. Remaining, all off-Mac: `docs/release-workplan.md` (cluster checkpoint inventory, Drive upload, push, default-branch switch) |

**Retired 2026-08-06.** Earlier revisions of this section leaned on 0.1668 landing near DeepSVG's published Chinese 0.167. With the raster-convention pipeline floor measured at 0.1422, any model scored this way inherits the same offset, so the agreement is arithmetic rather than corroboration. §2.4 says to drop it from the report.

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

It is a rasterized proxy for vector fidelity. It says nothing about command count, self-intersection, or control-point placement, only whether the inked region lands in the right place. Three further properties belong in the report:

1. It is scored after best-of-N_s selection, so it measures the best candidate rather than the average sample.
2. Sub-pixel coordinate accuracy is largely invisible to it. This caps how much any coordinate-level improvement can show up, and the discussion section should say so rather than let a small delta look like a small idea.
3. **It is undefined until the ground-truth side is specified, and on Chinese that choice is worth seven times the paper's own headline margin.** Added 2026-08-06.

**The two-rasterizer problem.** "Error" names a comparison, not a quantity, until both operands are pinned down. The candidate glyph is always rendered by our rasterizer. The ground truth can be either the dataset's stored bitmap, produced by a *different* rasterizer when the dataset was built (`--gt_source raster`, the convention this project used for its first thirteen days), or the ground-truth outline pushed through the same rasterizer as the candidate (`--gt_source svg`, added 2026-08-05). The gap between them is pure cross-rasterizer disagreement, carrying no information about any model.

Measured on the released checkpoints (§2.4), it is not a rounding term and it is not language-symmetric:

| | raster GT | svg GT | difference | pipeline floor (`l1_inf`) |
|---|---|---|---|---|
| Chinese, 34 fonts | 0.1629 | 0.1174 | **0.0455** | 0.1422 |
| English, 34-font subset | 0.0658 | 0.0584 | **0.0074** | 0.0253 |

The difference column tracks the floor column, which is what identifies the term as a property of the rasterization pipeline rather than of the glyphs being scored. Two consequences:

- **A Chinese Error is uninterpretable without its convention.** §1.3 puts the paper's whole margin over DeepVecFont on Chinese at 0.006. The convention choice moves the number by 0.0455.
- **Under the raster convention the Chinese metric is floor-dominated.** 0.1422 of any Chinese score is disagreement between two rasterizers over identical outlines, which is 88% of the baseline's 0.1621 and more than the paper's entire reported 0.080.

Every Stage 2 number in this document is raster-convention throughout, on both arms of every comparison, so the floor cancels in the paired differences and no conclusion in §3 or §5 depends on the choice. It matters only where an absolute value is compared against an external number, which is §2.4 and nowhere else.

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

**Three more, found 2026-08-08 when the paper itself (`DeepVecFontV2.pdf`) was added to the repo root and Sec. 4.1 was read against our own protocol. The first is material and touches every English number in this project.**

| Where | Paper says | We did | Consequence |
|---|---|---|---|
| Sec. 4.1 | `Ns` = **10 for English**, 50 for Chinese, with the final output the candidate of highest IoU | `n_samples 50` on **both** scripts, in every English row of `RESULTS.csv` | §2.4's English comparison is not like-for-like |
| Sec. 4.1 | the Chinese training set is "enlarged **ten times**" by affine augmentation | `data_utils/augment.py` implements exactly this but defaults to `--n_aug 5`, and it is an offline build step; **whether our Chinese data was built with it at all is unrecorded** | live lead on the open ~0.037 Chinese residual |
| Sec. 4.1 | **1425** English test fonts | §0 records the split as 1,386 | one sentence in §2.1; the English arm was scored on a 34-font subset regardless |

**Term 1, the `Ns` deviation.** "Error" is scored after best-of-N selection (§1.2, property 1), and N is not a free parameter — a larger N can only lower L1. Our English 0.0584 against the published 0.052 was therefore measured with five times the candidate budget, so the true gap is *wider* than §2.4 states. Note the direction carefully, because it is counter-intuitive: this **strengthens** §2.4's conclusion rather than weakening it. That section's argument rests on English nearly reproducing while Chinese does not, and English reproducing *less* well than we thought makes the two scripts more alike. The paired Stage 2 comparisons are untouched: both arms of every English delta in §5.2 used `n_samples 50`, so N cancels exactly the way the raster-convention floor does. `docs/day6-gpu-push.md` §4 fixes it going forward by screening at 10 and rescoring the seed-1111 anchor at 10, which measures the effect's size for free. Worth doing regardless of Job C-EN: a single checkpoint scored at N = 10 and N = 50 is a direct measurement of how much of a published best-of-N number is the model and how much is the sampling budget, and it belongs in §6 beside the two-rasterizer finding as a second instance of *a metric is a pipeline, not a formula*.

**Term 2, the augmentation, is the one that could close a standing open question.** §2.4's term 2 has carried a Chinese-specific residual of ~0.037 as unexplained since 2026-08-06, attributed vaguely to "the Chinese data or test protocol". A training set built without the paper's 10× affine augmentation — 212 fonts rather than 2,120 — is exactly the shape of cause that produces a Chinese-specific gap while English, whose 8,035 training fonts need no augmentation, reproduces cleanly on the same code path. **It costs no GPU to check**: count the Chinese training set through `get_loader` and see whether it reports 212 fonts or 2,120. Command in `docs/day6-gpu-push.md` §8.2. Retraining on a 10× augmented set is affordable on Chinese at ~1 h per run and would be the single most interesting result still available, but it is a scope change and gets an §8 entry before anything launches, not after.

**Two things that do match, checked so they are not re-litigated:** candidate selection is by highest IoU in both the paper and `test_few_shot.py` (`iou_tmp > iou_max[i]`); and Adam at lr 2e-4, 64×64 images, and 4 / 8 reference glyphs on English / Chinese all match `COMMANDS.md`. Also for §6: Tab. 3's sampling-point study spans 0.0557 → 0.0520 across six settings with individual steps of 0.0002 to 0.0014, so like Tab. 1's ablation **every step in it sits below our measured English floor of 0.0038.** That is two of the paper's own tables, not one, whose individual rows this instrument could not have resolved.

A fifth item is not a paper discrepancy but belongs in the same discussion. In `Transformer.forward` the loop unpacks `cross_attn, cross_ff, self_attns` and uses only `self_attns`. The cross-attention modules and `self.latents = nn.Parameter(torch.randn(256, 512))` are constructed, handed to the optimizer, and never called. With `depth=6` and `self_per_cross_attn=2` the encoder is 12 self-attention blocks and zero cross-attention blocks, so the model is not the Perceiver its configuration block advertises. The authors left a comment marking this as known.

**Measured (2026-08-05), `scripts/dead_params.py`.** The dead cross-attention path (`latents`, `to_logits`, `to_patch_embedding`, `pre_lstm_fc`, both `layers[*]` and `layers_cnnsvg[*]` cross-attention and cross-feedforward blocks) totals **23,204,402 parameters against a 136,055,207-parameter model — 17.06% dead**, carried through Adam as 185.6 MB of moment state (fp32, two moments) that trains nothing. Not run as a candidate — see §3.6's E6 note for why an ablation of it cannot be read against the seed floor — but the count itself is a clean reconstruction finding for §1.4.

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

#### `val_metric` and the KL term — decided 2026-08-05 (§8 item 8)

`val_metric` deliberately excludes the KL term, because KL does not enter the rendered output, so checkpoint selection tracks reconstruction quality rather than the full training objective; the consequence, that E12's checkpoint selection is blind to exactly what E12 varies, is stated wherever an E12 number is quoted.

That is the whole decision, and it is a choice rather than a fix, unlike the `svg_para` omission in the same function which was a genuine bug. It is recorded here rather than left open because leaving it open is precisely how the `svg_para` gap survived from the first run to day 2. Supporting evidence in §3.6: at one seed, removing the KL term entirely is the second-best row in Tier 3 Batch A and raising it 10× is the only row worse than the anchor, so the term moves the rendered metric very little in either direction.

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

| Grid | Bin width (numericalize units) | At 64×64 | Rounding error per coordinate |
|---|---|---|---|
| n = 256 (the paper's Sec. 3.1 text) | `30/256 = 0.1172` | 0.3125 px | uniform on ±0.15625 px |
| n = 128 (`models/transformers.py`) | `30/128 = 0.2344` | 0.625 px | uniform on ±0.3125 px |
| n = 64 (`data_utils/relax_rep.py`) | `30/64 = 0.4688` | 1.25 px | uniform on ±0.625 px |

One pass over the test set, and it becomes the reference row that tells you what fraction of your gap to 0.080 is even addressable by a coordinate-level change. At the 64-bin grid that fraction is roughly four times larger than at 128, so the answer matters.

**Written 2026-08-05: `scripts/quantization_oracle.py`.** Mac-side, runs on the cluster because it needs the dataset. Two things about it are worth stating in the report rather than buried in the script.

**The pixel arithmetic in the table above hides a trap, and the script asserts against it.** `numericalize` normalizes by **30**, while `SVG_PREFIX_BIG` in `data_utils/svg_utils.py` sets **`viewBox="0 0 24 24"`**. Neither constant is the image size and they are not the same number, so a bin is `(30/n) · (img_size/24)` pixels — 0.625 px at n=128, not the 0.5 px that `img_size/n` would give. The 24/30 factor is exactly the kind of thing that silently rescales a floor by 25%, so `--selftest` checks all three widths and the erroneous form explicitly.

**The oracle reports two floors, and only the difference is about quantization.** Rendering the *unquantized* ground-truth sequence through `render()` and scoring it against the dataset's own stored raster does not give L1 = 0: the two rasterizers disagree at glyph edges, `render()` truncates past `max_seq_len`, and the path is reconstructed from relaxed commands. The script therefore always evaluates `n=inf` alongside the requested grids.

```
L1(n=inf)                the pipeline floor -- rasterizer and representation
L1(n=128) - L1(n=inf)    the cost of quantization proper, at the released grid
L1(n=128) - L1(n=256)    the ceiling on what E13 could ever have bought
```

Quoting the raw oracle as "the quantization floor" without subtracting the pipeline term would attribute the whole residual to quantization. That is the same class of mistake as anchoring every delta on one seed, one level down, and it is worth a sentence in §6 next to the anchor bias for exactly that reason. The third line is the one that retires E13: it is an upper bound on a candidate that already screened null, computed without training anything.

**Run 2026-08-05, and a bug caught before it reached the table.** The first run of `scripts/quantization_oracle.py` (34 Chinese test fonts, 1768 glyphs) printed a pipeline floor of L1 = 0.1939 — *larger* than the trained model's own confirmation-budget score (0.1621 three-seed mean), which is not a coherent reading for a floor: the oracle replays the exact ground-truth sequence, so nothing should beat it. A rendered side-by-side of two glyphs (`char3`, `char7` of test font 19) showed why: every reconstructed glyph carried a spurious thin stroke from a corner of the canvas into the shape.

Traced to a real bug in the new script, not in the thirteen days of results behind it. The relaxed representation's 8 args per command are `[start_x, start_y, c1x, c1y, c2x, c2y, end_x, end_y]` — a redundant 4th control point (the start) kept only so the relaxation-consistency loss has something to constrain. `render()`'s `_make_simple_cmds_long` (unchanged from the original released code, confirmed identical in `data_utils/svg_utils_backup.py`) was written for that code's 6-arg predecessor and was never updated for the 8-arg relaxation: fed 8 args, it reads `arguments[8],[9]` for a Move/Line target and gets the *unsupervised* `c2x, c2y` slack values instead of the true `end_x, end_y` (`cmd_args_mask` never trains positions 2–5 for those command types), and for a Curve it shifts every control point back by one, silently dropping the true endpoint. `models/model_main.py:130` already knows this — `sampled_svg_2 = torch.cat([commands2, args2[:, :, 2:]], dim=-1)` explicitly drops the redundant start point before calling `render()` for every SVG `test_few_shot.py` has ever scored. `scripts/quantization_oracle.py` skipped that trim, since it reads `sequence_relaxed.npy` directly rather than going through the model's decode path. Confirmed the production path is unaffected: `experiments/seedfloor_1111_chn_main_model/results/150_6040.ckpt/0025/svgs_single/syn_50_23_refined.svg` — an actual scored file — rendered clean with no artifact. **This bug is confined to the oracle script written today; it does not touch any Tier 1–3, seed-floor, or confirmation number in this document.**

Fixed by the same trim `model_main.py` uses (`scripts/quantization_oracle.py`'s `score_font`, one line). Self-test still 18/18 after the fix (it only checks the quantization arithmetic, not this rendering path, so it couldn't have caught this — worth remembering next time a script's self-test looks green but its output doesn't). Re-rendering the same two glyphs confirmed the hairline gone and a plausible-looking reconstruction underneath it.

Re-run, 34 fonts / 1768 glyphs, corrected:

| Grid | Bin px | L1 | s-IoU | SSIM |
|---|---|---|---|---|
| n = ∞ (pipeline floor) | — | **0.1422** | 0.3713 | 0.4916 |
| n = 256 | 0.3125 | 0.1427 | 0.3698 | 0.4911 |
| n = 128 (released) | 0.6250 | 0.1443 | 0.3664 | 0.4884 |
| n = 64 | 1.2500 | 0.1453 | 0.3642 | 0.4834 |

This is now a coherent floor: 0.1422 sits *below* the three-seed baseline mean of 0.1621, as a floor should. Cost of quantization at the released grid, pipeline floor subtracted: **L1 +0.0021** at n=128, +0.0031 at n=64. **Ceiling on E13 (128→256 bins): L1 +0.0016**, well under the 0.0097 seed floor — E13 could not have cleared the floor no matter how it landed, which is consistent with (and now explains) its null screening result. Per-font rows in `oracle_chn.csv`.

### 2.4 The gap to 0.080

**Rewritten 2026-08-06.** This section carried an argument for three days and now carries a
measurement. The argument was that the gap is dominated by training budget. That is
**falsified for Chinese**, by the authors' own released checkpoint. What follows replaces it
outright. The superseded reasoning is in git history at `4a157e4`; do not quote its
conclusion from anywhere else, including `REPORT.md`.

**The decisive run.** `docs/official-checkpoints-and-600.md`, Job A: the released
DeepVecFont-v2 checkpoints, loaded strictly against 136,055,207 parameters, decoded and
scored through this repo's own harness at `n_samples 50`. Nothing was trained on our side,
so every number below is a statement about the evaluation rather than about our
reproduction.

| | paper | official 600 ep, raster GT | official 600 ep, svg GT | ours, 150 ep, 3-seed mean |
|---|---|---|---|---|
| Chinese, full 34 fonts | 0.080 | **0.1629** | **0.1174** | **0.1621** |
| English, 34-font subset | 0.052 | **0.0658** | **0.0584** | none — see below |

#### What the first row settles

**The reproduction is faithful.** The 150-epoch Chinese baseline scores 0.1621 against the
released 600-epoch checkpoint's 0.1629 under the identical convention, font set and sample
budget. The difference is **0.0008**, roughly a twelfth of this project's own seed-noise
floor of 0.0093, which is to say indistinguishable. Whatever separates this repo from the
published number, it is not our training run. This is what the assignment's "compare your
results to the reported results to within negligible differences" clause can actually be
answered with, and it is a considerably stronger position than the one held on 5 August.

**Training budget is eliminated.** The released checkpoint has four times our epoch budget
and scores no better through this harness. The three 600-epoch Chinese seeds trained on
2026-08-06 (Job B, `seedfloor600_<seed>_chn`, not yet scored) therefore have a known upper
bound before they are read: the authors' own 600-epoch weights sit at 0.1629. Scoring them
now confirms rather than decides, and a result near 0.162 is the expected one.

#### Where the gap actually lives

**Term 1 — the rasterizer, worth roughly half of it on Chinese.** `--gt_source raster`
compares the model's rendered glyph against the dataset's stored ground-truth bitmap, which
was produced by a *different* rasterizer at dataset-build time. `--gt_source svg` (added
2026-08-05) renders the ground-truth outline through the same rasterizer as the candidate,
which removes that term. The difference between the two columns is therefore the
cross-rasterizer disagreement, and it is strongly language-dependent: **0.0455 on Chinese**
(0.1629 → 0.1174) against **0.0074 on English**.

The pipeline floor says the same thing independently. Scoring ground truth against ground
truth, mean of the `l1_inf` column: **Chinese 0.1422** (`oracle_chn.csv`), **English 0.0253**
(`oracle_eng_subset34.csv`). The released 128-bin quantization grid adds only +0.0021 and
+0.0013 respectively on top. **The paper's 0.080 sits 0.062 below the Chinese floor**, so
under the raster convention it is unreachable by any model whatsoever, theirs included. The
English floor at 0.0253 sits well below its 0.052. The convention is load-bearing for
Chinese and nearly irrelevant for English.

**Term 2 — a Chinese-specific residual of about 0.037, still open.** Under the svg
convention the rasterizer term is gone by construction, and the two languages still diverge:
English reaches 0.0584 against 0.052, a 12% gap on a scoped subset; Chinese reaches 0.1174
against 0.080, still 47% high. Same harness, same eval code, same released weights, opposite
outcomes. What remains points at the Chinese data or test protocol rather than at the model
— the test font list, the character subset, `ref_char_ids`, or the dataset build itself.
None of that was measured, and none of it is worth GPU time at this point in the schedule.
It is reported as an open residual with its size stated.

**Read it against §1.3's scale.** The paper's entire margin over its own predecessor on
Chinese is 0.006 absolute. The rasterizer term alone is 0.0455, more than seven times that
margin. A Chinese Error quoted without its ground-truth convention is uninterpretable at the
scale the paper's own claims live at, which is why §1.2 now carries this as a property of
the metric rather than as a footnote here.

**Retire the DeepSVG coincidence.** Earlier drafts leaned on 0.1668 landing near DeepSVG's
published Chinese 0.167. With the raster-convention floor measured at 0.1422 that agreement
carries no information: any model scored this way inherits the same 0.14 offset, so landing
near another paper's number under a different pipeline is arithmetic, not corroboration.
Drop it from the report.

#### Caveats that travel with the English row

It is not a full run. English's test set is 1,386 fonts against Chinese's 34, and a full
3-checkpoint sweep projected to ~40 GPU-hours and >100 GB against a 200 GB quota near its
cap. Scope was cut to a deterministic 34-font subset for the 500/550/600 comparison plus an
862-font partial decode of checkpoint 500. **The subset reads optimistic against the larger
sample by 0.0074 (raster) and 0.0106 (svg).** Applied to the 600-epoch row that puts a
full-set English estimate nearer 0.073 / 0.069 than 0.0658 / 0.0584, which widens the gap to
the paper without changing the Chinese-versus-English contrast that the section rests on.
Quote 0.0658 as a scoped estimate with a known bias and its direction, never as a measured
number.

**Stage 2 is untouched by any of this.** E9 is a paired per-font comparison against our own
baseline, three seeds, one fixed convention throughout. A change in the metric's absolute
scale moves both arms of that comparison equally.

### 2.5 Stage 1 closes when

The extended eval script runs on the existing Chinese results and emits Error, SSIM, s-IoU, renderability, and a per-font CSV; the oracle row and bin histogram exist; and §2.4 has an answer. It can then be written up while Stage 2 trains.

**Closed 2026-08-05.** SSIM lands in the §5 table for all six confirmation rows (bit-identical L1/s-IoU rescore, so nothing but the new column moved), the oracle row is filled with a bug caught and fixed along the way (§2.3), the bin histogram has been in since day 1, and §2.4 has its answer with the oracle's pipeline floor folded in. Report §6 sections 1–3 can now be written from real numbers.

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

**Measured (2026-08-05), from `RESULTS.csv`: every delta in this document is referenced to the worst of the three baseline seeds.** All twenty-six single-seed candidate rows across Tiers 1, 2 and 3a are quoted as a delta against `seedfloor_1111_chn`. Twenty-one of the twenty-six come out *better* than that anchor — a two-sided sign test gives **p = 0.0025**, against the roughly thirteen-of-twenty-six expected if the changes did nothing. Read naively, that says almost every arbitrary single-factor change improves the model, which is not credible.

The actual cause is the anchor. Seed 1111 scored **0.1725**, seed 2222 **0.1631**, seed 3333 **0.1680**: the anchor is the worst of the three draws and sits **0.0046 above the baseline mean of 0.1679**. Re-reference the same twenty-six rows to that mean and the picture inverts — twenty-one of twenty-six are now *worse* than baseline, and only two rows (`e2_batchnorm_chn` −0.0034, `e12_kl000_chn` −0.0030) improve on it by more than 0.003:

| Reference | Candidates better than it | Largest improvement |
|---|---|---|
| `seedfloor_1111_chn` = 0.1725 (the anchor used throughout) | 21 / 26 | −0.0080 |
| Baseline mean = 0.1679 (three seeds) | 5 / 26 | −0.0034 |

Nothing already recorded is *wrong* — a delta against a named single-seed run is a well-defined quantity, and §3.6's paired reading of Batch B was never affected, since that one differences each candidate seed against its own matching baseline seed. But three consequences follow:

1. **Every "largest delta" ranking in this document was computed against an unlucky draw**, including the ranking that selected Batch B's three candidates. That selection has now been tested by replication and it went **1 for 3** (E9 survived, E1 and E13 flipped sign). Treat that 1/3 as the measured base rate for "the leading single-seed candidate survives replication", and size any future replicate-the-leader batch against it rather than against optimism.
2. `e2_batchnorm_chn` (−0.0080) and `e12_kl000_chn` (−0.0076) are now larger single-seed deltas than E9's −0.0059 ever was. That is not evidence they are better than E9 — it is the same measurement that Batch B just showed to be 2/3 misleading, and both are −0.003 rather than −0.008 once referenced to the mean.
3. **Report both references in §5.** Quote each delta against the baseline mean, with the per-seed anchor kept alongside so the earlier tables remain traceable. A results table that silently anchors on one seed of three is the exact failure mode this section exists to prevent, and having walked into it and caught it is worth a paragraph in §6.

**Measured (2026-08-05): changing the architecture moves the metric about half as much as changing the seed does.** Same twenty-six rows, treated as a sample rather than as individual claims. They span 0.1645 to 0.1746, a range of 0.0101 with sd 0.0026. The three baseline seeds span 0.1631 to 0.1725, a range of 0.0094 with sd 0.0047. If both sets were draws from the same distribution, twenty-six draws should span roughly **2.3×** the range of three (the expected range of a normal sample is ≈1.69σ at n = 3 and ≈3.90σ at n = 26); the observed ratio is **1.07**, implying a candidate-induced σ around 0.47 of the seed-induced σ. The direct sd ratio gives 0.56, which agrees. So: twenty-six deliberate single-factor architectural changes, spanning normalization, latent width, optimizer, quantization, loss weighting, LR schedule, regularization and capacity, collectively perturb the rendered metric **less than re-running the released model under a different random seed does.** State it with the caveat that an sd from three points is itself poorly determined; the range argument, which does not depend on that sd, carries the claim on its own. This is the strongest single sentence Stage 2 produced and it belongs in §6.

**Recomputed (2026-08-05, after the confirmation session): the screening anchor itself moved, and every delta in this document is now generated rather than typed.** Clearing `results/<ckpt>/` to fix the stale-budget bug in §3.7 also removed the seed-1111 screening tree, so `build_results_table.py` picked up a re-scored row on the next rebuild. The anchor reads **0.1728 / 0.2240** where every table above was written against 0.1725 / 0.2165, and `e9_sigma050_chn` moved 0.1665 → 0.1661.

Both shifts are decode noise, and the decision was to **recompute rather than caveat**. `scripts/recompute_deltas.py` (new) regenerates every screening delta directly from `RESULTS.csv`, printing a *vs anchor* and a *vs three-seed mean* column side by side so the disagreement between them stays visible. The hand-typed tables in §3.2, §3.5 and §3.6 are kept for provenance, since they are what every ranking decision was actually made on, but **the script is the live table and it wins on conflict.**

| | As recorded | Recomputed |
|---|---|---|
| Anchor `seedfloor_1111_chn`, L1 | 0.1725 | **0.1728** |
| Anchor, s-IoU | 0.2165 | **0.2240** |
| Three-seed mean, L1 | 0.1679 | **0.1680** |
| **L1 floor** | 0.0093 | **0.0097** |
| **s-IoU floor** | 0.0390–0.0401 | **0.0315** |
| Candidates beating the anchor | 21 / 26 | **22 / 26** |
| Candidates beating the mean | 5 / 26 | **6 / 26** |
| Candidate L1 span vs seed span | 0.0101 / 0.0094 | **0.0101 / 0.0097** |
| E9 σ=0.5 screening delta vs anchor | −0.0059 | **−0.0067** |

No conclusion changes: E9 remains the only same-sign candidate, every tier stays null by a wide margin, and the anchor still sits +0.0048 above the mean. Two readings sharpen.

**The s-IoU floor was inflated, so E1's negative result is larger than recorded.** Seed 1111's s-IoU moved 0.0075 between screening sessions on a bit-identical checkpoint, against 0.0003 for L1 on the same pair of runs. s-IoU's decode noise is therefore an order of magnitude larger in absolute terms than L1's, and the excursion happened to land in the direction that widened the floor. The correction runs both ways and both terms move. E1's own seed-1111 paired difference was computed against the old 0.2165 anchor and reads −0.0139 against 0.2240, so the deficit deepens from −0.0735 to **−0.0760** at the same time as the floor shrinks. **E1 is 2.4× its floor**, not the 1.9× quoted against 0.0401. Quote the floor as ≈0.03, not to four digits, and say in §6 that a floor estimated from three points is itself a noisy quantity — which is the same lesson as the anchor bias, one level up.

**The candidate span and the seed span have all but converged**, 0.0101 against 0.0097. The half-as-much reading above is unaffected in substance and slightly tighter in the ratio.

**Screen cheap, confirm expensive.** Do not run `test_few_shot.py --n_samples 50` over all 34 test fonts for every candidate.

- *Screening eval*: 8 fixed test fonts, `--n_samples 3`, at a fixed epoch. Minutes.
- *Confirmation eval*: all 34 fonts, `--n_samples 50`, matching the paper. Finalists only.

Freeze the screening font list and `ref_char_ids` before the first run and never change them.

**Check whether `val_metric` predicts the test metric.** *(Script written 2026-08-04: `scripts/val_metric_correlation.py`. Costs no GPU, runs against the existing Tier 1 and Tier 2 experiment dirs, and gates the E14-deep batch — see the E14 note above.)* `compute_val_loss` produces `val_metric` free at every checkpoint, and it now also goes to wandb as `CKPT/val_metric`. It is a teacher-forced loss; the test metric is a rendered, autoregressively decoded, best-of-N L1. How well they correlate on this model is unknown. After the first six experiments, compute the **rank correlation between `val_metric` and screening Error** across those six. High correlation means you screen everything else for free and spend the saved time on more candidates. Low correlation means the validation loss is not a usable proxy, which is a reportable finding in itself, and you screen on the rendered metric from then on.

The caveat from §1.5 applies: **E8 and E13 change the cross-entropy itself**, so those two are always screened on the rendered metric regardless of what the correlation says.

**Measured (2026-08-05).** `scripts/val_metric_correlation.py` across all 26 non-excluded Tier 1–3 runs: **Spearman ρ = 0.125**, Kendall τ = 0.083, rising only to ρ = 0.179 with E14/E15/E11 (the settling-confounded class from §3.2 (b)) dropped. Both readings are far under the 0.4 threshold this document set in advance. **`val_metric` does not predict the rendered metric on this model** — the teacher-forced loss and the autoregressive, best-of-N rollout diverge, which is itself the reportable finding §3.2 anticipated. Consequence for E14-deep: the batch shrinks from seven rows to the two peak-lr controls (`--lr 4e-4`, `--lr 8e-4`), which screen on the rendered metric regardless of the correlation result, per the rule set in the E14 note above. Every remaining candidate screens on the rendered metric, full stop — the cheap-proxy branch this section held open is closed.

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

**Launched 2026-08-04, rung 1 dropped one candidate before training started.** `--img_norm instance` (`nn.InstanceNorm2d`) crashes at the image encoder's deepest layer, which bottlenecks to a `[32, 1024, 1, 1]` feature map: `ValueError: Expected more than 1 spatial element when training`. InstanceNorm needs more than one spatial element to compute a per-instance variance; `GroupNorm` and `BatchNorm2d` in the same slot have no such constraint. This is an architectural incompatibility surfaced by rung 1's construction-then-shape probe (`docs/tier3-launch.md`), not a wiring bug `check_infra.py` could have caught (its E2 checks only verify the module constructs, not that it survives a forward pass at the bottleneck's actual resolution). Dropped from both `scripts/run_experiments.sh` and `scripts/test_experiments.sh`; the batch runs as fifteen rather than sixteen. Worth a line in the report's E2 discussion as its own finding: three normalization alternatives were tried, one is structurally incompatible with this architecture's 1×1 bottleneck. The other seven rung-1 probes (`--img_norm group/batch`, `--ngf 32`, `--bottleneck_bits 256/1024`, `--ema_decay 0.999`, `--optimizer adamw`) and the `--img_norm batch` checkpoint round-trip all passed cleanly.

**Measured (2026-08-04).** All fifteen trained (150 epochs) and screened alongside a re-screened three-seed baseline (`scripts/test_experiments.sh parallel`, GPUs 1-2, `n_samples 3`, all 34 fonts). Renderability was 100% (1768/1768, 34/34 fonts) for every one of the 21 runs.

Baseline, re-screened this session: seed 1111 L1 0.1725 / s-IoU 0.2165, seed 2222 L1 0.1631 / s-IoU 0.2410, seed 3333 L1 0.1680 / s-IoU 0.2555 — mean L1 0.1679, spread 0.0094, matching the standing 0.0093 floor.

*Batch A (breadth, single seed 1111):*

| Run | Checkpoint | L1 | s-IoU | delta vs baseline | clears 0.0094 |
|---|---|---|---|---|---|
| E12 `kl_beta=0.0` | `150_6040.ckpt` | 0.1649 | 0.2531 | −0.0076 | no |
| E12 `kl_beta=0.1` | `150_6040.ckpt` | 0.1735 | 0.2347 | +0.0010 | no |
| E11 AdamW, `wd=0.01` | `150_6040.ckpt` | 0.1698 | 0.2110 | −0.0027 | no |
| E2 `img_norm=group` | `150_6040.ckpt` | 0.1673 | 0.2349 | −0.0052 | no |
| E2 `img_norm=batch` | `150_6040.ckpt` | 0.1645 | 0.2515 | −0.0080 | no |
| E4 `ngf=32` | `125_5040.ckpt` | 0.1691 | 0.1965 | −0.0034 | no |
| E15 `ema_decay=0.999` | `150_6040.ckpt` | 0.1709 | 0.2278 | −0.0016 | no |
| E5 `bottleneck_bits=256` | `125_5040.ckpt` | 0.1687 | 0.2259 | −0.0038 | no |
| E5 `bottleneck_bits=1024` | `150_6040.ckpt` | 0.1699 | 0.2516 | −0.0026 | no |

**None of the nine clear the floor** — the same null as Tier 1 and Tier 2. Reportable as coverage of the assignment's change categories, per the batch's own design.

*Batch B (the three largest Tier 1/2 deltas, each at three seeds):*

| Candidate | Mean L1 (3 seeds) | Mean spread | Per-seed diff vs baseline (1111/2222/3333) | Same sign? |
|---|---|---|---|---|
| E9 `enc_noise_std_train=0.5` | 0.1645 | 0.0063 | −0.0060 / −0.0027 / −0.0013 | **yes, all negative** |
| E1 `enc_final_norm` | 0.1694 | 0.0075 | −0.0055 / +0.0037 / +0.0063 | no |
| E13 `n_args_bins=256` | 0.1699 | 0.0062 | −0.0050 / +0.0106 / +0.0006 | no |

Per §3.6's own reading rule, this is not read against the 0.0093/0.0094 floor — that number is baseline's own seed spread, the quantity being replaced. Read as mean-against-mean plus the paired per-seed sign.

**E9 σ=0.5 is the one candidate whose per-seed difference has the same sign at all three seeds** — L1 improves at 1111, 2222 and 3333 alike, even though the −0.0033 mean improvement is smaller than the baseline's own 0.0094 spread. E1 and E13 are both mixed-sign (one seed favorable, two not), which is the same inconclusive shape Tier 1 and Tier 2 produced throughout — Batch B's design exists precisely to tell these two cases apart, and this run separates them. Per `docs/tier3-launch.md`, a same-sign result "is worth the confirmation eval even if the means overlap" — E9 σ=0.5 is therefore the first Tier 1-3 candidate with a positive case for §3.7's confirmation eval, on grounds Tier 1/2 could not have supported.

Two things worth flagging rather than folding into the headline: E9 seed 2222 also shows the largest s-IoU in the whole batch (0.2838, +0.0428 over its own baseline seed) — echoes the E8 s-IoU-without-L1 pattern from §3.5, but is a different mechanism (train-time noise, not label smoothing) and hasn't been checked against the s-IoU floor here. And E1's seeds 2222 and 3333 selected earlier checkpoints (`125_5040`, `100_4040`) than every other Batch B row's `150_6040` — `best_checkpoint.py` picked them from the manifest, not hand-chosen, but it means E1's three seeds aren't even scoring the same epoch budget, which is a caveat worth carrying into any write-up of that row.

#### Read again on the second metric (2026-08-05)

The table above reads Batch B on L1 alone. The s-IoU column of the same six runs, paired against each seed's own baseline s-IoU (0.2165 / 0.2410 / 0.2555), changes what two of the three rows mean:

| Candidate | Paired s-IoU diff (1111 / 2222 / 3333) | Mean | Same sign? | Selected epoch |
|---|---|---|---|---|
| E9 `enc_noise_std_train=0.5` | +0.0357 / +0.0428 / +0.0027 | **+0.0271** | **yes, all positive** | 150 / 150 / 150 |
| E1 `enc_final_norm` | −0.0064 / −0.0681 / −0.1460 | **−0.0735** | **yes, all negative** | 150 / 125 / 100 |
| E13 `n_args_bins=256` | +0.0050 / +0.0152 / −0.0332 | −0.0043 | no | 150 / 150 / 150 |

**E9's case is stronger than the L1 table alone shows.** Six paired differences, two metrics, three seeds, and every one of the six points the same way. That matters because the two metrics turn out to be nearly independent instruments here rather than two views of the same thing: across the thirty-one 150-epoch Chinese rows in `RESULTS.csv` the Pearson correlation between L1 and s-IoU is only **r = −0.335** (negative meaning weak *agreement*, since lower L1 and higher s-IoU are both good), which is r² = 0.11 — about 11% shared variance and 89% that L1 cannot see. A permutation test puts that correlation at **p = 0.07**, so it is not even clearly distinguishable from zero at this sample size, which if anything sharpens the point: on this model, at this resolution, rasterized L1 and structural overlap are close to orthogonal readings of the same output. A candidate moving both in the favourable direction at every seed is therefore doing more than moving one number. This is also the measurement §1.2 and §6 have been missing: "what the metric can and cannot see" now has a number instead of an argument.

**E1 is not a null result. It is a same-sign negative one, and it should be reported as a finding.** §3.6 above records E1 as "mixed-sign", which is true of L1 and misses the s-IoU column entirely. On s-IoU, E1 degrades at all three seeds, monotonically, by a mean of −0.0735 — **1.8 to 1.9× the s-IoU seed floor**, and the only effect measured anywhere in this project that exceeds its own floor at all.

> **Superseded 2026-08-05 by the recompute in §3.2, and the effect gets larger.** Both terms of that ratio were computed against the pre-recompute seed-1111 anchor. E1's seed-1111 paired difference is **−0.0139** against the corrected 0.2240 anchor rather than −0.0064, so the three legs read −0.0139 / −0.0681 / −0.1460 and the mean deficit is **−0.0760**, against a floor of **0.0315**. **E1 is 2.4× its floor.** The paragraph below is kept as written because it is the reading that promoted E1 to a finding; quote 2.4× and −0.0760 in the report. Verified against `RESULTS.csv`.

*(A note on that floor, found while checking these numbers. §3.5 records it as 0.0401, from the Tier 1 session's baseline s-IoU of 0.2154 / 0.2410 / 0.2555. The Tier 3 re-screen in `RESULTS.csv` gives 0.2165 / 0.2410 / 0.2555, a spread of **0.0390**. The two differ because seed 1111's s-IoU moved by 0.0011 between screening sessions on an identical checkpoint — the same decode noise §3.2 measured on L1, and the same size. So the s-IoU floor is itself uncertain at the ±0.001 level and should be quoted as ≈0.039–0.040 rather than as 0.0401 to four digits. Nothing downstream changes: E1's −0.0735 is 1.83× the larger of the two and 1.88× the smaller.)* Its seed-3333 leg scores s-IoU 0.1095, the lowest value in the whole thirty-seven-row table by a margin, against a candidate range that otherwise bottoms out near 0.196. And the two degraded seeds are exactly the two that selected epochs 125 and 100 rather than 150, so late training stopped improving `val_metric` under the terminal LayerNorm on two of three seeds.

By the same rule that promotes E9, E1 earns a claim — a negative one: **a terminal LayerNorm on the sequence encoder leaves rasterized L1 unchanged while consistently degrading structural overlap and destabilizing late training.** Symmetry matters here. The promotion rule was written before the results were in; applying it only when it produces good news would be selection by another name. E1 was rank 2 of eighteen on the screening table and this is what replication did to it, which makes it the most instructive row in the report's section on why single points could not be trusted.

Caveat on both readings: the s-IoU differences are paired and read by sign, so the 0.0401 floor is not the applicable bar (it is the baseline's own unpaired spread, per the reading rule above). The floor is quoted for scale only.

**A note on `kl_beta=0`, and what it says about §8 item 8.** `e12_kl000_chn` is the second-largest single-seed delta in Batch A (−0.0076 against the anchor, −0.0030 against the baseline mean) and the largest of the two regularization rows, while `kl_beta=0.1` is the *only* Batch A row that is worse than the anchor. Removing the KL term entirely costs nothing measurable and may help slightly. That is consistent with §3.6's own prediction that with additive encoder noise the reparameterization is close to decorative, and it points at an answer for §8 item 8: if the KL term is doing little to the rendered output, `val_metric` omitting it is a defensible choice rather than a latent bug. One seed, so this is a direction rather than a result — but it is enough to decide a documentation question that has been open since 2026-08-04.

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

**Run 2026-08-05.** Only one candidate ever earned a confirmation slot (§5), so there is no combination to take — "combined model" is E9 σ_train=0.5 alone. Confirmation eval run at all three seeds, `n_samples 50`, σ_test 1.0 (§8 item 9), paired Wilcoxon on both L1 and s-IoU. Results and both Wilcoxon passes are in §5. Along the way, `test_few_shot.py`'s per-font resume check (`if os.path.exists(merge_outfile): skip`) turned out to key only on the merge HTML's existence, not on what `--n_samples` produced it — the seed-2222 and seed-3333 pairs initially silently reran on stale `n_samples=3` screening output left over from Tier 3, mislabeled as the n=50 confirmation. Caught by the 0 vs 34 "skipping" line count and the numbers matching `RESULTS.csv`'s screening rows exactly; fixed by clearing `results/<ckpt>/` before rerunning. Worth a line in §7 as a landmine distinct from the ones already in §1.5: any rerun of `test_few_shot.py` at a different `--n_samples` against an experiment directory that already has output needs its `results/<ckpt>/` cleared first, or it silently scores the old budget.

---

## 4. Schedule

**Deadline postponed 2026-08-11 to 1 September; this table stays dated against 15 August as a log of the original sprint, not current planning.** With the experimental programme closed and the report at full-draft prose across all seven sections (`_Open Tasks.md`, Day 9), the remaining critical-path items — push the commit, read the report end to end, add figures, review the `main..repro` diff — now have roughly three weeks of slack instead of days.

Dated against a 15 August deadline. Day 1 is Monday 3 August.

| Date | Day | Work |
|---|---|---|
| Mon 3 Aug | 1 | Sync the cluster, GPU dry run (§7.4). Time 5 epochs, size the matrix (§3.3). Launch the 3-seed baseline (§3.2). Bin histogram (§2.3). Resolve the §1.5 eval-script question. |
| Mon 3 – Tue 4 | 1–2 | Extend `eval_reconstruction_error.py`: SSIM, renderability, per-font CSV. Quantization oracle. Rescore the epoch-100 and epoch-125 Chinese results. Answer §2.4. **Stage 1 closes.** |
| Wed 5 – Fri 7 | 3–5 | Tier 1: E9 sweep, E1, E7 sweep, E10 sweep. Three GPUs in parallel. |
| Fri 7 Aug | 5 | Compute the `val_metric` vs rendered-Error rank correlation (§3.2). Decide the screening metric for everything after this point. |
| Sat 8 – Mon 10 | 6–8 | Tier 2, subject to §8: E8, E13, E3, E14. Plus the test-time σ sweep on the E9 winner. |
| Tue 11 Aug | 9 | Tier 3, as many as fit. |

**Actual, as of 2026-08-05 (day 3).** Tier 3 ran on day 2 night rather than day 9, so the whole experimental programme is six days ahead of this table and Stage 2's Chinese arm is answered. Days 3 to 9 are now report time plus the three things §8 item 10 kept in scope. Revised shape:

| Date | Day | Work |
|---|---|---|
| Wed 5 Aug | 3 | `docs/confirmation-launch.md`: σ_test ladder, E9 confirmation, paired Wilcoxon, `val_metric` correlation, `dead_params`. Mac-side: SSIM, the quantization oracle, §2.4 in prose. **Stage 1 closes.** |
| Thu 6 – Fri 7 | 4–5 | E14-deep if the correlation gates it open. English arm: time 6 epochs, 3-seed English baseline, freeze the budget from those curves *before* looking at a candidate. Report sections 1–3. |
| Sat 8 – Mon 10 | 6–8 | English generalization test on E9. Report sections 4–6, which is where the three §3.2/§3.6 findings do their work. |
| Tue 11 – Thu 13 | 9–11 | Figures, the `main..repro` diff review, full read-through. |
| Fri 14 – Sat 15 | 12–13 | Buffer and submit. |

The compression §4 warned about is gone. The risk has moved from "not enough GPU time" to "a null result written up thinly", which is a writing problem and is what the extra days are for.

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

**Filled 2026-08-05, `docs/confirmation-launch.md`.** This table is confirmation-budget
numbers (`n_samples 50`, all 34 fonts) for finalists only, per §3.2's "screen cheap, confirm
expensive" split. It stayed empty through Tiers 1 and 2 because nothing had earned a
confirmation run. Tier 3 Batch B produced one: **E9 `enc_noise_std_train=0.5`**, same-sign
paired improvement at all three seeds on L1 and on s-IoU alike (§3.6).

**σ_test = 1.0** (the released default). Both the E9 and the baseline σ_test ladders (six
points each, `scripts/sigma_test_sweep.sh`) landed in Rule 1's first case: neither improves
on σ_test=1.0 by more than the 0.0011 decode-noise band (E9's own best point, σ=1.5, gains
only +0.0003; baseline's own best point *is* σ=1.0). The knob was untuned and checking it
changed nothing — recorded as a measured null.

Screening-budget numbers (`n_samples 3`) live in §3.2, §3.5 and §3.6 and in `RESULTS.csv`,
and are **not comparable to anything in this table** — the §3.2 `n_samples` ladder showed a
systematic best-of-N drop from 0.1722 at n=3 to 0.1678 at n=20, larger than any candidate
delta in the project. Never mix the two.

| Row | n_samples | Error (L1) ↓ | SSIM ↑ | s-IoU ↑ | Render % | Wilcoxon p (L1, HL shift) |
|---|---|---|---|---|---|---|
| Paper, reported (CN) | 50 | 0.080 | — | — | — | — |
| Paper, DeepSVG (CN) | — | 0.167 | — | — | — | — |
| Quantization oracle, pipeline floor (n = ∞) | — | 0.1422 | 0.4916 | 0.3713 | 34/34 | — |
| Quantization oracle, 128-bin grid (released) | — | 0.1443 | 0.4884 | 0.3664 | 34/34 | — |
| **Baseline, seed 1111** | 50 | 0.1662 | 0.4375 | 0.2545 | 34/34 | — |
| **Baseline, seed 2222** | 50 | 0.1569 | 0.4487 | 0.2716 | 34/34 | — |
| **Baseline, seed 3333** | 50 | 0.1632 | 0.4413 | 0.2781 | 34/34 | — |
| Baseline mean of three seeds | 50 | 0.1621 | 0.4425 | 0.2681 | 34/34 | — |
| **E9 σ_train = 0.5**, seed 1111 | 50 | 0.1588 | 0.4474 | 0.2870 | 34/34 | 0.0012 (−0.0074) |
| **E9 σ_train = 0.5**, seed 2222 | 50 | 0.1531 | 0.4576 | 0.3170 | 34/34 | 0.0437 (−0.0039) |
| **E9 σ_train = 0.5**, seed 3333 | 50 | 0.1623 | 0.4388 | 0.2816 | 34/34 | 0.3050 (−0.0011) |
| E9 mean of three seeds | 50 | 0.1581 | 0.4479 | 0.2952 | 34/34 | all same sign |
| *Reference: Stage 1 reconstruction, epoch 135* | 50 | 0.1641 | — | 0.2467 | 33/34 | — |

**Delta vs baseline mean: L1 −0.0040, s-IoU +0.0271.** Smaller than the screening-budget
delta (−0.0059 L1) but the same direction, at four times the sample count per glyph.

**SSIM, added 2026-08-05: +0.0054 (0.4425 → 0.4479), same direction as the other two
metrics.** Rescored from the existing confirmation-eval SVGs with no re-decoding — L1 and
s-IoU came back bit-identical to the values above, which is the check that the rescore is
trustworthy. Smallest of the three deltas in relative terms, consistent with SSIM's
anti-aliased Gaussian window smoothing over exactly the sub-pixel placement noise §1.2 says
the other two metrics cannot see, so a candidate that helps structure more than it helps
raw pixel placement would be expected to show up more on s-IoU than on SSIM.

**s-IoU paired Wilcoxon (same three seeds, not pooled, `--metric iou`):** seed 1111
p=0.0001 (HL +0.0317), seed 2222 p=0.0000 (HL +0.0453), seed 3333 p=0.4417 (HL +0.0045) —
all three same sign, two strongly significant. Per §3.6's r=−0.335 finding, L1 and s-IoU are
close to independent instruments here, so a candidate moving both favourably at every seed
(six of six paired sign tests, two metrics) is doing more than moving one number. Seed 3333
is the weak leg on both metrics — consistent with it being the smallest-margin seed at
screening budget too (§3.6).

Rules for filling it, so the table cannot mislead the way the screening tables did:

- **Quote every candidate delta against the baseline *mean*, not against seed 1111.** §3.2's
  2026-08-05 measurement is that the seed-1111 anchor is 0.0046 below the mean and that
  anchoring on it makes 21 of 26 arbitrary changes look like improvements. Keep the per-seed
  rows visible so the earlier tables stay traceable, but the headline delta is against the mean.
- **One Wilcoxon per seed, never pooled.** Per-font differences across seeds share the same
  fonts and data and are not independent; `scripts/paired_wilcoxon.py` refuses to pool them
  and reports the sign pattern of the three Hodges–Lehmann shifts instead.
- **Report the Hodges–Lehmann shift beside every p-value.** Every effect here lives inside a
  0.0093 floor, so a p-value without a size is not interpretable.
- **State σ_test.** Every number in the project so far was taken at the released σ_test = 1.0
  without that being checked; whichever value this table is measured at goes in the caption.
  See §8 item 9 and Rule 1 of `docs/confirmation-launch.md`.
- **Both floors, not one.** L1's is 0.0093 and s-IoU's is 0.0401. Judging a candidate on one
  metric against the other's floor is the specific mistake the E8 follow-up walked into (§3.5).

The three baseline seed rows are what license every claim below them. Put them in the table, not in a footnote.

### 5.1 Official-checkpoint rows (Stage 1, not Stage 2)

**Added 2026-08-06** (`docs/official-checkpoints-and-600.md`, Job A). These are the authors'
released weights scored through this harness, so they belong to §2.4's reproduction argument
and **must never be mixed into the Stage 2 comparison above**: different budget, different
language in half the rows, and in English's case a different font sample.

Every row is `n_samples 50`. **The convention column is not optional** — per §1.2 a Chinese
Error without it is uninterpretable.

| Row | Lang | Ckpt | GT conv. | Fonts | Error (L1) ↓ | SSIM ↑ | s-IoU ↑ |
|---|---|---|---|---|---|---|---|
| Paper, reported | chn | — | unstated | unstated | 0.080 | — | — |
| `official_chn` | chn | 500 | raster | 34 | 0.1629 | 0.4394 | 0.3132 |
| `official_chn` | chn | 550 | raster | 34 | 0.1642 | 0.4352 | 0.3185 |
| `official_chn` | chn | 600 | raster | 34 | 0.1629 | 0.4373 | 0.3225 |
| `official_chn` | chn | 500 | **svg** | 34 | 0.1198 | 0.5414 | 0.4459 |
| `official_chn` | chn | 550 | **svg** | 34 | 0.1186 | 0.5420 | 0.4567 |
| `official_chn` | chn | 600 | **svg** | 34 | 0.1174 | 0.5450 | 0.4627 |
| Pipeline floor (`l1_inf`) | chn | — | raster | 34 | 0.1422 | — | — |
| Floor + 128-bin grid | chn | — | raster | 34 | 0.1443 | — | — |
| **Ours, 3-seed mean, 150 ep** | chn | 150 | raster | 34 | **0.1621** | 0.4425 | 0.2681 |
| Paper, reported | eng | — | unstated | unstated | 0.052 | — | — |
| `official_eng` ‡ | eng | 500 | raster | 34 sub | 0.0645 | 0.7224 | 0.7103 |
| `official_eng` ‡ | eng | 550 | raster | 34 sub | 0.0649 | 0.7191 | 0.7043 |
| `official_eng` ‡ | eng | 600 | raster | 34 sub | 0.0658 | 0.7181 | 0.7029 |
| `official_eng` ‡ | eng | 500 | **svg** | 34 sub | 0.0569 | 0.7427 | 0.7399 |
| `official_eng` ‡ | eng | 550 | **svg** | 34 sub | 0.0573 | 0.7399 | 0.7334 |
| `official_eng` ‡ | eng | 600 | **svg** | 34 sub | 0.0584 | 0.7388 | 0.7324 |
| `official_eng` ‡ | eng | 500 | raster | 862 part | 0.0719 | 0.7045 | 0.6463 |
| `official_eng` ‡ | eng | 500 | **svg** | 862 part | 0.0675 | 0.7201 | 0.6675 |
| Pipeline floor (`l1_inf`) ‡ | eng | — | raster | 34 sub | 0.0253 | — | — |
| Floor + 128-bin grid ‡ | eng | — | raster | 34 sub | 0.0266 | — | — |

‡ **Not a full-set number.** English's test split is 1,386 fonts; these are a deterministic
34-font prefix (`--max_fonts`, unshuffled split) or an 862-font partial decode. The subset
reads **optimistic** against the 862-font sample by 0.0074 raster / 0.0106 svg at checkpoint
500, so every ‡ row is an estimate with a known bias and a known direction. §2.4 states how
to quote them.

Three readings, all in §2.4: our 150-epoch baseline sits 0.0008 from the released 600-epoch
checkpoint (inside a 0.0093 floor, so the reproduction is faithful); 500/550/600 are flat
within 0.0013 in both languages, so the budget is saturated well before 600; and the
raster-to-svg gap is 0.0455 on Chinese against 0.0074 on English.

**Scored 2026-08-07/08 (Job A, `docs/english-candidate.md` §6).** `seedfloor600_<seed>_chn`,
trained 2026-08-06, three seeds, both GT conventions, `n_samples 50`, best-val-metric
checkpoint per seed (200/150/200 — pruning kept those, not epoch 600 itself, same
`best_checkpoint.py` selection every other candidate in this project uses):

| Row | Seed | Ckpt | GT conv. | Fonts | Error (L1) ↓ | s-IoU ↑ | SSIM ↑ |
|---|---|---|---|---|---|---|---|
| `seedfloor600_1111_chn` | 1111 | 200 | raster | 33/34 | 0.1635 | 0.3002 | 0.4377 |
| `seedfloor600_1111_chn` | 1111 | 200 | svg | 33/34 | 0.1236 | 0.4176 | 0.5334 |
| `seedfloor600_2222_chn` | 2222 | 150 | raster | 34/34 | 0.1614 | 0.2606 | 0.4444 |
| `seedfloor600_2222_chn` | 2222 | 150 | svg | 34/34 | 0.1324 | 0.3464 | 0.5189 |
| `seedfloor600_3333_chn` | 3333 | 200 | raster | 34/34 | 0.1500 | 0.3394 | 0.4574 |
| `seedfloor600_3333_chn` | 3333 | 200 | svg | 34/34 | 0.1237 | 0.4176 | 0.5250 |
| **3-seed mean** | — | — | raster | — | **0.1583** | 0.3001 | 0.4465 |
| **3-seed mean** | — | — | svg | — | **0.1266** | 0.3939 | 0.5258 |

**Confirms, does not decide.** Raster mean 0.1583 sits 0.0046 *below* the official 600-epoch
checkpoint's 0.1629 — the official row's own bound — and well inside the 0.0093 seed-noise
floor measured for the 150-epoch runs, so this is not a materially better result and §2.4's
gate does not fire. Training four times longer than the 150-epoch baseline still lands within
noise of the released weights, exactly as predicted. The one-font renderability miss for seed
1111 (33/34) is the same font that failed for the 135-epoch reconstruction row above; not a
new failure mode.

No "combined winners" row is planned. §3.7 says to combine the candidates that cleared the
floor, and exactly one candidate has a case at all — a combination of one is just E9. If the
report wants a combination it needs a stated rationale beyond stacking; the only mechanistically
motivated pairing available is E9 σ=0.5 with E12 `kl_beta=0`, since §3.6 argues the
reparameterization is close to decorative once additive encoder noise is present.

### 5.2 English generalization (E9), run and closed 2026-08-08

`docs/english-candidate.md`, Jobs B and C. Three E9 `enc_noise_std_train=0.5` seeds, each
paired to its same-seed English baseline at the matched epoch (§2.2 of that doc: 640/580/640,
the epoch each baseline was actually scored at, not the full 800-epoch budget). `n_samples 50`,
34-font subset, raster convention throughout — the only convention the English baselines were
ever scored at.

| Row | Seed | Ckpt | Error (L1) ↓ | s-IoU ↑ | SSIM ↑ | Fonts |
|---|---|---|---|---|---|---|
| `eng_seedfloor_1111` (baseline) | 1111 | 640 | 0.0583 | 0.7371 | 0.7427 | 33/34 |
| `e9_sigma050_1111_eng` | 1111 | 640 | 0.0602 | 0.7308 | 0.7360 | 33/34 |
| `eng_seedfloor_2222` (baseline) | 2222 | 580 | 0.0621 | 0.7242 | 0.7287 | 33/34 |
| `e9_sigma050_2222_eng` | 2222 | 580 | 0.0594 | 0.7344 | 0.7362 | 33/34 |
| `eng_seedfloor_3333` (baseline) | 3333 | 640 | 0.0587 | 0.7314 | 0.7408 | 34/34 |
| `e9_sigma050_3333_eng` | 3333 | 640 | 0.0608 | 0.7263 | 0.7353 | 33/34 |

Per-seed delta (candidate − baseline, negative is an improvement on L1):

| Seed | ΔL1 | Δs-IoU | L1 Wilcoxon p (HL shift) | s-IoU Wilcoxon p (HL shift) |
|---|---|---|---|---|
| 1111 | +0.0019 | −0.0063 | 0.0267 (+0.00196) | 0.0151 (−0.00762) |
| 2222 | −0.0027 | +0.0102 | 0.0124 (−0.00171) | 0.0930 (+0.00451) |
| 3333 | +0.0021 | −0.0051 | 0.6811 (+0.00048) | 0.5317 (−0.00220) |

**Mixed sign on both metrics — outcome 3 of the pre-committed reading rule in
`docs/english-candidate.md` §4.** One seed (2222) favours E9 on both L1 and s-IoU; the other
two disfavour it on both. All six per-seed deltas sit inside or barely outside the English
floor (L1 0.0038, s-IoU 0.0129). Per the rule, fixed before any run launched: **E9 does not
replicate on English, and this is reported as a result, not a failed run.** It sharpens
§3.2/§8 item 10's own finding that a single-seed leader on Chinese survived replication only
1 time in 3 — here, replicating across languages instead of across seeds, the same candidate
again lands inconclusive rather than confirmed. No re-tuning, no alternative checkpoint, no
fourth seed was applied, per the rule.

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
4. ~~**English.**~~ **Closed 2026-08-08.** Sized 2026-08-04, measured 2026-08-07, cut-off overruled the same day, run 2026-08-07/08. The rule in `docs/english-arm.md` Step 1 said `> 6 h` per run means drop English. Measured: `E_conv` 580, frozen budget 630, ~100 s/epoch, **~17.5 GPU-hours per run**. The rule fired, and Step 1's timing and Step 2's convergence measurement got there independently. **Overruled deliberately by Ben on 2026-08-07** with three GPUs free and six days of schedule slack that did not exist when the rule was written; three runs across three GPUs is one overnight rather than three days. Recorded as an overrule in `docs/english-arm.md` Step 3 and disclosed in `REPORT.md`'s methods, because a pre-committed rule dropped quietly the one time it is inconvenient is worth less than no rule.

   **What ran: E9 `enc_noise_std_train=0.5` at three seeds, paired to the three English baselines at the matched epoch (640/580/640), not the full 800-epoch budget.** That pairing choice was itself re-litigated before scoring — the obvious alternative was resuming all three past their matched epoch to the full 800 and picking each one's own best-val-metric checkpoint, matching how the baselines were nominally trained. Kept as designed: extending after two of three seeds were already showing a mixed-sign result would have been exactly the kind of post-hoc re-tuning §4 of `docs/english-candidate.md` forbids, and the matched-epoch pairing is the stronger comparison on its own terms (same seed, same epoch, no pruning-survivor confound) — see that doc's §2.2.

   **Result: mixed sign, outcome 3 of the pre-committed reading rule. E9 does not replicate on English.** One seed (2222) favours it on both L1 and s-IoU, two (1111, 3333) disfavour it on both, all six deltas at or inside the noise floor. Full table and Wilcoxon output in §5.2. This is reported as a finding, not a failed run — see §6.

   Two sub-questions this closed along the way. The English `--n_samples` discrepancy is settled at **50**, matching the baselines and the official-checkpoint eval rather than §4's 10 or `COMMANDS.md`'s 20. And the epoch budget was frozen from the baseline curves before any candidate was looked at, as required.

   *Original item, kept for provenance.* **Back in scope 2026-08-04, sized rather than committed.** A second language column and a stronger reconstruction section, against roughly a day. The cost of one English run has never been measured and the two relevant factors pull opposite ways — shorter sequences and half the reference shots make an epoch cheaper, while the budget is 4× longer — so this follows §3.3: time it, then pick the matrix from the measured cost. Plan and cut-off rule in `docs/english-arm.md`. Two things it resolves that are open right now: the `--n_samples` discrepancy for English confirmation (§4 says 10, `COMMANDS.md` uses 20, and the ladder in §3.2 showed that gap is larger than most candidate deltas), and the fact that the English epoch budget must be **frozen from the baseline curve before any candidate is looked at**, since `warmup_cosine` derives its schedule from `--n_epochs` and cannot be compared across budgets at all.

   Scope it as a generalization test, not a second sweep: carry across whatever Batch B promotes plus the §3.7 combination, two or three candidates, not eighteen. Note also that the paper's entire published English ablation spans 0.0069 absolute with individual steps of 0.003 or less, so if the English noise floor lands anywhere near the Chinese 0.0093 it will be wider than the whole effect range being chased.
5. ~~**Screening budget.**~~ **Resolved 2026-08-03: 150 epochs everywhere.** Superseded in part by item 3 — the open question is no longer epochs but `n_samples`, and §3.2's decomposition answers it.
6. ~~**Fix the §1.5 eval-script issue.**~~ **Resolved 2026-08-03**, layout is per-checkpoint and the script handles both.
7. **What `--max_ckpt_keep` means for the sweep.** Keeping one checkpoint per run is right for disk, but if you later want to score a candidate at both 60 and 150 epochs you need both. Consider 2 for the runs that feed §3.3's validation.

8. ~~**Does `val_metric` omitting the KL term stay a choice or become a fix?**~~ **Resolved 2026-08-05, in one sentence as asked: it stays a choice, and the sentence is now in §1.5** — *`val_metric` deliberately excludes the KL term because KL does not enter the rendered output, so checkpoint selection tracks reconstruction quality rather than the full training objective; the consequence, that E12's selection is blind to what E12 varies, is stated wherever an E12 number is quoted.* Evidence rather than taste: `e12_kl000_chn` (KL removed entirely) is the second-best row in Batch A and `e12_kl100_chn` is the only row worse than the anchor, so the term is doing little to the rendered metric in either direction — see §3.6. One seed, so this decides a documentation question and nothing more; it is not a claim that `kl_beta=0` is better.

   *Original item, kept for provenance.* New 2026-08-04. The training loss is `... + kl_beta * kl`; `val_metric` is the same sum without it. Defensible, since KL does not affect rendered output directly, but it means `val_metric` is not the training objective and that E12's checkpoint selection is blind to exactly what E12 varies. The `svg_para` omission in the same function was treated as a bug and fixed; this one has never been decided either way, which is how that one survived. Decide it, in one sentence, and put the sentence in §1.5.

9. ~~**What to do with the σ_test result.**~~ **Resolved 2026-08-05, before the sweep runs: re-screen the finalists only, caveat the rest.** The full rule, with all three cases, is in `docs/confirmation-launch.md` under "Rule 1" and is pre-committed there so it cannot be chosen to suit the outcome. In summary: a shift counts only if it beats σ=1.0 by more than the 0.0011 decode noise; if **both** ladders (E9's and the baseline's) shift to the same σ\*, that is a property of the eval procedure and the six finalist rows are re-measured at σ\* while Tiers 1–3 stay at 1.0 with a stated caveat; if only **one** ladder shifts, that is a σ_train × σ_test interaction rather than a free win, and the confirmation eval runs at σ_test = 1.0 regardless, because tuning the eval on the candidate's own ladder would confirm E9 on a footing it was never screened on. Running the sweep on the baseline as well as on E9 exists precisely to tell those two cases apart.

   **Run 2026-08-05.** Both six-point ladders landed in the rule's first case: neither improves on σ=1.0 by more than the 0.0011 decode-noise band (E9's best point, σ=1.5, is +0.0003 over σ=1.0; the baseline's own best point *is* σ=1.0). No re-screen triggered, no caveat needed on any existing table. σ_test = 1.0 for the confirmation eval in §5.

   *Original item, kept for provenance.* New 2026-08-04. `scripts/sigma_test_sweep.sh` is eval-only and answers a question §3.4 raised and §4 scheduled. If the optimum σ_test differs from the released 1.0 by more than the 0.0011 decode noise, every number in the results table was taken at an arbitrary point on that curve and the screening comparisons need re-running at the better value. That is cheap for the eval but it invalidates the table as a *set*, so decide before running whether a shifted optimum triggers a re-screen or gets reported as a caveat.

10. **How much of the remaining budget goes to more candidates.** New 2026-08-05, and the one live scope question. §3.2's two 2026-08-05 measurements say the candidate-induced spread is about half the seed-induced spread, and that the leading single-seed candidate survived replication 1 time in 3. Both point the same way: another breadth batch buys single points in a setup that has now been shown twice over not to resolve them. Scope confirmed 2026-08-05 as **Stage 1 closeout and the report, plus the English arm and E14-deep** — E14-deep because its two peak-lr rows change *which* optimum is reached rather than how tightly the weights settle into it, and English because a second script is a generalization claim rather than a twentieth single point. `e2_batchnorm_chn` and `e12_kl000_chn` are explicitly **not** promoted to a replication batch: they are the same kind of single-seed leader Batch B just showed to be 2/3 misleading.

11. **`--freq_ckpt`: 5 (chn) / 20 (eng) in `COMMANDS.md`, versus the codebase's own `--freq_ckpt 50` default.** New 2026-08-06, not decided here — revisit on Ben's local machine. The working commands checkpoint 10x (chn) and 2.5x (eng) more often than upstream's own default. This is not free: `train.py`'s checkpoint boundary (line ~322) runs its own full `compute_val_loss` pass over `val_loader` *and* writes a ~1.3 GB checkpoint to disk every time, on top of the separate step-based `--freq_val` validation — so a 10x higher save frequency is a real, uncosted wall-clock tax on every training run, not just a denser log. The upside is real too: a denser `checkpoint_metrics.csv` is what the English arm's `E_conv` convergence rule (`docs/english-arm.md` step 2) and best-checkpoint selection lean on, so reverting toward the paper's cadence trades away exploratory resolution to save time. The actual added wall-clock from the extra val passes + disk writes hasn't been isolated/measured yet — that measurement, not a guess, should decide this.

12. **The day-6 GPU push, and the two things it reopens.** New 2026-08-08, decided the same day. All GPUs became available (Ben runs on 3, 2 and 1, leaving 0 free), which changes the price of several items this document had priced under scarcity. Runbook: `docs/day6-gpu-push.md`. Two of the five jobs need justifying against decisions already made, and both are recorded here rather than slipped in.

    **Reopened: checkpoint selection is not neutral, and it may have manufactured a result.** Selection is by `val_metric`; scoring is by rendered L1 and s-IoU. No term in `val_metric` is the quantity being reported — `img_l1` is the image decoder branch, while what gets scored is the refinement decoder's SVG rasterized, and `svg_para_total` is cross-entropy over command types and bins rather than pixels. §1.5 already flags that `val_metric` was blind to the refinement decoder until 2026-08-03; this is the deeper version of the same problem, and `val_metric_correlation.py`'s claim that within-run selection "was never affected" is an argument about *scale* that says nothing about *ordering*, which is what `prune_checkpoints` acts on. It is untested. The concrete damage: every Chinese baseline was scored at epoch 150, while `e1_norm_2222_chn` was selected at 125 and `e1_norm_3333_chn` at 100 — **E1's two degrading legs are the two scored earliest**, and E1 is the only effect in this project that exceeds its own floor (§3.6, `REPORT.md` §6 finding 3). The effect is confounded with checkpoint epoch and the baseline's own s-IoU curve (0.2550 at 125, 0.2467 at 135, 0.2545 at 150) is not monotonic enough to correct it arithmetically. Job A retrains both E1 legs plus a matched baseline seed with `--max_ckpt_keep 10`, scores at 100/125/150, and reads E1 at matched 150 only, under a rule fixed before launch. ~4 GPU-h, and it is the highest value per hour anywhere in this document because it does not add a finding, it checks whether one is real. Job B is the cheap companion: `scripts/val_on_checkpoint.py` computes `val_metric` for the three released English checkpoints, whose rendered scores get monotonically *worse* with training (0.0645 / 0.0649 / 0.0658 raster), and asks whether `val_metric` agrees. ~15 GPU-min. **Scope guard: whatever both return, the project does not switch to rendered-L1 checkpoint selection.** That means decoding the validation split at every checkpoint and it invalidates all 37 rows of `RESULTS.csv` with seven days left. It is reported as a measured protocol defect and named as future work.

    **Reopened, and the largest single job here: one replicated candidate per assignment category (Job C), plus a one-seed English screen of the same six (Job C-EN).** The instructor's Part 2 brief lists seven example categories and asks for a table comparing paper, reconstruction and improved model. Tiers 1–3 cover six of the seven, but every row in that coverage is a single seed, and this document's own §3.2 says what a single seed is worth here. So: one representative per category — E2 `img_norm batch`, E4 `ngf 32`, E5 `bottleneck_bits 256`, E7 `loss_w_aux 0.1`, E11 AdamW, E3 `n_layers_refine 2`, with E9 and Tier 4 already replicated — at three seeds each on Chinese, ~18 GPU-h, all trained with `--max_ckpt_keep 10` and scored at matched epoch 150 so the epoch confound Job A found never enters the table. This is item 10 reopened a second time, and the purpose is stated so it cannot be mistaken: these are **category representatives for a required table**, not single-seed leaders believed to win. Expect null and write §4 that way in advance. **Job C-EN runs the same six at one seed on English (~105 GPU-h, ~35 h wall clock), at Ben's insistence 2026-08-08 and on a legitimate ground: the instructor requires coverage of both datasets as the original paper reports them.** It is screening, pre-committed as such — a leader earns a three-seed English confirmation only if the schedule allows, and otherwise is reported as an unconfirmed screening number with the 1-in-3 replication base rate quoted beside it. E7 is preferred over E12 as the loss representative despite E12's better number, because §8 item 8 records that `val_metric` excludes the KL term and E12's selection is therefore blind to exactly what E12 varies; E7 is a paper-versus-code discrepancy (§1.4) and makes the better report row.

    **Demoted the same evening: three more English baseline seeds.** Staged first, then challenged by Ben and dropped to optional (`docs/day6-gpu-push.md` §7). It cannot change a conclusion — §5.2 is mixed sign, and mixed sign is floor-independent. What it props up is two §6 claims that move in *opposite* directions: "the paper's own English ablation steps sit below our floor" strengthens if the floor widens, and "our English baseline beats the released checkpoints by 1.3× to 1.6× the floor" weakens. Range grows with sample size, so six seeds most likely widen the spread and weaken the second. It was also the only job blocking all three GPUs overnight, and the Chinese floor is likewise a three-seed estimate the project has quoted for six days. Cheaper substitute if §6 turns out to need it: `scripts/eval_noise.sh` against English, which measures the decode component with no training at all, the way §3.2 decomposed the Chinese floor.

    **Reopened: one three-seed capacity batch (Tier 4), against item 10.** Item 10 declined **single-seed breadth**, on the measured grounds that candidate spread is about half seed spread and that the leading single-seed candidate replicated 1 time in 3. That reasoning stands and is not overturned; a three-seed batch is simply not the thing it declined. The substantive reason to run it: of the assignment's seven example categories, Tiers 1–3 cover six cleanly, and *"add residual or attention layers"* is covered only by E3's refinement depth. The sequence transformer's own width and depth, which is the model's actual capacity, was never varied — E4 widened the image stacks (null, 0.1691) and E5 the latent (null, 0.1687/0.1699), neither touching the transformer. Two flags added 2026-08-08, both defaulting to the released value, both asserted by `check_infra.py` section 9: **E16 `--enc_depth`** 6→8 (12→16 self-attention blocks, `models/model_main.py`) and **E17 `--dec_d_ff`** 1024→2048 (`models/transformers.py:209`, a 2× expansion on d_model 512 where the literature default is 4×; the `ff` module is deep-copied into both decoder stacks, so one flag widens the AR decoder and the refinement decoder together). Three seeds each, ~6 GPU-h total on Chinese. **Three seeds is not optional here**: both flags change parameter count and therefore shift the global RNG stream for every module constructed after them, which is the exact objection that kept E6 off the candidate list, and averaging over three seeds is what converts an init shift into a seed draw and makes a capacity change readable at all. **Expect null, and pre-committed as such** — this is in the report for category coverage, not because a win is expected.

    **Not reopened, and listed so it stays closed:** no re-run of E9 on English at a longer budget, a different checkpoint, or a fourth seed (declined 2026-08-08 in §9 item 5, *before* the GPUs freed up — hardware availability is not new evidence); no new single-seed breadth batch; no chasing the published 0.080. Job C (three more English baseline seeds, 4444/5555/6666, ~17.5 GPU-h each) re-measures the English floor and explicitly **does not** reopen §5.2: E9's English deltas are paired same-seed and same-epoch, so more baselines change the yardstick and not the six deltas, and mixed sign is mixed sign at any floor.

13. **Whether Chinese is being rendered/scored correctly at all, separate from the known rasterizer and augmentation gaps.** New 2026-08-12, from Ben looking at the real decoded glyphs in `fig8_best`/`fig9_failures` (§6 report figures, built for `docs/pull-figure-assets.md`): the Chinese baseline panels in the failure strip render as solid filled blocks rather than the ground truth's open rectangular outlines, which reads as a bigger and more structural disagreement than a few percent of pixels. **Explicitly scoped as its own session, not a report task and not a reopening of §2.4.** §2.4 and this item ask different questions — §2.4 is "why is our number 0.162 instead of 0.080", closed by measurement; this is "is the number we compute even measuring what we think it measures on Chinese", open. Three things already on record bear on it and should be the starting point rather than re-derived from scratch:

    - **The two-rasterizer gap (§1.2) is 0.0455 on Chinese against 0.0074 on English**, six times larger, and it is a property of the rasterization pipeline, not the model.
    - **The quantization-oracle floor (§2.4, "the gap actually lives", and `REPORT.md` §6.5) is 0.1422 at unlimited precision** — encoding the ground truth into the model's own representation and decoding it with *no model involved* still scores 0.1422 on Chinese under the raster convention. That is 88% of the 0.1621 baseline. A perfect Chinese model, scored this way, would still show as roughly seven-eighths wrong.
    - **`render_model_output.py`'s own docstring records a 42%-of-pixels disagreement** between a from-scratch SVG-fill renderer and `render_svg_mask` (cairosvg) on Chinese glyphs with several overlapping subpaths, which is exactly the kind of glyph the failure strip shows. That comparison was between two *renderers* of the same outline, not a claim about the model, but it is direct evidence that Chinese glyph geometry is unusually sensitive to fill-rule and self-intersection handling, and it has not been checked against what the paper's own rendering convention does with the same outlines.

    None of this says the 0.162 number is wrong — the released-checkpoint check in §2.4 says our harness reproduces the authors' own pipeline almost exactly, which argues the *pipeline* is at least self-consistent. What it does not rule out is that the pipeline, self-consistent or not, might be scoring something other than "does this glyph look like a person drew it": a model that always fills dense strokes solid could be genuinely worse, or could be losing an unrecoverable amount to a fill-rule mismatch inherited from the dataset's SVGs, and the current evidence cannot tell those apart. A dedicated session should start from the three points above rather than re-opening the model-quality question, and a reasonable first move is comparing a handful of Chinese ground-truth SVGs' fill-rule/winding behavior against what the paper's figures imply, before touching any model output.

14. **An external review, and the batch staged in answer to it.** New 2026-08-12. `docs/review-gemini.md` is the review verbatim; `docs/review-response.md` is the runbook. Overall verdict "not ready to submit", with three critical issues: E9 is promoted as the improved model on a delta inside its own floor, checkpoint selection runs on a criterion the project itself measured at ρ = 0.125 against the reported metric, and the reproduction is incomplete while the Chinese training set sits at 6× augmentation against the paper's 10×.

    **Decided 2026-08-12, all three taken.** The deadline moved to 1 September and three GPUs are free, so the answer is a measurement rather than a paragraph of limitations.

    - **Rendered-metric checkpoint selection**, which required first noticing that `train.py:129` builds its validation loader on the **test split** — so `val_metric` has always been computed on the scored fonts, and doing the same with a rendered metric would be oracle selection rather than a fix. A held-out val split is carved out of train instead (`scripts/make_val_split.py`, 20 Chinese base fonts, by base id so no augmented twin straddles the split), and `render_val.py` scores it at every checkpoint with the same decoder, rasterizer and binarization `eval_reconstruction_error.py` uses. `--ckpt_select val_render_l1` makes pruning and `scripts/best_checkpoint.py` rank on it. **The test-split validation is a finding in its own right** and belongs in §6 next to the two-rasterizer problem and Job B: it is the third instance of the same class, an evaluation apparatus that does not measure what it is taken to measure.
    - **The 10× Chinese rebuild, as the new baseline** rather than as a second arm. It is the review's "incomplete reproduction" point and the live lead on the ~0.037 residual (§8's 2026-08-08 augmentation check). Making it the baseline means **nothing in the new batch is comparable to any row in `RESULTS.csv`**, which is accepted deliberately: every comparison is internal to the batch, against its own three baseline seeds at a floor re-measured from them.
    - **Found while implementing it:** the released `aug_rules` has five branches ending in a bare `else`, so every index ≥ 4 returns `rotate(-5)`. `--n_aug 9` against the original code would have written five identical copies of one transform and reported itself as 10× augmentation. Nine distinct rules now, indices 0–4 byte-identical to the released ones so 6× stays a strict subset of 10×, and an unknown index raises. A silent-duplication bug in the upstream data pipeline, and a §6 line.

    **Scope, fixed before the numbers.** Chinese: three baseline seeds plus E9, E1 and the six Job C category representatives, three seeds each, 27 runs. English: three baseline seeds plus E9 and `e3_refine2`, three seeds each, 9 runs, at `--n_samples 10` — which closes the review's English-protocol point in the same motion. Reading rules in `docs/review-response.md` §4, including the three pre-written outcomes for E9 and the gate that stops the batch if the rebuilt dataset turns out to have a wider seed floor than the old one. **Not in scope:** a fourth seed, a longer budget, a new candidate, or re-scoring the old 6× rows under the new rule. Hardware availability is not evidence, and §4.6 of the runbook says so before the launch rather than after.

    **The cheap half.** `scripts/selection_disagreement.py` answers the review's question 2 — whether the 26-candidate spread is an artifact of mismatched checkpoint sampling — from the manifests alone, no GPU, once the batch has trained with every checkpoint kept. Its reading rule is printed by the script and fixed in the runbook §8.

    **Closed 2026-08-14. The §4.5 gate fired; English never launched.** All 27 Chinese runs trained and scored on the cluster 2026-08-12/13 (`test_experiments_review_batch.out`), but the batch's own floor computation crashed (`scripts/test_experiments.sh:321` indexed `L1S[seedfloor_1111_chn]`, missing the `rv_` prefix every other reference in the same script uses — fixed, one-line). Recomputed independently from the per-font eval CSVs: **re-measured Chinese L1 floor 0.0151** (seed L1s 0.1534 / 0.1560 / 0.1685), against the old 0.0093 — **63% larger**, past §4.5's "materially larger" line. Checked anyway, not just gated on: none of the eight candidates (E9, E1, and the six Job C representatives) clears 0.0151 on L1 at the required same-sign three-seed bar; E2/E3/E4 agree in sign but stay inside the floor, E9 and E1 don't even agree in sign across seeds on L1. **Ben's call, 2026-08-14: follow §4.5 as written rather than overrule it.** Per §4.5: the floor measurement is the finding, the pre-review 6× tables in `RESULTS.csv`/`REPORT.md` stand as the substantive results with their selection caveat, and the English arm does not run — no `rv_*_eng` runs exist. §9's other exclusions (fourth seed, longer budget, new candidate, re-scoring the old rows) were never in question since the gate closed the batch before any candidate was read.

    **s-IoU disclosed, not promoted.** The s-IoU floor came back **narrower** than before, 0.0223 against the old 0.0315 — the opposite direction from L1, so §4.5's "blunter instrument" reading does not extend to this metric on its own terms. Read that way, two candidates clear it with same-sign agreement at all three seeds: **E1 (terminal LayerNorm), mean Δs-IoU −0.0310, degrading** — the same candidate, same direction, as the pre-review report's one confirmed degrading result (−0.0376 there) — and **E2 (batch norm), mean Δs-IoU +0.0357, improving**, null in the pre-review sweep. Both are null on L1. **Ben's call, 2026-08-14: disclose, do not promote.** §4.5 names the old L1 figure specifically and its instruction is to decide on the baseline seeds *before* reading a single candidate; crediting a metric that happens to still clear after the primary one has gated the batch is the same move the review's §2 objected to in E9's sign-agreement pivot. Recorded here for anyone re-reading this batch later, not carried into `RESULTS.csv` or `REPORT.md` as a result.

    **The cheap half ran anyway and confirms the review's question 2 independently.** `scripts/selection_disagreement.py --floor 0.0151`: 21/27 runs disagree between `val_metric` and rendered-metric checkpoint selection, mean cost **+0.0186** rendered L1, above the re-measured floor on 20/27 runs (`selection_audit.csv`). By the script's own pre-committed rule, that means checkpoint selection *was* what decided the pre-review 26-candidate sweep — the review's harshest reading of question 2 holds on this project's own data, independent of the floor-gate outcome above.

    All 27 runs recorded in `RESULTS.csv` as `batch=rv-review` (`scripts/build_results_table.py`), tagged explicitly non-comparable to every earlier batch — different training set, different training-set size, different selection rule, and a null reading throughout. `data_splits/chn_val_split.json` and `selection_audit.csv` committed.

---

## 9. Do this first

**Current, as of 2026-08-15 (day 13).** The review-response batch closed 2026-08-14 (§0, §8 item 14): the L1 floor gate fired at 0.0151, no candidate cleared it, and Ben's call was to follow §4.5 as written rather than overrule it. `REPORT.md` §5.6 answers the review point by point, but `REPORT.pdf` had gone stale against it — last built 2026-08-12, before the closing commit landed. Rebuilt this session. One stale number caught and fixed along the way: the References section still quoted 113 scored checkpoints, the pre-review-batch count; `RESULTS.csv` is 140 rows since the 27 `rv-review` runs landed, and the report now says so. `verify_report.py` passes clean at the new count, 0 failed claims.

1. **Nothing left needs the cluster.** The experimental programme is closed twice over — Stage 2's original sweep on day 9, and the review's re-read on day 12–14. Everything remaining is Mac-side.
2. **Read the report end to end, once, as a continuous document.** §1–3 were written days 3–5, §4–5.5 on day 9, §5.6 added day 12–14. It has never been confirmed as one piece, and `_Open Tasks.md`'s Day 9 checklist has carried this item unchecked since 2026-08-11.
3. **The `main..repro` diff review.** 112 files, ~16,300 insertions, 95 deletions. A graded section in its own right (§10, "Code and data") and never read as one. Worth a pass for stray debug code, dead flags, and anything that has drifted from what §1.1 and §4.2 claim the code does.
4. **§8 item 13 (Chinese render correctness) stays its own session.** Scoped that way on 2026-08-12, explicitly not a report task, and untouched by the review batch's dataset and selection fixes — those answered "is checkpoint selection biased" and "is augmentation short", not "does the rasterized comparison mean what we think it means on Chinese".
5. **Optional, unchanged:** English checkpoint 500's decode from font 862; E14-deep's two peak-lr rows. Neither changes a conclusion.
6. **Push whatever this session commits**, from Ben's own terminal — the Cowork sandbox has no GitHub credentials.

*Day 10's list, kept for provenance; all seven items are done:*

1. **Push.** The Mac session of 2026-08-12 committed twice and could not push (no credentials in the sandbox). `git push origin repro` from Ben's own terminal, before the cluster pulls.
2. **Cluster preflight.** `git pull origin repro`, then `python scripts/check_infra.py` — expect **224 passed, 0 failed**, up from 196; section 10 is the new rendered-validation path. Then `df -h /data/bens`, which matters more than usual: this batch keeps every checkpoint of 27 runs.
3. **Rebuild the Chinese dataset.** `docs/review-response.md` §3, in order, with the backup first. Val split before augmentation, never after — the whole point of the split is that no held-out font's sheared twin is left in train, and §3.4's leak check is what proves it.
4. **Smoke test the rendered-validation path** before committing three GPUs to twelve hours. Nothing in `render_val.py` has run on a GPU. `docs/review-response.md` §5.1 lists the four things to check in the manifest, including the one that catches a wrong ground-truth side: an untrained Chinese model scoring *below* the 0.1422 pipeline floor means the metric is broken, not that the model is good.
5. **Read §4 of the runbook before launching, not after scoring.** The review's charge is that a pre-committed rule got bent. The three outcomes for E9 are written down there already, including the one where it comes out of the improved column.
6. **Then the Chinese batch**, then the free selection audit, then English only if the §4.5 gate has not fired.
7. **Report edits that need no GPU at all** and can happen in parallel: standard deviations in §5.1, the broken arrow glyphs in the metric headers, and §4's heading calling E9 a training-dynamics intervention rather than an architectural change. All three are the review's §4, all three are ten minutes.

*Day 6's list, kept for provenance; items 1–5 are done, 6 and 7 stand as written:*

**As of 2026-08-08 evening (day 6).** All GPUs are now available (3, 2, 1; GPU 0 stays free). Six jobs are staged in `docs/day6-gpu-push.md`, ordered by value per GPU-hour, and the scope decisions behind two of them are §8 item 12. **None of them is on the critical path** — the critical path is still §4, §5 and §6 of `REPORT.md`, all three unwritten, and that is Mac-side. The GPU column and the writing column do not touch: jobs run against Ben's sleep, not his report time. If a job overruns, kill it.

1. **Write §6.** Nine findings as of day 5, and Job A may make it ten or may remove one. None of the nine depends on anything still running — this is the critical path and it stays first.
2. **Write §4 and §5's connecting prose**, including §5.2's new table. §4 gains a Tier 4 paragraph if Job D runs.
3. **Cluster, in this order, and the order matters.** Job A first (E1 de-confound, ~4 GPU-h): it is the only job that can *change* something already written, so its answer is wanted before §5 and §6 are drafted around E1. Job B alongside it (~15 GPU-min). Then Job C overnight (three English baseline seeds, ~18 h wall clock). Then Job D (Tier 4 capacity, ~6 GPU-h). Job E last and only if wanted. Every reading rule is pre-committed in the runbook before the corresponding launch; **do not deviate after seeing a result**, which is the rule that held on day 5 and is the reason Stage 2's argument is worth anything.
4. **Before the first launch:** `git pull origin repro`, then `python scripts/check_infra.py` (expect 196 passed, 0 failed), then set `GPUS=(3 2 1)` in both `scripts/run_experiments.sh` and `scripts/test_experiments.sh` and delete the stale "GPU 3 is off limits going forward" comment from 2026-08-04.
5. **Recording checklist from `docs/english-candidate.md` §9** — repo done (`PROJECT_PLAN.md` §0/§5.2/§8, `RESULTS.csv`), vault still open: `Runs/Run Log.md` (six new training/eval rows), `Experiments/Experiment Tracker.md` (E9 English status → does-not-replicate), `_Open Tasks.md` checkboxes and `updated:` frontmatter. `docs/day6-gpu-push.md` §10 extends this checklist with the day-6 rows.
6. **Optional, and only after 1–5:** finish English checkpoint 500's decode from font 862 to convert the ‡ rows from estimate to measurement, and E14-deep's two peak-lr rows. Neither changes a conclusion. **Checkpoint 500, not 600** — 500 is the best-scoring released English checkpoint (0.0645 raster / 0.0569 svg) and 600 the worst, the partial decode already ran on 500, and the job measures subset bias rather than model quality, so consistency with the existing partial run is what matters. Asked and answered 2026-08-08; see `docs/day6-gpu-push.md` §5.
7. **Not planned:** re-running E9 English to the full 800-epoch budget with independent best-checkpoint selection. Considered and declined 2026-08-08 — see §8 item 4's note on why the matched-epoch pairing was kept as designed rather than extended after a partial result was already in. **Re-confirmed the same evening** when all GPUs became available: hardware freeing up is not new evidence, and §8 item 12 keeps this closed.

*Day 5's list, kept for provenance; every item on it is done:*

1. ~~**Cluster, in this order, and the order matters.**~~ **Done 2026-08-08.** Job A (`seedfloor600_*_chn`, both conventions) closed first, confirmed rather than overturned §2.4 — see §5.1. Job B (three E9 English seeds) trained cleanly to their matched epochs overnight. Job C scored them; mixed sign, §5.2.
2. ~~**Do not deviate from `docs/english-candidate.md` §4 after seeing a result.**~~ **Held 2026-08-08.** No re-tuning, no alternative checkpoint, no fourth seed, even after the epoch-budget question was reopened before scoring (§8 item 4's note).
3. **Write §6.** Superseded by day 6's item 1 above — the two English findings this pointed at are now joined by the E9-non-replication result.
4. **Write §4 and §5's connecting prose.** Superseded by day 6's item 2.
5. **Optional item.** Carried forward unchanged to day 6's item 4.

*Day 4's list, kept for provenance. Items 1, 2 and 3 are done; item 4 is now day 5's item 1:*

1. ~~**§2.4, §1.2, §5.1, §0.**~~ **Done 2026-08-06, Mac-side.** §2.4 rewritten outright around the official-checkpoint measurement (training budget eliminated, two terms measured, the DeepSVG coincidence retired); §1.2 given the two-rasterizer property with its own table; §5.1 added with every official row labelled by GT convention; §0 brought current.
2. ~~**`REPORT.md` §2.3 and §2.4 still argue the superseded case.**~~ **Done 2026-08-06, committed 2026-08-07 in `bb79e57`.** This item was written before the rewrite landed and was stale within hours; the rewrite was sitting uncommitted while the cluster pushed on top of it. §2.4 now argues from the official-checkpoint measurement. *Original text:* They were drafted 2026-08-05 against "the gap is dominated by training budget", which is now falsified. This is the highest-value remaining item and it needs no cluster: the new §2.4 is strictly better material, because "our reproduction sits 0.0008 from the authors' released weights" is a stronger answer to the assignment's comparison clause than any account of why we fell short. Rewrite §2.3–2.4, then check §6's discussion for anything leaning on the old conclusion.
3. **The vault is a day behind.** `_Open Tasks.md` is still on day 3 and its "Day 3 evening" block reads as pending work that has since been done and partly overtaken. `Runs/Run Log.md` needs the fourteen official-checkpoint rows and the **retraction of the `dvf_base_exp_eng` epoch-600 row** — that checkpoint was the released one, not a run of ours, so an unverifiable row should be dropped rather than annotated.
4. **Cluster, cheap and bounded: score `seedfloor600_*_chn`.** Three seeds trained 2026-08-06, never decoded. Both GT conventions, `n_samples 50`, 34 fonts, to match §5.1. Read it as confirmation: the official 600-epoch checkpoint at 0.1629 already bounds what a 600-epoch run of ours can return through this harness, and anything near 0.162 is the expected result. **If it comes back materially better than 0.1629, stop and re-read §2.4** — that would mean the released checkpoint is not what it appears to be, which changes the section again.
5. **Optional, and only if items 2 and 3 are done:** finish English checkpoint 500's decode from font 862 (`test_few_shot.py` resumes where it stopped) to convert the ‡ rows from estimate to measurement. It sharpens a number that already supports the argument; it does not change any conclusion.

*Day 3's list, kept for provenance; every item on it is done:*

1. ~~**Next cluster session: `docs/confirmation-launch.md`.**~~ **Run and closed 2026-08-05.** Both CPU items done (`val_metric_correlation.py`: ρ=0.125, doesn't predict the rendered metric, cuts E14-deep to two rows; `dead_params.py`: 17.06% dead, §1.4). σ_test ladder null on both arms, σ_test=1.0 stands (§8 item 9). E9's confirmation eval, three seeds, `n_samples 50`, plus `scripts/paired_wilcoxon.py` on L1 and s-IoU — results in §5. One landmine found and fixed along the way: `test_few_shot.py`'s resume-skip check ignores `--n_samples`, so two of the six confirmation runs initially silently rescored stale `n_samples=3` output (§3.7).
2. **Mac-side, no cluster: close Stage 1.** SSIM in `eval_reconstruction_error.py`, then rescore. The quantization oracle on the 128-bin grid — still no script, and it is now the last thing standing between Stage 1 and being written up. §2.4 in prose: why 0.1668 and not 0.080, with two of four explanations already ruled out.
3. **Start writing.** §6 sections 1 to 3 can be written today from §1, §2 and `RESULTS.csv`. Section 6's discussion now has four measured findings to carry it rather than an argument: what the anchor bias did to the rankings (§3.2), that architectural change moves the metric about half as much as the seed does (§3.2), that L1 and s-IoU are nearly independent instruments at r = −0.335 (§3.6), and that E9's confirmation-budget gain replicates the screening-budget direction on both metrics at reduced but still same-sign magnitude (§5).
4. **English arm** (`docs/english-arm.md`) and **E14-deep** are in scope per §8 item 10. English starts with a 6-epoch timing run that can share a wave with the eval steps in item 1; E14-deep waits on the correlation from item 1.

*Day 1's list, kept for provenance:*

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
