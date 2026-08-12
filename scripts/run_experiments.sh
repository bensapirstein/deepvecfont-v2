#!/usr/bin/env bash
# Centralizes launching a batch of train.py runs (the 3-seed baseline today,
# a Tier 1/2 parameter sweep later) so there is one place to edit the params
# and one switch to flip between running them one after another (the common
# case: one GPU) and running them all at once (when several GPUs are free).
#
# Every run gets exactly one GPU, never more — sharing a GPU across
# concurrent runs has caused problems before.
#
# parallel mode runs EXPERIMENTS in waves of len(GPUS): it launches up to
# that many at once, waits for the wave to finish, then launches the next
# wave. EXPERIMENTS can be longer than GPUS; it doesn't need editing between
# waves.
#
# Usage:
#   ./scripts/run_experiments.sh sequential
#   ./scripts/run_experiments.sh parallel
#
# Edit GPUS and EXPERIMENTS below before each use.

set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-sequential}"   # sequential | parallel
if [[ "$MODE" != "sequential" && "$MODE" != "parallel" ]]; then
  echo "Usage: $0 [sequential|parallel]" >&2
  exit 1
fi

# GPU ids to use, in order. Check `nvidia-smi` and set these by hand.
# - sequential: only GPUS[0] is ever used, one experiment at a time.
# - parallel: EXPERIMENTS runs in waves of len(GPUS) -- one experiment per id
#   per wave, however many waves it takes to get through the whole array.
# GPU allocation set by Ben 2026-08-08 (docs/day6-gpu-push.md): use 3, 2, 1.
# GPU 0 stays free.
GPUS=(3 2 1)

# One entry per experiment: "name_exp  <extra args appended to COMMON_ARGS>".
# This is the loop-over-params spot — add/edit lines here for a sweep.
#
# Tier 3 + multi-seed replication, 2026-08-04 (PROJECT_PLAN.md §3.6, §8 items 2-3).
# Sixteen runs in two batches, fifteen after e2_instancenorm_chn was dropped at
# rung 1 (see the E2 note below). Two GPUs, ~1 h per run, so ~7-8 h — one overnight.
#
# The batch is split because the two halves answer different questions and the
# second is the one that survives a null:
#
#   Batch A (10 runs, waves 1-5). Tier 3 breadth, all at --seed 1111, each one
#   factor off seedfloor_1111_chn. Expect these to land inside the floor, same as
#   Tier 1 and Tier 2 did; they are in the table for coverage of the assignment's
#   change categories, not because a single point is expected to resolve.
#
#   Batch B (6 runs, waves 6-8). The three largest deltas measured so far, each
#   re-run at seeds 2222 and 3333 so it has a mean against the baseline's mean
#   (0.1724 / 0.1631 / 0.1680) rather than a point against a point. §3.2's
#   decomposition put ~88% of the 0.0093 floor in training-seed variance, so this
#   is the only lever that can shrink the bar, and it is what §8 item 3 leans to.
#
# Run A then B in one go: leave both blocks uncommented and launch parallel. To
# stop after A, comment out the Batch B block.
# Renamed 2026-08-12: this was a second live `EXPERIMENTS=` assignment that the
# review batch below silently overrode (bash takes the last one). Closed job, kept
# for provenance, no longer able to be launched by accident.
EXPERIMENTS_ARCHIVE_JOBA=(
  # Job A, 2026-08-08 (docs/day6-gpu-push.md §1). Retrains of E1's two
  # degrading legs plus one matched baseline, all with every checkpoint kept
  # (--max_ckpt_keep 10 below), so E1 can be read at matched epoch 150 against
  # a baseline at 150 and the val_metric-vs-rendered ordering can be checked
  # within a run. a_ prefix keeps these separate from the originals in
  # experiments/ and RESULTS.csv -- the confound is itself a finding and the
  # old rows are its evidence.
  "a_e1_norm_2222_chn --seed 2222 --enc_final_norm True"
  "a_e1_norm_3333_chn --seed 3333 --enc_final_norm True"
  "a_seedfloor_3333_chn --seed 3333"
)

EXPERIMENTS_ARCHIVE_JOB_A="${EXPERIMENTS_ARCHIVE_JOBA[*]}"
EXPERIMENTS_ARCHIVE_JOB_C=(
  "c_e2_batchnorm_1111_chn --seed 1111 --img_norm batch"
  "c_e2_batchnorm_2222_chn --seed 2222 --img_norm batch"
  "c_e2_batchnorm_3333_chn --seed 3333 --img_norm batch"
  "c_e4_ngf32_1111_chn --seed 1111 --ngf 32"
  "c_e4_ngf32_2222_chn --seed 2222 --ngf 32"
  "c_e4_ngf32_3333_chn --seed 3333 --ngf 32"
  "c_e5_bneck256_1111_chn --seed 1111 --bottleneck_bits 256"
  "c_e5_bneck256_2222_chn --seed 2222 --bottleneck_bits 256"
  "c_e5_bneck256_3333_chn --seed 3333 --bottleneck_bits 256"
  "c_e7_aux01_1111_chn --seed 1111 --loss_w_aux 0.1"
  "c_e7_aux01_2222_chn --seed 2222 --loss_w_aux 0.1"
  "c_e7_aux01_3333_chn --seed 3333 --loss_w_aux 0.1"
  "c_e11_adamw_1111_chn --seed 1111 --optimizer adamw --weight_decay 0.01"
  "c_e11_adamw_2222_chn --seed 2222 --optimizer adamw --weight_decay 0.01"
  "c_e11_adamw_3333_chn --seed 3333 --optimizer adamw --weight_decay 0.01"
  "c_e3_refine2_1111_chn --seed 1111 --n_layers_refine 2"
  "c_e3_refine2_2222_chn --seed 2222 --n_layers_refine 2"
  "c_e3_refine2_3333_chn --seed 3333 --n_layers_refine 2"
)

# ===========================================================================
# REVIEW BATCH, staged 2026-08-12. docs/review-response.md is the runbook and
# reads this array; PROJECT_PLAN.md §8 item 14 is the scope decision.
#
# What is different about this batch, and it is not a small difference:
#
#   1. It trains on a REBUILT Chinese dataset -- 10x augmentation (--n_aug 9)
#      per the paper's Sec. 4.1, against the 6x every earlier run used, and with
#      20 base fonts carved out into a held-out val split. Nothing here is
#      comparable to any row in RESULTS.csv from before this date. Every
#      comparison in this batch is internal to this batch: candidates against
#      THESE three baseline seeds, at a floor re-measured from THESE three.
#
#   2. Checkpoints are selected on --ckpt_select val_render_l1, the rendered
#      Error on the held-out val fonts, not on val_metric. That is the review's
#      single highest-priority fix. See render_val.py.
#
#   3. --max_ckpt_keep 10 keeps every checkpoint a 151-epoch run produces at
#      --freq_ckpt 25, so both selections can be compared after the fact by
#      scripts/selection_disagreement.py without retraining anything. Check
#      `df -h /data/bens` before launching: 27 runs x 6 checkpoints is the
#      largest single batch this project has held on disk at once.
#
# 27 runs, ~1 GPU-h each plus the rendered validation pass, three GPUs, so
# roughly 12 hours of wall clock in waves of three.
EXPERIMENTS=(
  # --- the three baseline seeds. Everything else is read against their mean,
  # and the seed floor is re-measured from their spread before any candidate is
  # looked at. §3.2's rule, unchanged.
  "rv_seedfloor_1111_chn --seed 1111"
  "rv_seedfloor_2222_chn --seed 2222"
  "rv_seedfloor_3333_chn --seed 3333"

  # --- E9, the finalist the review says does not clear its own floor. This is
  # the run that decides whether it survives honest checkpointing. The reading
  # rule is pre-committed in docs/review-response.md §4 and does not move after
  # the numbers land.
  "rv_e9_sigma050_1111_chn --seed 1111 --enc_noise_std_train 0.5"
  "rv_e9_sigma050_2222_chn --seed 2222 --enc_noise_std_train 0.5"
  "rv_e9_sigma050_3333_chn --seed 3333 --enc_noise_std_train 0.5"

  # --- E1, the project's one confirmed degrading effect, and the only one that
  # ever cleared a floor. Job A already de-confounded it for epoch; this asks
  # whether it survives selection too. A degrading result that evaporates under
  # correct checkpointing is as much a finding as one that holds.
  "rv_e1_norm_1111_chn --seed 1111 --enc_final_norm True"
  "rv_e1_norm_2222_chn --seed 2222 --enc_final_norm True"
  "rv_e1_norm_3333_chn --seed 3333 --enc_final_norm True"

  # --- the six assignment-category representatives from Job C. All six read
  # null on Chinese under val_metric selection. Re-run here because "null under
  # a defective selection rule" is not the same claim as "null", and the
  # coverage table in REPORT §5 rests on all six.
  "rv_e2_batchnorm_1111_chn --seed 1111 --img_norm batch"
  "rv_e2_batchnorm_2222_chn --seed 2222 --img_norm batch"
  "rv_e2_batchnorm_3333_chn --seed 3333 --img_norm batch"
  "rv_e4_ngf32_1111_chn --seed 1111 --ngf 32"
  "rv_e4_ngf32_2222_chn --seed 2222 --ngf 32"
  "rv_e4_ngf32_3333_chn --seed 3333 --ngf 32"
  "rv_e5_bneck256_1111_chn --seed 1111 --bottleneck_bits 256"
  "rv_e5_bneck256_2222_chn --seed 2222 --bottleneck_bits 256"
  "rv_e5_bneck256_3333_chn --seed 3333 --bottleneck_bits 256"
  "rv_e7_aux01_1111_chn --seed 1111 --loss_w_aux 0.1"
  "rv_e7_aux01_2222_chn --seed 2222 --loss_w_aux 0.1"
  "rv_e7_aux01_3333_chn --seed 3333 --loss_w_aux 0.1"
  "rv_e11_adamw_1111_chn --seed 1111 --optimizer adamw --weight_decay 0.01"
  "rv_e11_adamw_2222_chn --seed 2222 --optimizer adamw --weight_decay 0.01"
  "rv_e11_adamw_3333_chn --seed 3333 --optimizer adamw --weight_decay 0.01"
  "rv_e3_refine2_1111_chn --seed 1111 --n_layers_refine 2"
  "rv_e3_refine2_2222_chn --seed 2222 --n_layers_refine 2"
  "rv_e3_refine2_3333_chn --seed 3333 --n_layers_refine 2"
)

# English arm of the same batch. Swap this in after the Chinese one finishes,
# and swap COMMON_ARGS for COMMON_ARGS_ENG below at the same time. Nine runs at
# ~17.5 GPU-h each is ~53 h of wall clock on three GPUs -- start it before a
# weekend, not before a deadline.
#
# Three candidates rather than eight, because English costs 17x what Chinese
# costs per run and the two arms disagree: E9 is the finalist (mixed sign on
# English under the old protocol), E3_refine2 is the one candidate confirmed
# degrading on English at three seeds, and the baselines are the yardstick.
EXPERIMENTS_ENG=(
  "rv_seedfloor_1111_eng --seed 1111"
  "rv_seedfloor_2222_eng --seed 2222"
  "rv_seedfloor_3333_eng --seed 3333"
  "rv_e9_sigma050_1111_eng --seed 1111 --enc_noise_std_train 0.5"
  "rv_e9_sigma050_2222_eng --seed 2222 --enc_noise_std_train 0.5"
  "rv_e9_sigma050_3333_eng --seed 3333 --enc_noise_std_train 0.5"
  "rv_e3_refine2_1111_eng --seed 1111 --n_layers_refine 2"
  "rv_e3_refine2_2222_eng --seed 2222 --n_layers_refine 2"
  "rv_e3_refine2_3333_eng --seed 3333 --n_layers_refine 2"
)

EXPERIMENTS_ARCHIVE_JOBD=(
  # Job D, 2026-08-08 (docs/day6-gpu-push.md §5). Tier 4 capacity: E16 encoder
  # depth, E17 decoder feedforward width, three seeds each -- never at one seed,
  # since both flags shift the global RNG stream for every module constructed
  # after them (the same objection that kept E6 off the candidate list).
  # Closed 2026-08-08: both null, mixed sign on both metrics.
  "e17_dff2048_1111_chn --seed 1111 --dec_d_ff 2048"
  "e17_dff2048_2222_chn --seed 2222 --dec_d_ff 2048"
  "e17_dff2048_3333_chn --seed 3333 --dec_d_ff 2048"
  "e16_depth8_1111_chn --seed 1111 --enc_depth 8"
  "e16_depth8_2222_chn --seed 2222 --enc_depth 8"
  "e16_depth8_3333_chn --seed 3333 --enc_depth 8"
)

EXPERIMENTS_ARCHIVE_TIER3=(
  # ---- Batch A: Tier 3 breadth ------------------------------------------------

  # E12, KL weight. kl_beta has always been 0.01 and never tuned. It interacts
  # with E9 head-on: with a sigma=1.0 additive perturbation already on the encoder
  # output, the reparameterization trick is close to decorative, so the KL term may
  # be regularizing something that is already noisy by construction. Bracketed wide
  # (0 and 10x) rather than sampled finely — nothing in two tiers has resolved at
  # this scale, so a fine sweep would buy resolution the setup does not have.
  "e12_kl000_chn --seed 1111 --kl_beta 0.0"
  "e12_kl100_chn --seed 1111 --kl_beta 0.1"

  # E11, AdamW. torch.optim.AdamW is imported in train.py and unused. Adam applies
  # --weight_decay as an L2 term inside the gradient, where the adaptive per-
  # parameter rate rescales it; AdamW decouples it. At the released weight_decay=0
  # the two are identical, so this run is the first time the flag does anything.
  "e11_adamw_chn --seed 1111 --optimizer adamw --weight_decay 0.01"

  # E2, image-stack normalization. The released norm is a spatial LayerNorm over
  # [C, H, W], which pools channel and spatial statistics together and so discards
  # per-channel scale. This is the assignment's "add normalization layers" bullet
  # done on the image branch, where E1 did it on the sequence branch.
  #
  # --img_norm instance dropped 2026-08-04, rung 1: the image encoder's deepest
  # layer bottlenecks to a 1x1 spatial feature map, and nn.InstanceNorm2d raises
  # ValueError("Expected more than 1 spatial element when training, got input
  # size torch.Size([32, 1024, 1, 1])") there -- it needs >1 spatial element to
  # compute a per-instance variance. group and batch have no such constraint.
  # Architectural incompatibility, not a wiring bug; itself worth a line in the
  # report. See PROJECT_PLAN.md 3.6.
  "e2_groupnorm_chn --seed 1111 --img_norm group"
  "e2_batchnorm_chn --seed 1111 --img_norm batch"

  # E4, width. ngf 16 -> 32 doubles both image encoder and decoder. Unlike every
  # other row in this batch it changes capacity, so read it as a capacity control
  # rather than as a like-for-like architectural factor, and say so in the report.
  "e4_ngf32_chn --seed 1111 --ngf 32"

  # E15, weight EMA. The one candidate here with a reliable prior in the
  # literature, and the only one that changes nothing about training — the EMA is
  # evaluated and checkpointed, the optimizer still steps the raw weights. If
  # anything in Tier 3 clears the floor it is most likely this.
  "e15_ema999_chn --seed 1111 --ema_decay 0.999"

  # E5, latent width. --bottleneck_bits looked like a free flag and was not: the
  # latent is written into a 512-wide slot in ModalityFusion, so any other value
  # crashed until the z_proj projection added 2026-08-04. Bracketed either side of
  # 512. Note both rows change parameter count in the image decoder too, whose
  # input_nc is bottleneck_bits + char_num.
  "e5_bneck256_chn --seed 1111 --bottleneck_bits 256"
  "e5_bneck1024_chn --seed 1111 --bottleneck_bits 1024"

  # E6 has no row here on purpose. The dead Perceiver cross-attention parameters
  # are constructed before several live modules, and every construction draws from
  # the global RNG stream, so deleting them shifts the initialization of everything
  # after them. A "dead params removed" run is numerically a seed change, and the
  # seed floor is larger than anything the sweep is chasing, so the run could not
  # be read either way. It is a reconstruction finding instead:
  #   python scripts/dead_params.py

  # ---- Batch B: multi-seed replication of the three largest deltas -------------
  # Seed 1111 is already trained for all three; only 2222 and 3333 are missing.
  # Deltas vs the seed-1111 baseline at screening: E9 sigma=0.5 -0.0059,
  # E1 -0.0054, E13 bins256 -0.0050. All roughly half the 0.0093 floor, and
  # statistically indistinguishable from each other, which is exactly why picking
  # one would have been arbitrary and all three get the same treatment.
  "e9_sigma050_2222_chn --seed 2222 --enc_noise_std_train 0.5"
  "e9_sigma050_3333_chn --seed 3333 --enc_noise_std_train 0.5"
  "e1_norm_2222_chn --seed 2222 --enc_final_norm True"
  "e1_norm_3333_chn --seed 3333 --enc_final_norm True"
  "e13_bins256_2222_chn --seed 2222 --n_args_bins 256"
  "e13_bins256_3333_chn --seed 3333 --n_args_bins 256"
)

# ---------------------------------------------------------------------------
# E14-deep, staged 2026-08-04, NOT yet launched. Swap this in after Tier 3.
#
# Motivation: the wandb val curves show warmup_cosine consistently below baseline,
# but E14's rendered screening delta was -0.0010 -- rank 10 of the 18 candidates
# run so far, and at the 0.0011 decode-noise level, i.e. indistinguishable from
# baseline on the metric that is actually reported. It also has the worst s-IoU in
# Tier 2 (-0.0178). So val_metric and the rendered metric disagree about E14
# specifically.
#
# DO NOT LAUNCH THIS UNTIL `python scripts/val_metric_correlation.py` HAS RUN.
# It costs no GPU and it decides whether these runs are screened on val_metric at
# all. If val_metric does not predict rendered Error, rows 1-5 below are measuring
# a quantity nobody reports and the batch should shrink to rows 6-7.
#
# Rows 1-5 are controls: they decompose what E14 actually changed. Rows 6-7 are
# the only ones with a mechanism for a real gain, and they screen on the rendered
# metric regardless of what the correlation says.
# E14_DEEP=(
#   # 1. THE control. warmup_cosine ends at 0.05x lr; the released ExponentialLR
#   #    ends at 0.997^150 = 0.635x. That is a 12.7x difference in terminal step
#   #    size, and late-training val loss falls as the lr decays and the weights
#   #    stop bouncing around the minimum -- whether or not rollout improves.
#   #    gamma = 0.05^(1/150) = 0.98023 lands exp on the same terminal lr, so this
#   #    run isolates "warmup + cosine shape" from "anneal the lr at all".
#   #    If this reproduces E14's val curve, E14 is an lr-annealing result and the
#   #    schedule shape contributed nothing.
#   "e14_ctrl_gamma_chn --seed 1111 --lr_gamma 0.98023"
#
#   # 2-3. Split the bundle. E14 changed warmup AND decay shape at once, which
#   #    violates the one-factor rule the rest of the sweep follows.
#   #    lr_min_factor 1.0 collapses the cosine to a constant, giving warmup only;
#   #    lr_warmup_steps 1 gives cosine only. Neither needs new code.
#   "e14_warmonly_chn --seed 1111 --lr_schedule warmup_cosine --lr_min_factor 1.0"
#   "e14_cosonly_chn --seed 1111 --lr_schedule warmup_cosine --lr_warmup_steps 1"
#
#   # 4-5. How much of the effect is just terminal lr, within warmup_cosine.
#   "e14_min000_chn --seed 1111 --lr_schedule warmup_cosine --lr_min_factor 0.0"
#   "e14_min020_chn --seed 1111 --lr_schedule warmup_cosine --lr_min_factor 0.2"
#
#   # 6-7. The actual reason warmup exists, and the only rows here likely to move
#   #    the rendered metric. Warmup's payoff is that it makes a LARGER peak lr
#   #    stable; running warmup_cosine at the baseline's 2e-4 adds the machinery
#   #    without collecting the benefit. A higher peak changes which optimum is
#   #    reached, not merely how tightly the weights settle into it, so unlike
#   #    rows 1-5 this has a mechanism for improving autoregressive rollout.
#   "e14_lr4e4_chn --seed 1111 --lr_schedule warmup_cosine --lr 0.0004"
#   "e14_lr8e4_chn --seed 1111 --lr_schedule warmup_cosine --lr 0.0008"
# )
# ---------------------------------------------------------------------------

# Tier 2 batch, 2026-08-04, kept for provenance. Results in §3.5: nothing cleared
# the 0.0093 floor; E13 bins256 was the largest single delta at -0.0050.
#   "e8_ls05_chn --seed 1111 --args_label_smooth_sigma 0.5"
#   "e8_ls10_chn --seed 1111 --args_label_smooth_sigma 1.0"
#   "e8_ls20_chn --seed 1111 --args_label_smooth_sigma 2.0"
#   "e13_bins256_chn --seed 1111 --n_args_bins 256"
#   "e13_nopad_chn --seed 1111 --arg_embed_pad_idx False"
#   "e3_refine2_chn --seed 1111 --n_layers_refine 2"
#   "e3_refine3_chn --seed 1111 --n_layers_refine 3"
#   "e14_wucos_chn --seed 1111 --lr_schedule warmup_cosine"

# Tier 1 batch, 2026-08-03 night, kept for provenance. Re-run under the fixed
# compute_val_loss (PROJECT_PLAN.md §1.5) after val_metric was found to omit the
# refinement-decoder loss. Results in §3.2: noise floor 0.0093, nothing cleared it.
#   "seedfloor_1111_chn --seed 1111"
#   "seedfloor_2222_chn --seed 2222"
#   "seedfloor_3333_chn --seed 3333"
#   "e7_aux01_chn --seed 1111 --loss_w_aux 0.1"
#   "e7_aux03_chn --seed 1111 --loss_w_aux 0.3"
#   "e7_aux10_chn --seed 1111 --loss_w_aux 1.0"
#   "e9_sigma000_chn --seed 1111 --enc_noise_std_train 0.0"
#   "e9_sigma010_chn --seed 1111 --enc_noise_std_train 0.1"
#   "e9_sigma025_chn --seed 1111 --enc_noise_std_train 0.25"
#   "e9_sigma050_chn --seed 1111 --enc_noise_std_train 0.5"
#   "e10_drop01_chn --seed 1111 --dropout 0.1"
#   "e10_drop02_chn --seed 1111 --dropout 0.2"
#   "e1_norm_chn --seed 1111 --enc_final_norm True"

# Args shared by every experiment in this batch. Identical across all of them,
# which is what makes every row comparable to the batch's own baseline seeds.
#
# --render_val_freq 25 matches --freq_ckpt, so every checkpoint carries a rendered
# score and selection never has to fall back. --render_val_samples 1 is one decode
# per glyph: the test protocol's best-of-50 at every checkpoint is unaffordable, and
# a single decode is still the same decoder, rasterizer and binary masks as the
# reported number, which val_metric never was.
#
# --max_ckpt_keep 10 keeps all six checkpoints of a 151-epoch run, so
# scripts/selection_disagreement.py can compare the two selection rules after the
# fact. Prune after scoring, not before.
# --ref_char_ids is new to a TRAINING command and is not optional here. The rendered
# pass runs the model at mode='test', which reads its reference glyphs from this flag
# and asserts there are exactly --ref_nshot of them. The default is the English set of
# four; a Chinese run at --ref_nshot 8 dies at the first checkpoint without it. These
# are the same eight ids every Chinese test run in COMMANDS.md uses, so the validation
# decode is conditioned exactly as the scored decode will be.
COMMON_ARGS="--mode train --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29 --batch_size 32 --n_epochs 151 --freq_ckpt 25 --max_ckpt_keep 10 --render_val_freq 25 --render_val_samples 1 --ckpt_select val_render_l1"

# English. 631 epochs is the budget frozen 2026-08-07 in docs/english-arm.md Step 1
# (E_conv per seed 400/580/420, rounded up), kept unchanged so this batch differs
# from the English arm in selection rule alone. --freq_ckpt 20 gives 31 checkpoints;
# --render_val_freq 40 scores every other one, because English decodes are slower and
# the curve is flat enough by then that 16 scored points locate the minimum.
COMMON_ARGS_ENG="--mode train --model_name main_model --language eng --max_seq_len 51 --ref_nshot 4 --ref_char_ids 0,1,26,27 --batch_size 32 --n_epochs 631 --freq_ckpt 20 --max_ckpt_keep 10 --render_val_freq 40 --render_val_samples 1 --ckpt_select val_render_l1"

# Appends the launched PID to the global `pids` array. Must be called directly
# (not via `$(launch_one ...)`) -- command substitution forks a subshell, and a
# background job started inside one is not a child of *this* shell, so `wait`
# on its pid later fails with "not a child of this shell".
launch_one() {
  local name="$1" extra_args="$2" gpu="$3"
  local logfile="nohup_${name}.out"
  echo "[$name] GPU $gpu -> $logfile"
  CUDA_VISIBLE_DEVICES=$gpu nohup python train.py $COMMON_ARGS --name_exp "$name" $extra_args \
    > "$logfile" 2>&1 &
  pids+=("$!")
}

if [[ "$MODE" == "sequential" ]]; then
  # One GPU (GPUS[0]) reused for every experiment, one at a time.
  for entry in "${EXPERIMENTS[@]}"; do
    name="${entry%% *}"
    extra_args="${entry#* }"
    pids=()
    launch_one "$name" "$extra_args" "${GPUS[0]}"
    wait "${pids[0]}"
    echo "[$name] done"
  done
else
  # parallel: waves of len(GPUS), so EXPERIMENTS can be longer than GPUS
  # without editing the array between waves.
  n_gpu=${#GPUS[@]}
  n_exp=${#EXPERIMENTS[@]}
  wave=1
  for (( start=0; start<n_exp; start+=n_gpu )); do
    pids=()
    for (( j=0; j<n_gpu && start+j<n_exp; j++ )); do
      entry="${EXPERIMENTS[$((start + j))]}"
      name="${entry%% *}"
      extra_args="${entry#* }"
      launch_one "$name" "$extra_args" "${GPUS[$j]}"
    done
    echo "Wave $wave: launched ${#pids[@]} runs: ${pids[*]}"
    wait "${pids[@]}"
    echo "Wave $wave done."
    wave=$((wave + 1))
  done
  echo "All ${n_exp} runs finished."
fi
