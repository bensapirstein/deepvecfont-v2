# Tier 3 launch runbook

2026-08-04, day 2. Sixteen training runs planned in two batches, plus one eval-only
audit. Fifteen actually launched: rung 1 dropped `e2_instancenorm_chn` (see below).
Follows `docs/tier2-launch.md`; read `PROJECT_PLAN.md` §3.6 and §8 items 2-3 first.

## Why the batch has this shape

Tier 1 and Tier 2 put eighteen single-point candidates inside a 0.0093 L1 noise
floor. Nothing cleared it. §3.2's decomposition then put roughly 88% of that floor
in training-seed variance and only 12% in decode noise, which rules out the cheap
remedy (raise `--n_samples`) and leaves exactly one lever: run candidates at
multiple seeds.

So Tier 3 is deliberately two halves. **Batch A** is breadth, and its job is to
cover the assignment's change categories with the remaining untested factors; a
null there is the expected outcome and is reportable as one. **Batch B** is depth,
and it is the half that produces a usable number whichever way it lands, because a
mean against a mean can be compared where a point against a point could not.

E6 has no training run. See the note at the bottom.

## Before launching

```bash
cd ~/deepvecfont-v2
git pull
conda activate dvf_v2

# 1. The diff is bigger than Tier 2's -- new file models/norms.py, a new class in
#    train.py, a projection in modality_fusion.py. Section 8 covers all of it.
python scripts/check_infra.py

# 2. Storage. Sixteen runs at --max_ckpt_keep 2.
df -h /data/bens && du -sh /data/bens/deepvecfont-v2/*
```

`check_infra.py` must print zero failures before anything launches. The specific
failure it exists to catch is a flag that parses, logs itself into `opts.txt` and
wandb, and never reaches the model — that produces a baseline-identical number
that reads as a result.

**Expect 190 passed, 0 failed here, not 182.** The Mac has no torch, so the last
block of section 8 — the one that actually constructs each norm module and checks
the group counts — prints `[SKIP]` there and was never exercised before this batch.
On the cluster it runs. If you see `[SKIP]` on the cluster, torch is not importable
in `dvf_v2` and nothing below this line is meaningful.

## Rung 0: the defaults are still the released model

The Tier 1 and Tier 2 tables and the seed floor are only valid references while
every new flag's default reproduces the released behaviour. Check it directly
rather than trusting it, on 2 epochs:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --mode train --name_exp t3_sanity_default \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --seed 1111 --n_epochs 2 --freq_ckpt 1 --max_ckpt_keep 1
```

The epoch-0 and epoch-1 losses should match the first two epochs of
`nohup_seedfloor_1111_chn.out` to the digit. If they do not, stop: something in the
Tier 3 diff changed the default path, and every comparison in the plan is affected,
not just Tier 3.

## Rung 1: the four new code paths construct

Each of these is 2 epochs and exists only to catch a shape error before it wastes a
wave. `--img_norm batch` and `--bottleneck_bits` are the two most likely to break.

```bash
for extra in "--img_norm group" "--img_norm batch" "--img_norm instance" \
             "--ngf 32" "--bottleneck_bits 256" "--bottleneck_bits 1024" \
             "--ema_decay 0.999" "--optimizer adamw --weight_decay 0.01"; do
  echo "=== $extra ==="
  CUDA_VISIBLE_DEVICES=1 python train.py --mode train --name_exp t3_probe \
    --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
    --batch_size 32 --seed 1111 --n_epochs 1 --freq_ckpt 1 --max_ckpt_keep 1 \
    $extra 2>&1 | tail -3
done
rm -rf experiments/t3_probe_main_model experiments/t3_sanity_default_main_model
```

**Run 2026-08-04: `--img_norm instance` failed here, everything else passed.**
`ValueError: Expected more than 1 spatial element when training, got input size
torch.Size([32, 1024, 1, 1])` -- the image encoder's deepest layer bottlenecks to
a 1x1 feature map, and `nn.InstanceNorm2d` needs more than one spatial element to
compute a per-instance variance. `check_infra.py`'s E2 checks only construct the
module, they don't forward a tensor at the bottleneck's actual shape, so this was
rung 1's to catch and it did. Dropped `e2_instancenorm_chn` from both
`scripts/run_experiments.sh` and `scripts/test_experiments.sh`; batch runs as
fifteen. See `PROJECT_PLAN.md` §3.6 for the report-facing note.

Then confirm one checkpoint round-trips, since strict loading is where a
train/test flag mismatch shows up:

```bash
CUDA_VISIBLE_DEVICES=1 python train.py --mode train --name_exp t3_probe_bn \
  --model_name main_model --language chn --max_seq_len 71 --ref_nshot 8 \
  --batch_size 32 --seed 1111 --n_epochs 1 --freq_ckpt 1 --max_ckpt_keep 1 \
  --img_norm batch
CUDA_VISIBLE_DEVICES=1 python test_few_shot.py --mode test --name_exp t3_probe_bn \
  --language chn --max_seq_len 71 --model_name main_model --batch_size 1 \
  --n_samples 3 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29 \
  --img_norm batch --name_ckpt 0_40.ckpt
rm -rf experiments/t3_probe_bn_main_model
```

## Launch

`scripts/run_experiments.sh` is already filled with both batches, in order.
`GPUS=(1 2)`, so fifteen runs is eight waves (the last with one idle GPU) at
roughly an hour each.

```bash
cd ~/deepvecfont-v2
nohup ./scripts/run_experiments.sh parallel > nohup_tier3_batch.out 2>&1 &
tail -f nohup_tier3_batch.out
```

To stop after Batch A, comment out the Batch B block in `EXPERIMENTS` before
launching. Batch B is the half worth protecting if the night gets cut short, so if
only one half can run, invert the order rather than dropping it.

## Screening

`scripts/test_experiments.sh` is filled to match, including all three seed-floor
runs. Screening budget: `--n_samples 3`, all 34 fonts.

```bash
nohup ./scripts/test_experiments.sh parallel > nohup_tier3_test.out 2>&1 &
```

Which flags have to be repeated at test time is the one thing worth re-reading in
that script's header. `--img_norm`, `--ngf` and `--bottleneck_bits` change what
`ModelMain` constructs, so omitting them is a strict-load crash. `--ema_decay`,
`--optimizer`, `--weight_decay` and `--kl_beta` do not, because the EMA weights
were already written into the checkpoint under the ordinary key names.

## Reading Batch B

Batch B is not read against the 0.0093 floor. That number is the spread of the
baseline across seeds, and it is the thing being replaced. Compute instead:

- baseline mean L1 across seeds 1111 / 2222 / 3333, and its spread;
- candidate mean L1 across the same three seeds, and its spread;
- the paired difference per seed, since the seeds are matched.

A candidate whose per-seed differences all carry the same sign is worth the
confirmation eval even if the means overlap, and that is a different and stronger
statement than anything Tier 1 or Tier 2 could support. Record it in §3.6 as a
dated **Measured** paragraph, then in the vault `Runs/Run Log.md` and
`Experiments/Experiment Tracker.md`.

## E6, which is not a run

The Perceiver cross-attention path is constructed and never called: `cross_attn`
and `cross_ff` are unpacked in both loops in `Transformer.forward` and
`att_residual` and never invoked, alongside `latents`, `to_logits`,
`to_patch_embedding` and `pre_lstm_fc`. The parameters are allocated, handed to the
optimizer in `parameters_all`, and carry Adam moment buffers.

Deleting them is not a numerical no-op, though, and that is the reason there is no
training row. The dead modules are constructed *before* several live ones, and every
`nn.Linear` and `nn.Parameter` draws from the global RNG stream, so removing them
shifts the initialization of everything built afterwards. A "dead parameters
removed" run differs from baseline by an effective seed change, and the seed floor
is larger than any effect in play, so the run could not be interpreted in either
direction. It goes in the reconstruction section as a count:

```bash
python scripts/dead_params.py
```

Runs on CPU in seconds. Put the number in §1.4 next to the other paper-versus-code
deviations.
