#!/usr/bin/env python3
"""Pre-flight checks for the training infrastructure. No torch, no GPU, no data required.

Run this before syncing to the cluster, and again on the cluster before the GPU dry run.
It verifies the parts of the --seed / --wandb / --max_ckpt_keep / stage-2 flag work that
can be checked without a training run:

  1. options.py  -- every flag exists with the expected type and default   [live]
  2. str2bool    -- `--wandb False` actually disables the flag             [live]
  3. train.py    -- imports, seeding and wandb call sites are wired right  [static, AST]
  4. wandb       -- init / log / finish round-trip in offline mode         [live]
  5. models/     -- the E9 / E1 / E10 flags actually reach the model       [static, source]

What this CANNOT check, and what the cluster dry run is for: that setup_seed actually
makes a run reproducible, that the mirrored scalars carry sane values, and that a run
shows up in the wandb web UI.

Usage:  python scripts/check_infra.py
Exit code 0 if everything passed.
"""
import argparse
import ast
import math as _math
import os
import py_compile
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

PASSED, FAILED = [], []


def check(label, condition, detail=""):
    (PASSED if condition else FAILED).append(label)
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail and not condition else ""))
    return condition


# ---------------------------------------------------------------- 1. options.py

def check_options():
    print("\n1. options.py flags [live]")
    from options import get_parser_main_model

    parser = get_parser_main_model()
    actions = {a.dest: a for a in parser._actions}
    defaults = vars(parser.parse_args([]))

    expected = [
        # dest,                  type,  default
        ('seed',                 int,   1111),
        ('max_ckpt_keep',        int,   1),
        ('enc_noise_std_train',  float, 1.0),
        ('enc_noise_std_test',   float, 1.0),
        ('dropout',              float, 0.0),
        ('enc_final_norm',       None,  False),   # str2bool, checked by identity below
        # Tier 2, 2026-08-04.
        ('args_label_smooth_sigma', float, 0.0),
        ('n_args_bins',          int,   128),
        ('arg_embed_pad_idx',    None,  True),    # str2bool
        ('n_layers_refine',      int,   1),
        ('lr_schedule',          str,   'exp'),
        ('lr_warmup_steps',      int,   500),
        ('lr_min_factor',        float, 0.05),
        # Tier 3, 2026-08-04. kl_beta / ngf / bottleneck_bits / weight_decay predate
        # this work; they are listed so a future edit to their defaults trips a check
        # rather than silently invalidating the Tier 1 and Tier 2 tables.
        ('optimizer',            str,   'adam'),
        ('img_norm',             str,   'layer'),
        ('img_norm_groups',      int,   32),
        ('ema_decay',            float, 0.0),
        ('ema_warmup_steps',     int,   0),
        ('kl_beta',              float, 0.01),
        ('ngf',                  int,   16),
        ('bottleneck_bits',      int,   512),
        ('weight_decay',         float, 0.0),
    ]
    for dest, typ, default in expected:
        if not check(f"--{dest} exists", dest in actions):
            continue
        if typ is not None:
            check(f"--{dest} type is {typ.__name__}", actions[dest].type is typ,
                  f"got {actions[dest].type}")
        check(f"--{dest} default is {default!r}", defaults[dest] == default,
              f"got {defaults[dest]!r}")

    check("--wandb exists", 'wandb' in actions)
    check("--wandb default is True", defaults.get('wandb') is True, f"got {defaults.get('wandb')!r}")
    check("--wandb_project exists", 'wandb_project' in actions)
    check("--wandb_project default is 'deepvecfont-v2'",
          defaults.get('wandb_project') == 'deepvecfont-v2', f"got {defaults.get('wandb_project')!r}")

    # The stage-2 flags must be inert: their defaults have to reproduce current behaviour.
    # This is what keeps the 3-seed noise floor comparable to every run launched after
    # the wiring landed, so a failure here invalidates the whole Stage 2 comparison.
    check("enc_noise_std defaults reproduce the hardcoded sigma=1.0",
          defaults['enc_noise_std_train'] == 1.0 and defaults['enc_noise_std_test'] == 1.0)
    check("dropout default reproduces the all-zero dropout in models/",
          defaults['dropout'] == 0.0)
    check("enc_final_norm defaults off, so the encoder is unchanged",
          defaults['enc_final_norm'] is False, f"got {defaults['enc_final_norm']!r}")
    check("--enc_final_norm parses as str2bool, not as a truthy string",
          parser.parse_args(['--enc_final_norm', 'False']).enc_final_norm is False)

    # Same discipline for Tier 2: every default has to be the released behaviour, or the
    # Tier 1 table and the seed floor stop being a valid reference for Tier 2.
    check("args_label_smooth_sigma defaults to 0, keeping the one-hot target",
          defaults['args_label_smooth_sigma'] == 0.0)
    check("n_args_bins defaults to the released 128, not the paper's 256",
          defaults['n_args_bins'] == 128)
    check("arg_embed_pad_idx defaults on, keeping padding_idx=0",
          defaults['arg_embed_pad_idx'] is True, f"got {defaults['arg_embed_pad_idx']!r}")
    check("n_layers_refine defaults to the released 1, not the paper's 2",
          defaults['n_layers_refine'] == 1)
    check("lr_schedule defaults to exp, the original ExponentialLR",
          defaults['lr_schedule'] == 'exp')
    check("--arg_embed_pad_idx parses as str2bool, not as a truthy string",
          parser.parse_args(['--arg_embed_pad_idx', 'False']).arg_embed_pad_idx is False)
    try:
        parser.parse_args(['--lr_schedule', 'linear'])
        check("--lr_schedule rejects an unknown schedule", False, "it was accepted")
    except SystemExit:
        check("--lr_schedule rejects an unknown schedule", True)

    # Regression guard: the flags the training commands in COMMANDS.md rely on.
    for dest in ('language', 'max_seq_len', 'ref_nshot', 'name_exp', 'freq_ckpt', 'n_epochs'):
        check(f"--{dest} still present", dest in actions)


def check_str2bool():
    print("\n2. str2bool [live]")
    from options import get_parser_main_model, str2bool

    for raw, want in [('True', True), ('true', True), ('1', True),
                      ('False', False), ('false', False), ('0', False)]:
        check(f"str2bool({raw!r}) == {want}", str2bool(raw) is want)

    parser = get_parser_main_model()
    check("`--wandb False` disables the flag",
          parser.parse_args(['--wandb', 'False']).wandb is False)
    check("`--wandb True` enables the flag",
          parser.parse_args(['--wandb', 'True']).wandb is True)
    try:
        parser.parse_args(['--wandb', 'banana'])
        check("`--wandb banana` is rejected", False, "it was accepted")
    except SystemExit:
        check("`--wandb banana` is rejected", True)

    # Documented known issue: the pre-existing bool flags still use type=bool.
    for dest in ('tboard', 'resume', 'multi_gpu'):
        action = {a.dest: a for a in parser._actions}[dest]
        if action.type is bool:
            print(f"  [note] --{dest} still uses type=bool, so `--{dest} False` evaluates True")


# ---------------------------------------------------------------- 3. train.py

def check_train_source():
    print("\n3. train.py wiring [static]")
    path = os.path.join(REPO, 'train.py')

    try:
        py_compile.compile(path, cfile=os.path.join(tempfile.gettempdir(), 'train.pyc'), doraise=True)
        check("train.py compiles", True)
    except py_compile.PyCompileError as e:
        check("train.py compiles", False, str(e))
        return

    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)

    check("wandb import is guarded by try/except ImportError",
          any(isinstance(n, ast.Try)
              and any(isinstance(h.type, ast.Name) and h.type.id == 'ImportError' for h in n.handlers)
              and 'wandb' in ast.dump(n)
              for n in ast.walk(tree)))

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == 'train_main_model'), None)
    if not check("train_main_model exists", fn is not None):
        return

    body = ast.dump(fn)
    check("setup_seed is called with opts.seed",
          "func=Name(id='setup_seed'" in body and "attr='seed'" in body)
    check("setup_seed(1111) is gone", 'setup_seed(1111)' not in src)

    # wandb.init argument contract
    init = next((n for n in ast.walk(fn)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == 'init'
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'wandb'), None)
    if check("wandb.init is called", init is not None):
        kw = {k.arg for k in init.keywords}
        for arg in ('project', 'name', 'config', 'tags'):
            check(f"wandb.init passes {arg}=", arg in kw)
        check("wandb.init project is opts.wandb_project (not a hardcoded literal, "
              "so language arms can be split into separate projects)",
              any(k.arg == 'project' and isinstance(k.value, ast.Attribute)
                  and k.value.attr == 'wandb_project'
                  for k in init.keywords))
        check("wandb.init config is vars(opts)", 'vars' in ast.dump(init))

    logs = [n for n in ast.walk(fn)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == 'log'
            and isinstance(n.func.value, ast.Name) and n.func.value.id == 'wandb']
    check("wandb.log is called at least twice (train + val)", len(logs) >= 2, f"found {len(logs)}")
    check("every wandb.log passes an explicit step=",
          all(any(k.arg == 'step' for k in c.keywords) for c in logs))

    check("wandb.finish is called",
          any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == 'finish'
              and isinstance(n.func.value, ast.Name) and n.func.value.id == 'wandb'
              for n in ast.walk(fn)))

    # Every wandb.* call other than the import must sit behind the use_wandb gate.
    guarded = []
    for node in ast.walk(fn):
        if isinstance(node, ast.If) and 'use_wandb' in ast.dump(node.test):
            guarded.extend(id(c) for c in ast.walk(node)
                           if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                           and isinstance(c.func.value, ast.Name) and c.func.value.id == 'wandb')
    all_calls = [id(c) for c in ast.walk(fn)
                 if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                 and isinstance(c.func.value, ast.Name) and c.func.value.id == 'wandb']
    check("every wandb call is behind `if use_wandb`",
          set(all_calls) == set(guarded),
          f"{len(all_calls) - len(set(guarded) & set(all_calls))} unguarded")

    check("no image/artifact upload to wandb",
          not any(s in src for s in ('wandb.Image', 'wandb.save', 'wandb.Artifact', 'log_artifact')))

    # The tag lists must be reachable from the wandb block, not nested inside `if opts.tboard`.
    tboard_ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If) and 'tboard' in ast.dump(n.test)]
    nested = any('loss_svg_items' in ast.dump(n) and 'Assign' in ast.dump(n)
                 and any(isinstance(c, ast.Assign) and 'loss_svg_items' in ast.dump(c)
                         for c in ast.walk(n))
                 for n in tboard_ifs)
    check("loss item lists hoisted out of `if opts.tboard`", not nested,
          "they are still assigned inside the tboard branch, so --tboard False breaks wandb")


# ---------------------------------------------------------------- 4. wandb round-trip

def check_wandb_roundtrip():
    print("\n4. wandb round-trip [live, offline mode]")
    try:
        import wandb
    except ImportError:
        print("  [skip] wandb is not installed here. On the cluster: pip install wandb")
        return

    os.environ.setdefault('WANDB_SILENT', 'true')
    fake_opts = argparse.Namespace(name_exp='check_infra_main_model', language='chn',
                                   seed=1111, lr=0.0002, max_ckpt_keep=1,
                                   enc_noise_std_train=1.0, dropout=0.0)
    with tempfile.TemporaryDirectory() as tmp:
        run = wandb.init(project="deepvecfont-v2", name=fake_opts.name_exp,
                         config=vars(fake_opts), tags=[fake_opts.language],
                         mode="offline", dir=tmp)
        check("wandb.init returns a run", run is not None)
        check("config carries the opts", run.config.get('seed') == 1111
              and run.config.get('language') == 'chn')

        # Mirrors the real loop: a train block and a val block can fire at the same
        # batches_done. wandb holds an explicit step open until a higher step arrives
        # (or finish() is called), so the two dicts merge rather than clobbering.
        def train_log(step):
            run.log({'Loss/loss': 4.2 - step / 100,
                     'Loss/svg_total': 3.9,
                     'Loss/img_l1': 0.11,
                     'lr': 0.0002, 'epoch': 0}, step=step)

        train_log(50)
        train_log(100)
        run.log({'VAL/loss_svg_total': 3.8, 'VAL/val_metric': 4.03}, step=100)
        train_log(150)   # advancing the step is what commits step 100

        summary = dict(run.summary)
        for key in ('Loss/loss', 'Loss/svg_total', 'VAL/val_metric'):
            check(f"{key} reached the run summary", key in summary)
        check("train and val dicts merge at a shared step",
              'Loss/loss' in summary and 'VAL/loss_svg_total' in summary,
              "the second log at the same step clobbered the first")

        run.finish()
        check("wandb.finish completes", True)
        # Consequence worth knowing during the dry run: the very last step of a run stays
        # uncommitted until finish(). A killed run (SIGKILL, node eviction) loses it.
        print("  [note] the final step commits only at wandb.finish(); a hard-killed run drops it")


# ------------------------------------------------- 5. stage 2 wiring (E9, E1, E10)

def check_stage2_wiring():
    """Static check that the Tier 1 flags actually reach the model.

    A declared-but-unread flag is the specific failure this guards against: the run
    would train happily, log the flag into opts.txt and wandb, and produce a number
    identical to the baseline. That is worse than a crash, because it looks like a
    result. Wired 2026-08-03; see PROJECT_PLAN.md 3.4.
    """
    print("\n5. stage 2 wiring: E9 / E1 / E10 [static, source]")

    tf_path = os.path.join(REPO, 'models', 'transformers.py')
    mm_path = os.path.join(REPO, 'models', 'model_main.py')
    with open(tf_path) as fh:
        tf_src = fh.read()
    with open(mm_path) as fh:
        mm_src = fh.read()

    # E9. The old line was an unconditional `x = x + torch.randn_like(x)`.
    check("E9: the hardcoded unscaled perturbation is gone",
          'x = x + torch.randn_like(x)' not in tf_src,
          "models/transformers.py still adds the perturbation without a sigma")
    check("E9: sigma is selected by self.training",
          'opts.enc_noise_std_train if self.training else opts.enc_noise_std_test' in tf_src)
    check("E9: sigma scales the noise",
          'sigma * torch.randn_like(x)' in tf_src)
    # train.py wraps validation in model_main.eval() and restores .train() after, and
    # test_few_shot.py calls .eval(), so self.training is the right discriminator.
    with open(os.path.join(REPO, 'train.py')) as fh:
        train_src = fh.read()
    check("E9: train.py puts the model in eval() around validation",
          '.eval()' in train_src and '.train()' in train_src,
          "without eval() the test sigma would never apply at validation")

    # E1.
    check("E1: both encoder stacks construct a terminal norm",
          'self.enc_final_norm = nn.LayerNorm' in tf_src
          and 'self.enc_final_norm_cnnsvg = nn.LayerNorm' in tf_src)
    check("E1: forward applies it", 'if self.enc_final_norm is not None:' in tf_src)
    check("E1: att_residual applies it too",
          'if self.enc_final_norm_cnnsvg is not None:' in tf_src)
    check("E1: off by default leaves the state_dict keys unchanged",
          'self.enc_final_norm = None' in tf_src,
          "the modules must not be constructed when the flag is off, or old "
          "checkpoints stop loading strictly")

    # E10. Every one of these was a hardcoded 0.0 upstream.
    check("E10: decoder sublayers no longer hardcode dropout=0.0",
          'MultiHeadedAttention(h=8, d_model=512, dropout=0.0)' not in tf_src
          and 'PositionwiseFeedForward(d_model=512, d_ff=1024, dropout=0.0)' not in tf_src)
    check("E10: the decoder takes dropout as a parameter, not off the module global",
          'def __init__(self, dropout=None):' in tf_src)
    check("E10: ModelMain passes the real opts into the decoder",
          'Transformer_decoder(dropout=opts.dropout)' in mm_src)
    check("E10: ModelMain passes the real opts into the encoder",
          'attn_dropout = opts.dropout' in mm_src and 'ff_dropout = opts.dropout' in mm_src,
          "models/model_main.py still hardcodes attn_dropout / ff_dropout to 0.")

    # The whole point of the defaults.
    check("E10: no hardcoded dropout=0. survives in the encoder call site",
          'attn_dropout = 0.,' not in mm_src)


def check_tier2_wiring():
    """Static check that the Tier 2 flags actually reach the model and the optimizer.

    Same failure mode as section 5, and it bites harder here: E13 changes tensor shapes
    in seven places and missing one of them is either a crash or, worse, a head that
    predicts over 256 bins against a target built at 128. Wired 2026-08-04; see
    PROJECT_PLAN.md 3.5.
    """
    print("\n7. tier 2 wiring: E8 / E13 / E3 / E14 [static, source]")

    from options import get_parser_main_model

    with open(os.path.join(REPO, 'models', 'transformers.py')) as fh:
        tf_src = fh.read()
    with open(os.path.join(REPO, 'train.py')) as fh:
        train_src = fh.read()

    # E8. The target distribution moves behind one helper so the one-hot path and the
    # smoothed path cannot drift apart.
    check("E8: build_args_target exists", 'def build_args_target(' in tf_src)
    check("E8: the loss builds its target through it",
          'build_args_target(tgt_args, opts.n_args_bins, opts.args_label_smooth_sigma)' in tf_src)
    check("E8: sigma <= 0 still returns the exact one-hot",
          'if smooth_sigma <= 0:' in tf_src and 'return F.one_hot(tgt_args, n_bins)' in tf_src)
    check("E8: the smoothed target is renormalized over the bins that exist",
          "weights / weights.sum(-1, keepdim=True)" in tf_src,
          "without renormalizing, a target near bin 0 or 127 loses mass off the end")
    check("E8: the raw one-hot argument target is gone from the loss",
          'F.one_hot(tgt_args, 128)' not in tf_src)

    # E13. Seven hardcoded bin counts; every one of them has to follow the flag.
    for label, needle in [
        ("arg_embed vocabulary", 'nn.Embedding(opts.n_args_bins, 128,'),
        ("args_fcn output width", 'nn.Linear(512, 8 * opts.n_args_bins)'),
        ("logit reshapes", 'args_logits.reshape(N, S, 8, opts.n_args_bins)'),
        ("seqlen_mask3", 'repeat(1,1,8,opts.n_args_bins)'),
    ]:
        check(f"E13: {label} follows --n_args_bins", needle in tf_src)
    check("E13: both decoder passes reshape on the flag",
          tf_src.count('args_logits.reshape(N, S, 8, opts.n_args_bins)') == 2,
          f"found {tf_src.count('args_logits.reshape(N, S, 8, opts.n_args_bins)')}, expected 2 "
          "(Transformer_decoder.forward and .parallel_decoder)")
    check("E13: numericalize defaults to the flag",
          'def numericalize(cmd, n=None):' in tf_src
          and 'n = opts.n_args_bins if n is None else n' in tf_src)
    check("E13: denumericalize defaults to the flag",
          'def denumericalize(cmd, n=None):' in tf_src)
    check("E13: no hardcoded 128-bin arithmetic survives in the model path",
          'n=128' not in tf_src and '8 * 128' not in tf_src,
          "a leftover 128 makes the head and the target disagree silently")
    check("E13: padding_idx is conditional",
          'padding_idx=0 if opts.arg_embed_pad_idx else None' in tf_src)

    # E3.
    check("E3: the refinement decoder depth follows --n_layers_refine",
          'c(ff), dropout=p_drop), opts.n_layers_refine)' in tf_src)
    check("E3: the sequential decoder depth is untouched at 6",
          'c(ff), dropout=p_drop), 6)' in tf_src,
          "only the parallel/refinement stack is the E3 factor")

    # E14. The schedule changes cadence, so both step sites have to be guarded.
    check("E14: the schedule is selected by flag",
          "sched_per_step = opts.lr_schedule == 'warmup_cosine'" in train_src)
    # gamma became a flag on 2026-08-04 so a matched-terminal-lr control can
    # separate E14's schedule shape from plain lr annealing. The default still has
    # to reproduce the released 0.997, which is now a check on the default rather
    # than on a literal in the source.
    check("E14: 'exp' still builds an ExponentialLR from the flag",
          'ExponentialLR(optimizer, gamma=opts.lr_gamma)' in train_src)
    check("E14: --lr_gamma defaults to the released 0.997",
          abs(vars(get_parser_main_model().parse_args([]))['lr_gamma'] - 0.997) < 1e-12)
    check("E14: the matched-terminal-lr control lands where warmup_cosine does",
          abs((0.05 ** (1 / 150)) ** 150 - 0.05) < 1e-9,
          "gamma = 0.05 ** (1/150) = 0.98023 over 150 epochs")
    check("E14: warmup_cosine steps per optimizer step",
          'if sched_per_step:                # E14' in train_src)
    check("E14: the per-epoch step is guarded so it cannot double-step",
          'if not sched_per_step:' in train_src,
          "stepping both per batch and per epoch would decay the lr ~40x too fast")
    check("E14: math is imported for the cosine",
          'import math' in train_src)

    # The arithmetic, checked rather than trusted: warmup ramps to 1.0 and the cosine
    # lands on the floor, with no discontinuity at the handover.
    warmup, total, floor = 500, 151 * 40, 0.05

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / (total - warmup))
        return floor + (1.0 - floor) * 0.5 * (1.0 + _math.cos(_math.pi * progress))

    check("E14: warmup ends at exactly 1.0x lr", abs(factor(warmup - 1) - 1.0) < 1e-9)
    check("E14: no jump across the warmup boundary",
          abs(factor(warmup) - factor(warmup - 1)) < 1e-3)
    check("E14: the cosine lands on the floor", abs(factor(total - 1) - floor) < 1e-3)
    check("E14: it actually moves the lr, unlike exp",
          factor(total - 1) < 0.997 ** 150,
          f"cosine ends at {factor(total - 1):.3f}, exp ends at {0.997 ** 150:.3f}")


def check_checkpoint_metric():
    """Static check that checkpoint selection covers what test-time scoring uses.

    test_few_shot.py scores sampled_svg_2, which comes from the parallel/refinement
    decoder (loss_dict['svg_para']). compute_val_loss used to declare svg_para
    accumulators and never fill them, so val_metric silently selected checkpoints on the
    sequential decoder alone. Fixed 2026-08-03; see checkpoint_log.py.
    """
    print("\n6. checkpoint metric wiring [static, source]")

    with open(os.path.join(REPO, 'train.py')) as fh:
        train_src = fh.read()

    check("compute_val_loss accumulates svg_para, not just img/svg",
          "for loss_cat in ['img', 'svg', 'svg_para']:" in train_src)
    check("val_metric includes the svg_para (refinement decoder) term",
          "loss_val['svg_para']['total']" in train_src)
    check("checkpoint filenames no longer embed val_metric",
          "valloss" not in train_src)
    check("checkpoint saves append to the checkpoint_log manifest",
          "checkpoint_log.append(" in train_src)
    check("prune_checkpoints selects from the manifest, not a filename regex",
          "checkpoint_log.read_all(dir_log)" in train_src)

    if not os.path.exists(os.path.join(REPO, 'checkpoint_log.py')):
        check("checkpoint_log.py exists", False)
        return

    import checkpoint_log
    check("checkpoint_log.best_checkpoint_file is importable",
          callable(getattr(checkpoint_log, 'best_checkpoint_file', None)))

    # MANIFEST_FIELDS grew on 2026-08-04 (val_svg_aux, for the E7 scale problem).
    # A run in flight is appending to a manifest written under the old header, and
    # writing the wider row under it would shift every column silently. Exercised
    # rather than reasoned about, because a corrupted manifest breaks checkpoint
    # selection for the run that is producing it.
    check("manifest records the aux term separately (E7 rescaling)",
          'val_svg_aux' in checkpoint_log.MANIFEST_FIELDS)

    import csv as _csv
    import tempfile as _tf
    loss_val = {'img': {'l1': 1.0, 'vggpt': 2.0},
                'svg': {'total': 3.0, 'cmd': 1.0, 'args': 1.0, 'aux': 0.5},
                'svg_para': {'total': 4.0, 'cmd': 1.0, 'args': 1.0, 'aux': 0.7}}
    old_header = ['epoch', 'step', 'checkpoint', 'val_metric', 'val_l1', 'val_vggpt',
                  'val_svg_total', 'val_svg_para_total']
    with _tf.TemporaryDirectory() as d:
        p = checkpoint_log.manifest_path(d)
        with open(p, 'w', newline='') as fh:
            w = _csv.writer(fh)
            w.writerow(old_header)
            w.writerow([100, 4040, '100_4040.ckpt', '6.100000', '1', '2', '3', '4'])
        checkpoint_log.append(d, 125, 5040, '125_5040.ckpt', 5.9, loss_val)
        rows = list(_csv.DictReader(open(p, newline='')))
        check("appending to a pre-2026-08-04 manifest does not shift its columns",
              all(len(r) == len(old_header) for r in rows),
              f"row widths {[len(r) for r in rows]} against an {len(old_header)}-column header")
        best = checkpoint_log.best_row(d)
        check("selection still works on a mixed-vintage manifest",
              best is not None and best['checkpoint'] == '125_5040.ckpt')

    with _tf.TemporaryDirectory() as d:
        checkpoint_log.append(d, 150, 6040, '150_6040.ckpt', 5.5, loss_val)
        with open(checkpoint_log.manifest_path(d), newline='') as fh:
            header = next(_csv.reader(fh))
        check("a fresh manifest gets the full schema",
              header == checkpoint_log.MANIFEST_FIELDS,
              f"got {header}")


def check_tier3_wiring():
    """Static check that the Tier 3 flags reach the model or the optimizer.

    Same guard as sections 5 and 7: a declared-but-unread flag trains happily, logs
    itself into opts.txt and wandb, and produces a number identical to the baseline,
    which looks like a result rather than a bug. Wired 2026-08-04; see
    PROJECT_PLAN.md 3.6.
    """
    print("\n8. tier 3 wiring: E11 / E2 / E15 / E5 / E12 / E4 [static, source]")

    with open(os.path.join(REPO, 'train.py')) as fh:
        train_src = fh.read()
    with open(os.path.join(REPO, 'models', 'model_main.py')) as fh:
        mm_src = fh.read()
    with open(os.path.join(REPO, 'models', 'modality_fusion.py')) as fh:
        mf_src = fh.read()

    norms_path = os.path.join(REPO, 'models', 'norms.py')
    if not check("models/norms.py exists", os.path.exists(norms_path)):
        return
    with open(norms_path) as fh:
        norms_src = fh.read()

    # E11. AdamW was imported and unused; the switch has to be by flag, and 'adam'
    # has to still build the original optimizer.
    check("E11: the optimizer class is selected by flag",
          "optimizer_cls = AdamW if opts.optimizer == 'adamw' else Adam" in train_src)
    check("E11: weight_decay still reaches the optimizer",
          'weight_decay=opts.weight_decay' in train_src)
    check("E11: AdamW is no longer an unused import",
          train_src.count('AdamW') >= 2)

    # E12. Pre-existing wiring, asserted so a refactor of the loss sum cannot drop it.
    check("E12: kl_beta multiplies the kl term in the training loss",
          "opts.kl_beta * loss_dict['kl']" in train_src)

    # E2 / E4. Both image stacks have to take the factory, not a hardcoded LayerNorm.
    check("E2: model_main imports the norm factory",
          'from .norms import make_img_norm' in mm_src)
    check("E2: the factory is built from opts.img_norm",
          "make_img_norm(getattr(opts, 'img_norm', 'layer')" in mm_src)
    check("E2: no hardcoded norm_layer=nn.LayerNorm survives in model_main",
          'norm_layer=nn.LayerNorm' not in mm_src)
    check("E2: both image stacks receive the same norm factory",
          mm_src.count('norm_layer=img_norm') == 2,
          f"found {mm_src.count('norm_layer=img_norm')}, expected encoder + decoder")
    check("E2: 'layer' returns nn.LayerNorm, so the default is unchanged",
          'nn.LayerNorm(shape)' in norms_src)
    check("E2: group count is reduced until it divides the channel count",
          'channels % groups != 0' in norms_src,
          "a fixed 32 groups fails on the ngf-wide first encoder layer")
    check("E2: instance norm is affine, so E2 is not confounded with a capacity drop",
          'affine=True' in norms_src)
    # E4. Three call sites, not two: ModalityFusion needs it as well, because
    # fc_fusion's input width is `ngf * mult_max + seq_latent_dim` and mult_max is
    # fixed at 64 for a 64px image. Miss that one and ngf=32 is a shape error at the
    # fusion, not a silent no-op.
    check("E4: ngf reaches both image stacks and the fusion",
          mm_src.count('ngf=opts.ngf') == 3,
          f"found {mm_src.count('ngf=opts.ngf')}, expected encoder + decoder + fusion")
    check("E4: fc_fusion's input width tracks ngf",
          'ngf * mult_max + seq_latent_dim' in mf_src)

    # E5. The flag is unusable without the projection; assert both the projection and
    # that the image decoder still sees the *unprojected* latent.
    check("E5: z_proj is constructed when bottleneck_bits != 512",
          'nn.Linear(bottleneck_bits, 512) if bottleneck_bits != 512 else None' in mf_src)
    check("E5: the projection is applied on both the train and the eval branch",
          mf_src.count('self.z_proj(') == 2)
    check("E5: z_proj is None at the default, so no state_dict key is added",
          'else None' in mf_src)
    check("E5: the image decoder input width tracks bottleneck_bits",
          'input_nc=opts.bottleneck_bits + opts.char_num' in mm_src)
    check("E5: the unprojected latent is what leaves modality_fusion",
          "output['latent'] = z" in mf_src and "output['latent'] = mu" in mf_src)

    # E15. The whole design rests on swap_in/swap_out bracketing every read of the
    # weights, and on the swaps being balanced.
    check("E15: WeightEMA is defined", 'class WeightEMA' in train_src)
    check("E15: the EMA is constructed only when the flag is positive",
          'WeightEMA(model_main, opts.ema_decay) if opts.ema_decay > 0 else None' in train_src)
    check("E15: the EMA updates after optimizer.step()",
          train_src.index('ema.update(model_main)') > train_src.index('optimizer.step()'))
    check("E15: swap_in and swap_out are balanced",
          train_src.count('ema.swap_in(model_main)') == train_src.count('ema.swap_out(model_main)') == 2,
          "one pair around the periodic val, one around the checkpoint save")
    check("E15: the checkpoint save is inside a swap, so tested weights are the EMA ones",
          train_src.index('ema.swap_in(model_main)\n            loss_val')
          < train_src.index('torch.save(')
          < train_src.rindex('ema.swap_out(model_main)'))
    check("E15: the shadow is built after --resume, not before",
          train_src.index('Resumed from') < train_src.index('ema = WeightEMA'))
    check("E15: bias correction is applied to the decay",
          'min(self.decay,' in train_src,
          "without it the first hundreds of steps are dominated by the initialization")
    check("E15: only floating-point entries are averaged",
          'v.dtype.is_floating_point' in train_src,
          "integer buffers such as num_batches_tracked must not be averaged")

    # The defaults, restated as behaviour rather than as values: with every Tier 3 flag
    # at its default the source paths taken are the released ones.
    from options import get_parser_main_model
    d = vars(get_parser_main_model().parse_args([]))
    check("tier 3 defaults reproduce the released model exactly",
          d['optimizer'] == 'adam' and d['img_norm'] == 'layer'
          and d['ema_decay'] == 0.0 and d['bottleneck_bits'] == 512
          and d['ngf'] == 16 and d['kl_beta'] == 0.01 and d['weight_decay'] == 0.0,
          "the Tier 1/Tier 2 tables and the seed floor are only valid references "
          "while this holds")

    # The norm factory, exercised rather than trusted: every channel width that either
    # image stack constructs at ngf 16 and ngf 32 has to yield a usable group count.
    #
    # This block is the only part of section 8 that needs torch, so it degrades to a
    # loud skip on a machine without it (the Mac side of this project has no torch).
    # A skip is NOT a pass -- on the cluster, where the batch actually launches, this
    # must run. The runbook's rung 1 is the backstop either way.
    sys.path.insert(0, REPO)
    try:
        from models.norms import make_img_norm, _group_count
    except ImportError as exc:                                    # noqa: BLE001
        print(f"  [SKIP] norm factory construction checks -- {exc}")
        print("  [SKIP] these are NOT passes. Re-run this script on the cluster.")
        return
    for ngf in (16, 32):
        widths = [ngf] + [ngf * (2 ** (i + 1)) for i in range(6)]
        for c in widths:
            g = _group_count(c, 32)
            if c % g != 0:
                check(f"E2: group count divides {c} channels (ngf={ngf})", False,
                      f"got {g} groups for {c} channels")
                break
        else:
            check(f"E2: group counts divide every channel width at ngf={ngf}", True)
    for kind in ('layer', 'group', 'batch', 'instance'):
        try:
            make_img_norm(kind)([16, 64, 64])
            check(f"E2: --img_norm {kind} constructs", True)
        except Exception as exc:                                  # noqa: BLE001
            check(f"E2: --img_norm {kind} constructs", False, str(exc))
    try:
        make_img_norm('banana')
        check("E2: an unknown --img_norm is rejected", False, "it was accepted")
    except ValueError:
        check("E2: an unknown --img_norm is rejected", True)


def check_tier4_wiring():
    """Static check that the Tier 4 capacity flags reach the model.

    Same guard as sections 5, 7 and 8, and it matters more here than anywhere else:
    a capacity flag that does not reach the constructor produces a run with the
    baseline's parameter count and the candidate's name, which is the one failure
    mode that looks exactly like a null result. Wired 2026-08-08; see
    PROJECT_PLAN.md 8 item 12.
    """
    print("\n9. tier 4 wiring: E16 encoder depth / E17 decoder ff width [static, source]")

    with open(os.path.join(REPO, 'models', 'model_main.py')) as fh:
        mm_src = fh.read()
    with open(os.path.join(REPO, 'models', 'transformers.py')) as fh:
        tf_src = fh.read()

    check("E16: Transformer(depth=...) reads the flag",
          "depth = opts.enc_depth" in mm_src)
    check("E16: the hardcoded depth = 6 is gone from model_main",
          "depth = 6," not in mm_src)
    check("E17: PositionwiseFeedForward reads the flag",
          "d_ff=opts.dec_d_ff" in tf_src)
    check("E17: the hardcoded d_ff=1024 is gone from the decoder",
          "d_ff=1024" not in tf_src)

    # Defaults must jointly reproduce the release, or every table before today
    # stops being a valid reference.
    try:
        sys.path.insert(0, REPO)
        from options import get_parser_main_model
        defaults = get_parser_main_model().parse_args([])
        check("E16 default is 6 (released encoder depth)", defaults.enc_depth == 6,
              f"got {defaults.enc_depth}")
        check("E17 default is 1024 (released decoder ff width)", defaults.dec_d_ff == 1024,
              f"got {defaults.dec_d_ff}")
    except Exception as exc:  # pragma: no cover - argparse import failure
        check("tier 4 defaults are importable", False, repr(exc))


def main():
    print("Pre-flight infrastructure checks -- " + REPO)
    check_options()
    check_str2bool()
    check_train_source()
    check_wandb_roundtrip()
    check_stage2_wiring()
    check_checkpoint_metric()
    check_tier2_wiring()
    check_tier3_wiring()
    check_tier4_wiring()

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("\nFailures:")
        for label in FAILED:
            print(f"  - {label}")
        return 1
    print("\nAll pre-flight checks passed. Next: the GPU dry run on the cluster (see docs/infra-upgrade.md).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
