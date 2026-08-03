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
        check("wandb.init project is 'deepvecfont-v2'",
              any(k.arg == 'project' and getattr(k.value, 'value', None) == 'deepvecfont-v2'
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


def main():
    print("Pre-flight infrastructure checks -- " + REPO)
    check_options()
    check_str2bool()
    check_train_source()
    check_wandb_roundtrip()
    check_stage2_wiring()

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
