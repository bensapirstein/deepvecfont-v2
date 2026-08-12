#!/usr/bin/env python3
"""Carve a held-out validation split out of the train split, by base font id.

Why
---
train.py builds its validation loader with `get_loader(..., 'test')`. What this codebase
calls validation has always been the test split, so `val_metric` was computed on the
scored fonts and "select the checkpoint with the best rendered metric" would, without
this script, mean selecting on the set the report reports. See `render_val.py`'s
docstring and `docs/review-response.md`.

By base font id, and why that is the whole point
------------------------------------------------
The Chinese train split is augmented in place: font `000` sits next to `000_0` ...
`000_k`, each an affine transform of the same outlines (`data_utils/augment.py`). Moving
`000` to val while leaving `000_4` in train would put a sheared copy of a validation
glyph in the training set — the selection signal would then be partly memorized, and
the protocol this whole exercise exists to fix would be broken in a subtler way than it
was before. So the unit of the split is the base id, and every augmented copy of a
held-out font leaves the train split with it.

Held-out fonts are moved *unaugmented*: their `_k` copies go to a discard directory
rather than into val. Validation measures reconstruction on real fonts, and scoring the
same font five more times under five affine transforms would just weight it five times
over.

Usage
-----
    python scripts/make_val_split.py --language chn --n_val 20          # dry run
    python scripts/make_val_split.py --language chn --n_val 20 --apply

Writes `data_splits/<lang>_val_split.json` into the repo (tracked, so which fonts were
held out survives the next `/data/bens` cleanup) and a copy beside the data. Idempotent:
re-running against an existing split verifies it instead of resampling.
"""
import argparse
import json
import os
import random
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_DIR = os.path.join(REPO, 'data_splits')

# Fixed unless you have a reason. Changing it resamples the split, which invalidates
# every checkpoint selected against the old one.
DEFAULT_SEED = 20260812


def base_id(name):
    """`000` -> `000`, `000_3` -> `000`."""
    return name.split('_')[0]


def scan(train_dir):
    """{base id: [dir names]} for every font directory in the split."""
    groups = {}
    for name in sorted(os.listdir(train_dir)):
        if not os.path.isdir(os.path.join(train_dir, name)):
            continue
        groups.setdefault(base_id(name), []).append(name)
    return groups


def choose(base_ids, n_val, seed):
    """Deterministic sample. Sorted first so filesystem order cannot leak in."""
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(base_ids), n_val))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', default=os.path.join(REPO, 'data', 'vecfont_dataset'))
    ap.add_argument('--language', default='chn', choices=['chn', 'eng'])
    ap.add_argument('--n_val', type=int, default=20,
                    help='number of BASE fonts to hold out (not directories)')
    ap.add_argument('--seed', type=int, default=DEFAULT_SEED)
    ap.add_argument('--apply', action='store_true',
                    help='actually move directories; without it this is a dry run')
    opts = ap.parse_args()

    lang_dir = os.path.join(opts.data_root, opts.language)
    train_dir = os.path.join(lang_dir, 'train')
    val_dir = os.path.join(lang_dir, 'val')
    discard_dir = os.path.join(lang_dir, '_val_aug_discard')
    manifest_repo = os.path.join(MANIFEST_DIR, f'{opts.language}_val_split.json')

    if not os.path.isdir(train_dir):
        print(f"no such train split: {train_dir}", file=sys.stderr)
        return 2

    groups = scan(train_dir)
    n_dirs = sum(len(v) for v in groups.values())
    print(f"train split: {n_dirs} directories, {len(groups)} base fonts "
          f"({n_dirs / max(1, len(groups)):.1f}x augmentation)")

    if os.path.isdir(val_dir) and os.listdir(val_dir):
        existing = sorted(d for d in os.listdir(val_dir)
                          if os.path.isdir(os.path.join(val_dir, d)))
        print(f"val split already exists: {len(existing)} fonts")
        if os.path.exists(manifest_repo):
            recorded = json.load(open(manifest_repo))['val_fonts']
            ok = sorted(recorded) == existing
            print(f"manifest {'matches' if ok else 'DOES NOT MATCH'} {manifest_repo}")
            return 0 if ok else 1
        print(f"no manifest at {manifest_repo}; not touching an existing split",
              file=sys.stderr)
        return 1

    if opts.n_val >= len(groups):
        print(f"--n_val {opts.n_val} >= {len(groups)} base fonts", file=sys.stderr)
        return 2

    val_fonts = choose(groups.keys(), opts.n_val, opts.seed)
    moved_val = [(f, os.path.join(train_dir, f), os.path.join(val_dir, f)) for f in val_fonts]
    moved_discard = [(d, os.path.join(train_dir, d), os.path.join(discard_dir, d))
                     for f in val_fonts for d in groups[f] if d != f]

    remaining = sum(len(groups[f]) for f in groups if f not in set(val_fonts))
    print(f"\nhold out {len(val_fonts)} base fonts -> {val_dir}")
    print(f"  {val_fonts[:6]}{' ...' if len(val_fonts) > 6 else ''}")
    print(f"discard {len(moved_discard)} augmented copies of them -> {discard_dir}")
    print(f"train keeps {remaining} directories over {len(groups) - len(val_fonts)} base fonts")

    if not opts.apply:
        print("\ndry run. re-run with --apply to move anything.")
        return 0

    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(discard_dir, exist_ok=True)
    for _, src, dst in moved_val + moved_discard:
        shutil.move(src, dst)

    payload = {
        'language': opts.language,
        'seed': opts.seed,
        'n_val': opts.n_val,
        'val_fonts': val_fonts,
        'train_base_fonts_remaining': len(groups) - len(val_fonts),
        'augmented_copies_discarded': len(moved_discard),
    }
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    for path in (manifest_repo, os.path.join(lang_dir, 'val_split.json')):
        with open(path, 'w') as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {path}")

    print(f"\ndone. {discard_dir} is safe to delete once you have checked the split.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
