"""Per-experiment checkpoint metrics manifest.

Checkpoint filenames used to embed val_metric (`{epoch}_{step}_valloss{x}.ckpt`), so
selecting the best one meant parsing it back out of a string. That breaks the moment two
things drift apart -- which they did: the embedded number and the number that should have
governed selection were computed from different code paths.

Instead, every checkpoint save appends a row to `<dir_log>/checkpoint_metrics.csv`, and
`best_checkpoint_file` reads that manifest to answer "best so far". Filenames go back to
being plain `{epoch}_{step}.ckpt`.
"""
import csv
import os
import re

# val_svg_aux / val_svg_para_aux added 2026-08-04. Reason: loss_w_aux (E7) is a
# weight *inside* val_metric, via svg['total'] and svg_para['total'] both, so an E7
# run's val_metric is on a different scale from the baseline's and the two cannot be
# compared. Logging the aux term separately makes that rescalable after the fact.
# Runs trained before this date cannot be corrected -- the term was never persisted.
# val_render_* added 2026-08-12. These are the rendered Error and s-IoU on the held-out
# val split -- the quantity the report reports, not a proxy for it. Everything to their
# left is a training-loss term. They are blank on runs trained with --render_val_freq 0,
# which is every run before this date, and `best_row` refuses to rank on a blank column
# rather than silently falling back to val_metric. See render_val.py.
MANIFEST_FIELDS = ['epoch', 'step', 'checkpoint', 'val_metric', 'val_l1', 'val_vggpt',
                   'val_svg_total', 'val_svg_para_total', 'val_svg_aux', 'val_svg_para_aux',
                   'val_render_l1', 'val_render_siou', 'val_render_renderability']

# Lower is better for these, higher for these. Selection has to know which.
CRITERIA_LOWER = ('val_metric', 'val_render_l1')
CRITERIA_HIGHER = ('val_render_siou',)

# Pre-fix checkpoints (baseline, seedfloor runs) have no manifest -- only the old
# filename-embedded score. Fall back to it so those directories don't need retraining.
LEGACY_CKPT_RE = re.compile(r'^(\d+)_(\d+)_valloss([\d.]+)\.ckpt$')


def manifest_path(dir_log):
    return os.path.join(dir_log, 'checkpoint_metrics.csv')


def existing_fields(path):
    """Header of an existing manifest, or None.

    MANIFEST_FIELDS grew on 2026-08-04. A manifest written under the old header is
    still open and being appended to by any run in flight, and writing the new,
    wider row under the old header would silently shift every column. So an existing
    file keeps its own header and the extra columns are dropped for that file; only
    fresh manifests get the full schema.
    """
    if not os.path.exists(path):
        return None
    with open(path, newline='') as f:
        header = next(csv.reader(f), None)
    return header or None


def append(dir_log, epoch, step, checkpoint, val_metric, loss_val, render=None):
    path = manifest_path(dir_log)
    fields = existing_fields(path)
    write_header = fields is None
    if write_header:
        fields = MANIFEST_FIELDS
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow({
            'epoch': epoch,
            'step': step,
            'checkpoint': checkpoint,
            'val_metric': f'{val_metric:.6f}',
            'val_l1': f'{float(loss_val["img"]["l1"]):.6f}',
            'val_vggpt': f'{float(loss_val["img"]["vggpt"]):.6f}',
            'val_svg_total': f'{float(loss_val["svg"]["total"]):.6f}',
            'val_svg_para_total': f'{float(loss_val["svg_para"]["total"]):.6f}',
            'val_svg_aux': f'{float(loss_val["svg"]["aux"]):.6f}',
            'val_svg_para_aux': f'{float(loss_val["svg_para"]["aux"]):.6f}',
            'val_render_l1': '' if render is None else f'{render["l1"]:.6f}',
            'val_render_siou': '' if render is None else f'{render["siou"]:.6f}',
            'val_render_renderability': '' if render is None else f'{render["renderability"]:.6f}',
        })


def read_all(dir_log):
    path = manifest_path(dir_log)
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def rank(rows, criterion='val_metric'):
    """Manifest rows sorted best-first on `criterion`, dropping rows that lack a value.

    Dropping rather than defaulting is deliberate. A run trained with
    --render_val_freq 0 has no val_render_l1 at all, and a run with
    --render_val_freq 50 against --freq_ckpt 25 has one on every other checkpoint. In
    both cases treating a blank as 0.0 would rank an unmeasured checkpoint first, which
    is the failure mode that makes a selection bug look like a good result.
    """
    if criterion not in CRITERIA_LOWER + CRITERIA_HIGHER:
        raise ValueError(f"unknown selection criterion {criterion!r}")
    sign = -1.0 if criterion in CRITERIA_HIGHER else 1.0
    scored = []
    for r in rows:
        raw = (r.get(criterion) or '').strip()
        if not raw:
            continue
        scored.append((sign * float(raw), r))
    scored.sort(key=lambda pair: pair[0])
    return [r for _, r in scored]


def best_row(dir_log, criterion='val_metric'):
    """Best manifest row on `criterion`, or None if no row carries that column."""
    ranked = rank(read_all(dir_log), criterion)
    return ranked[0] if ranked else None


def best_checkpoint_file(exp_dir, criterion='val_metric'):
    """Best checkpoint filename for an experiment dir (`experiments/<name>_main_model`).

    Prefers the manifest; falls back to legacy filename parsing for experiments trained
    before this manifest existed, provided the checkpoint is still on disk. The legacy
    fallback only ever knew val_metric, so asking for a rendered criterion and getting
    the fallback is an error rather than a silent substitution.
    """
    dir_log = os.path.join(exp_dir, 'logs')
    dir_ckpt = os.path.join(exp_dir, 'checkpoints')

    # Walk the ranking rather than testing only its first row. The manifest is
    # append-only and outlives pruning, so the best-scoring checkpoint may well have
    # been deleted -- pruning under one criterion, then selecting under another, is
    # exactly how that happens. Returning the best SURVIVING checkpoint is the right
    # answer; giving up because the very best one is gone is not.
    ranked = rank(read_all(dir_log), criterion)
    for row in ranked:
        if os.path.exists(os.path.join(dir_ckpt, row['checkpoint'])):
            return row['checkpoint']

    if criterion != 'val_metric':
        if ranked:
            raise FileNotFoundError(
                f"{len(ranked)} checkpoints in {dir_log} carry a {criterion} value but "
                f"none of them is still on disk in {dir_ckpt}. The run was pruned under a "
                f"different criterion, or --max_ckpt_keep was too small."
            )
        raise FileNotFoundError(
            f"no checkpoint in {dir_log} carries a {criterion} value. Either the run was "
            f"trained with --render_val_freq 0, or its rendered-validation pass never ran. "
            f"Do not fall back to val_metric here -- that is the selection this criterion exists to replace."
        )

    candidates = []
    if os.path.isdir(dir_ckpt):
        for fname in os.listdir(dir_ckpt):
            m = LEGACY_CKPT_RE.match(fname)
            if m:
                candidates.append((float(m.group(3)), fname))
    if not candidates:
        raise FileNotFoundError(
            f"no checkpoint_metrics.csv in {dir_log} and no legacy-named checkpoints in {dir_ckpt}"
        )
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]
