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
MANIFEST_FIELDS = ['epoch', 'step', 'checkpoint', 'val_metric', 'val_l1', 'val_vggpt',
                   'val_svg_total', 'val_svg_para_total', 'val_svg_aux', 'val_svg_para_aux']

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


def append(dir_log, epoch, step, checkpoint, val_metric, loss_val):
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
        })


def read_all(dir_log):
    path = manifest_path(dir_log)
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def best_row(dir_log):
    """Manifest row with the lowest val_metric, or None if there is no manifest yet."""
    rows = read_all(dir_log)
    if not rows:
        return None
    return min(rows, key=lambda r: float(r['val_metric']))


def best_checkpoint_file(exp_dir):
    """Best checkpoint filename for an experiment dir (`experiments/<name>_main_model`).

    Prefers the manifest; falls back to legacy filename parsing for experiments trained
    before this manifest existed, provided the checkpoint is still on disk.
    """
    dir_log = os.path.join(exp_dir, 'logs')
    dir_ckpt = os.path.join(exp_dir, 'checkpoints')

    row = best_row(dir_log)
    if row is not None and os.path.exists(os.path.join(dir_ckpt, row['checkpoint'])):
        return row['checkpoint']

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
