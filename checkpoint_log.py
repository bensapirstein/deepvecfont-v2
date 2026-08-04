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

MANIFEST_FIELDS = ['epoch', 'step', 'checkpoint', 'val_metric', 'val_l1', 'val_vggpt', 'val_svg_total', 'val_svg_para_total']

# Pre-fix checkpoints (baseline, seedfloor runs) have no manifest -- only the old
# filename-embedded score. Fall back to it so those directories don't need retraining.
LEGACY_CKPT_RE = re.compile(r'^(\d+)_(\d+)_valloss([\d.]+)\.ckpt$')


def manifest_path(dir_log):
    return os.path.join(dir_log, 'checkpoint_metrics.csv')


def append(dir_log, epoch, step, checkpoint, val_metric, loss_val):
    path = manifest_path(dir_log)
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
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
