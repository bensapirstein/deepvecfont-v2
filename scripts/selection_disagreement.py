#!/usr/bin/env python3
"""How often does val_metric pick a different checkpoint than the rendered metric does?

This is the direct answer to question 2 of the external review (docs/review-gemini.md
§5): "given that checkpoint selection was functionally random with respect to the
rendered L1 metric, how can we trust that the 0.0101 spread of the 26 candidates isn't
purely an artifact of mismatched checkpoint sampling?"

The honest answer is a measurement, not an argument. For every run trained with
--render_val_freq > 0, both columns are in `checkpoint_metrics.csv`, so the two
selections can be compared without a single GPU-second: which epoch each criterion
picks, whether they agree, and how far apart the two checkpoints score on the rendered
val metric when they disagree. That last number is the one that matters. If the
disagreement is real but costs 0.0002 of rendered L1, selection was never the problem.
If it costs more than the seed floor, the review's concern is confirmed in our own data.

No GPU, no decode. Reads the manifests only.

    python scripts/selection_disagreement.py experiments/rv_*_chn_main_model
    python scripts/selection_disagreement.py --csv_out selection_audit.csv experiments/rv_*
"""
import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import checkpoint_log

FIELDS = ['run', 'n_ckpts_scored', 'epoch_val_metric', 'epoch_render', 'agree',
          'render_l1_at_val_metric_pick', 'render_l1_at_render_pick', 'cost_of_val_metric']


def audit(exp_dir):
    dir_log = os.path.join(exp_dir, 'logs')
    rows = [r for r in checkpoint_log.read_all(dir_log) if (r.get('val_render_l1') or '').strip()]
    if not rows:
        return None
    by_val = checkpoint_log.rank(rows, 'val_metric')[0]
    by_render = checkpoint_log.rank(rows, 'val_render_l1')[0]
    l1_val = float(by_val['val_render_l1'])
    l1_render = float(by_render['val_render_l1'])
    return {
        'run': os.path.basename(exp_dir.rstrip('/')),
        'n_ckpts_scored': len(rows),
        'epoch_val_metric': by_val['epoch'],
        'epoch_render': by_render['epoch'],
        'agree': 'yes' if by_val['checkpoint'] == by_render['checkpoint'] else 'no',
        'render_l1_at_val_metric_pick': f'{l1_val:.6f}',
        'render_l1_at_render_pick': f'{l1_render:.6f}',
        # Positive = what selecting on val_metric costs, in rendered L1, against the
        # best checkpoint the run actually had. Compare it to the seed-noise floor
        # (Chinese 0.0093, English 0.0038) before calling it large or small.
        'cost_of_val_metric': f'{l1_val - l1_render:+.6f}',
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp_dirs', nargs='+')
    ap.add_argument('--csv_out', default=None)
    ap.add_argument('--floor', type=float, default=0.0093,
                    help='seed-noise floor to compare the cost against (chn 0.0093, eng 0.0038)')
    opts = ap.parse_args()

    results = []
    for d in opts.exp_dirs:
        row = audit(d)
        if row is None:
            print(f"skipping {d}: no checkpoint carries val_render_l1", file=sys.stderr)
            continue
        results.append(row)

    if not results:
        print("nothing to audit", file=sys.stderr)
        return 1

    w = max(len(r['run']) for r in results)
    print(f"{'run':<{w}}  {'n':>3}  {'ep(val)':>7}  {'ep(rnd)':>7}  {'agree':>5}  "
          f"{'L1@val':>8}  {'L1@rnd':>8}  {'cost':>9}")
    for r in results:
        print(f"{r['run']:<{w}}  {r['n_ckpts_scored']:>3}  {r['epoch_val_metric']:>7}  "
              f"{r['epoch_render']:>7}  {r['agree']:>5}  "
              f"{r['render_l1_at_val_metric_pick']:>8}  {r['render_l1_at_render_pick']:>8}  "
              f"{r['cost_of_val_metric']:>9}")

    n_dis = sum(1 for r in results if r['agree'] == 'no')
    costs = [float(r['cost_of_val_metric']) for r in results]
    mean_cost = sum(costs) / len(costs)
    over = sum(1 for c in costs if c > opts.floor)
    print(f"\ndisagreement: {n_dis}/{len(results)} runs")
    print(f"mean cost of val_metric selection: {mean_cost:+.6f} rendered L1")
    print(f"runs where that cost exceeds the {opts.floor} floor: {over}/{len(results)}")
    print("\nReading rule, fixed before the numbers: a mean cost inside the floor means "
          "checkpoint selection was\nnot what decided the sweep, and the review's question 2 "
          "is answered in the negative. A mean cost\nabove it means it was, and every "
          "val_metric-selected delta in RESULTS.csv is reported as such.")

    if opts.csv_out:
        with open(opts.csv_out, 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=FIELDS)
            wr.writeheader()
            wr.writerows(results)
        print(f"\nwrote {opts.csv_out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
