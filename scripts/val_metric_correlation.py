"""Does `val_metric` predict the rendered screening Error?

PROJECT_PLAN.md §3.2 has carried this as an open question since day 1, and §4 put it
on day 5. It matters more now than when it was written, for a specific reason: the
two instruments have very different noise.

    rendered Error   stochastic best-of-n_samples decoding, unseeded.
                     Measured decode noise 0.0011, on top of a 0.0093 seed floor.
    val_metric       deterministic given a checkpoint, computed over the whole
                     validation set, and logged at every checkpoint rather than once.

So `val_metric` is the lower-variance instrument by a wide margin, and a consistent
separation in a wandb val curve can look far more convincing than a single rendered
number that is buried in noise. That is only useful if the two agree about *which*
checkpoint is better. This script measures whether they do.

    python scripts/val_metric_correlation.py
    python scripts/val_metric_correlation.py --exclude e8 e13   # loss-scale changers

Reads, per experiment under `experiments/`:
    logs/checkpoint_metrics.csv     -> val_metric of the best (selected) checkpoint
    results/eval_*.csv              -> per-font rendered L1, averaged

Reports Spearman and Kendall rank correlation, and the per-candidate residual, so a
single disagreeing candidate is visible rather than averaged away.

## The caveat that decides how to read the output

E8 and E13 change the cross-entropy itself, so their `val_metric` is on a different
scale by construction and they are not comparable to anything else. Pass them to
`--exclude`; the default already does.

E14 is the interesting case and is *not* excluded, because it changes no loss term at
all -- only the LR schedule -- so its `val_metric` is directly comparable. Read its
residual specifically. A warmup_cosine run ends at 0.05x the base lr where the
released ExponentialLR ends at 0.635x, a 12.7x difference in terminal step size.
Late-training validation loss falls almost mechanically as the lr decays and the
weights stop bouncing around the minimum, and that happens whether or not
autoregressive rollout quality improves. `val_metric` is teacher-forced; the rendered
metric is an autoregressive rollout through the refinement decoder, and teacher
forcing cannot see exposure bias. So E14 is exactly the candidate where a lower val
curve is most likely to be measuring the schedule rather than the model.

If E14 is a large residual and the rest correlate well, the conclusion is not "the
proxy is broken" -- it is "the proxy is usable except across LR schedules", which is
a sharper and more reportable finding than either alone.
"""

import argparse
import csv
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ranks(xs):
    """Ranks, averaging ties -- required for Spearman to be correct on ties."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float('nan')


def spearman(xs, ys):
    return _pearson(_ranks(xs), _ranks(ys))


def kendall_tau(xs, ys):
    n, conc, disc = len(xs), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = xs[i] - xs[j], ys[i] - ys[j]
            if a == 0 or b == 0:
                continue
            if (a > 0) == (b > 0):
                conc += 1
            else:
                disc += 1
    total = conc + disc
    return (conc - disc) / total if total else float('nan')


def best_val_metric(exp_dir):
    path = os.path.join(exp_dir, 'logs', 'checkpoint_metrics.csv')
    if not os.path.exists(path):
        return None, None
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    best = min(rows, key=lambda r: float(r['val_metric']))
    return float(best['val_metric']), best['checkpoint']


def rendered_l1(exp_dir):
    """Mean per-font L1 from the most recent eval CSV in results/."""
    paths = sorted(glob.glob(os.path.join(exp_dir, 'results', 'eval_*.csv')),
                   key=os.path.getmtime)
    if not paths:
        return None, None
    with open(paths[-1], newline='') as f:
        rows = list(csv.reader(f))
    if len(rows) < 2:
        return None, None
    header, body = rows[0], rows[1:]
    # eval_reconstruction_error.py writes (font, l1, iou, expected, failed).
    try:
        col = next(i for i, h in enumerate(header) if h.strip().lower() in ('l1', 'mean_l1'))
    except StopIteration:
        col = 1
    vals = [float(r[col]) for r in body if len(r) > col and r[col].strip()]
    if not vals:
        return None, None
    return sum(vals) / len(vals), os.path.basename(paths[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiments_root', default=os.path.join(REPO, 'experiments'))
    ap.add_argument('--exclude', nargs='*', default=['e8', 'e13'],
                    help='name prefixes to drop; default drops the two candidates that '
                         'change the cross-entropy scale, whose val_metric is not '
                         'comparable to anything else')
    ap.add_argument('--flag', nargs='*', default=['e14'],
                    help='name prefixes to keep but report separately as residuals')
    args = ap.parse_args()

    if not os.path.isdir(args.experiments_root):
        sys.exit(f"no such directory: {args.experiments_root}\n"
                 f"Run this on the cluster, where experiments/ is populated.")

    collected, skipped = [], []
    for d in sorted(os.listdir(args.experiments_root)):
        exp_dir = os.path.join(args.experiments_root, d)
        if not os.path.isdir(exp_dir) or d.startswith('archive'):
            continue
        name = d[:-len('_main_model')] if d.endswith('_main_model') else d
        vm, ckpt = best_val_metric(exp_dir)
        l1, src = rendered_l1(exp_dir)
        if vm is None or l1 is None:
            skipped.append((name, 'no manifest' if vm is None else 'no eval csv'))
            continue
        collected.append({'name': name, 'val_metric': vm, 'l1': l1,
                          'ckpt': ckpt, 'eval': src})

    if skipped:
        print("Skipped (incomplete):")
        for n, why in skipped:
            print(f"  {n:<28} {why}")
        print()

    def excluded(n):
        return any(n.lower().startswith(p.lower()) for p in args.exclude)

    def flagged(n):
        return any(n.lower().startswith(p.lower()) for p in args.flag)

    used = [r for r in collected if not excluded(r['name'])]
    if len(used) < 4:
        sys.exit(f"only {len(used)} usable runs; need at least 4 for this to mean anything")

    print(f"{'run':<28}{'val_metric':>12}{'rendered L1':>13}{'vm rank':>9}{'L1 rank':>9}{'resid':>8}")
    print("-" * 79)
    vms = [r['val_metric'] for r in used]
    l1s = [r['l1'] for r in used]
    vr, lr_ = _ranks(vms), _ranks(l1s)
    for r, a, b in sorted(zip(used, vr, lr_), key=lambda t: t[1]):
        mark = '  <-- flagged' if flagged(r['name']) else ''
        print(f"{r['name']:<28}{r['val_metric']:>12.4f}{r['l1']:>13.4f}"
              f"{a:>9.1f}{b:>9.1f}{b - a:>+8.1f}{mark}")

    rho, tau = spearman(vms, l1s), kendall_tau(vms, l1s)
    print("-" * 79)
    print(f"\nn = {len(used)} runs (excluded: {', '.join(args.exclude) or 'none'})")
    print(f"Spearman rho = {rho:+.3f}")
    print(f"Kendall  tau = {tau:+.3f}")

    # Drop the flagged runs and recompute, so a single schedule-changing outlier is
    # visible as a shift rather than silently dragging the whole coefficient down.
    rest = [r for r in used if not flagged(r['name'])]
    if len(rest) >= 4 and len(rest) < len(used):
        rho2 = spearman([r['val_metric'] for r in rest], [r['l1'] for r in rest])
        print(f"Spearman rho = {rho2:+.3f}  excluding {', '.join(args.flag)} "
              f"(n = {len(rest)})")
        print(f"  -> the flagged runs move rho by {rho2 - rho:+.3f}")

    print("""
How to read this:

  rho >= 0.7 and the flagged residuals small
      val_metric is a usable screen. Screen the remaining candidates on it for
      free, and spend the saved GPU time on more candidates or more seeds.

  rho >= 0.7 but E14 a large residual
      The proxy works except across LR schedules, which is the most likely
      outcome and the most useful one. It means the wandb val curves can be
      trusted for candidates that share the baseline schedule, and cannot be
      used to compare a schedule change against it. Say exactly that in §6.

  rho < 0.4
      val_metric does not predict the rendered metric on this model. That is a
      reportable finding in its own right -- it says the teacher-forced loss and
      the autoregressive rollout diverge -- and it means every remaining
      candidate screens on the rendered metric.

Negative rho would mean the two disagree systematically; check the sign
convention before believing it, since both metrics are lower-is-better and a
positive correlation is the expected direction.
""")


if __name__ == '__main__':
    main()
