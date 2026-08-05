#!/usr/bin/env python3
"""Paired Wilcoxon signed-rank test on per-font reconstruction error.

PROJECT_PLAN.md §3.7 ends the sweep with "the confirmation eval, 34 fonts at
`n_samples 50`, on the baseline and on the combined model, with the paired
Wilcoxon." This is that test. Nothing else in the repo computes it.

Why paired, and why a rank test:

  Paired, because both models are scored on the *same* 34 fonts. Font-to-font
  difficulty dominates the spread of the per-font L1 column -- some fonts are
  simply harder to reconstruct than others, for both models -- and pairing
  removes that variance entirely instead of leaving it in the error term. An
  unpaired comparison of two 34-value columns throws away the one piece of
  structure the design has.

  A rank test, because the per-font L1 distribution is right-skewed: a handful
  of fonts carry errors several times the median, and a t-test on 34 values is
  led around by them. Wilcoxon asks the weaker question -- are the differences
  symmetrically distributed about zero -- and answers it without assuming a
  shape.

What it does NOT do: pool the three seeds into one test. Per-font differences
from different training seeds are not independent draws (same fonts, same data,
same everything but initialization), so pooling 102 rows and calling it n=102
would inflate significance by roughly the seed correlation. Each seed gets its
own test, and the across-seed summary is the sign pattern of the three -- which
is the same rule §3.6 used to promote E9 in the first place.

Usage:

    python scripts/paired_wilcoxon.py \\
        --baseline experiments/seedfloor_1111_chn_main_model/results/eval_150_6040_n50.csv \\
        --candidate experiments/e9_sigma050_chn_main_model/results/eval_150_6040_n50.csv

    # all three seeds at once, baseline/candidate paired positionally
    python scripts/paired_wilcoxon.py \\
        --baseline  .../seedfloor_1111_chn_.../eval_150_6040_n50.csv \\
                    .../seedfloor_2222_chn_.../eval_150_6040_n50.csv \\
                    .../seedfloor_3333_chn_.../eval_150_6040_n50.csv \\
        --candidate .../e9_sigma050_chn_.../eval_150_6040_n50.csv \\
                    .../e9_sigma050_2222_chn_.../eval_150_6040_n50.csv \\
                    .../e9_sigma050_3333_chn_.../eval_150_6040_n50.csv \\
        --labels 1111 2222 3333

Metric direction is handled explicitly: `--metric l1` is lower-is-better, so a
negative median difference favours the candidate; `--metric iou` is
higher-is-better and the reported direction flips with it. Say which one a
number came from, always.

No scipy dependency -- the exact null distribution is enumerated for n <= 20 and
the tie-corrected normal approximation with continuity correction is used above
that. At 34 fonts the approximation is the one in force, and it is checked
against the exact value on small fixtures by `--selftest`.
"""
import argparse
import csv
import itertools
import math
import os
import sys


def read_per_font(path, metric):
    """-> {font: value}. The CSV is eval_reconstruction_error.py's per-font output."""
    if not os.path.exists(path):
        sys.exit(f"no such CSV: {path}")
    col = {"l1": "l1", "iou": "iou"}[metric]
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            font = row["font"]
            if font in out:
                sys.exit(f"duplicate font {font!r} in {path}")
            out[font] = float(row[col])
    if not out:
        sys.exit(f"no rows in {path}")
    return out


def rankdata_average(values):
    """Ranks 1..n, ties averaged. Returns ranks in the input order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def wilcoxon_signed_rank(diffs, exact_max_n=20):
    """Two-sided Wilcoxon signed-rank on nonzero differences.

    Returns (W, n_effective, p, method). W is the smaller of the two signed rank
    sums, which is the statistic the tables are built on.

    Zero differences are dropped (Wilcoxon's original handling). With float L1
    values exact zeros essentially never occur, but dropping is stated rather
    than assumed, and n_effective reports what was actually tested.
    """
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return 0.0, 0, 1.0, "degenerate (all differences zero)"

    ranks = rankdata_average([abs(d) for d in nz])
    w_pos = sum(r for d, r in zip(nz, ranks) if d > 0)
    w_neg = sum(r for d, r in zip(nz, ranks) if d < 0)
    w = min(w_pos, w_neg)

    has_ties = len(set(abs(d) for d in nz)) != n

    if n <= exact_max_n and not has_ties:
        # Enumerate every sign assignment: under the null each |d| is equally
        # likely to have carried a + or a -.
        target = w
        count = 0
        base = list(range(1, n + 1))
        for signs in itertools.product((0, 1), repeat=n):
            s = sum(r for r, keep in zip(base, signs) if keep)
            if min(s, n * (n + 1) / 2 - s) <= target:
                count += 1
        p = count / 2 ** n
        return w, n, min(p, 1.0), f"exact (n={n})"

    mu = n * (n + 1) / 4.0
    # Tie correction on the variance, the standard form.
    tie_term = 0.0
    absd = sorted(abs(d) for d in nz)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and absd[j + 1] == absd[i]:
            j += 1
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0)
    if sigma == 0:
        return w, n, 1.0, "degenerate (zero variance)"
    z = (abs(w - mu) - 0.5) / sigma          # continuity correction
    p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    note = "normal approx, tie-corrected" + (", ties present" if has_ties else "")
    return w, n, min(max(p, 0.0), 1.0), f"{note} (n={n}, z={z:.3f})"


def hodges_lehmann(diffs):
    """Median of the Walsh averages -- the location estimate Wilcoxon inverts to.

    Reported alongside the p-value because a p-value alone says nothing about
    size, and every effect in this project lives inside a 0.0093 noise floor.
    """
    n = len(diffs)
    walsh = [(diffs[i] + diffs[j]) / 2.0 for i in range(n) for j in range(i, n)]
    walsh.sort()
    m = len(walsh)
    return walsh[m // 2] if m % 2 else (walsh[m // 2 - 1] + walsh[m // 2]) / 2.0


def sign_test(diffs):
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return pos, neg, 1.0
    k = min(pos, neg)
    p = 2 * sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return pos, neg, min(p, 1.0)


def compare_one(base_path, cand_path, metric, label):
    base = read_per_font(base_path, metric)
    cand = read_per_font(cand_path, metric)

    shared = sorted(set(base) & set(cand))
    dropped = sorted((set(base) | set(cand)) - set(shared))
    if not shared:
        sys.exit(f"[{label}] no fonts in common between the two CSVs")
    if dropped:
        # A font that failed to render in one run and not the other cannot be
        # paired. Report it loudly: silently dropping it biases the comparison
        # toward whichever model rendered the harder font.
        print(f"  ! {len(dropped)} font(s) present in only one run, excluded "
              f"from the pairing: {', '.join(dropped)}")

    diffs = [cand[f] - base[f] for f in shared]
    better = "lower" if metric == "l1" else "higher"
    favourable = (lambda d: d < 0) if metric == "l1" else (lambda d: d > 0)

    w, n_eff, p, method = wilcoxon_signed_rank(diffs)
    hl = hodges_lehmann(diffs)
    pos, neg, p_sign = sign_test(diffs)
    n_fav = sum(1 for d in diffs if favourable(d))
    mean_base = sum(base[f] for f in shared) / len(shared)
    mean_cand = sum(cand[f] for f in shared) / len(shared)

    print(f"\n[{label}]  {os.path.basename(os.path.dirname(os.path.dirname(cand_path)))}"
          f"  vs  {os.path.basename(os.path.dirname(os.path.dirname(base_path)))}")
    print(f"  fonts paired          : {len(shared)}")
    print(f"  mean {metric:<4} baseline  : {mean_base:.5f}")
    print(f"  mean {metric:<4} candidate : {mean_cand:.5f}   ({mean_cand - mean_base:+.5f})")
    print(f"  fonts favouring cand  : {n_fav}/{len(shared)}   ({better} {metric} is better)")
    print(f"  Hodges-Lehmann shift  : {hl:+.5f}   (median paired difference, cand - base)")
    print(f"  Wilcoxon W            : {w:g}")
    print(f"  Wilcoxon p (2-sided)  : {p:.4f}   [{method}]")
    print(f"  sign test p (2-sided) : {p_sign:.4f}   ({pos} up / {neg} down)")
    return {"label": label, "n": len(shared), "hl": hl, "p": p,
            "mean_delta": mean_cand - mean_base, "n_fav": n_fav}


def selftest():
    """Check the normal approximation against the exact enumeration."""
    import random
    random.seed(0)
    ok = True
    for n in (8, 12, 16, 20):
        d = [random.gauss(0.3, 1.0) for _ in range(n)]
        _, _, p_exact, m1 = wilcoxon_signed_rank(d, exact_max_n=n)
        _, _, p_approx, m2 = wilcoxon_signed_rank(d, exact_max_n=0)
        agree = abs(p_exact - p_approx) < 0.06
        ok &= agree
        print(f"  n={n:>3}  exact {p_exact:.4f}  approx {p_approx:.4f}  "
              f"{'ok' if agree else 'MISMATCH'}")
    # A known-answer case: differences all one sign -> W = 0, p = 2/2^n.
    d = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    w, _, p, _ = wilcoxon_signed_rank(d, exact_max_n=6)
    expect = 2 / 2 ** 6
    hit = w == 0 and abs(p - expect) < 1e-12
    ok &= hit
    print(f"  all-positive n=6: W={w:g} p={p:.6f} (expect 0, {expect:.6f})  "
          f"{'ok' if hit else 'MISMATCH'}")
    # Hodges-Lehmann of a symmetric set is its centre.
    hl = hodges_lehmann([-2.0, -1.0, 0.0, 1.0, 2.0])
    hit = abs(hl) < 1e-12
    ok &= hit
    print(f"  HL of symmetric set: {hl:+.6f}  {'ok' if hit else 'MISMATCH'}")
    print("selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", nargs="+", help="per-font eval CSV(s) for the baseline")
    ap.add_argument("--candidate", nargs="+", help="per-font eval CSV(s), paired positionally")
    ap.add_argument("--labels", nargs="*", default=None, help="one per pair, e.g. seed ids")
    ap.add_argument("--metric", choices=["l1", "iou"], default="l1")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.baseline or not args.candidate:
        ap.error("--baseline and --candidate are required (or use --selftest)")
    if len(args.baseline) != len(args.candidate):
        ap.error(f"{len(args.baseline)} baseline CSVs vs {len(args.candidate)} candidate CSVs")

    labels = args.labels or [str(i + 1) for i in range(len(args.baseline))]
    if len(labels) != len(args.baseline):
        ap.error("--labels must have one entry per pair")

    print(f"Paired Wilcoxon signed-rank, metric = {args.metric} "
          f"({'lower' if args.metric == 'l1' else 'higher'} is better)")

    results = [compare_one(b, c, args.metric, l)
               for b, c, l in zip(args.baseline, args.candidate, labels)]

    if len(results) > 1:
        print("\n=== across seeds ===")
        print("Each seed is its own test. These are NOT pooled: per-font differences")
        print("from different seeds share the same fonts and data, so they are not")
        print("independent, and pooling would inflate significance.\n")
        print(f"  {'seed':<8} {'n':>4} {'mean delta':>12} {'HL shift':>12} "
              f"{'fonts fav':>10} {'p':>8}")
        for r in results:
            print(f"  {r['label']:<8} {r['n']:>4} {r['mean_delta']:>+12.5f} "
                  f"{r['hl']:>+12.5f} {r['n_fav']:>10} {r['p']:>8.4f}")
        favourable = (lambda x: x < 0) if args.metric == "l1" else (lambda x: x > 0)
        signs = [favourable(r["hl"]) for r in results]
        if all(signs):
            print("\n  All seeds shift in the candidate's favour. This is the §3.6")
            print("  same-sign criterion, now at per-font resolution rather than on")
            print("  a single aggregate number per seed.")
        elif not any(signs):
            print("\n  All seeds shift against the candidate.")
        else:
            print(f"\n  Mixed sign across seeds ({sum(signs)}/{len(signs)} favourable).")
            print("  Per §3.6 that is the inconclusive shape, not a result.")


if __name__ == "__main__":
    main()
