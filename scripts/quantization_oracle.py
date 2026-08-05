#!/usr/bin/env python3
"""The quantization oracle: the reconstruction floor no model with this head can beat.

PROJECT_PLAN.md 2.3. Written 2026-08-05, the last open Stage 1 item.

WHAT IT MEASURES
----------------
The model predicts each coordinate as a class over a quantized grid, then
`denumericalize` maps the class back to a viewBox coordinate. Even a model that
picks the correct bin every single time therefore emits a rounded outline. This
script pushes the GROUND-TRUTH sequences through that same round trip, renders
them through the same rasterizer the eval uses, and scores them against the same
ground-truth images -- so the number it returns is the best score achievable at
that grid resolution.

    ground truth sequence  ->  numericalize(n)  ->  denumericalize(n)
                           ->  render()         ->  cal_iou_l1 / cal_ssim
                           vs  rendered_<size>.npy

TWO FLOORS, NOT ONE
-------------------
The script always evaluates `n=None` (no quantization at all) alongside the
requested grids, and this is the point of it. Rendering the true outline through
`render()` -> cairosvg and comparing against the dataset's own stored raster does
not give L1 = 0: the two rasterizers disagree at glyph edges, `render()` drops
commands past `max_seq_len`, and `_vector_to_svg` reconstructs the path from
relaxed commands. That residual is a PIPELINE floor and has nothing to do with
quantization.

    L1(n=inf)                the pipeline floor -- rasterizer and representation
    L1(n=128) - L1(n=inf)    the cost of quantization proper, at the released grid
    L1(n=128) - L1(n=256)    the most E13 could ever have bought

Reporting the raw oracle without subtracting the pipeline floor would attribute
the whole thing to quantization, which is the same class of error as anchoring
every delta on one seed. Both numbers go in PROJECT_PLAN.md 2.3 and section 5.

GRIDS
-----
    n = 256   what the paper's Sec. 3.1 text describes
    n = 128   what the released code does, and what every number in this project used
    n =  64   the grid in data_utils/relax_rep.py, ruled out as unreachable (1.5)

UNITS, WHICH ARE NOT SELF-EVIDENT
---------------------------------
`numericalize` divides by 30, but `SVG_PREFIX_BIG` in data_utils/svg_utils.py sets
`viewBox="0 0 24 24"`. The two constants are different and neither is the image
size. So one quantization step is 30/n in the numericalize domain, and the render
maps 24 of those units onto `img_size` pixels:

    bin width in px = (30 / n) * (img_size / 24)

At 64x64 that is 0.625 px for n=128, 1.25 px for n=64 and 0.3125 px for n=256,
matching PROJECT_PLAN.md 2.3's table. Using img_size/30 instead gives 0.5 px and
is wrong by a factor of 24/30; the check is in --selftest so it stays wrong-proof.
Rounding error per coordinate is uniform on +/- half a bin, so +/- 0.3125 px at the
released grid. PROJECT_PLAN.md 1.2 notes the reported metric is largely blind to
sub-pixel placement, so a small oracle gap is the expected outcome and is itself
the finding: it caps how much of the gap to 0.080 any coordinate-level change
could ever have closed.

USAGE (cluster -- needs the dataset)
------------------------------------
    python scripts/quantization_oracle.py --language chn --img_size 64
    python scripts/quantization_oracle.py --language chn --grids 64 128 256 --csv_out oracle_chn.csv
    python scripts/quantization_oracle.py --selftest        # no dataset needed
"""

import argparse
import csv
import importlib.util
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _load_eval_module():
    """Import the metrics from eval_reconstruction_error.py.

    Imported by path rather than by name so the two scripts cannot drift: the
    oracle row in the results table has to be produced by the identical L1,
    s-IoU and SSIM code that produced every model row, or it is not a floor for
    those numbers.
    """
    path = os.path.join(REPO, "eval_reconstruction_error.py")
    spec = importlib.util.spec_from_file_location("_dvf_eval", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def numericalize(cmd, n):
    """Copy of models.transformers.numericalize, on numpy, with n always explicit.

    Not imported from there because that module parses `opts` at import time and
    would pull in torch and the whole option surface. The arithmetic is asserted
    identical by --selftest.
    """
    return (cmd / 30 * n).round().clip(min=0, max=n - 1).astype(np.int64)


def denumericalize(cmd, n):
    return cmd / n * 30


# numericalize's normalizer and the SVG viewBox are different constants. See the
# module docstring; --selftest asserts the resulting pixel widths.
NUMERICALIZE_SCALE = 30.0
VIEWBOX = 24.0


def bin_px(n, img_size):
    """Width of one quantization bin, in rendered pixels."""
    return (NUMERICALIZE_SCALE / n) * (img_size / VIEWBOX)


def round_trip(args, n):
    """Quantize and de-quantize the 8 coordinate arguments. n=None means no-op."""
    if n is None:
        return args
    return denumericalize(numericalize(args, n), n)


def font_dirs(data_root, language, mode="test"):
    root = os.path.join(data_root, language, mode)
    if not os.path.isdir(root):
        raise SystemExit(f"ERROR: no dataset at {root}. This script needs the cluster.")
    dirs = sorted(
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    )
    if not dirs:
        raise SystemExit(f"ERROR: {root} has no font directories")
    return dirs


def score_font(font_dir, grids, char_num, max_seq_len, img_size, ev, render):
    """Return {grid: (l1s, ious, ssims)} for one font, or None if unusable."""
    seq_path = os.path.join(font_dir, "sequence_relaxed.npy")
    img_path = os.path.join(font_dir, f"rendered_{img_size}.npy")
    if not (os.path.exists(seq_path) and os.path.exists(img_path)):
        return None

    seq = np.load(seq_path).reshape(char_num, max_seq_len, -1).astype(np.float64)
    gt = np.load(img_path).reshape(char_num, img_size, img_size).astype(np.float64)
    # dataloader.py loads this raster, divides by 255 and inverts once; the metric's
    # ground truth image is that raster back in its stored orientation, ink low.
    if gt.max() <= 1.0 + 1e-9:
        gt = gt * 255.0

    out = {g: ([], [], []) for g in grids}
    for g in grids:
        one = seq.copy()
        one[:, :, 4:] = round_trip(one[:, :, 4:], g)
        for c in range(char_num):
            try:
                svg = render(one[c])
                synth = ev.render_svg_mask(svg, img_size)
            except Exception:
                continue
            iou, l1 = ev.cal_iou_l1(synth, gt[c])
            out[g][0].append(l1)
            out[g][1].append(iou)
            out[g][2].append(ev.cal_ssim(ev._to_ink(synth), ev._to_ink(gt[c])))
    return out


def selftest():
    """Check the round trip's arithmetic and its stated error bounds. No dataset."""
    checks, failed = [], 0

    def check(name, cond):
        nonlocal failed
        checks.append((name, bool(cond)))
        if not cond:
            failed += 1

    rng = np.random.default_rng(1111)
    x = rng.uniform(0, 30, size=(4096,))

    for n in (64, 128, 256):
        y = round_trip(x, n)
        err = np.abs(y - x)
        # Round-to-nearest on a grid of width 30/n bounds the error by half a bin,
        # except where .clip() truncates -- so exclude the top bin's overflow.
        interior = x < 30 * (n - 0.5) / n
        check(f"n={n}: max interior error <= half a bin",
              err[interior].max() <= 30 / n / 2 + 1e-9)
        check(f"n={n}: mean error near a quarter bin",
              abs(err[interior].mean() - 30 / n / 4) < 30 / n / 40)
        check(f"n={n}: output lands on the grid",
              np.allclose(y * n / 30, np.round(y * n / 30)))

    check("n=None is a no-op", np.array_equal(round_trip(x, None), x))
    check("doubling the bins halves the error",
          abs(np.abs(round_trip(x, 256) - x).mean() * 2
              - np.abs(round_trip(x, 128) - x).mean()) < 1e-3)
    check("clip destroys coordinates above 30",
          round_trip(np.array([31.0]), 128)[0] < 30.0)
    check("clip destroys coordinates below 0",
          round_trip(np.array([-1.0]), 128)[0] == 0.0)

    # Bin width in pixels at 64x64, the numbers quoted in PROJECT_PLAN.md 2.3.
    # numericalize normalizes by 30; the SVG viewBox is 24. Both appear here.
    for n, px in ((64, 1.25), (128, 0.625), (256, 0.3125)):
        check(f"n={n}: bin width is {px} px at 64x64", abs(bin_px(n, 64) - px) < 1e-9)
    check("using img_size/30 would be wrong by 24/30",
          abs(bin_px(128, 64) / (30 / 128 * 64 / 30) - 30 / 24) < 1e-9)

    try:
        ev = _load_eval_module()
        check("eval metrics import cleanly", hasattr(ev, "cal_ssim") and hasattr(ev, "cal_iou_l1"))
    except Exception as exc:                                  # pragma: no cover
        check(f"eval metrics import cleanly ({exc})", False)

    for name, ok in checks:
        print(f"  {' OK ' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(1 for _, o in checks if o)} passed, {failed} failed")
    return 1 if failed else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default=os.path.join(REPO, "data", "vecfont_dataset"))
    ap.add_argument("--language", default="chn", choices=["chn", "eng"])
    ap.add_argument("--mode", default="test")
    ap.add_argument("--grids", type=int, nargs="+", default=[64, 128, 256],
                    help="quantization grids to score; n=inf is always added")
    ap.add_argument("--char_num", type=int, default=None, help="default 52 chn / 52 eng")
    ap.add_argument("--max_seq_len", type=int, default=None, help="default 71 chn / 51 eng")
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--max_fonts", type=int, default=None, help="cap for a quick pass")
    ap.add_argument("--csv_out", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())

    char_num = args.char_num or 52
    max_seq_len = args.max_seq_len or (71 if args.language == "chn" else 51)

    ev = _load_eval_module()
    if ev.cairosvg is None:
        raise SystemExit("ERROR: cairosvg is required to render the oracle outlines")
    from data_utils.svg_utils import render

    grids = [None] + sorted(set(args.grids))
    dirs = font_dirs(args.data_root, args.language, args.mode)
    if args.max_fonts:
        dirs = dirs[: args.max_fonts]

    print(f"Oracle over {len(dirs)} {args.language} {args.mode} fonts, "
          f"{char_num} glyphs each, rendered at {args.img_size}x{args.img_size}")
    print(f"Grids: {['inf' if g is None else g for g in grids]}\n")

    totals = {g: ([], [], []) for g in grids}
    per_font_rows = []
    for i, d in enumerate(dirs):
        got = score_font(d, grids, char_num, max_seq_len, args.img_size, ev, render)
        if got is None:
            print(f"  WARN {os.path.basename(d)}: missing sequence or raster, skipped")
            continue
        for g in grids:
            for k in range(3):
                totals[g][k].extend(got[g][k])
        row = {"font": os.path.basename(d)}
        for g in grids:
            label = "inf" if g is None else g
            row[f"l1_{label}"] = float(np.mean(got[g][0])) if got[g][0] else ""
            row[f"iou_{label}"] = float(np.mean(got[g][1])) if got[g][1] else ""
            row[f"ssim_{label}"] = float(np.mean(got[g][2])) if got[g][2] else ""
        per_font_rows.append(row)
        if (i + 1) % 5 == 0:
            print(f"  ... {i + 1}/{len(dirs)} fonts")

    if not per_font_rows:
        raise SystemExit("ERROR: nothing scored")

    print(f"\n{'grid':>6}  {'bin px':>7}  {'L1':>8}  {'s-IoU':>8}  {'SSIM':>8}  {'glyphs':>7}")
    print("  " + "-" * 52)
    summary = {}
    for g in grids:
        l1s, ious, ssims = totals[g]
        label = "inf" if g is None else str(g)
        px = "-" if g is None else f"{bin_px(g, args.img_size):.4f}"
        summary[label] = (float(np.mean(l1s)), float(np.mean(ious)), float(np.mean(ssims)))
        print(f"{label:>6}  {px:>7}  {np.mean(l1s):8.4f}  {np.mean(ious):8.4f}  "
              f"{np.mean(ssims):8.4f}  {len(l1s):7d}")

    base = summary["inf"][0]
    print(f"\nPipeline floor (no quantization): L1 = {base:.4f}")
    print("This is rasterizer and representation loss, not quantization. Subtract it.")
    for g in sorted(set(args.grids), reverse=True):
        cost = summary[str(g)][0] - base
        print(f"  cost of quantization at n={g:<4} L1 = {cost:+.4f}")
    if 128 in args.grids and 256 in args.grids:
        gain = summary["128"][0] - summary["256"][0]
        print(f"\nCeiling on E13 (128 -> 256 bins): L1 = {gain:+.4f}. "
              f"Compare against the seed floor before reading anything into it.")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_font_rows[0].keys()))
            w.writeheader()
            w.writerows(per_font_rows)
        print(f"\nPer-font rows written to {args.csv_out}")


if __name__ == "__main__":
    main()
