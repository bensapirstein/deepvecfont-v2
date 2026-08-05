"""
Reproduces the "reconstruction error" metric from Table 2 of the DeepVecFont-v2 paper:
the average L1 distance (on binarized 64x64 masks) between each synthesized glyph's
rasterized image and its ground-truth glyph image, using the best-of-N_s candidate
already selected by test_few_shot.py (highest IOU vs. the image decoder's own output).

Run test_few_shot.py first so that the merge HTMLs exist.

Two results layouts are supported, because test_few_shot.py changed at commit a8e3cf6:

    per-checkpoint (current)  experiments/{name_exp}/results/{name_ckpt}/{font_idx}/svgs_merge/*.html
    flat (pre-a8e3cf6)        experiments/{name_exp}/results/{font_idx}/svgs_merge/*.html

The layout is detected automatically; --name_ckpt selects one when several are present.
Numbers scored under the two layouts are NOT interchangeable: the flat tree is whatever the
last test run left behind, with no record of which checkpoint produced it.

Three metrics are reported, which is the triple DualVector (CVPR 2023) uses on this task:

    Error (L1)  the paper's number. Mean absolute difference of the two binary masks,
                i.e. the fraction of the 4096 pixels that disagree.
    s-IoU       intersection over union of the same two binary masks.
    SSIM        Wang et al. structural similarity, 11x11 Gaussian window, sigma 1.5,
                data_range 1.0, computed on the ANTI-ALIASED grayscale renders rather
                than on the binarized masks. Added 2026-08-05.

The SSIM/binarization choice is deliberate and belongs in the report. L1 and s-IoU throw
away the alpha channel before comparing, so both are blind to edge placement finer than one
pixel -- PROJECT_PLAN.md 1.2's point that sub-pixel coordinate accuracy is largely invisible
to the reported metric. SSIM on the grayscale render is the one instrument here that can see
it, because a coordinate shift of a third of a pixel changes the anti-aliased edge intensity
without flipping a single thresholded pixel. `ssim_bin`, over the same binary masks the other
two metrics use, is written to the per-font CSV alongside it: the gap between `ssim` and
`ssim_bin` is a direct measure of how much of the structural signal binarization discards.
"""
import argparse
import csv
import glob
import os
import re
import sys
from io import BytesIO

import numpy as np
from PIL import Image

try:                                    # not needed by --selftest
    import cairosvg
except ImportError:                     # pragma: no cover
    cairosvg = None

IOU_THRESH = 255 * 3 / 4

# Wang et al. 2004 defaults, matching skimage's gaussian_weights=True path.
SSIM_WIN, SSIM_SIGMA = 11, 1.5
SSIM_K1, SSIM_K2 = 0.01, 0.03


def extract_svgs(html_path):
    content = open(html_path, encoding='utf-8').read()
    return re.findall(r'<svg.*?</svg>', content, flags=re.DOTALL)


def render_svg_mask(svg_str, img_size):
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), output_width=img_size, output_height=img_size)
    arr = np.array(Image.open(BytesIO(png_bytes)))
    return 255 - arr[:, :, 3]  # matches data_utils.common_utils.trans2_white_bg


def cal_iou_l1(mask_arr1, mask_arr2):
    m1 = mask_arr1 < IOU_THRESH
    m2 = mask_arr2 < IOU_THRESH
    iou = np.sum(m1 * m2) / np.sum(m1 + m2)
    l1 = np.mean(np.abs(m1.astype(float) - m2.astype(float)))
    return iou, l1


def _gaussian_window(size=SSIM_WIN, sigma=SSIM_SIGMA):
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _filter_valid(img, kernel_1d):
    """Separable convolution, 'valid' mode -- no padding, as skimage's SSIM does.

    Implemented with numpy alone so the script keeps its two-dependency footprint
    (cairosvg, PIL) and runs unchanged in the cluster's dvf_v2 env.
    """
    k = len(kernel_1d)
    if img.shape[0] < k or img.shape[1] < k:
        raise ValueError(f"image {img.shape} smaller than the {k}x{k} SSIM window")
    # rows, then columns
    windows = np.lib.stride_tricks.sliding_window_view(img, k, axis=1)
    out = windows @ kernel_1d
    windows = np.lib.stride_tricks.sliding_window_view(out, k, axis=0)
    return np.einsum('ijk,k->ij', windows, kernel_1d)


def cal_ssim(img1, img2, data_range=1.0):
    """Mean SSIM between two float arrays. Gaussian window, no sample-covariance bias."""
    a = np.asarray(img1, dtype=np.float64)
    b = np.asarray(img2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    win = _gaussian_window()
    c1 = (SSIM_K1 * data_range) ** 2
    c2 = (SSIM_K2 * data_range) ** 2

    mu_a, mu_b = _filter_valid(a, win), _filter_valid(b, win)
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    # Population (not sample) covariance, matching use_sample_covariance=False.
    var_a = _filter_valid(a * a, win) - mu_aa
    var_b = _filter_valid(b * b, win) - mu_bb
    cov_ab = _filter_valid(a * b, win) - mu_ab

    num = (2 * mu_ab + c1) * (2 * cov_ab + c2)
    den = (mu_aa + mu_bb + c1) * (var_a + var_b + c2)
    return float(np.mean(num / den))


def _to_ink(mask_arr):
    """255-background/0-ink uint8 render -> float ink coverage in [0, 1]."""
    return (255.0 - np.asarray(mask_arr, dtype=np.float64)) / 255.0


def selftest():
    """Check cal_ssim against its defining properties and, if available, skimage."""
    rng = np.random.default_rng(1111)
    checks, failed = [], 0

    def check(name, cond):
        nonlocal failed
        checks.append((name, bool(cond)))
        if not cond:
            failed += 1

    img = rng.random((64, 64))
    check('SSIM(x, x) == 1', abs(cal_ssim(img, img) - 1.0) < 1e-9)
    check('symmetry', abs(cal_ssim(img, 1 - img) - cal_ssim(1 - img, img)) < 1e-12)
    check('identical beats perturbed',
          cal_ssim(img, img) > cal_ssim(img, np.clip(img + rng.normal(0, .1, img.shape), 0, 1)))
    check('constant images', abs(cal_ssim(np.zeros((64, 64)), np.zeros((64, 64))) - 1.0) < 1e-9)
    check('window normalized', abs(_gaussian_window().sum() - 1.0) < 1e-12)
    check("'valid' output shape", _filter_valid(img, _gaussian_window()).shape == (54, 54))

    # A glyph-like case: the two failure modes SSIM is here to separate.
    base = np.zeros((64, 64)); base[20:44, 26:32] = 1.0          # a stem
    thick = np.zeros((64, 64)); thick[20:44, 25:33] = 1.0        # uniformly too thick
    broken = base.copy(); broken[30:34, 26:32] = 0.0             # one mangled span
    l1_thick = np.mean(np.abs(base - thick))
    l1_broken = np.mean(np.abs(base - broken))
    check('constructed pair has near-equal L1', abs(l1_thick - l1_broken) < 0.006)
    check('SSIM separates them', abs(cal_ssim(base, thick) - cal_ssim(base, broken)) > 0.02)

    # Cross-check against the reference implementation on identical inputs. Verified
    # 2026-08-05 to agree to 1.1e-16 across eight noise levels and a glyph-like mask
    # pair, so this asserts a tight bound rather than a loose one. skimage is not a
    # dependency of the script; the check skips when it is absent (as on the cluster).
    try:
        from skimage.metrics import structural_similarity as sk_ssim
        worst = 0.0
        cases = [np.clip(img + rng.normal(0, s, img.shape), 0, 1) for s in (0.02, 0.1, 0.4)]
        cases.append(broken)
        for other in cases:
            a = img if other is not broken else base
            ref = sk_ssim(a, other, data_range=1.0, gaussian_weights=True,
                          sigma=SSIM_SIGMA, use_sample_covariance=False)
            worst = max(worst, abs(ref - cal_ssim(a, other)))
        check(f'matches skimage to {worst:.1e}', worst < 1e-12)
    except ImportError:
        checks.append(('skimage cross-check (not installed)', None))

    for name, ok in checks:
        print(f"  {'SKIP' if ok is None else ' OK ' if ok else 'FAIL'}  {name}")
    print(f"\n{sum(1 for _, o in checks if o)} passed, {failed} failed")
    return 1 if failed else 0


def is_font_dir(path):
    return os.path.isdir(os.path.join(path, 'svgs_merge'))


def resolve_font_dirs(exp_dir, name_ckpt=None):
    """Return (font_dirs, layout, ckpt_label), handling both results layouts."""
    results_dir = os.path.join(exp_dir, 'results')
    if not os.path.isdir(results_dir):
        raise SystemExit(f"ERROR: no results directory at {results_dir}")

    entries = sorted(p for p in glob.glob(os.path.join(results_dir, '*')) if os.path.isdir(p))
    if not entries:
        raise SystemExit(f"ERROR: {results_dir} is empty. Run test_few_shot.py first.")

    flat = [p for p in entries if is_font_dir(p)]
    if flat:
        if name_ckpt:
            print(f"WARN: --name_ckpt {name_ckpt} ignored; {results_dir} uses the flat "
                  f"(pre-a8e3cf6) layout, which does not record a checkpoint")
        return flat, 'flat', 'flatlayout'

    # Per-checkpoint layout: each entry is a checkpoint directory of font directories.
    ckpt_dirs = {os.path.basename(p): p for p in entries
                 if any(is_font_dir(q) for q in glob.glob(os.path.join(p, '*')))}
    if not ckpt_dirs:
        raise SystemExit(
            f"ERROR: no svgs_merge found under {results_dir} at either depth.\n"
            f"       Entries: {[os.path.basename(p) for p in entries]}")

    if name_ckpt:
        match = ckpt_dirs.get(name_ckpt) or ckpt_dirs.get(name_ckpt + '.ckpt')
        if match is None:
            raise SystemExit(f"ERROR: checkpoint {name_ckpt!r} not found. "
                             f"Available: {sorted(ckpt_dirs)}")
        chosen = os.path.basename(match)
    elif len(ckpt_dirs) == 1:
        chosen = next(iter(ckpt_dirs))
    else:
        raise SystemExit(f"ERROR: {len(ckpt_dirs)} checkpoints present, pass --name_ckpt. "
                         f"Available: {sorted(ckpt_dirs)}")

    font_dirs = sorted(p for p in glob.glob(os.path.join(ckpt_dirs[chosen], '*')) if is_font_dir(p))
    return font_dirs, 'per-checkpoint', chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', help='e.g. experiments/dvf_base_exp_chn_main_model')
    parser.add_argument('--name_ckpt', default=None,
                        help='checkpoint subdirectory under results/; required when several are present')
    parser.add_argument('--char_num', type=int, default=52)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--csv_out', default=None,
                        help='write per-font rows here (default: <exp_dir>/results/eval_<ckpt>.csv)')
    parser.add_argument('--selftest', action='store_true',
                        help='check the SSIM implementation and exit; needs no results tree')
    args = parser.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.exp_dir:
        parser.error('--exp_dir is required (or pass --selftest)')
    if cairosvg is None:
        raise SystemExit('ERROR: cairosvg is not installed; it is required to render candidates')

    font_dirs, layout, ckpt_label = resolve_font_dirs(args.exp_dir, args.name_ckpt)
    print(f"Layout: {layout}  |  checkpoint: {ckpt_label}  |  font dirs found: {len(font_dirs)}")
    if layout == 'flat':
        print("NOTE: flat layout. This tree predates the per-checkpoint change, so the "
              "checkpoint that produced it is not recorded. Re-run test_few_shot.py to "
              "attribute these numbers to a checkpoint.")

    all_l1, all_iou, all_ssim, all_ssim_bin, per_font = [], [], [], [], []
    n_fonts_skipped = 0
    n_glyphs_expected = n_glyphs_scored = 0

    for font_dir in font_dirs:
        merge_htmls = glob.glob(os.path.join(font_dir, 'svgs_merge', '*.html'))
        if not merge_htmls:
            print(f"WARN {font_dir}: no svgs_merge html, skipping")
            n_fonts_skipped += 1
            continue
        svgs = extract_svgs(merge_htmls[0])
        if len(svgs) != 2 * args.char_num:
            print(f"WARN {font_dir}: expected {2 * args.char_num} svgs, got {len(svgs)}, skipping")
            n_fonts_skipped += 1
            continue
        synth_svgs = svgs[:args.char_num]

        font_l1, font_iou, font_ssim, font_ssim_bin = [], [], [], []
        font_expected = font_failed = 0
        for i in range(args.char_num):
            gt_path = os.path.join(font_dir, 'imgs', f"{i:02d}_gt.png")
            if not os.path.exists(gt_path):
                continue
            font_expected += 1
            gt_mask = np.array(Image.open(gt_path))[:, :, 0]
            try:
                synth_mask = render_svg_mask(synth_svgs[i], args.img_size)
            except Exception as e:
                print(f"WARN render failed {font_dir} char {i}: {e}")
                font_failed += 1
                continue
            iou, l1 = cal_iou_l1(synth_mask, gt_mask)
            font_l1.append(l1)
            font_iou.append(iou)
            # Grayscale SSIM sees the anti-aliased edge; binary SSIM sees what L1 and
            # s-IoU see. The gap between them is reported below.
            font_ssim.append(cal_ssim(_to_ink(synth_mask), _to_ink(gt_mask)))
            font_ssim_bin.append(cal_ssim((synth_mask < IOU_THRESH).astype(float),
                                          (gt_mask < IOU_THRESH).astype(float)))

        if not font_l1:
            print(f"WARN {font_dir}: no glyph scored, skipping")
            n_fonts_skipped += 1
            continue

        n_glyphs_expected += font_expected
        n_glyphs_scored += len(font_l1)
        all_l1.extend(font_l1)
        all_iou.extend(font_iou)
        all_ssim.extend(font_ssim)
        all_ssim_bin.extend(font_ssim_bin)
        per_font.append((os.path.basename(font_dir), float(np.mean(font_l1)),
                         float(np.mean(font_iou)), float(np.mean(font_ssim)),
                         float(np.mean(font_ssim_bin)), font_expected, font_failed))

    if not per_font:
        raise SystemExit("ERROR: nothing was scored. Check the layout note above.")

    print(f"\nEvaluated {len(per_font)} fonts, {len(all_l1)} glyph samples\n")
    for name, l1, iou, ssim, ssim_bin, exp, failed in per_font:
        flag = f"  ({failed}/{exp} render fails)" if failed else ""
        print(f"  font {name}: L1={l1:.4f}  IOU={iou:.4f}  SSIM={ssim:.4f}{flag}")

    n_fonts_total = len(font_dirs)
    render_rate = n_glyphs_scored / n_glyphs_expected if n_glyphs_expected else 0.0
    font_rate = len(per_font) / n_fonts_total if n_fonts_total else 0.0

    print(f"\nReconstruction Error (L1, matches paper's 'Error'): {np.mean(all_l1):.4f}")
    print(f"Mean IOU (s-IoU): {np.mean(all_iou):.4f}")
    print(f"SSIM (grayscale, Gaussian window): {np.mean(all_ssim):.4f}")
    print(f"SSIM (binarized, same masks as L1/s-IoU): {np.mean(all_ssim_bin):.4f}  "
          f"[gap {np.mean(all_ssim_bin) - np.mean(all_ssim):+.4f}]")
    print(f"Renderability: {render_rate:.4f} glyphs ({n_glyphs_scored}/{n_glyphs_expected}), "
          f"{font_rate:.4f} fonts ({len(per_font)}/{n_fonts_total}, {n_fonts_skipped} skipped)")
    if render_rate < 1.0 or font_rate < 1.0:
        print("NOTE: renderability is below 1.0. Error is averaged over what rendered, so it is "
              "only comparable against a candidate with the same rate.")

    csv_path = args.csv_out or os.path.join(
        args.exp_dir, 'results', f"eval_{ckpt_label.replace('/', '_')}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['font', 'l1', 'iou', 'ssim', 'ssim_bin',
                    'glyphs_expected', 'glyphs_render_failed'])
        w.writerows(per_font)
    print(f"\nPer-font rows written to {csv_path}")
    print("Use these for the paired Wilcoxon against a candidate (PROJECT_PLAN.md 2.2).")


if __name__ == '__main__':
    main()
