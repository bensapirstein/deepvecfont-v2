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
"""
import argparse
import csv
import glob
import os
import re
from io import BytesIO

import cairosvg
import numpy as np
from PIL import Image

IOU_THRESH = 255 * 3 / 4


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
    parser.add_argument('--exp_dir', required=True, help='e.g. experiments/dvf_base_exp_chn_main_model')
    parser.add_argument('--name_ckpt', default=None,
                        help='checkpoint subdirectory under results/; required when several are present')
    parser.add_argument('--char_num', type=int, default=52)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--csv_out', default=None,
                        help='write per-font rows here (default: <exp_dir>/results/eval_<ckpt>.csv)')
    args = parser.parse_args()

    font_dirs, layout, ckpt_label = resolve_font_dirs(args.exp_dir, args.name_ckpt)
    print(f"Layout: {layout}  |  checkpoint: {ckpt_label}  |  font dirs found: {len(font_dirs)}")
    if layout == 'flat':
        print("NOTE: flat layout. This tree predates the per-checkpoint change, so the "
              "checkpoint that produced it is not recorded. Re-run test_few_shot.py to "
              "attribute these numbers to a checkpoint.")

    all_l1, all_iou, per_font = [], [], []
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

        font_l1, font_iou = [], []
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

        if not font_l1:
            print(f"WARN {font_dir}: no glyph scored, skipping")
            n_fonts_skipped += 1
            continue

        n_glyphs_expected += font_expected
        n_glyphs_scored += len(font_l1)
        all_l1.extend(font_l1)
        all_iou.extend(font_iou)
        per_font.append((os.path.basename(font_dir), float(np.mean(font_l1)),
                         float(np.mean(font_iou)), font_expected, font_failed))

    if not per_font:
        raise SystemExit("ERROR: nothing was scored. Check the layout note above.")

    print(f"\nEvaluated {len(per_font)} fonts, {len(all_l1)} glyph samples\n")
    for name, l1, iou, exp, failed in per_font:
        flag = f"  ({failed}/{exp} render fails)" if failed else ""
        print(f"  font {name}: L1={l1:.4f}  IOU={iou:.4f}{flag}")

    n_fonts_total = len(font_dirs)
    render_rate = n_glyphs_scored / n_glyphs_expected if n_glyphs_expected else 0.0
    font_rate = len(per_font) / n_fonts_total if n_fonts_total else 0.0

    print(f"\nReconstruction Error (L1, matches paper's 'Error'): {np.mean(all_l1):.4f}")
    print(f"Mean IOU (s-IoU): {np.mean(all_iou):.4f}")
    print(f"Renderability: {render_rate:.4f} glyphs ({n_glyphs_scored}/{n_glyphs_expected}), "
          f"{font_rate:.4f} fonts ({len(per_font)}/{n_fonts_total}, {n_fonts_skipped} skipped)")
    if render_rate < 1.0 or font_rate < 1.0:
        print("NOTE: renderability is below 1.0. Error is averaged over what rendered, so it is "
              "only comparable against a candidate with the same rate.")

    csv_path = args.csv_out or os.path.join(
        args.exp_dir, 'results', f"eval_{ckpt_label.replace('/', '_')}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['font', 'l1', 'iou', 'glyphs_expected', 'glyphs_render_failed'])
        w.writerows(per_font)
    print(f"\nPer-font rows written to {csv_path}")
    print("Use these for the paired Wilcoxon against a candidate (PROJECT_PLAN.md 2.2).")


if __name__ == '__main__':
    main()
