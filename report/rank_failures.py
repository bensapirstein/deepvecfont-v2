"""Rank glyphs in report/assets/model_output/ by per-glyph reconstruction error.

Usage (cluster only, needs cairosvg):  python report/rank_failures.py
Writes: report/assets/model_output_l1.csv

This exists so the failure-strip figure (make_model_comparison_figures.py) can pick
its glyphs from a number, not from scanning by eye -- the reading rule in
docs/pull-figure-assets.md forbids choosing the flattering examples, and the same
rule cuts the other way for a failure strip: it should not be hand-picked either.

Reuses cal_iou_l1 and render_svg_mask from eval_reconstruction_error.py directly, so
a glyph's L1 here is computed by the identical formula as every number in RESULTS.csv.
--gt_source svg (the ground-truth outline through the same rasterizer as the
candidate): the merge HTML holds both halves, so this needs nothing beyond what is
already committed. This script needs cairosvg and only runs where that is installed
(the cluster's dvf_v2 env); the CSV it writes is what the figure script actually reads,
and that one is plain csv + matplotlib.
"""
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, from report/
sys.path.insert(0, ROOT)
from eval_reconstruction_error import extract_svgs, render_svg_mask, cal_iou_l1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets", "model_output")
OUT = os.path.join(HERE, "assets", "model_output_l1.csv")

# (language, model label, experiment dir name, checkpoint dir name, char_num)
RUNS = [
    ("chn", "baseline", "seedfloor_1111_chn_main_model", "150_6040_ckpt", 52),
    ("chn", "e9", "e9_sigma050_chn_main_model", "150_6040_ckpt", 52),
    ("eng", "baseline", "eng_seedfloor_1111_main_model", "640_205761_ckpt", 52),
    ("eng", "e9", "e9_sigma050_1111_eng_main_model", "640_205761_ckpt", 52),
]


def main():
    rows = []
    for lang, model, exp, ckpt, char_num in RUNS:
        font_dirs = sorted(glob.glob(os.path.join(ASSETS, exp, "results", ckpt, "*")))
        for font_dir in font_dirs:
            font_idx = os.path.basename(font_dir)
            htmls = glob.glob(os.path.join(font_dir, "svgs_merge", "*.html"))
            if not htmls:
                continue
            svgs = extract_svgs(htmls[0])
            if len(svgs) != 2 * char_num:
                # Matches eval_reconstruction_error.py's own skip-and-warn: a font that
                # failed to render (e.g. eng font 0004 for e9, empty html on disk since
                # 2026-08-08) is a real renderability failure, not a bug in this script.
                print(f"WARN {font_dir}: expected {2 * char_num} svgs, got {len(svgs)}, skipping")
                continue
            synth_svgs, gt_svgs = svgs[:char_num], svgs[char_num:]
            for i in range(char_num):
                synth_mask = render_svg_mask(synth_svgs[i], 64)
                gt_mask = render_svg_mask(gt_svgs[i], 64)
                iou, l1 = cal_iou_l1(synth_mask, gt_mask)
                rows.append((lang, model, font_idx, i, f"{l1:.5f}", f"{iou:.5f}"))
        print(f"{lang}/{model}: {len(font_dirs)} fonts scored")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lang", "model", "font_idx", "char_idx", "l1", "iou"])
        w.writerows(rows)
    print(f"wrote {OUT}, {len(rows)} rows")


if __name__ == "__main__":
    main()
