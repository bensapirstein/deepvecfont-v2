"""Rasterize the glyphs the two model-comparison figures need, cluster-side.

Usage (cluster only, needs cairosvg):  python report/render_model_output.py
Writes: report/assets/model_output_renders.npz

Why this exists rather than parsing the SVG path data into a matplotlib Path
directly (as make_glyph_figures.py does for the dataset's own command sequences):
a from-scratch SVG-path-fill implementation was tried and checked against
render_svg_mask on a handful of real glyphs, and it disagreed by as much as 42% of
pixels on paths with several overlapping subpaths, which Chinese glyphs have
routinely. The reading rule in docs/pull-figure-assets.md is that the figure and
the table cannot disagree; an approximate renderer risks exactly that. Reusing
render_svg_mask -- the same function every number in RESULTS.csv goes through --
removes the risk entirely, at the cost of needing cairosvg here, once, on the
cluster. The npz this writes is plain uint8 arrays, so the figure script that
reads it (make_model_comparison_figures.py) needs nothing beyond numpy.
"""
import glob
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, from report/
sys.path.insert(0, ROOT)
from eval_reconstruction_error import extract_svgs, render_svg_mask  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets", "model_output")
OUT = os.path.join(HERE, "assets", "model_output_renders.npz")
RES = 256  # print-quality; the reported metric itself renders at 64, this is display only

RUN_DIRS = {
    ("chn", "baseline"): ("seedfloor_1111_chn_main_model", "150_6040_ckpt"),
    ("chn", "e9"): ("e9_sigma050_chn_main_model", "150_6040_ckpt"),
    ("eng", "baseline"): ("eng_seedfloor_1111_main_model", "640_205761_ckpt"),
    ("eng", "e9"): ("e9_sigma050_1111_eng_main_model", "640_205761_ckpt"),
}
CHAR_NUM = 52

# The "compare" figure: font 0000 (first test font, i.e. no selection at all),
# characters 10/20/30/40 -- fixed before any glyph was looked at, and clear of
# every reference-character id used at decode time (chn 0,1,2,3,26,27,28,29;
# eng 0,1,26,27), so none of these is a glyph the model was simply handed.
COMPARE_FONT = "0000"
COMPARE_CHARS = [10, 20, 30, 40]

# The "failures" figure: the worst-L1 Chinese baseline glyphs, per
# report/assets/model_output_l1.csv (rank_failures.py) -- a number choosing the
# examples, not a person scanning for the ugliest ones.
N_FAILURES = 4


def load_pair(lang, model, font_idx):
    exp, ckpt = RUN_DIRS[(lang, model)]
    font_dir = os.path.join(ASSETS, exp, "results", ckpt, font_idx)
    htmls = glob.glob(os.path.join(font_dir, "svgs_merge", "*.html"))
    if not htmls:
        return None
    svgs = extract_svgs(htmls[0])
    if len(svgs) != 2 * CHAR_NUM:
        return None
    return svgs[:CHAR_NUM], svgs[CHAR_NUM:]


def worst_chn_baseline_glyphs(n):
    import csv
    csv_path = os.path.join(HERE, "assets", "model_output_l1.csv")
    rows = [r for r in csv.DictReader(open(csv_path))
            if r["lang"] == "chn" and r["model"] == "baseline"]
    rows.sort(key=lambda r: -float(r["l1"]))
    return [(r["font_idx"], int(r["char_idx"]), float(r["l1"])) for r in rows[:n]]


def main():
    out = {}
    meta_compare = []
    meta_failures = []

    # -- compare figure: font 0000, both languages, both models, fixed chars --
    for lang in ("chn", "eng"):
        pairs = {model: load_pair(lang, model, COMPARE_FONT) for model in ("baseline", "e9")}
        for ci in COMPARE_CHARS:
            gt_svg = pairs["baseline"][1][ci]  # GT is identical across models; take either
            out[f"compare_{lang}_gt_{ci}"] = render_svg_mask(gt_svg, RES)
            for model in ("baseline", "e9"):
                synth_svg = pairs[model][0][ci]
                out[f"compare_{lang}_{model}_{ci}"] = render_svg_mask(synth_svg, RES)
        meta_compare.append((lang, COMPARE_FONT, COMPARE_CHARS))
        print(f"compare: {lang} font {COMPARE_FONT}, chars {COMPARE_CHARS}")

    # -- failures figure: worst Chinese baseline glyphs, baseline + E9 + GT ------
    worst = worst_chn_baseline_glyphs(N_FAILURES)
    pairs_by_font = {}
    for font_idx, ci, l1 in worst:
        for model in ("baseline", "e9"):
            if (font_idx, model) not in pairs_by_font:
                pairs_by_font[(font_idx, model)] = load_pair("chn", model, font_idx)
            synth_svgs, gt_svgs = pairs_by_font[(font_idx, model)]
            key_model = f"fail_{font_idx}_{ci}_{model}"
            out[key_model] = render_svg_mask(synth_svgs[ci], RES)
        out[f"fail_{font_idx}_{ci}_gt"] = render_svg_mask(
            pairs_by_font[(font_idx, "baseline")][1][ci], RES)
        meta_failures.append((font_idx, ci, l1))
        print(f"failure: font {font_idx} char {ci}, baseline L1={l1:.4f}")

    out["meta_compare_chars"] = np.array(COMPARE_CHARS)
    out["meta_compare_font"] = np.array(COMPARE_FONT)
    out["meta_failures"] = np.array(meta_failures, dtype=object)

    np.savez_compressed(OUT, **out)
    size_kb = os.path.getsize(OUT) / 1024
    print(f"wrote {OUT}, {len(out)} arrays, {size_kb:.0f} KB")


if __name__ == "__main__":
    main()
