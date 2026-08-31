"""Draw the qualitative model-output figures: ground truth vs. baseline vs. E9.

Usage:  python report/make_model_comparison_figures.py
Writes: report/figures/fig7_compare.png
        report/figures/fig8_best.png
        report/figures/fig9_failures.png

Reads report/assets/model_output_renders.npz, which holds real rasters of real
model output -- the same font 0000 / char 10,20,30,40 selection, the same
best-L1 and worst-L1 Chinese glyphs docs/pull-figure-assets.md's reading rule
commits to before anything is drawn (see render_model_output.py for how the
npz was made). Nothing here re-renders an SVG or picks a glyph; it only lays
out arrays that already exist. No cairosvg dependency, so this runs anywhere
numpy does.

Reading rule, from docs/pull-figure-assets.md, applied here:
  1. Fonts/characters were chosen before any glyph was looked at (font 0000,
     chars 10/20/30/40 -- see render_model_output.py).
  2. Where E9 looks better in fig7, that is one case, not evidence: section 5.2
     of REPORT.md already finds E9's Chinese mean does not clear its floor.
  3. fig9 is the failure strip this rule requires: real worst-case Chinese
     glyphs, chosen by L1, not by eye. fig8 is its mirror at the other tail,
     chosen the same way, so the report does not show only one side of the
     distribution.
  4. Chinese and English get equal space in fig7.
  5. fig7 appears before fig8 and fig9 in REPORT.md: the qualitative baseline
     comparison comes first, then what the distribution's two tails look like.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

D = np.load(os.path.join(HERE, "assets", "model_output_renders.npz"), allow_pickle=True)
L1 = list(csv.DictReader(open(os.path.join(HERE, "assets", "model_output_l1.csv"))))

plt.rcParams.update({
    "font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
INK = "#1a1a1a"
GREY = "#9a9a9a"
E9_COLOR = "#1f6f8b"

ENG_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

COMPARE_CHARS = [int(c) for c in D["meta_compare_chars"]]
COMPARE_FONT = str(D["meta_compare_font"])


def show(ax, img, title=None):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(.5); s.set_color(GREY)
    if title:
        ax.set_title(title, fontsize=7.6, color=INK, pad=3)


def mean_l1(lang, model):
    vals = [float(r["l1"]) for r in L1 if r["lang"] == lang and r["model"] == model]
    return sum(vals) / len(vals)


# ==================================================== figure 7, GT/baseline/E9
def compare():
    n = len(COMPARE_CHARS)
    # row layout: 0 chn header, 1-3 chn GT/baseline/E9, 4 eng header, 5-7 eng images
    fig = plt.figure(figsize=(1.55 * n, 9.4))
    gs = fig.add_gridspec(8, n, height_ratios=[.42, 1, 1, 1, .42, 1, 1, 1],
                          hspace=.18, wspace=.08)

    row_labels = ["ground truth", "baseline", "+ E9 (σ=0.5)"]
    header_rows = {"chn": 0, "eng": 4}
    image_rows = {"chn": (1, 2, 3), "eng": (5, 6, 7)}

    for lang in ("chn", "eng"):
        hdr = fig.add_subplot(gs[header_rows[lang], :]); hdr.axis("off")
        lang_name = "Chinese" if lang == "chn" else "English"
        l1_bl, l1_e9 = mean_l1(lang, "baseline"), mean_l1(lang, "e9")
        hdr.text(.5, .1, f"{lang_name}  ·  font {COMPARE_FONT}  ·  "
                 f"mean L1 over these 6 fonts -- baseline {l1_bl:.3f}, +E9 {l1_e9:.3f}",
                 ha="center", va="bottom", fontsize=9.5, color=INK, weight="bold",
                 transform=hdr.transAxes)

        rows = image_rows[lang]
        for col, ci in enumerate(COMPARE_CHARS):
            gt = D[f"compare_{lang}_gt_{ci}"]
            bl = D[f"compare_{lang}_baseline_{ci}"]
            e9 = D[f"compare_{lang}_e9_{ci}"]
            label = ENG_CHARSET[ci] if lang == "eng" else f"#{ci}"
            title = f"char {label}" if lang == "eng" else f"glyph {label}"
            for r, img in zip(rows, (gt, bl, e9)):
                ax = fig.add_subplot(gs[r, col])
                show(ax, img, title=title if r == rows[0] else None)
                if col == 0:
                    ax.set_ylabel(row_labels[rows.index(r)], fontsize=7.6, color=INK)

    fig.text(.5, -.01,
             "Same font, same four characters, both languages. Not a claim: section 5.2's "
             "reading is that E9's Chinese mean does not beat the margin of error, and a "
             "picture cannot upgrade that -- this is what the numbers in section 5 look like "
             "as glyphs, nothing more.",
             ha="center", va="top", fontsize=7.6, color=GREY, wrap=True)
    fig.savefig(os.path.join(OUT, "fig7_compare.png")); plt.close(fig)


# ============================================ figure 8, best (good performance)
def best():
    meta = D["meta_best"]  # (font_idx, char_idx, l1) rows, best first
    n = len(meta)
    fig, axes = plt.subplots(3, n, figsize=(1.7 * n, 4.6))
    fig.subplots_adjust(hspace=.30, wspace=.10)

    for col, (font_idx, ci, l1) in enumerate(meta):
        ci = int(ci)
        gt = D[f"best_{font_idx}_{ci}_gt"]
        bl = D[f"best_{font_idx}_{ci}_baseline"]
        e9 = D[f"best_{font_idx}_{ci}_e9"]
        show(axes[0, col], gt, title=f"font {font_idx}, glyph #{ci}\nL1 = {float(l1):.3f}")
        show(axes[1, col], bl)
        show(axes[2, col], e9)

    for r, name in enumerate(["ground truth", "baseline (this L1)", "+ E9"]):
        axes[r, 0].set_ylabel(name, fontsize=7.6, color=INK)

    fig.suptitle("Best Chinese baseline glyphs by L1, out of the 6 decoded test fonts "
                 "(chosen by the number, not by eye)", fontsize=9, color=INK, y=1.01)
    fig.text(.5, -.04,
             "The other tail of the same distribution as the failure strip below: these are "
             "what the low end of a 0.163 Chinese mean looks like, picked by the identical "
             "rule -- lowest L1 among the same six decoded fonts, not the most flattering "
             "glyphs found by scanning. Simple strokes with little overlap area reconstruct "
             "close to the ground truth; the failure strip shows what the model does with "
             "denser, more overlapping strokes instead.",
             ha="center", fontsize=7.4, color=GREY, wrap=True)
    fig.savefig(os.path.join(OUT, "fig8_best.png")); plt.close(fig)


# ======================================================= figure 9, failures
def worst():
    meta = D["meta_failures"]  # (font_idx, char_idx, l1) rows, worst first
    n = len(meta)
    fig, axes = plt.subplots(3, n, figsize=(1.7 * n, 4.6))
    fig.subplots_adjust(hspace=.30, wspace=.10)

    for col, (font_idx, ci, l1) in enumerate(meta):
        ci = int(ci)
        gt = D[f"fail_{font_idx}_{ci}_gt"]
        bl = D[f"fail_{font_idx}_{ci}_baseline"]
        e9 = D[f"fail_{font_idx}_{ci}_e9"]
        show(axes[0, col], gt, title=f"font {font_idx}, glyph #{ci}\nL1 = {float(l1):.3f}")
        show(axes[1, col], bl)
        show(axes[2, col], e9)

    for r, name in enumerate(["ground truth", "baseline (this L1)", "+ E9"]):
        axes[r, 0].set_ylabel(name, fontsize=7.6, color=INK)

    fig.suptitle("Worst Chinese baseline glyphs by L1, out of the 6 decoded test fonts "
                 "(chosen by the number, not by eye)", fontsize=9, color=INK, y=1.01)
    fig.text(.5, -.04,
             "The reported Chinese Error is 0.163 averaged over 34 fonts; these are what "
             "0.16-ish actually looks like at the tail, not the average case.",
             ha="center", fontsize=7.4, color=GREY, wrap=True)
    fig.savefig(os.path.join(OUT, "fig9_failures.png")); plt.close(fig)


if __name__ == "__main__":
    compare()
    best()
    worst()
    print("wrote fig7_compare.png, fig8_best.png, fig9_failures.png")
