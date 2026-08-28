"""Draw the two explanatory figures from real test-split glyphs.

Usage:  python report/make_glyph_figures.py
Writes: report/figures/fig1_teaser.png
        report/figures/fig2_anatomy.png

Both are built from assets/glyphs.npz, which holds genuine outlines from the
English test split (see extract_assets.py). Nothing here is drawn by hand or
traced; every curve on the page is the command sequence the model consumes.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

D = np.load(os.path.join(HERE, "assets", "glyphs.npz"), allow_pickle=True)
CHARSET = str(D["charset"])
VIEWBOX = 24.0          # svg_utils.SVG_PREFIX_BIG
NUMERICALIZE_SCALE = 30.0   # models.transformers.numericalize
CMDS = ["EOS", "M", "L", "C"]

plt.rcParams.update({
    "font.size": 9, "figure.dpi": 200, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
INK   = "#1a1a1a"
CURVE = "#1f6f8b"   # cubic segments
LINE  = "#b4472e"   # straight segments
CTRL  = "#c58a2e"   # control points and their handles
GREY  = "#9a9a9a"


def commands(fid, ch):
    """Decode one glyph into [(kind, start, c1, c2, end), ...] in y-up plot space."""
    seq = D[f"{fid}_sequence_relaxed"]
    n = int(D[f"{fid}_seq_len"][CHARSET.index(ch)].item())
    rows = seq[CHARSET.index(ch)][:n]
    out = []
    flip = lambda p: (p[0], VIEWBOX - p[1])      # dataset y grows downward
    for r in rows:
        kind = CMDS[int(np.argmax(r[:4]))]
        if kind == "EOS":
            break
        out.append((kind, flip(r[4:6]), flip(r[6:8]), flip(r[8:10]), flip(r[10:12])))
    return out


def glyph_path(cmds):
    verts, codes = [], []
    for kind, start, c1, c2, end in cmds:
        if kind == "M":
            if verts:
                verts.append((0, 0)); codes.append(Path.CLOSEPOLY)
            verts.append(end); codes.append(Path.MOVETO)
        elif kind == "L":
            verts.append(end); codes.append(Path.LINETO)
        else:
            verts += [c1, c2, end]; codes += [Path.CURVE4] * 3
    if verts:
        verts.append((0, 0)); codes.append(Path.CLOSEPOLY)
    return Path(verts, codes)


def frame(ax, title=None, pad=1.0):
    ax.set_xlim(-pad, VIEWBOX + pad); ax.set_ylim(-pad, VIEWBOX + pad)
    ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8.5, color=INK, pad=6)


def draw_filled(ax, cmds, color=INK):
    ax.add_patch(PathPatch(glyph_path(cmds), facecolor=color, edgecolor="none"))


def draw_outline(ax, cmds, lw=1.4, nodes=True):
    ax.add_patch(PathPatch(glyph_path(cmds), facecolor="none", edgecolor=INK, lw=lw))
    if nodes:
        pts = np.array([c[4] for c in cmds])
        ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.4, mfc="white", mec=INK, mew=1.1, zorder=5)


def draw_beziers(ax, cmds, handles=True):
    """Segments coloured by command type, with control points and their handles."""
    for kind, start, c1, c2, end in cmds:
        if kind == "M":
            continue
        if kind == "L":
            ax.plot([start[0], end[0]], [start[1], end[1]], color=LINE, lw=2.0, zorder=3)
        else:
            t = np.linspace(0, 1, 60)[:, None]
            p = ((1 - t) ** 3 * np.array(start) + 3 * (1 - t) ** 2 * t * np.array(c1)
                 + 3 * (1 - t) * t ** 2 * np.array(c2) + t ** 3 * np.array(end))
            ax.plot(p[:, 0], p[:, 1], color=CURVE, lw=2.0, zorder=3)
            if handles:
                for a, b in ((start, c1), (end, c2)):
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=CTRL, lw=.7, ls="-", zorder=2)
                ax.plot([c1[0], c2[0]], [c1[1], c2[1]], "s", ms=2.8,
                        mfc=CTRL, mec="none", zorder=4)
    pts = np.array([c[4] for c in cmds])
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=3.6, mfc="white", mec=INK, mew=1.2, zorder=6)


def draw_raster(ax, fid, ch, grid=True):
    img = D[f"{fid}_rendered_64"][CHARSET.index(ch)]
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, extent=[0, VIEWBOX, 0, VIEWBOX],
              interpolation="nearest", origin="upper")
    if grid:
        for k in np.linspace(0, VIEWBOX, 17):
            ax.axhline(k, color=GREY, lw=.25, alpha=.7)
            ax.axvline(k, color=GREY, lw=.25, alpha=.7)


# ============================================================ figure 1, teaser
def teaser(fid="0013", ch="B"):
    cmds = commands(fid, ch)
    fig, axes = plt.subplots(1, 5, figsize=(10.2, 2.4))
    fig.subplots_adjust(wspace=.06)

    frame(axes[0], "1. a glyph")
    draw_filled(axes[0], cmds)

    frame(axes[1], "2. its outline")
    draw_outline(axes[1], cmds)

    frame(axes[2], "3. Bézier segments")
    draw_beziers(axes[2], cmds)

    # 4. the command sequence, which is what the model actually predicts
    ax = axes[3]; frame(ax, "4. a command sequence")
    rows = []
    for kind, s, c1, c2, e in cmds[:12]:
        tail = " +2 ctrl" if kind == "C" else ""
        rows.append((kind, f"{kind} ({e[0]:4.1f},{e[1]:4.1f}){tail}"))
    if len(cmds) > 12:
        rows.append(("", f"... {len(cmds) - 12} more"))
    for i, (kind, txt) in enumerate(rows):
        col = {"C": CURVE, "L": LINE, "M": INK}.get(kind, GREY)
        ax.text(1.0, VIEWBOX - 0.4 - i * 1.85, txt, fontsize=6.0, family="monospace",
                color=col, va="top", ha="left")

    frame(axes[4], "5. rendered to 64x64")
    draw_raster(axes[4], fid, ch)

    # fig.text(.5, .015,
    #          "The model reads panel 4 and writes panel 4. The reported metric compares "
    #          "panel 5. Everything this report measures happens in the gap between them.",
    #          ha="center", fontsize=8.2, color=INK)
    fig.savefig(os.path.join(OUT, "fig1_teaser.png")); plt.close(fig)


# =========================================================== figure 2, anatomy
def anatomy(fid="0013", ch="a"):
    cmds = commands(fid, ch)
    fig = plt.figure(figsize=(10.0, 5.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 1, 1], height_ratios=[1, 1],
                          hspace=.52, wspace=.20)

    # -- A. the big labelled glyph -----------------------------------------
    ax = fig.add_subplot(gs[:, 0])
    frame(ax, None, pad=4.6)
    draw_beziers(ax, cmds)
    ax.set_title("glyph  ·  one character, drawn in one font",
                 fontsize=8.8, color=INK, pad=10)

    on_pts = np.array([c[4] for c in cmds if c[0] != "M"])
    ctrl_pts = np.array([c[2] for c in cmds if c[0] == "C"]
                        + [c[3] for c in cmds if c[0] == "C"])
    curves = [c for c in cmds if c[0] == "C"]
    lines_ = [c for c in cmds if c[0] == "L"]

    def label(target, text, corner, color):
        """Leader from a corner of the panel to a feature, chosen so it never crosses."""
        fx, fy = corner
        ax.annotate(text, xy=target, xycoords="data",
                    xytext=(fx, fy), textcoords="axes fraction",
                    fontsize=7.3, color=color, ha="left" if fx < .5 else "right",
                    va="bottom" if fy < .5 else "top",
                    arrowprops=dict(arrowstyle="-", color=color, lw=.7,
                                    shrinkA=1, shrinkB=3,
                                    connectionstyle="arc3,rad=0.12"))

    # pick features at the extremes so the leaders stay outside the letter
    top_on = on_pts[np.argmax(on_pts[:, 1])]
    right_ctrl = ctrl_pts[np.argmax(ctrl_pts[:, 0])]
    left_curve = min(curves, key=lambda c: (c[1][0] + c[4][0]) / 2)
    left_mid = np.mean([left_curve[1], left_curve[4]], axis=0)

    label(top_on, "on-curve point\nthe outline goes through it", (.02, .985), INK)
    label(right_ctrl, "control point\npulls the curve, sits off it", (.985, .985), CTRL)
    label(left_mid, "cubic Bézier segment\nfour points make a curve", (.02, .015), CURVE)
    if lines_:
        long_line = max(lines_, key=lambda c: np.hypot(*(np.array(c[4]) - np.array(c[1]))))
        label(np.mean([long_line[1], long_line[4]], axis=0),
              "straight segment", (.985, .015), LINE)

    # -- B. vector, then rendered ------------------------------------------
    ax = fig.add_subplot(gs[0, 1]); frame(ax, "vector  ·  no resolution")
    draw_outline(ax, cmds, lw=1.3, nodes=False)
    ax = fig.add_subplot(gs[0, 2]); frame(ax, "raster  ·  64 × 64 pixels")
    draw_raster(ax, fid, ch)
    ax.annotate("", xy=(-1.4, 12), xytext=(-6.2, 12), xycoords="data",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2), annotation_clip=False)
    ax.text(-3.8, 13.4, "render", fontsize=7.6, ha="center", color=INK)
    ax.text(VIEWBOX / 2, -3.0,
            "the reported metric compares two of these",
            fontsize=7.2, ha="center", color=GREY)

    # -- C. quantization, zoomed so the grid and the snapping are both visible ----
    ax = fig.add_subplot(gs[1, 1])
    step = NUMERICALIZE_SCALE / 128.0          # one bin, in viewBox units
    # zoom on a window holding several on-curve points, so the grid has something to bite
    # centre on the point whose true position is furthest from any bin, so the
    # rounding the panel is about is the largest one actually present in the glyph
    off = np.abs(on_pts - np.round(on_pts / step) * step).max(axis=1)
    cx, cy = on_pts[np.argmax(off)]
    half = step * 7
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half * .74, cy + half * .74)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(.6); sp.set_color(GREY)
    for k in np.arange(np.floor((cx - half) / step) * step, cx + half + step, step):
        ax.axvline(k, color=GREY, lw=.35, alpha=.85)
    for k in np.arange(np.floor((cy - half) / step) * step, cy + half + step, step):
        ax.axhline(k, color=GREY, lw=.35, alpha=.85)
    draw_beziers(ax, cmds, handles=False)
    # the true coordinate, and the only place the model is able to put it
    snapped = np.round(on_pts / step) * step
    ax.plot(on_pts[:, 0], on_pts[:, 1], "o", ms=5.0, mfc="none", mec=INK, mew=1.3,
            zorder=6, label="true coordinate")
    ax.plot(snapped[:, 0], snapped[:, 1], "o", ms=3.2, mfc=LINE, mec="none",
            zorder=7, label="nearest bin")
    for a, b in zip(on_pts, snapped):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=LINE, lw=.8, zorder=5)
    ax.legend(frameon=False, fontsize=6.6, ncol=2, loc="upper center",
              bbox_to_anchor=(.5, -.03), handletextpad=.3, columnspacing=1.2,
              borderpad=.1)
    ax.set_title("every coordinate snaps to a bin", fontsize=8.8, color=INK, pad=8)
    ax.text(.5, -.20, "128 bins per axis  ·  one bin ≈ 0.63 px at 64 × 64\n"
                      "the model chooses a bin, so it can never draw between them",
            transform=ax.transAxes, fontsize=7.2, ha="center", va="top", color=GREY)

    # -- D. few-shot --------------------------------------------------------
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    ax.set_title("few-shot  ·  4 glyphs in, 52 out", fontsize=8.8, color=INK, pad=8)
    for i, c in enumerate("ABab"):
        sub = ax.inset_axes([i * .105, .58, .098, .34])
        frame(sub); draw_filled(sub, commands(fid, c))
        for s in sub.spines.values():
            s.set_visible(True); s.set_linewidth(.6); s.set_color(INK)
    ax.text(.455, .75, "→", fontsize=14, transform=ax.transAxes, va="center", ha="center")
    strip = np.hstack([D[f"{fid}_rendered_64"][CHARSET.index(c)] for c in "CDEFGHIJKL"])
    sub = ax.inset_axes([.52, .58, .48, .34]); sub.axis("off")
    sub.imshow(strip, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.text(0, .40, "font  ·  a whole alphabet in one consistent style",
            fontsize=7.4, transform=ax.transAxes, color=INK)
    ax.text(0, .26, "the model sees four references and must draw\nthe remaining forty-eight",
            fontsize=7.2, transform=ax.transAxes, color=GREY, va="top")

    fig.savefig(os.path.join(OUT, "fig2_anatomy.png")); plt.close(fig)


if __name__ == "__main__":
    teaser()
    anatomy()
    print("wrote fig1_teaser.png, fig2_anatomy.png")
