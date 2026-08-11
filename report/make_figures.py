"""Regenerate every figure in REPORT.md from RESULTS.csv.

Usage:  python report/make_figures.py
Writes: report/figures/fig1_spread.png .. fig4_convention.png

No figure in the report is drawn by hand. If RESULTS.csv changes, rerun this.
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))        # report/
ROOT = os.path.dirname(HERE)                             # repo root
OUT  = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
R = list(csv.DictReader(open(os.path.join(ROOT, "RESULTS.csv"))))

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": .25, "grid.linewidth": .5,
})
INK, ACC, BAD = "#222222", "#1f6f8b", "#b4472e"


def rows(**kw):
    out = []
    for r in R:
        ok = True
        for k, v in kw.items():
            if k == "gtsvg":
                if ("gtsvg" in r["checkpoint"]) != v: ok = False
            elif k == "batch_in":
                if r["batch"] not in v: ok = False
            elif str(r.get(k)) != str(v): ok = False
        if ok: out.append(r)
    return out


def one(**kw):
    m = rows(**kw)
    assert len(m) == 1, (kw, len(m))
    return m[0]


f = lambda r, k: float(r[k])
mean = lambda x: sum(x) / len(x)

# ---------------------------------------------------------------- figure 1
# The headline: 26 architectural changes span no more than 3 re-seedings do.
fig, ax = plt.subplots(figsize=(7.2, 2.9))
base = [f(one(name_exp=f"seedfloor_{s}_chn", epoch=150, n_samples=3), "l1") for s in (1111, 2222, 3333)]
cands = sorted(f(r, "l1") for r in rows(language="chn", n_samples=3,
                                        batch_in=("tier1", "tier2", "tier3a"))
               if not r["name_exp"].startswith("seedfloor"))
ax.axhspan(min(base), max(base), color=ACC, alpha=.18, zorder=0)
ax.scatter(range(len(cands)), cands, s=26, color=INK, zorder=3, label=f"{len(cands)} single-factor changes")
for i, b in enumerate(base):
    ax.axhline(b, color=ACC, lw=1, ls="--", zorder=2, label="baseline, 3 seeds" if i == 0 else None)
ax.set_xticks([]); ax.set_ylabel("Error (L1), lower is better")
ax.set_xlabel("each dot is one architectural change, sorted by score")
ax.set_title(f"Changing the architecture moves the metric as much as changing the seed\n"
             f"candidates span {max(cands)-min(cands):.4f}   ·   3 re-seedings span {max(base)-min(base):.4f}",
             fontsize=9.5, loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig(f"{OUT}/fig1_spread.png"); plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Paper vs reconstruction vs improved, both languages.
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
cb = [f(one(name_exp=f"seedfloor_{s}_chn", epoch=150, n_samples=50), "l1") for s in (1111, 2222, 3333)]
ce = [f(one(name_exp=n, epoch=150, n_samples=50), "l1")
      for n in ("e9_sigma050_chn", "e9_sigma050_2222_chn", "e9_sigma050_3333_chn")]
eb = [f(one(name_exp=n, epoch=e, n_samples=50), "l1")
      for n, e in (("eng_seedfloor_1111", 640), ("eng_seedfloor_2222", 580), ("eng_seedfloor_3333", 640))]
ee = [f(one(name_exp=f"e9_sigma050_{s}_eng", epoch=e, n_samples=50), "l1")
      for s, e in ((1111, 640), (2222, 580), (3333, 640))]
oc = f(one(name_exp="official_chn", epoch=600, n_samples=50, gtsvg=False), "l1")
oe = f(one(name_exp="official_eng", epoch=600, n_samples=50, n_fonts=34, gtsvg=False), "l1")

for ax, (lang, paper, offi, rec, imp) in zip(axes, [
        ("Chinese", 0.080, oc, cb, ce), ("English", 0.052, oe, eb, ee)]):
    vals = [paper, offi, mean(rec), mean(imp)]
    labs = ["paper", "released\nweights", "ours", "ours\n+ E9"]
    cols = [BAD, "#999999", INK, ACC]
    bars = ax.bar(range(4), vals, color=cols, width=.62)
    # seed spread as an error bar on the two columns we trained ourselves
    for i, series in ((2, rec), (3, imp)):
        ax.errorbar(i, mean(series), yerr=[[mean(series)-min(series)], [max(series)-mean(series)]],
                    color="white", capsize=3, lw=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_title(lang, fontsize=10, loc="left"); ax.set_ylim(0, max(vals)*1.28)
    ax.grid(axis="x", alpha=0)
axes[0].set_ylabel("Error (L1), lower is better")
fig.suptitle("Our reconstruction matches the released weights; neither reaches the published number",
             fontsize=9.5, x=.02, ha="left", y=1.04)
fig.savefig(f"{OUT}/fig2_threeway.png"); plt.close(fig)

# ---------------------------------------------------------------- figure 3
# E9 per seed, paired. Chinese all same sign; English mixed.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=False)
seeds = ["1111", "2222", "3333"]
chn_d = [c - b for c, b in zip(ce, cb)]
eng_d = [c - b for c, b in zip(ee, eb)]
for ax, (lang, d, floor) in zip(axes, [("Chinese", chn_d, 0.0097), ("English", eng_d, 0.0038)]):
    cols = [ACC if x < 0 else BAD for x in d]
    ax.bar(seeds, d, color=cols, width=.55)
    ax.axhspan(-floor, floor, color="#000000", alpha=.07, zorder=0)
    ax.axhline(0, color=INK, lw=.8)
    for i, x in enumerate(d):
        ax.text(i, x, f"{x:+.4f}", ha="center", va="bottom" if x > 0 else "top", fontsize=8)
    verdict = "same sign at all 3 seeds" if all(x < 0 for x in d) else "mixed sign: does not replicate"
    ax.set_title(f"{lang} — {verdict}", fontsize=9, loc="left")
    ax.set_xlabel("training seed"); ax.grid(axis="x", alpha=0)
    lo, hi = min(d + [-floor]), max(d + [floor])
    pad = (hi - lo) * .22
    ax.set_ylim(lo - pad, hi + pad)
axes[0].set_ylabel("Δ Error (L1) vs paired baseline\nnegative = better")
fig.suptitle("The one improvement we found does not survive a change of script\n"
             "shaded band is the seed-noise floor", fontsize=9.5, x=.02, ha="left", y=1.09)
fig.savefig(f"{OUT}/fig3_e9.png"); plt.close(fig)

# ---------------------------------------------------------------- figure 4
# The metric depends on an unstated choice of rasterizer.
fig, ax = plt.subplots(figsize=(5.4, 2.6))
pairs = []
for lang, nm, ep, nf in (("Chinese", "official_chn", 600, 34), ("English", "official_eng", 600, 34)):
    kw = dict(name_exp=nm, epoch=ep, n_samples=50)
    if nm == "official_eng": kw["n_fonts"] = nf
    pairs.append((lang, f(one(gtsvg=False, **kw), "l1"), f(one(gtsvg=True, **kw), "l1")))
x = range(len(pairs)); w = .34
b1 = ax.bar([i - w/2 for i in x], [p[1] for p in pairs], w, color=INK, label="ground truth = dataset raster")
b2 = ax.bar([i + w/2 for i in x], [p[2] for p in pairs], w, color=ACC, label="ground truth = same rasterizer")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.4f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels([p[0] for p in pairs])
ax.set_ylabel("Error (L1)"); ax.set_ylim(0, max(max(p[1], p[2]) for p in pairs)*1.42)
# gap annotations on one shared baseline, clear of every bar and of the legend
gy = ax.get_ylim()[1] * .60
for i, p in enumerate(pairs):
    ax.annotate("", xy=(i-w/2, gy), xytext=(i+w/2, gy),
                arrowprops=dict(arrowstyle="<->", color=BAD, lw=.9))
    ax.text(i, gy*1.03, f"gap {p[1]-p[2]:.4f}", ha="center", fontsize=8, color=BAD)
ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.grid(axis="x", alpha=0)
ax.set_title("One fixed checkpoint, one formula, two answers\n"
             "the paper's whole margin over its predecessor on Chinese is 0.006",
             fontsize=9.5, loc="left")
fig.savefig(f"{OUT}/fig4_convention.png"); plt.close(fig)

print("wrote:", *sorted(os.listdir(OUT)), sep="\n  ")
