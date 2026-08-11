# report/

Everything produced for the submission lives here. Nothing in this folder is
hand-maintained: the figures and the PDF are generated, and the numbers are checked.

| File | What it is |
|---|---|
| `REPORT.md` | The report. Source of truth, edited by hand. |
| `REPORT.pdf` | Built from `REPORT.md`. Do not edit; rebuild it. |
| `build.sh` | Regenerates figures, verifies every number, then builds the PDF. |
| `make_figures.py` | Draws the results figures (3-6) from `../RESULTS.csv`. |
| `make_glyph_figures.py` | Draws the explanatory figures (1-2) from real test-split outlines. |
| `extract_assets.py` | Pulls those outlines out of `../data.zip` into `assets/glyphs.npz`. Rarely needed; the npz is committed. |
| `verify_report.py` | Re-derives every number quoted in `REPORT.md` from `../RESULTS.csv`. Exits non-zero on any mismatch. |
| `figures/` | Generated output. Safe to delete; `build.sh` recreates it. |
| `assets/` | Inputs to the figures: the glyph npz, plus provenance for choices made by eye. |

## To rebuild

```bash
bash report/build.sh
```

That runs the figures, then the verifier, then pandoc. **The verifier gates the
build**, so a number that has drifted from `RESULTS.csv` stops the PDF rather than
reaching the submission.

## The rule this folder exists to enforce

Results are stored once, in `../RESULTS.csv`, one row per scored checkpoint,
rebuilt by `../scripts/build_results_table.py`. Every table and every figure in the
report is derived from that file at build time. If a number needs to change, it
changes in `RESULTS.csv` and everything downstream follows. Nothing is typed twice.

The same rule covers the glyphs. Every curve in figures 1 and 2 is a real command sequence
from the English **test** split, decoded by the same code path the model and the evaluation
use, so no shape in this report is hand-drawn, traced, or illustrative. `assets/render_selftest.png`
is the check: our decode of the stored outline against the dataset's own raster of it.
`assets/font_choice_contact_sheet.png` records the one judgement made by eye, which font to
draw, so that choice is visible rather than asserted.

Model-output glyphs are not here. They exist only under `experiments/` on the cluster; see
`../docs/pull-figure-assets.md`, which also fixes the rules for what such a figure may show
before any of it is drawn.
