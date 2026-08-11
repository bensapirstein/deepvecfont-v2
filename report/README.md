# report/

Everything produced for the submission lives here. Nothing in this folder is
hand-maintained: the figures and the PDF are generated, and the numbers are checked.

| File | What it is |
|---|---|
| `REPORT.md` | The report. Source of truth, edited by hand. |
| `REPORT.pdf` | Built from `REPORT.md`. Do not edit; rebuild it. |
| `build.sh` | Regenerates figures, verifies every number, then builds the PDF. |
| `make_figures.py` | Draws every figure from `../RESULTS.csv`. No figure is drawn by hand. |
| `verify_report.py` | Re-derives every number quoted in `REPORT.md` from `../RESULTS.csv`. Exits non-zero on any mismatch. |
| `figures/` | Generated output. Safe to delete; `build.sh` recreates it. |

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
