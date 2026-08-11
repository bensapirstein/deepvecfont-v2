#!/usr/bin/env bash
# Build the submission PDF. Run from anywhere:  bash report/build.sh
#
# Everything the report needs is in this folder. Figures are regenerated from
# RESULTS.csv and every quoted number is re-checked against it before pandoc
# runs, so a stale figure or a stale number cannot reach the PDF.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

python3 "$HERE/make_figures.py"
python3 "$HERE/verify_report.py"

cd "$HERE"   # relative figure paths in REPORT.md resolve from here
pandoc REPORT.md -o REPORT.pdf \
  --pdf-engine=xelatex --from=gfm \
  -V geometry:a4paper -V geometry:margin=2.4cm \
  -V mainfont="DejaVu Serif" -V sansfont="DejaVu Sans" -V monofont="DejaVu Sans Mono" \
  -V fontsize=10pt -V linkcolor=black -V urlcolor=black -V colorlinks=true
echo "wrote report/REPORT.pdf"
