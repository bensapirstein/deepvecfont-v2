# Where the citations in the code point

Comments and docstrings throughout this repository cite `PROJECT_PLAN.md` by section
number, and cite runbooks under `docs/` by filename — `docs/day6-gpu-push.md`,
`docs/review-response.md`, `docs/english-arm.md` and about ten others. Roughly eighty such
references exist, mostly in `options.py`, `scripts/` and the two batch drivers.

**Those files are not on this branch.** They are the working record of a thirty-day sprint:
dated status tables, cluster paths, GPU allocations, and decisions logged as they were
taken. They are provenance rather than deliverable, and carrying 200 KB of them into a
release would bury the code they annotate.

They live on the `repro` branch, unchanged:

```bash
git show repro:PROJECT_PLAN.md | less
git show repro:docs/review-response.md | less
git log repro --oneline                       # the whole working history, in order
```

Or browse them at
[github.com/bensapirstein/deepvecfont-v2/tree/repro](https://github.com/bensapirstein/deepvecfont-v2/tree/repro).

The citations were left in place rather than rewritten. Editing eighty comments would have
added a large diff of pure churn to `main..submission`, which is the diff a reader uses to
see what this project actually changed, and it would have cost the thing the citations are
for: a comment that says *why* a default is what it is, and where that reasoning was
written down.

## What replaced them here

| For | Read |
|---|---|
| Running anything | `docs/REPRODUCE.md` |
| What was found | `report/REPORT.md` |
| The numbers behind it | `RESULTS.csv`, and `report/verify_report.py` which checks them |
| The head we designed and dropped | `archive/FLOW_MATCHING_PLAN.md`, cited by report §6.5 |

One rule that lived in a dropped runbook is worth restating, because it governs three
figures in the report and a reader cannot check it from the pictures alone. Qualitative
glyph figures pick their fonts and characters **before** any glyph is looked at, show both
tails of the error distribution rather than only the flattering end, and give Chinese and
English equal space. `report/make_model_comparison_figures.py` states the full rule in its
own docstring and the figures are laid out from arrays that already exist, so no shape in
the report was chosen by eye after the fact.
