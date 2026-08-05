#!/usr/bin/env python3
"""Aggregate every experiment's per-font eval CSV into one RESULTS.csv row.

Source of truth is experiments/<name>_main_model/results/eval_<ckpt>.csv
(written by eval_reconstruction_error.py) plus opts.txt (seed, language) for
each experiment. Re-run any time after a new evaluation:

    python scripts/build_results_table.py

New experiment directories are picked up automatically. Their `batch` and
`n_samples` are looked up in the tables below; anything not listed there
gets `batch=unclassified` and the default screening budget (3), printed as
a warning so it can be classified by hand.
"""
import csv
import os
import re
import sys

EXP_ROOT = "experiments"
OUT_PATH = "RESULTS.csv"

# Which round of the project each experiment belongs to. Manual, not
# inferred from the name: several names are ambiguous across rounds (e.g.
# e9_sigma050_chn was trained for Tier 1, then re-screened as Tier 3 Batch
# B's seed-1111 anchor). Add new entries here as new batches launch.
BATCH = {
    "dvf_base_exp_chn": "stage1",
    "seedfloor_1111_chn": "baseline",
    "seedfloor_2222_chn": "baseline",
    "seedfloor_3333_chn": "baseline",
    "e7_aux01_chn": "tier1", "e7_aux03_chn": "tier1", "e7_aux10_chn": "tier1",
    "e9_sigma000_chn": "tier1", "e9_sigma010_chn": "tier1", "e9_sigma025_chn": "tier1",
    "e10_drop01_chn": "tier1", "e10_drop02_chn": "tier1",
    "e1_norm_chn": "tier1",  # re-screened as tier3b's seed-1111 anchor
    "e8_ls05_chn": "tier2", "e8_ls10_chn": "tier2", "e8_ls20_chn": "tier2",
    "e13_bins256_chn": "tier2",  # re-screened as tier3b's seed-1111 anchor
    "e13_nopad_chn": "tier2",
    "e3_refine2_chn": "tier2", "e3_refine3_chn": "tier2",
    "e14_wucos_chn": "tier2",
    "e12_kl000_chn": "tier3a", "e12_kl100_chn": "tier3a",
    "e11_adamw_chn": "tier3a",
    "e2_groupnorm_chn": "tier3a", "e2_batchnorm_chn": "tier3a",
    "e4_ngf32_chn": "tier3a",
    "e15_ema999_chn": "tier3a",
    "e5_bneck256_chn": "tier3a", "e5_bneck1024_chn": "tier3a",
    "e9_sigma050_chn": "tier3b",  # seed 1111 leg, trained in tier1
    "e9_sigma050_2222_chn": "tier3b", "e9_sigma050_3333_chn": "tier3b",
    "e1_norm_2222_chn": "tier3b", "e1_norm_3333_chn": "tier3b",
    "e13_bins256_2222_chn": "tier3b", "e13_bins256_3333_chn": "tier3b",
}

# Eval budget (--n_samples) each experiment was screened/confirmed at.
# Everything not listed defaults to 3, the screening budget every Tier
# 1-3 candidate has used since scripts/test_experiments.sh was introduced.
N_SAMPLES = {
    "dvf_base_exp_chn": 50,  # Stage 1 confirmation budget, matches the paper
}

NOTES = {
    "e1_norm_chn": "re-screened as tier3b seed-1111 anchor",
    "e9_sigma050_chn": "re-screened as tier3b seed-1111 anchor",
    "e13_bins256_chn": "re-screened as tier3b seed-1111 anchor",
    "dvf_base_exp_chn": "epoch 135, pre --seed flag (hardcoded seed 1111)",
}

CKPT_EPOCH_RE = re.compile(r"^(\d+)_")


def read_opts(exp_dir):
    path = os.path.join(exp_dir, "opts.txt")
    opts = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if ":" in line:
                    k, v = line.split(":", 1)
                    opts[k.strip()] = v.strip()
    return opts


def main():
    rows = []
    if not os.path.isdir(EXP_ROOT):
        sys.exit(f"no {EXP_ROOT}/ here -- run from the repo root")

    for entry in sorted(os.listdir(EXP_ROOT)):
        if not entry.endswith("_main_model"):
            continue
        name_exp = entry[: -len("_main_model")]
        exp_dir = os.path.join(EXP_ROOT, entry)
        results_dir = os.path.join(exp_dir, "results")
        if not os.path.isdir(results_dir):
            continue
        csvs = [f for f in os.listdir(results_dir) if f.startswith("eval_") and f.endswith(".csv")]
        if not csvs:
            continue
        # One eval CSV per checkpoint that's been scored; usually just one.
        for csv_name in sorted(csvs):
            csv_path = os.path.join(results_dir, csv_name)
            checkpoint = csv_name[len("eval_") : -len(".csv")]
            m = CKPT_EPOCH_RE.match(checkpoint)
            epoch = int(m.group(1)) if m else ""

            with open(csv_path) as f:
                font_rows = list(csv.DictReader(f))
            if not font_rows:
                continue
            n_fonts = len(font_rows)
            l1 = sum(float(r["l1"]) for r in font_rows) / n_fonts
            s_iou = sum(float(r["iou"]) for r in font_rows) / n_fonts
            glyphs_expected = sum(int(r["glyphs_expected"]) for r in font_rows)
            glyphs_failed = sum(int(r["glyphs_render_failed"]) for r in font_rows)
            fonts_rendered = sum(1 for r in font_rows if int(r["glyphs_render_failed"]) == 0)

            opts = read_opts(exp_dir)
            seed = opts.get("seed", "1111" if name_exp == "dvf_base_exp_chn" else "")
            language = opts.get("language", "chn")

            batch = BATCH.get(name_exp)
            if batch is None:
                batch = "unclassified"
                print(f"warning: {name_exp} not in BATCH map, tagged 'unclassified'", file=sys.stderr)

            eval_date = None
            try:
                import datetime

                eval_date = datetime.date.fromtimestamp(os.path.getmtime(csv_path)).isoformat()
            except OSError:
                eval_date = ""

            rows.append(
                {
                    "name_exp": name_exp,
                    "batch": batch,
                    "seed": seed,
                    "language": language,
                    "checkpoint": checkpoint,
                    "epoch": epoch,
                    "n_samples": N_SAMPLES.get(name_exp, 3),
                    "n_fonts": n_fonts,
                    "fonts_rendered": fonts_rendered,
                    "glyphs_expected": glyphs_expected,
                    "glyphs_rendered": glyphs_expected - glyphs_failed,
                    "l1": f"{l1:.4f}",
                    "s_iou": f"{s_iou:.4f}",
                    "ssim": "",  # not implemented yet, see PROJECT_PLAN.md 2.2
                    "eval_date": eval_date,
                    "notes": NOTES.get(name_exp, ""),
                }
            )

    rows.sort(key=lambda r: (r["batch"], r["name_exp"]))

    fieldnames = [
        "name_exp", "batch", "seed", "language", "checkpoint", "epoch",
        "n_samples", "n_fonts", "fonts_rendered", "glyphs_expected",
        "glyphs_rendered", "l1", "s_iou", "ssim", "eval_date", "notes",
    ]
    with open(OUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
