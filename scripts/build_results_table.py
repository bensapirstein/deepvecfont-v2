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
    "official_chn": "official-checkpoint", "official_eng": "official-checkpoint",
    "eng_seedfloor_1111": "baseline", "eng_seedfloor_2222": "baseline",
    "eng_seedfloor_3333": "baseline",
    # E9 carried to English as a generalization test, three seeds paired to the
    # three baselines above (docs/english-candidate.md). Its own batch rather than
    # tier3b: same candidate, different script, and the two must never be pooled.
    "e9_sigma050_1111_eng": "eng-generalization",
    "e9_sigma050_2222_eng": "eng-generalization",
    "e9_sigma050_3333_eng": "eng-generalization",
    # 600-epoch Chinese seeds, trained 2026-08-06 (PROJECT_PLAN.md §9 item 4).
    "seedfloor600_1111_chn": "baseline-600", "seedfloor600_2222_chn": "baseline-600",
    "seedfloor600_3333_chn": "baseline-600",
    # Job A, 2026-08-08 (docs/day6-gpu-push.md §1): de-confound E1's epoch
    # selection. --max_ckpt_keep 10, scored at 100/125/150 for all three.
    "a_e1_norm_2222_chn": "job-a-deconfound",
    "a_e1_norm_3333_chn": "job-a-deconfound",
    "a_seedfloor_3333_chn": "job-a-deconfound",
    # Job C, 2026-08-08 (docs/day6-gpu-push.md §3): one replicated candidate
    # per assignment category, three seeds each, matched epoch 150.
    "c_e2_batchnorm_1111_chn": "job-c-category", "c_e2_batchnorm_2222_chn": "job-c-category",
    "c_e2_batchnorm_3333_chn": "job-c-category",
    "c_e4_ngf32_1111_chn": "job-c-category", "c_e4_ngf32_2222_chn": "job-c-category",
    "c_e4_ngf32_3333_chn": "job-c-category",
    "c_e5_bneck256_1111_chn": "job-c-category", "c_e5_bneck256_2222_chn": "job-c-category",
    "c_e5_bneck256_3333_chn": "job-c-category",
    "c_e7_aux01_1111_chn": "job-c-category", "c_e7_aux01_2222_chn": "job-c-category",
    "c_e7_aux01_3333_chn": "job-c-category",
    "c_e11_adamw_1111_chn": "job-c-category", "c_e11_adamw_2222_chn": "job-c-category",
    "c_e11_adamw_3333_chn": "job-c-category",
    "c_e3_refine2_1111_chn": "job-c-category", "c_e3_refine2_2222_chn": "job-c-category",
    "c_e3_refine2_3333_chn": "job-c-category",
    # Job D, 2026-08-08 (docs/day6-gpu-push.md §5): Tier 4 capacity, three
    # seeds each, matched epoch 150 (same epoch-selection confound as Job A --
    # val_metric picked 125 for 5 of 6 runs; scored at matched 150 instead).
    "e17_dff2048_1111_chn": "job-d-capacity", "e17_dff2048_2222_chn": "job-d-capacity",
    "e17_dff2048_3333_chn": "job-d-capacity",
    "e16_depth8_1111_chn": "job-d-capacity", "e16_depth8_2222_chn": "job-d-capacity",
    "e16_depth8_3333_chn": "job-d-capacity",
    # Job C-EN, 2026-08-09 (docs/day6-gpu-push.md §4): same category
    # representatives, one seed, English, screened at n_samples 10 (matching
    # the paper's own English budget) against eng_seedfloor_1111 rescored at
    # the same budget. cen_e7_aux01_1111_eng stopped early by Ben's call on
    # its wandb curve, no results to score.
    "cen_e2_batchnorm_1111_eng": "job-c-en", "cen_e4_ngf32_1111_eng": "job-c-en",
    "cen_e5_bneck256_1111_eng": "job-c-en", "cen_e11_adamw_1111_eng": "job-c-en",
    "cen_e3_refine2_1111_eng": "job-c-en",
    # cen_e3_refine2_{2222,3333}_eng, 2026-08-10/11: two more seeds for the one
    # candidate whose 1111 seed cleared the English floor in the degrading
    # direction, queued to get a 3-seed same-sign-or-not reading (§0).
    "cen_e3_refine2_2222_eng": "job-c-en", "cen_e3_refine2_3333_eng": "job-c-en",
    # rv-review, 2026-08-12/13 (docs/review-response.md): answers the external
    # review, docs/review-gemini.md. Rebuilt Chinese dataset at 10x augmentation
    # (was 6x), a real held-out val split (20 base fonts, never trained on or
    # scored), and checkpoint selection on the rendered metric instead of
    # val_metric. NOT comparable to any batch above: different training set,
    # different training-set size, different selection rule. The re-measured
    # floor (0.0151) came back 63% above the pre-review 0.0093, §4.5's gate
    # fired, and no candidate here clears it -- reported as a null/instrument
    # finding, not a result. Do not read these rows against tier1-3/job-c deltas.
    "rv_seedfloor_1111_chn": "rv-review", "rv_seedfloor_2222_chn": "rv-review",
    "rv_seedfloor_3333_chn": "rv-review",
    "rv_e9_sigma050_1111_chn": "rv-review", "rv_e9_sigma050_2222_chn": "rv-review",
    "rv_e9_sigma050_3333_chn": "rv-review",
    "rv_e1_norm_1111_chn": "rv-review", "rv_e1_norm_2222_chn": "rv-review",
    "rv_e1_norm_3333_chn": "rv-review",
    "rv_e2_batchnorm_1111_chn": "rv-review", "rv_e2_batchnorm_2222_chn": "rv-review",
    "rv_e2_batchnorm_3333_chn": "rv-review",
    "rv_e4_ngf32_1111_chn": "rv-review", "rv_e4_ngf32_2222_chn": "rv-review",
    "rv_e4_ngf32_3333_chn": "rv-review",
    "rv_e5_bneck256_1111_chn": "rv-review", "rv_e5_bneck256_2222_chn": "rv-review",
    "rv_e5_bneck256_3333_chn": "rv-review",
    "rv_e7_aux01_1111_chn": "rv-review", "rv_e7_aux01_2222_chn": "rv-review",
    "rv_e7_aux01_3333_chn": "rv-review",
    "rv_e11_adamw_1111_chn": "rv-review", "rv_e11_adamw_2222_chn": "rv-review",
    "rv_e11_adamw_3333_chn": "rv-review",
    "rv_e3_refine2_1111_chn": "rv-review", "rv_e3_refine2_2222_chn": "rv-review",
    "rv_e3_refine2_3333_chn": "rv-review",
}

# Eval budget (--n_samples) each experiment was screened/confirmed at.
# Everything not listed defaults to 3, the screening budget every Tier
# 1-3 candidate has used since scripts/test_experiments.sh was introduced.
#
# This map is per-EXPERIMENT and is therefore only correct while an experiment
# has been scored at exactly one budget. The confirmation eval breaks that:
# seedfloor_1111_chn and e9_sigma050_chn are each scored at n_samples 3
# (screening) and at n_samples 50 (confirmation), and §3.2's rule is that a
# screening number and a confirmation number must never be compared. So the
# budget is read off the CSV *filename* first -- write confirmation output to
#     results/eval_<ckpt>_n50.csv   (--csv_out on eval_reconstruction_error.py)
# and this map is only the fallback for files with no suffix.
N_SAMPLES = {
    "dvf_base_exp_chn": 50,  # Stage 1 confirmation budget, matches the paper
    # official_{chn,eng}: every eval_*.csv here (n50, n50_gtsvg, n50_partial862,
    # n50_subset34, ...) was decoded at --n_samples 50; the extra suffixes after
    # _n50 don't match CKPT_NSAMPLES_RE (deliberately -- stripping them would
    # collide the raster and svg rows under one checkpoint label), so this
    # fallback is what actually supplies the correct budget for these two.
    "official_chn": 50,
    "official_eng": 50,
    # eng_seedfloor_*: confirmation-style n_samples 50 (docs/english-arm.md), same
    # _n50_subset34 suffix issue as official_eng above.
    "eng_seedfloor_1111": 50, "eng_seedfloor_2222": 50, "eng_seedfloor_3333": 50,
    # e9_sigma050_*_eng: scored at the same budget and on the same 34-font subset as
    # the baselines they pair against, so the same _n50_subset34 fallback applies.
    # These go straight to confirmation budget with no screening pass: there is no
    # screening/confirmation split on the English arm, because the whole arm is three
    # runs of one already-screened candidate (docs/english-candidate.md §1).
    "e9_sigma050_1111_eng": 50, "e9_sigma050_2222_eng": 50, "e9_sigma050_3333_eng": 50,
    # seedfloor600_*_chn: same _n50/_n50_gtsvg suffix issue as official_chn above,
    # scored at the confirmation budget (docs/english-candidate.md §6).
    "seedfloor600_1111_chn": 50, "seedfloor600_2222_chn": 50, "seedfloor600_3333_chn": 50,
    # rv-review batch: every eval_<ckpt>.csv here has no _n<N> suffix (see
    # eval_reconstruction_error.py's default --csv_out), but COMMON_ARGS in
    # scripts/test_experiments.sh runs the whole batch at --n_samples 50, the
    # confirmation budget, not the default screening fallback of 3.
    "rv_seedfloor_1111_chn": 50, "rv_seedfloor_2222_chn": 50, "rv_seedfloor_3333_chn": 50,
    "rv_e9_sigma050_1111_chn": 50, "rv_e9_sigma050_2222_chn": 50, "rv_e9_sigma050_3333_chn": 50,
    "rv_e1_norm_1111_chn": 50, "rv_e1_norm_2222_chn": 50, "rv_e1_norm_3333_chn": 50,
    "rv_e2_batchnorm_1111_chn": 50, "rv_e2_batchnorm_2222_chn": 50, "rv_e2_batchnorm_3333_chn": 50,
    "rv_e4_ngf32_1111_chn": 50, "rv_e4_ngf32_2222_chn": 50, "rv_e4_ngf32_3333_chn": 50,
    "rv_e5_bneck256_1111_chn": 50, "rv_e5_bneck256_2222_chn": 50, "rv_e5_bneck256_3333_chn": 50,
    "rv_e7_aux01_1111_chn": 50, "rv_e7_aux01_2222_chn": 50, "rv_e7_aux01_3333_chn": 50,
    "rv_e11_adamw_1111_chn": 50, "rv_e11_adamw_2222_chn": 50, "rv_e11_adamw_3333_chn": 50,
    "rv_e3_refine2_1111_chn": 50, "rv_e3_refine2_2222_chn": 50, "rv_e3_refine2_3333_chn": 50,
}

# eval_<ckpt>_n<N>.csv -> (ckpt, N). Anything without the suffix falls back to
# the N_SAMPLES map above and keeps its filename as the checkpoint label.
CKPT_NSAMPLES_RE = re.compile(r"^(?P<ckpt>.+?)_n(?P<n>\d+)$")

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
            n_samples = None
            m_ns = CKPT_NSAMPLES_RE.match(checkpoint)
            if m_ns:
                checkpoint = m_ns.group("ckpt")
                n_samples = int(m_ns.group("n"))
            m = CKPT_EPOCH_RE.match(checkpoint)
            epoch = int(m.group(1)) if m else ""

            with open(csv_path) as f:
                font_rows = list(csv.DictReader(f))
            if not font_rows:
                continue
            n_fonts = len(font_rows)
            l1 = sum(float(r["l1"]) for r in font_rows) / n_fonts
            s_iou = sum(float(r["iou"]) for r in font_rows) / n_fonts
            # SSIM landed in eval_reconstruction_error.py on 2026-08-05, so per-font
            # CSVs written before then have no ssim column. Left blank rather than
            # zero-filled: a missing measurement and a measured zero are different.
            ssim = ""
            if font_rows[0].get("ssim") not in (None, ""):
                ssim = f"{sum(float(r['ssim']) for r in font_rows) / n_fonts:.4f}"
            glyphs_expected = sum(int(r["glyphs_expected"]) for r in font_rows)
            glyphs_failed = sum(int(r["glyphs_render_failed"]) for r in font_rows)
            fonts_rendered = sum(1 for r in font_rows if int(r["glyphs_render_failed"]) == 0)

            opts = read_opts(exp_dir)
            seed = opts.get("seed", "1111" if name_exp == "dvf_base_exp_chn" else "")
            # opts.txt is written by train.py; experiments that only ever ran
            # test_few_shot.py (e.g. official_{chn,eng}, symlinked checkpoints
            # with no training here) have none, so the old "chn" default silently
            # mislabeled every official_eng row. Fall back to the name_exp suffix.
            language = opts.get("language") or ("eng" if name_exp.endswith("_eng") else "chn")

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
                    "n_samples": n_samples if n_samples is not None
                                 else N_SAMPLES.get(name_exp, 3),
                    "n_fonts": n_fonts,
                    "fonts_rendered": fonts_rendered,
                    "glyphs_expected": glyphs_expected,
                    "glyphs_rendered": glyphs_expected - glyphs_failed,
                    "l1": f"{l1:.4f}",
                    "s_iou": f"{s_iou:.4f}",
                    "ssim": ssim,
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
