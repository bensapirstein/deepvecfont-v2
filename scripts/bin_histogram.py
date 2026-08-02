#!/usr/bin/env python3
"""Histogram the quantized coordinate arguments of the training data.

Answers two questions, both from PROJECT_PLAN.md:

  1. (1.5) Which quantization grid is the data actually on? There are two numericalize
     definitions in this codebase: n=128 in models/transformers.py, which the model path
     uses, and n=64 in data_utils/relax_rep.py, which runs during preprocessing and
     writes back through a numpy view inside cal_aux_bezier_pts. If that mutation reached
     the persisted sequences, the 128-bin histogram will be a COMB with every odd bin
     empty, the model is predicting over 128 bins on 64 bins' worth of information, and
     E13 (256 bins) is pointless without rebuilding the dataset.

  2. (2.3) How much mass sits in the clipped boundary bins? numericalize does
     .clip(min=0, max=n-1), so coordinates outside [0, 30] are destroyed. Bin 0 is also
     the padding_idx of SVGEmbedding.arg_embed, so it maps to a frozen zero vector.

Reads sequence_relaxed.npy directly. numpy only: no torch, no GPU, no model.

Usage:
    python scripts/bin_histogram.py --language chn
    python scripts/bin_histogram.py --language eng --max_seq_len 51 --plot
"""
import argparse
import os
import sys

import numpy as np


def numericalize(cmd, n=128):
    """Copy of models/transformers.py:596, on numpy instead of torch."""
    return (cmd / 30 * n).round().clip(min=0, max=n - 1).astype(int)


def load_args(data_root, language, mode, char_num, max_seq_len, dim_seq, limit=None):
    """Yield the 8 coordinate arguments (columns 4:12) of every stroke of every font."""
    dir_path = os.path.join(data_root, language, mode)
    if not os.path.isdir(dir_path):
        raise SystemExit(f"ERROR: no data at {dir_path}")

    font_dirs = sorted(os.path.join(dir_path, d) for d in os.listdir(dir_path)
                       if os.path.isdir(os.path.join(dir_path, d)))
    if limit:
        font_dirs = font_dirs[:limit]
    if not font_dirs:
        raise SystemExit(f"ERROR: no font directories under {dir_path}")

    print(f"Reading {len(font_dirs)} fonts from {dir_path}")
    chunks, n_missing = [], 0
    for i, fd in enumerate(font_dirs):
        path = os.path.join(fd, 'sequence_relaxed.npy')
        if not os.path.exists(path):
            n_missing += 1
            continue
        seq = np.load(path).reshape(char_num, max_seq_len, dim_seq)
        chunks.append(seq[:, :, 4:].reshape(-1, 8))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(font_dirs)}")
    if n_missing:
        print(f"WARN: {n_missing} fonts had no sequence_relaxed.npy")
    if not chunks:
        raise SystemExit("ERROR: nothing loaded")
    return np.concatenate(chunks, axis=0)


def report(args_arr, n_bins):
    flat = args_arr.reshape(-1)
    # Padding rows are exactly zero across all 8 args; they would swamp bin 0.
    nonpad = args_arr[~np.all(args_arr == 0, axis=1)].reshape(-1)

    print(f"\n{'=' * 62}\n{n_bins}-bin grid\n{'=' * 62}")
    print(f"Coordinate values: {len(flat)} total, {len(nonpad)} after dropping all-zero "
          f"(padding) strokes")
    print(f"Raw range before quantization: [{flat.min():.4f}, {flat.max():.4f}]")

    binned = numericalize(nonpad, n=n_bins)
    hist = np.bincount(binned, minlength=n_bins)
    occupied = int((hist > 0).sum())

    print(f"\nOccupied bins: {occupied}/{n_bins}")
    print(f"  bin 0    : {hist[0]:>10}  ({100 * hist[0] / len(nonpad):.3f}%)   "
          f"clipped low, and padding_idx of arg_embed")
    print(f"  bin {n_bins - 1:<4} : {hist[-1]:>10}  ({100 * hist[-1] / len(nonpad):.3f}%)   clipped high")

    even, odd = hist[0::2].sum(), hist[1::2].sum()
    odd_frac = odd / max(even + odd, 1)
    odd_occupied = int((hist[1::2] > 0).sum())
    print(f"\nComb test (the 1.5 question)")
    print(f"  mass in even bins : {even:>10}  ({100 * (1 - odd_frac):.3f}%)")
    print(f"  mass in odd bins  : {odd:>10}  ({100 * odd_frac:.3f}%)")
    print(f"  odd bins occupied : {odd_occupied}/{n_bins // 2}")

    if odd_frac < 0.001:
        verdict = (f"COMB. The data sits on the {n_bins // 2}-bin grid. The n=64 preprocessing "
                   f"round-trip DID reach the persisted sequences.\n"
                   f"    => The {n_bins}-bin head has no information below the {n_bins // 2}-bin "
                   f"resolution.\n"
                   f"    => DROP E13. Adding bins cannot help without rebuilding the dataset.\n"
                   f"    => Use the {n_bins // 2}-bin row of the 2.3 oracle table.")
    elif odd_frac < 0.4:
        verdict = (f"PARTIAL comb ({100 * odd_frac:.1f}% odd). Unexpected. Some fonts may have been "
                   f"preprocessed differently.\n    => Investigate before scoping E13.")
    else:
        verdict = (f"FILLED. Odd and even bins are both populated, so the data was NOT round-tripped "
                   f"through the 64-bin grid.\n"
                   f"    => The model genuinely operates at {n_bins} bins.\n"
                   f"    => E13 stays on the list, gated only on the oracle result.")
    print(f"\n  VERDICT: {verdict}")

    top = np.argsort(hist)[::-1][:8]
    print(f"\n  Most populated bins: " + ", ".join(f"{b}({hist[b]})" for b in top))
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='./data/vecfont_dataset/')
    p.add_argument('--language', default='chn', choices=['eng', 'chn'])
    p.add_argument('--mode', default='train', choices=['train', 'test'])
    p.add_argument('--char_num', type=int, default=52)
    p.add_argument('--max_seq_len', type=int, default=None,
                   help='defaults to 71 for chn, 51 for eng')
    p.add_argument('--dim_seq', type=int, default=12)
    p.add_argument('--limit', type=int, default=None, help='only read the first N fonts')
    p.add_argument('--plot', action='store_true', help='save a PNG next to the script')
    args = p.parse_args()

    if args.max_seq_len is None:
        args.max_seq_len = 71 if args.language == 'chn' else 51
        print(f"max_seq_len defaulted to {args.max_seq_len} for {args.language}")

    arr = load_args(args.data_root, args.language, args.mode,
                    args.char_num, args.max_seq_len, args.dim_seq, args.limit)

    hist128 = report(arr, 128)
    report(arr, 64)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("\nWARN: matplotlib not installed, skipping the plot")
            return 0
        fig, ax = plt.subplots(figsize=(11, 3.4))
        ax.bar(np.arange(128), hist128, width=1.0)
        ax.set_xlabel('quantization bin (n=128)')
        ax.set_ylabel('count')
        ax.set_title(f'Coordinate bin occupancy, {args.language} {args.mode}')
        fig.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           f'bin_histogram_{args.language}_{args.mode}.png')
        fig.savefig(out, dpi=150)
        print(f"\nPlot written to {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
