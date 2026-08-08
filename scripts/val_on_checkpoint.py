#!/usr/bin/env python3
"""Compute `val_metric` for arbitrary checkpoints, including ones we did not train.

PROJECT_PLAN.md 8 item 12. `checkpoint_metrics.csv` only exists for runs trained
after 2026-08-03, and never for the authors' released weights, so there has been no
way to ask the one question that decides whether checkpoint selection is sound:

    does `val_metric` rank the checkpoints of a single run the same way the rendered
    Error ranks them?

`val_metric_correlation.py` answers the *cross-run* version of this and found rho =
0.125. It says in its own docstring that within-run selection "was never affected",
because a constant loss-scale factor cancels inside one run. That reasoning is about
scale and does not touch ordering, and the ordering is what `prune_checkpoints` acts
on. This script supplies the missing side of the comparison.

What makes the question live, from RESULTS.csv:

    official_eng 500 / 550 / 600  ->  rendered L1 0.0645 / 0.0649 / 0.0658 (raster)
                                                  0.0569 / 0.0573 / 0.0584 (svg)

Three checkpoints of one training run, monotonically WORSE on the reported metric the
longer it trains, and the authors shipped 600. If `val_metric` also ranks 500 best
then selection is fine and the authors simply released their last checkpoint. If
`val_metric` prefers 600, the released code selects on a criterion that does not track
its own reported metric, and every checkpoint in this project was chosen the same way.

Usage, one checkpoint per invocation (the model is rebuilt each time, which is cheap
next to the validation pass):

    CUDA_VISIBLE_DEVICES=3 python scripts/val_on_checkpoint.py \
        --ckpt_path experiments/official_eng_main_model/checkpoints/500_160821.ckpt \
        --language eng --max_seq_len 51 --ref_nshot 4 \
        --csv_out val_metric_audit.csv --tag official_eng_500

Appends one row per call to --csv_out, so a loop over checkpoints builds the table.

NOTE. This is a validation pass, not a decode. It costs a couple of minutes, not the
hours `test_few_shot.py` costs, which is the entire reason the audit is affordable.

NOTE. The flags that define the *architecture* must match the checkpoint being loaded
(--n_args_bins, --enc_depth, --dec_d_ff, --bottleneck_bits, --ngf, --img_norm,
--n_layers_refine, --enc_final_norm). Loading with the wrong ones fails at
load_state_dict rather than silently, which is the behaviour we want, but read the
error as "wrong flags" before reading it as "bad checkpoint".
"""
import csv
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from dataloader import get_loader          # noqa: E402
from models.model_main import ModelMain    # noqa: E402
from options import get_parser_main_model  # noqa: E402
from train import compute_val_loss         # noqa: E402


FIELDS = ['tag', 'ckpt_path', 'language', 'val_metric',
          'val_img_l1', 'val_img_vggpt', 'val_svg_total', 'val_svg_para_total',
          'val_svg_cmd', 'val_svg_args', 'val_svg_aux']


def main():
    parser = get_parser_main_model()
    parser.add_argument('--ckpt_path', required=True,
                        help='path to the .ckpt to score; may be any checkpoint, including a released one')
    parser.add_argument('--csv_out', default=None,
                        help='append one row here; created with a header if absent')
    parser.add_argument('--tag', default=None,
                        help='label for the row, e.g. official_eng_500. Defaults to the checkpoint basename')
    opts = parser.parse_args()

    # compute_val_loss reads opts.loss_w_l1 and opts.loss_w_pt_c, and both decoder
    # passes run, so mode must be one the model's forward understands as validation.
    opts.mode = 'val'

    if not os.path.exists(opts.ckpt_path):
        print(f"no such checkpoint: {opts.ckpt_path}", file=sys.stderr)
        return 2

    val_loader = get_loader(opts.data_root, opts.img_size, opts.language, opts.char_num,
                            opts.max_seq_len, opts.dim_seq, opts.batch_size_val, 'test')

    model_main = ModelMain(opts)
    model_main.cuda()

    state = torch.load(opts.ckpt_path, map_location='cuda')
    # train.py saves {'model': state_dict, 'opt': ..., ...}; released checkpoints use
    # the same layout. Fall back to treating the file as a bare state_dict.
    sd = state.get('model', state) if isinstance(state, dict) else state
    missing, unexpected = model_main.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"WARNING  missing={len(missing)} unexpected={len(unexpected)} keys. "
              f"If either is large the architecture flags do not match this checkpoint.",
              file=sys.stderr)
        for k in list(missing)[:5]:
            print(f"  missing:    {k}", file=sys.stderr)
        for k in list(unexpected)[:5]:
            print(f"  unexpected: {k}", file=sys.stderr)

    loss_val, val_metric = compute_val_loss(model_main, val_loader, opts)

    tag = opts.tag or os.path.basename(opts.ckpt_path)
    row = {
        'tag': tag,
        'ckpt_path': opts.ckpt_path,
        'language': opts.language,
        'val_metric': f"{val_metric:.6f}",
        'val_img_l1': f"{float(loss_val['img']['l1']):.6f}",
        'val_img_vggpt': f"{float(loss_val['img']['vggpt']):.6f}",
        'val_svg_total': f"{float(loss_val['svg']['total']):.6f}",
        'val_svg_para_total': f"{float(loss_val['svg_para']['total']):.6f}",
        'val_svg_cmd': f"{float(loss_val['svg']['cmd']):.6f}",
        'val_svg_args': f"{float(loss_val['svg']['args']):.6f}",
        'val_svg_aux': f"{float(loss_val['svg']['aux']):.6f}",
    }

    print(f"{tag}: val_metric = {val_metric:.6f}  "
          f"(img_l1 {row['val_img_l1']}, svg {row['val_svg_total']}, "
          f"svg_para {row['val_svg_para_total']})")

    if opts.csv_out:
        exists = os.path.exists(opts.csv_out)
        with open(opts.csv_out, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"appended to {opts.csv_out}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
