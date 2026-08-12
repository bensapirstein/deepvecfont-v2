"""Rendered-metric validation: the selection signal `val_metric` was never able to be.

Why this exists
---------------
`compute_val_loss` in train.py returns `val_metric`, a weighted sum of six training-loss
terms, and every checkpoint in this project was selected by it. `PROJECT_PLAN.md` §0
records what that turned out to be worth: cross-run Spearman rho = 0.125 against the
rendered Error the report actually quotes (Job B, 2026-08-08), and switching to
best-`val_metric` selection on the English arm never improved a single candidate's
rendered score while flipping two of five from null to degrading (2026-08-10). The
reason is structural, not statistical: no term inside `val_metric` *is* the scored
quantity. `img_l1` is the image decoder's branch; the number in the report comes from
rasterizing the refinement decoder's SVG output.

This module computes the scored quantity itself, at every checkpoint, on a held-out
split: decode, rasterize with cairosvg, threshold, compare against the ground-truth
bitmap. Same rasterizer and same binarization as `eval_reconstruction_error.py`'s
`--gt_source raster` path, so the number is directly comparable to a test score, just
measured on different fonts.

The split it reads
------------------
`data/vecfont_dataset/<lang>/val`, carved out of the train split by
`scripts/make_val_split.py` and never trained on. This matters more than it looks:
train.py's own `val_loader` is built with `get_loader(..., 'test')`, so what the
codebase calls validation has always been the test split. Selecting checkpoints on a
rendered metric computed there would be selecting on the scored set. The whole point of
the held-out split is that the resulting protocol is honest.

What it is a proxy for, and what it is not
------------------------------------------
The reported number is best-of-N_s over 50 candidates on the test split. This runs at
`--render_val_samples 1` by default, so it measures a *single* decode rather than the
best of fifty. It is a proxy, and a much closer one than `val_metric` — same decoder,
same rasterizer, same binary masks, same units — but a proxy. At `--render_val_samples
> 1` the best-of-N selection mirrors test_few_shot.py's: candidates are ranked by IoU
against the image decoder's own output, not against the ground truth, because that is
what the test protocol does and a validation signal that used GT for selection would
measure something the test run cannot.

Cost. One font is one autoregressive decode of `char_num` glyphs plus `char_num`
cairosvg calls. At 20 Chinese fonts and one sample that is a couple of GPU-minutes,
against ~1 GPU-h for a 150-epoch Chinese run at six checkpoints. Budget for it in the
runbook rather than assuming it is free.
"""
import io

import numpy as np
import torch
from PIL import Image

from data_utils.svg_utils import render

# Same threshold as eval_reconstruction_error.py and models/util_funcs.cal_iou, so a
# number from here and a number from there mean the same thing.
IOU_THRESH = 255 * 3 / 4


def rasterize_mask(svg_str, img_size):
    """SVG string -> white-background uint8 array, matching common_utils.trans2_white_bg."""
    import cairosvg  # imported here so a CPU-only checkout can import this module
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'),
                                 output_width=img_size, output_height=img_size)
    arr = np.array(Image.open(io.BytesIO(png_bytes)))
    return 255 - arr[:, :, 3]


def _seq_to_mask(one_seq, img_size):
    """One decoded sequence -> ink mask, or None if it does not render.

    `render()` raises on malformed command sequences and cairosvg raises on the SVG
    strings it can produce, which is the same failure eval_reconstruction_error.py
    reports as renderability. Both are caught here and counted rather than raised.
    """
    try:
        svg = render(one_seq.cpu().numpy())
        arr = rasterize_mask(svg, img_size)
    except Exception:
        return None
    return arr < IOU_THRESH


def _iou(m1, m2):
    union = np.sum(m1 + m2)
    if union == 0:
        return 1.0
    return float(np.sum(m1 * m2) / union)


@torch.no_grad()
def rendered_val_metrics(model_main, val_loader, opts, n_samples=1, max_fonts=0):
    """Decode the val split, rasterize, and score it the way the report scores things.

    Returns a dict with `l1`, `siou`, `renderability` and `n_glyphs`. `l1` is the
    quantity the paper calls Error: the fraction of the img_size^2 pixels on which the
    two binary masks disagree, averaged over glyphs.

    Glyphs that fail to render are counted in `renderability` and, following
    eval_reconstruction_error.py, excluded from the L1 and s-IoU means rather than
    scored as total failures. A run whose renderability drops is therefore reporting an
    L1 over a smaller and easier population, which is why `renderability` is returned
    alongside and logged next to the other two.
    """
    was_training = model_main.training
    model_main.eval()

    l1s, ious = [], []
    n_rendered, n_total = 0, 0

    for font_idx, data in enumerate(val_loader):
        if max_fonts and font_idx >= max_fonts:
            break
        for key in data:
            data[key] = data[key].cuda()

        # Ground truth, straight off the loader. SVGDataset applies `1 - x/255`, so ink
        # sits at 1.0 and background at 0.0; `> 0.25` is the same cut as
        # `raw < 255*3/4` on the stored bitmap.
        gt_masks = (data['rendered'][0] > 0.25).cpu().numpy()

        best_masks = [None] * opts.char_num
        best_iou = [-1.0] * opts.char_num

        for _ in range(max(1, n_samples)):
            ret_dict, _ = model_main(data, mode='test')
            sampled = ret_dict['svg']['sampled_2'].clone().detach()   # refinement decoder
            # Reference for best-of-N selection, mirroring test_few_shot.py: the image
            # decoder's own output, NOT the ground truth.
            img_out = ret_dict['img']['out'].detach()
            ref_masks = (img_out.reshape(opts.char_num, opts.img_size, opts.img_size)
                         > 0.25).cpu().numpy()

            for i, one_seq in enumerate(sampled):
                mask = _seq_to_mask(one_seq, opts.img_size)
                if mask is None:
                    continue
                score = _iou(mask, ref_masks[i]) if n_samples > 1 else 1.0
                if score > best_iou[i]:
                    best_iou[i] = score
                    best_masks[i] = mask

        for i in range(opts.char_num):
            n_total += 1
            if best_masks[i] is None:
                continue
            n_rendered += 1
            gt = gt_masks[i]
            l1s.append(float(np.mean(np.abs(best_masks[i].astype(float) - gt.astype(float)))))
            ious.append(_iou(best_masks[i], gt))

    if was_training:
        model_main.train()

    return {
        'l1': float(np.mean(l1s)) if l1s else float('nan'),
        'siou': float(np.mean(ious)) if ious else float('nan'),
        'renderability': (n_rendered / n_total) if n_total else float('nan'),
        'n_glyphs': n_total,
    }
