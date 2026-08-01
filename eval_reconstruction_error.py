"""
Reproduces the "reconstruction error" metric from Table 2 of the DeepVecFont-v2 paper:
the average L1 distance (on binarized 64x64 masks) between each synthesized glyph's
rasterized image and its ground-truth glyph image, using the best-of-N_s candidate
already selected by test_few_shot.py (highest IOU vs. the image decoder's own output).

Run test_few_shot.py first so that experiments/{name_exp}/results/{name_ckpt}/*/svgs_merge/*.html exist.
"""
import argparse
import glob
import os
import re
from io import BytesIO

import cairosvg
import numpy as np
from PIL import Image

IOU_THRESH = 255 * 3 / 4


def extract_svgs(html_path):
    content = open(html_path, encoding='utf-8').read()
    return re.findall(r'<svg.*?</svg>', content, flags=re.DOTALL)


def render_svg_mask(svg_str, img_size):
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'), output_width=img_size, output_height=img_size)
    arr = np.array(Image.open(BytesIO(png_bytes)))
    return 255 - arr[:, :, 3]  # matches data_utils.common_utils.trans2_white_bg


def cal_iou_l1(mask_arr1, mask_arr2):
    m1 = mask_arr1 < IOU_THRESH
    m2 = mask_arr2 < IOU_THRESH
    iou = np.sum(m1 * m2) / np.sum(m1 + m2)
    l1 = np.mean(np.abs(m1.astype(float) - m2.astype(float)))
    return iou, l1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', required=True, help='e.g. experiments/dvf_base_exp_chn_main_model')
    parser.add_argument('--name_ckpt', required=True, help='e.g. 125_5040_valloss3.8273.ckpt; results live under exp_dir/results/name_ckpt/')
    parser.add_argument('--char_num', type=int, default=52)
    parser.add_argument('--img_size', type=int, default=64)
    args = parser.parse_args()

    results_dir = os.path.join(args.exp_dir, 'results', args.name_ckpt)
    font_dirs = sorted(glob.glob(os.path.join(results_dir, '*')))

    all_l1, all_iou, per_font = [], [], []

    for font_dir in font_dirs:
        merge_htmls = glob.glob(os.path.join(font_dir, 'svgs_merge', '*.html'))
        if not merge_htmls:
            print(f"WARN {font_dir}: no svgs_merge html, skipping")
            continue
        svgs = extract_svgs(merge_htmls[0])
        if len(svgs) != 2 * args.char_num:
            print(f"WARN {font_dir}: expected {2 * args.char_num} svgs, got {len(svgs)}, skipping")
            continue
        synth_svgs = svgs[:args.char_num]

        font_l1, font_iou = [], []
        for i in range(args.char_num):
            gt_path = os.path.join(font_dir, 'imgs', f"{i:02d}_gt.png")
            if not os.path.exists(gt_path):
                continue
            gt_mask = np.array(Image.open(gt_path))[:, :, 0]
            try:
                synth_mask = render_svg_mask(synth_svgs[i], args.img_size)
            except Exception as e:
                print(f"WARN render failed {font_dir} char {i}: {e}")
                continue
            iou, l1 = cal_iou_l1(synth_mask, gt_mask)
            font_l1.append(l1)
            font_iou.append(iou)

        all_l1.extend(font_l1)
        all_iou.extend(font_iou)
        per_font.append((os.path.basename(font_dir), np.mean(font_l1), np.mean(font_iou)))

    print(f"\nEvaluated {len(per_font)} fonts, {len(all_l1)} glyph samples\n")
    for name, l1, iou in per_font:
        print(f"  font {name}: L1={l1:.4f}  IOU={iou:.4f}")

    print(f"\nReconstruction Error (L1, matches paper's 'Error'): {np.mean(all_l1):.4f}")
    print(f"Mean IOU: {np.mean(all_iou):.4f}")


if __name__ == '__main__':
    main()
