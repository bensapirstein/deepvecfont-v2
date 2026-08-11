"""Pull the handful of real glyphs the figures need out of ../data.zip.

Usage:  python report/extract_assets.py

Writes report/assets/glyphs.npz, about 1 MB, so the figure scripts run without
unpacking the 5 GB dataset archive and without the cluster. The npz is committed,
so this script only needs rerunning if you want different fonts.

WHICH FONTS, AND WHY
--------------------
Both are from the English *test* split, so no glyph in any figure is data the
model trained on.

  0013  a light geometric sans. Its curves stay legible at figure size, which is
        what the anatomy and teaser figures need. Chosen by eye from a contact
        sheet of the first fifteen test fonts, kept at assets/font_choice_contact_sheet.png
        so the choice is visible rather than asserted.
  0007  a serif, so a figure can show that a font is a style rather than a fixed
        set of shapes.

WHAT IS IN THE npz
------------------
  charset                 the 52 characters, 'A'..'Z' then 'a'..'z'
  <fid>_sequence_relaxed  (52, 51, 12)  the outline as a command sequence.
                          Dims 0:4 are a one-hot over [EOS, M, L, C]; dims 4:12
                          are [start_xy, ctrl1_xy, ctrl2_xy, end_xy] in the
                          0..24 viewBox space that svg_utils renders into.
  <fid>_rendered_64       (52, 64, 64)  the dataset's own raster. This is the
                          ground truth the reported metric compares against, and
                          it was produced by a *different* rasterizer than the one
                          the evaluation renders candidates with. That mismatch is
                          worth 0.0455 on Chinese and is a finding in section 6.3.
  <fid>_seq_len           (52, 1)  commands per glyph, before padding.
"""
import os
import sys
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZIP = os.path.join(ROOT, "data.zip")
OUT = os.path.join(HERE, "assets", "glyphs.npz")

FONTS = ["0013", "0007"]
BASE = "deepvecfont_v2/data/vecfont_dataset/eng/test"
CHAR_NUM, MAX_SEQ_LEN, FEAT = 52, 51, 12

if not os.path.exists(ZIP):
    sys.exit(f"ERROR: {ZIP} not found. This script needs the dataset archive.")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
store = {}
with zipfile.ZipFile(ZIP) as z:
    store["charset"] = np.array(
        z.read("deepvecfont_v2/data/char_set/eng.txt").decode().strip().replace("\n", ""))
    for fid in FONTS:
        for name, shape in (("sequence_relaxed", (CHAR_NUM, MAX_SEQ_LEN, FEAT)),
                            ("rendered_64", (CHAR_NUM, 64, 64)),
                            ("seq_len", (CHAR_NUM, 1))):
            with z.open(f"{BASE}/{fid}/{name}.npy") as fh:
                store[f"{fid}_{name}"] = np.load(fh).reshape(shape)

np.savez_compressed(OUT, **store)
print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1e6:.2f} MB)  fonts={FONTS}")
