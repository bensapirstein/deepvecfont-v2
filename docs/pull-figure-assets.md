# Pulling model-output glyphs off the cluster, for the comparison figures

Written 2026-08-11 (day 9). Cheap, bounded, no GPU, no training. Run it on the cluster and
bring back one tarball, on the order of a few hundred KB.

## Why

The report's explanatory figures (`fig1_teaser`, `fig2_anatomy`) are built from the dataset's
own outlines, which the Mac has via `data.zip`. The **model comparison** figures need
something the Mac does not have: the outlines the models actually drew. Those live only in
`experiments/*/results/`, which is a symlink into `/data/bens` on the cluster.

## What we need, and why it is so small

`test_few_shot.py` writes one merge HTML per test font:

```
experiments/{name_exp}_main_model/results/{name_ckpt}/{font_idx}/svgs_merge/{name_ckpt}_syn_merge_{font_idx}.html
```

Each of those files already contains **both halves of the comparison**: the first `char_num`
SVGs are the model's selected best-of-N output after parallel refinement, and the next
`char_num` are the ground truth. `eval_reconstruction_error.py` reads them with
`extract_svgs()` and scores exactly these, so a figure drawn from them shows the same glyphs
the numbers in section 5 were computed on. That is the point: the figure and the table cannot
disagree.

They are text, so they are tiny. We do not need checkpoints, images, or anything else.

## The pull

```bash
cd ~/deepvecfont-v2

# The four runs the report compares. Chinese first: baseline and E9 at the matched epoch,
# then the same pair on English. Checkpoint names come straight from RESULTS.csv.
PAIRS="
seedfloor_1111_chn_main_model/results/150_6040.ckpt
e9_sigma050_chn_main_model/results/150_6040.ckpt
eng_seedfloor_1111_main_model/results/640_205921.ckpt
e9_sigma050_1111_eng_main_model/results/640_205921.ckpt
"

# Look before pulling: confirm each path exists and see how many fonts are under it.
for p in $PAIRS; do
  n=$(ls -d experiments/$p/*/svgs_merge 2>/dev/null | wc -l)
  echo "$n fonts   experiments/$p"
done
```

**If a path prints `0`**, the checkpoint directory is named differently. List what is actually
there and use that name instead, rather than guessing:

```bash
ls experiments/seedfloor_1111_chn_main_model/results/
ls experiments/eng_seedfloor_1111_main_model/results/
```

The English checkpoint name in particular is a guess from the epoch number: `RESULTS.csv` has
English seed 1111 at epoch 640, and the step count in the filename depends on the loader
length, so read it off the directory rather than trusting the line above.

Then take **only the first six fonts of each**, which is more than any figure will use:

```bash
rm -f /tmp/figure-assets.tar.gz
FILES=""
for p in $PAIRS; do
  for d in $(ls -d experiments/$p/*/svgs_merge 2>/dev/null | sort | head -6); do
    FILES="$FILES $d"
  done
done
tar czf /tmp/figure-assets.tar.gz $FILES
du -h /tmp/figure-assets.tar.gz
```

Expect well under 5 MB. If it comes back much larger, something other than HTML got swept in;
check with `tar tzf /tmp/figure-assets.tar.gz | head`.

Bring `/tmp/figure-assets.tar.gz` back to the Mac and unpack it into `report/assets/model_output/`.

## Reading rule, fixed here before any figure is drawn

Worth stating in advance, because a glyph figure is the easiest place in a report to cheat
without meaning to.

1. **The fonts and characters shown are chosen before the glyphs are looked at.** Take the
   first six fonts in test order and a fixed character set. Do not scan for the flattering ones.
2. **If a figure shows a case where E9 looks better, it is captioned as one case, not as
   evidence.** Section 5.3 already says E9's Chinese mean does not clear its floor, and a
   picture cannot upgrade that.
3. **Show a failure too.** The honest version of this figure includes at least one glyph where
   both models are visibly wrong, and Chinese at 0.16 L1 will not be short of candidates.
4. **English and Chinese get equal space.** The report's own finding is that they disagree, so
   a figure showing only the language that behaves better would misrepresent it.

## What gets built from it

Two figures, both from `report/make_glyph_figures.py` once the assets are present:

- **Ground truth against baseline against E9**, same font, same characters, both languages.
  This is the qualitative companion to the table in section 5.1.
- **A failure strip**: the glyphs where reconstruction error is worst, which is the fastest way
  to show what an L1 of 0.16 on Chinese actually looks like, and why the metric being
  floor-dominated (section 3) matters.
