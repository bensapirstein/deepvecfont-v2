# DeepVecFont-v2 — reproduction and architectural sweep

Final project for **Generative Models for Text and Images**, Reichman University.

A fork of [`yizhiwang96/deepvecfont-v2`](https://github.com/yizhiwang96/deepvecfont-v2)
(Wang et al., *DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with
Higher Quality*, CVPR 2023, [arXiv:2303.14585](https://arxiv.org/abs/2303.14585)).

We reproduce the model on Chinese and English, check the reproduction against the authors'
released weights, and run 21 single-factor architectural changes against it — each judged
against the spread the unmodified model produces when it is simply re-seeded.

**Report: [`report/REPORT.pdf`](report/REPORT.pdf)** (source: [`report/REPORT.md`](report/REPORT.md))

---

## How to read this fork

| Branch | What it is |
|---|---|
| `main` | Pristine mirror of upstream. Never committed to. |
| `submission` | The delivered project, tagged `v1.0-submission`. Cut from `repro` with the working record removed. |
| `repro` | The full working branch, kept for provenance: the project plan, the cluster runbooks, and the commit history behind every number. |

**Everything we wrote is the `main..submission` diff.** One command separates the entire
contribution from the upstream code it sits on:

```bash
git diff main..submission --stat
git diff main..submission -- train.py models/ options.py     # the model changes
```

Upstream's own documentation — installation, the dataset build, the custom-dataset
pipeline, font copyrights — is unchanged and still authoritative. Where this README is
brief, [upstream's README](https://github.com/yizhiwang96/deepvecfont-v2/blob/main/README.md)
is the reference.

---

## Required submission items

| # | Item | Where |
|---|---|---|
| 1 | **Reconstructed model** + training and evaluation code | `models/`, `train.py`, `test_few_shot.py`, `eval_reconstruction_error.py`. Trained with the baseline command in [`docs/REPRODUCE.md`](docs/REPRODUCE.md) §3. |
| 2 | **Improved model** + training and evaluation code | Same code path. The improved configuration (E9) is `--enc_noise_std_train 0.5`; all 21 candidates are flags on the same entry points, so a candidate's default reproduces the baseline exactly. [`docs/REPRODUCE.md`](docs/REPRODUCE.md) §4. |
| 3 | **Dataset** | The authors' released dataset, download links below. Our Chinese rebuild at the paper's 10× augmentation is one command over it, and the held-out validation splits we carved are committed in [`data_splits/`](data_splits/). |
| — | Trained checkpoints | Google Drive, link below. |

---

## Getting the inputs

### Dataset

We use the authors' released dataset unchanged. From
[upstream](https://github.com/yizhiwang96/deepvecfont-v2#dataset):

- [OneDrive](https://1drv.ms/u/s!AkDQSKsmQQCghdBAA2WANQ3KcNV6uQ?e=p6NMIP)
- [Baiduyun](https://pan.baidu.com/s/1zyVBDazvSVIAGnmHQHO1GA) (password `pmr2`)

Unpack so that `data/` sits at the repository root. It contains `char_set/`,
`font_ttfs/`, `font_sfds/` and `vecfont_dataset/`.

Two notes that matter for reproducing our numbers rather than upstream's:

- The released train/test split differs from the paper's. The paper's split is in
  [`statics/v1_train_font_ids.txt`](statics/v1_train_font_ids.txt) and
  [`statics/v1_test_font_ids.txt`](statics/v1_test_font_ids.txt).
- The released Chinese `vecfont_dataset` is built at **6× augmentation**, not the 10× the
  paper states in Sec. 4.1. Section 5.6 of the report re-runs the sweep on a 10× rebuild;
  [`docs/REPRODUCE.md`](docs/REPRODUCE.md) §6 is the one command that produces it.

### Checkpoints

Our trained weights, four runs, one seed each:

> **Google Drive: [dvf_release](https://drive.google.com/drive/folders/1UNJARldxCeu_Af5_QDHYFK910dFLAObg?usp=sharing)**

| Archive | What it is | Report table |
|---|---|---|
| `chn_baseline_1111.ckpt` | Chinese reconstruction, 150 epochs | §5.1, *Our reconstruction* |
| `chn_e9_sigma050_1111.ckpt` | Chinese improved model, `--enc_noise_std_train 0.5` | §5.1, *E9* |
| `eng_baseline_1111.ckpt` | English reconstruction, 630 epochs | §5.1, *Our reconstruction* |
| `eng_e9_sigma050_1111.ckpt` | English improved model | §5.1, *E9* |

Each archive carries the checkpoint and the run's `checkpoint_metrics.csv`, so the
selection criterion is auditable rather than asserted. The report's three-seed means come
from three such runs each; seed 1111 is released as the representative draw, and
`RESULTS.csv` carries all 140 scored checkpoints as rows.

The **authors'** English and Chinese checkpoints, which we also score in §5.1, stay at
[upstream's own links](https://github.com/yizhiwang96/deepvecfont-v2#trained-checkpoints).

---

## Quickstart

```bash
conda create -n dvf_v2 python=3.9 && conda activate dvf_v2
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 \
  --extra-index-url https://download.pytorch.org/whl/cu117
pip install tensorboardX einops timm scikit-image cairosvg pandas scipy
```

Score a released checkpoint, which is the shortest path to a number in the report:

```bash
# decode the test split, best-of-50 candidates per glyph
CUDA_VISIBLE_DEVICES=0 python test_few_shot.py --mode test \
  --name_exp chn_baseline_1111 --model_name main_model --language chn \
  --max_seq_len 71 --ref_nshot 8 --ref_char_ids 0,1,2,3,26,27,28,29 \
  --batch_size 1 --n_samples 50 --name_ckpt <ckpt>

# the paper's Table 2 metric, plus s-IoU, SSIM and renderability
python eval_reconstruction_error.py \
  --exp_dir experiments/chn_baseline_1111_main_model \
  --name_ckpt <ckpt> --gt_source raster
```

Training, the improved model, the seed floor and the full sweep are all in
**[`docs/REPRODUCE.md`](docs/REPRODUCE.md)**.

---

## Results

Reconstruction error (L1) on the paper's protocol, lower is better. Our columns are
three-seed means ± sample standard deviation.

| | Paper | Released weights, our harness | Our reconstruction | Improved (E9) |
|---|---|---|---|---|
| Chinese | 0.080 | 0.1629 | 0.1621 ± 0.0047 | 0.1581 ± 0.0046 |
| English | 0.052 | 0.0658 | 0.0597 ± 0.0021 | 0.0601 ± 0.0007 |

Two things this table is doing at once. Our Chinese reconstruction lands **0.0008** from
the authors' own released weights scored through the same code, so the reproduction is
faithful and the distance to the published 0.080 is a property of the evaluation rather
than of our training — report §3 measures where it lives. And the improved model's Chinese
gain, while same-signed at every seed on all three metrics, sits **inside** the 0.0097
seed-noise floor we committed to before running anything, so §5.3 reports it as a
direction rather than as a gain of a stated size.

That floor is the point of the project. Twenty-six candidate configurations span 0.0101
between them; three re-seedings of the unmodified model span 0.0097.

Every figure and every number in the report is regenerated from
[`RESULTS.csv`](RESULTS.csv) at build time and checked by
[`report/verify_report.py`](report/verify_report.py), which fails the build on any drift.

---

## Layout

```
models/  train.py  test_few_shot.py  options.py   upstream model, our flags on top
eval_reconstruction_error.py                      the paper's metric, our implementation
render_val.py                                     rendered-metric checkpoint selection
data_utils/                                       upstream dataset pipeline (+ augment fix)
data_splits/                                      the held-out val splits we carved
scripts/                                          sweep drivers, the seed floor, analysis
report/                                           REPORT.md, figures, the number verifier
RESULTS.csv                                       every scored checkpoint, one row each
docs/REPRODUCE.md                                 end-to-end reproduction
docs/PROVENANCE.md                                where the code's PROJECT_PLAN citations point
archive/FLOW_MATCHING_PLAN.md                     the head we designed, costed and dropped (§6.5)
```

## What is deliberately not here

Model outputs, checkpoints and decode trees are not committed. They are large, they
regenerate from the commands above, and the numbers derived from them are in `RESULTS.csv`
instead. The papers themselves are linked, not vendored. The working branch `repro` holds
the project plan and the cluster runbooks if you want the process rather than the result.

## Upstream

Font copyrights, acknowledgments and the custom-dataset pipeline are upstream's and
unchanged; see [their README](https://github.com/yizhiwang96/deepvecfont-v2/blob/main/README.md).
The Chinese fonts come from [Founder](https://www.foundertype.com/) and cannot be used
commercially without their permission.

```bibtex
@inproceedings{wang2023deepvecfont,
  title={DeepVecFont-v2: Exploiting Transformers to Synthesize Vector Fonts with Higher Quality},
  author={Wang, Yuqing and Wang, Yizhi and Yu, Longhui and Zhu, Yuesheng and Lian, Zhouhui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18320--18328},
  year={2023}
}
```
