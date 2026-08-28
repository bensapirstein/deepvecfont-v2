# Release workplan

What is left before the public repository is complete. Everything Claude could do on the
Mac is done: branches `repro` and `submission` exist, `submission` is tagged
`v1.0-submission`, and a fresh clone of it builds `report/REPORT.pdf` with the verifier
passing. Three things remain, and all three need something this sandbox does not have:
the cluster, a browser, or GitHub credentials.

Order matters. §1 can invalidate §4, so run §1 first even if you plan to upload later.

---

## 1. Cluster: does the hardware evidence still exist?

**Read this before running it.** On 2026-08-06 the disk went from 182 GB to 75 GB by
dropping checkpoints and decode trees "for everything already in `RESULTS.csv`". Both
Chinese runs below were scored before that date. They may be gone. The English runs were
trained on 2026-08-07 and after, so they are likelier to have survived. Find out before
promising anything in the README.

These are the four runs the report's §5.1 table and figures 7 through 9 come from, with the
exact checkpoint each was scored at:

| Role | `name_exp` | Directory | Checkpoint | Seed |
|---|---|---|---|---|
| Chinese reconstruction | `seedfloor_1111_chn` | `experiments/seedfloor_1111_chn_main_model` | `150_6040.ckpt` | 1111 |
| Chinese improved (E9) | `e9_sigma050_chn` | `experiments/e9_sigma050_chn_main_model` | `150_6040.ckpt` | 1111 |
| English reconstruction | `eng_seedfloor_1111` | `experiments/eng_seedfloor_1111_main_model` | `640_205761.ckpt` | 1111 |
| English improved (E9) | `e9_sigma050_1111_eng` | `experiments/e9_sigma050_1111_eng_main_model` | `640_205761.ckpt` | 1111 |

`e9_sigma050_chn` carries no seed in its name and is seed 1111; that is the original Tier 3b
run, not a later replicate.

On the cluster, in `~/deepvecfont-v2`:

```bash
for d in seedfloor_1111_chn e9_sigma050_chn eng_seedfloor_1111 e9_sigma050_1111_eng; do
  echo "== $d"
  ls -lh experiments/${d}_main_model/*.ckpt 2>/dev/null || echo "   NO CHECKPOINTS"
  ls -l  experiments/${d}_main_model/checkpoint_metrics.csv 2>/dev/null || echo "   no metrics csv"
done
df -h /data/bens
```

**Decision rule, fixed here before you look:**

- **All four present.** Go to §2.
- **The two Chinese present, English gone.** Release the Chinese pair and say so in the
  README rather than quietly shipping half a table. Retraining English is ~17.5 GPU-h per
  run and buys a download convenience, not a result; it is not worth 35 GPU-h this late.
- **Chinese gone.** Retrain them. A 150-epoch Chinese run is about one GPU-hour, and both
  can run at once on separate GPUs. The commands are §3 and §4 of `docs/REPRODUCE.md` with
  `--name_exp seedfloor_1111_chn` and `--name_exp e9_sigma050_chn --enc_noise_std_train 0.5`,
  both at `--seed 1111`.
  **A retrained checkpoint is not the one that produced the numbers in the report.** Seed
  and code are identical, so it should land within the decode-noise term, but it is a
  re-run, not the artifact. If you release a retrain, score it once and put its actual L1
  in the README's checkpoint table beside the report's, whatever it says. Do not present a
  fresh run as the run that was measured.
- **Nothing survives and you do not want to retrain.** Ship the repository without weights.
  Say it in one line in the README: the checkpoints did not outlive the disk quota, and
  everything needed to retrain them is in `docs/REPRODUCE.md`. That is an honest release.
  Delete the Drive block rather than leaving a dead link.

## 2. Cluster: package what survived

One archive per run, each carrying the checkpoint and its `checkpoint_metrics.csv`, so a
reader can see the selection criterion rather than take the choice on trust.

```bash
cd ~/deepvecfont-v2
mkdir -p /tmp/dvf_release

pack () {   # pack <dir> <ckpt> <outname>
  local d=experiments/$1_main_model
  tar -czf /tmp/dvf_release/$3.tar.gz \
      -C "$d" "$2" checkpoint_metrics.csv
  echo "$3.tar.gz  $(du -h /tmp/dvf_release/$3.tar.gz | cut -f1)"
}

pack seedfloor_1111_chn     150_6040.ckpt    chn_baseline_1111
pack e9_sigma050_chn        150_6040.ckpt    chn_e9_sigma050_1111
pack eng_seedfloor_1111     640_205761.ckpt  eng_baseline_1111
pack e9_sigma050_1111_eng   640_205761.ckpt  eng_e9_sigma050_1111

cd /tmp/dvf_release && sha256sum *.tar.gz | tee SHA256SUMS && ls -lh
```

Keep `SHA256SUMS`. It goes in the Drive folder next to the archives and costs nothing.

Then pull them to the Mac:

```bash
# from the Mac, not the cluster
scp -r <cluster>:/tmp/dvf_release ~/Downloads/dvf_release
```

## 3. Google Drive

Upload the folder, then set link sharing to **anyone with the link, viewer**. A Drive link
that needs an access request is worse than no link: the grader will not wait.

Check it in a private window before you paste it anywhere.

## 4. Paste the link into the README

The placeholder is one line, on both branches:

```
> **Google Drive: _(link to be added)_**
```

Replace it on `repro`, then bring `submission` up to it. `submission` is `repro` plus one
cleanup commit, and that relationship is worth keeping:

```bash
cd "~/Projects/ACADEMIC/MLDS/Generative Models for Text and Images/deepvecfont-v2"
git checkout repro
# edit README.md
git commit -am "Add the Drive link for the released checkpoints"
git checkout submission && git rebase repro
git tag -f -a v1.0-submission -m "Final project submission"
```

If §1 sent you down a branch where fewer than four archives exist, fix the checkpoint table
in the README in the same edit. It currently promises four rows.

**One thing to watch on any future rebase.** The cleanup commit deletes the runbooks by
name, including this file. A file added to `docs/` on `repro` after the cut will follow a
rebase onto `submission` unless you add it to that commit's deletion list.

## 5. Push

The sandbox has no GitHub credentials, so this is yours. Everything is committed locally.

```bash
git push origin repro
git push origin submission
git push origin v1.0-submission
```

`origin/repro` is currently four commits behind; `main` is untouched and needs no push.

## 6. GitHub settings, in the browser

- **Default branch: `submission`.** GitHub shows the default branch's README on the
  repository page, and today that is `main`, which is upstream's mirror and upstream's
  README. A visitor would land on the wrong project. Changing the default does not touch
  `main`, so the pristine mirror and `git diff main..submission` both survive it.
- **Description:** something like *Reproduction and architectural sweep of DeepVecFont-v2
  (CVPR 2023). Final project, Generative Models for Text and Images, Reichman University.*
- **Un-watch upstream's issues** if the fork is noisy.
- Optionally cut a **release** from `v1.0-submission` and attach `report/REPORT.pdf`, so the
  report has a stable download URL that is not a branch path.

## 7. Final check, five minutes

```bash
git clone --branch submission --single-branch \
  https://github.com/bensapirstein/deepvecfont-v2.git /tmp/relcheck
cd /tmp/relcheck && bash report/build.sh && python3 scripts/check_infra.py | tail -3
```

Expect `0 failed claim(s)`, `wrote report/REPORT.pdf`, and 224 preflight checks passing.
Then open the repository page in a private window, confirm the README renders, the Drive
link opens, and `report/REPORT.pdf` displays.
