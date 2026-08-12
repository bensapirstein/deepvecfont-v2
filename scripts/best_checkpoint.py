#!/usr/bin/env python3
"""Print the best checkpoint filename for one experiment.

Reads experiments/<name>_main_model/logs/checkpoint_metrics.csv (written by train.py at
every checkpoint save) and falls back to legacy filename parsing for experiment dirs that
predate that manifest. See checkpoint_log.py.

    python scripts/best_checkpoint.py experiments/<name>_main_model
    python scripts/best_checkpoint.py experiments/<name>_main_model --criterion val_render_l1

--criterion val_metric (the default) is the training-loss proxy every run before
2026-08-12 was selected on. --criterion val_render_l1 is the rendered Error on the
held-out val split, i.e. the quantity the report reports, and only exists for runs
trained with --render_val_freq > 0. Asking for it on a run that never logged it is an
error, deliberately: falling back to val_metric would hide exactly the difference this
flag exists to measure.
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import checkpoint_log


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('exp_dir')
    ap.add_argument('--criterion', default='val_metric',
                    choices=list(checkpoint_log.CRITERIA_LOWER + checkpoint_log.CRITERIA_HIGHER))
    opts = ap.parse_args()
    try:
        print(checkpoint_log.best_checkpoint_file(opts.exp_dir, opts.criterion))
    except (FileNotFoundError, ValueError) as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
