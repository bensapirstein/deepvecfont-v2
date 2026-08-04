#!/usr/bin/env python3
"""Print the best checkpoint filename for one experiment, by val_metric.

Reads experiments/<name>_main_model/logs/checkpoint_metrics.csv (written by train.py at
every checkpoint save) and falls back to legacy filename parsing for experiment dirs that
predate that manifest. See checkpoint_log.py.

Usage: python scripts/best_checkpoint.py experiments/<name>_main_model
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import checkpoint_log


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    exp_dir = sys.argv[1]
    try:
        print(checkpoint_log.best_checkpoint_file(exp_dir))
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
