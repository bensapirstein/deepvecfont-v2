# Held-out validation splits

One JSON per language, written by `scripts/make_val_split.py`: which base fonts were
moved out of the train split, the seed that chose them, and how many augmented copies
went with them.

Tracked in git on purpose. The data itself lives on `/data/bens`, which has been pruned
twice already, and a checkpoint selected on a validation split nobody can reconstruct is
a checkpoint selected on nothing. If these files and the data ever disagree,
`make_val_split.py` re-run without `--apply` says so and exits non-zero.
