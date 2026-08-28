# archive/

One file, kept because the report cites it.

`FLOW_MATCHING_PLAN.md` is the design for a flow-matching output head over continuous
coordinates, which would remove the decoder's coordinate quantization and its sequential
factorization together. We designed it, costed it, and dropped it on schedule grounds
rather than on merit. Section 6.5 of the report proposes it as the next thing to try and
points here for what "it" is, so the plan travels with the report instead of being
described twice.

It is a plan, not an implementation. Nothing in this repository runs it, and the numbers
that motivate it are in section 6.5 and `scripts/quantization_oracle.py`.

The rest of this project's working record, including the plan this one was carved out of,
is on the `repro` branch. See `../docs/PROVENANCE.md`.
