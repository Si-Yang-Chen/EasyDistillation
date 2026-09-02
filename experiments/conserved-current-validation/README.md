# Conserved-current validation package

This directory separates two stages:

1. `produce_real_observables.py` validates an audited manifest and packages already
   contracted per-configuration/time Ward--Takahashi inputs into `observables.npz`.
2. `run_real_gauge_validation.py` analyzes that NPZ and writes tables, plots, and
   gated JSON results.

The producer intentionally **does not invent** Ward--Takahashi sides or charge
ratios from raw tensors. A valid real-data manifest must provide an independently
audited contraction NPZ with finite `wt_lhs`, `wt_rhs`, `charge_ratio`, and optional
`contact_term` arrays, plus hashes and configuration provenance. Existing
perambulators are inputs; this workflow does not recompute them.

Start from `manifest.template.json`. Replace every `REPLACE_*` value, use absolute
paths, and create a new result directory. First run `--dry-run`; only then run the
producer and analyzer. The analyzer's `--require-real-gauge` option rejects
synthetic fixtures and incomplete provenance.

The Slurm template is intentionally site-neutral. Fill scheduler directives,
module setup, and absolute variables only after local resource authorization.
