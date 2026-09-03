# Wilson directed-current V2V workflows

This directory contains three separate layers:

1. `generate_current_artifact_dcu.py` generates one eight-direction directed Wilson-current V2V artifact on a DCU node.
2. `contract_existing_vsv.py` and `contract_existing_vsv_pair.py` perform one-time or one-pair artifact smoke checks with an already audited VSV input.
3. `audit_fullsize_coverage.py`, `prepare_fulltime_execution_record.py`, `contract_fulltime_vsv_pair.py`, and `run_fulltime_vsv_pair_ensemble.py` prepare and run the full-`t` ensemble matrix once full-time VSV data exist.

The full-time output has axes `(first_current_anchor, second_current_anchor)` and is an ordered, connected, unflavored, unsigned raw V2V trace. It does not add a Wick sign, flavor/electric-charge factor, normalization, conjugation, real-part selection, source averaging, fit, Ward--Takahashi test, or charge-normalization interpretation.

## Full-time input gate

`audit_fullsize_coverage.py` is read-only. It checks the VSV/PSV/PSP rank slabs, manifests, shapes, dtypes, byte sizes, eight-configuration coverage, and the gauge/eigenvector/point/overlap headers. Default mode checks headers and file sizes. `--hash-files` additionally computes SHA-256 over all requested files and can read hundreds of GB.

For the current localized data layout, VSV and PSP have only 18 source times (`0,4,...,68`), while PSV has all 72. That is not sufficient for a full-time direct V2V pair. The full-time VSV manifest must declare every source time `0..71` before the pair production driver accepts it. PSV/PSP presence does not silently substitute for a missing VSV block.

Example read-only audit after the new storage is mounted:

```bash
python experiments/directed-current-v2v/audit_fullsize_coverage.py \
  --data-root /public/home/siyangchen/qedinf/data/beta6.20_mu-0.2770_ms-0.2400_L24x72 \
  --output /public/home/siyangchen/BASE/wilson-j4-fulltime/audit/coverage.json \
  --current-times 0-71
```

Run with `--hash-files` only when the additional I/O has been approved. The report is classified as a real-input coverage/cost audit, not a physics result.

## Full-time producer

The full-time VSV directory must use the `source-time-rank-slab` manifest convention. For each configuration, first generate a fresh source manifest from the exact committed source snapshot using the repository handover tool, then add the full-time scripts to that manifest's dependency set. The production source manifest must be generated after all script changes are committed and must record the actual Git commit/dirty state.

A typical source manifest preparation sequence is:

```bash
python tools/build_handover_manifest.py check-files --root . --files-from docs/delivery-files.list
python tools/build_handover_manifest.py build --root . \
  --output /absolute/audit/source-manifest.json \
  --files-from docs/delivery-files.list \
  --retention-decision docs/data-retention-decision.json \
  --accepted-git-ref origin/master \
  --accepted-git-ref origin/feature-stochastic \
  --published-git-ref origin/feature/wilson-current-j4
```

For a production source manifest, the delivery allowlist must include the full-time scripts and this README before the manifest is built.

The full-time VSV directory must use the `source-time-rank-slab` manifest convention:

```text
{configuration}.t{source_time:03d}.rank{rank:04d}.npy
shape: (local_time, 4, 4, sink_ne, source_ne)
disk axes: (t_sink_local, sink_spin, source_spin, sink_ne, source_ne)
time convention: absolute-global-source-and-sink
```

The producer requires:

- a content-addressed directed-current artifact for the same configuration;
- a source manifest covering the exact code dependencies and Git snapshot;
- an execution record generated inside the real Slurm job after `SLURM_JOB_ID` is available;
- a dedicated, previously nonexistent result directory.

The result directory is published atomically only after `correlator-<sha256>.npy`, `manifest.json`, `result.json`, and `DONE` are flushed. The monitor completion contract can therefore validate the result without repairing or guessing missing files.

The one-configuration wrapper is:

```bash
python experiments/directed-current-v2v/run_fulltime_vsv_pair_ensemble.py \
  --configuration 10000 \
  --current-artifact /absolute/current/10000/current-artifact \
  --vsv-directory /absolute/fulltime-vsv \
  --source-manifest /absolute/audit/source-manifest.json \
  --result-root /absolute/results \
  --current-ne 128 \
  --wilson-r 1.0 \
  --current-direction 3
```

The wrapper creates `result-root/<configuration>/execution-record.json` and `result-root/<configuration>/result/`. It is deliberately one configuration per process so an ensemble array can give every configuration an independent result lineage.

## Slurm preparation

`fulltime-production-contract.template.json` is the input/resource contract to fill after storage and a resource pilot are approved. `submit_directed_current_artifact_array.slurm.template` first produces one Current artifact per configuration; `submit_fulltime_vsv_pair_array.slurm.template` then consumes those artifacts and the full-time VSV directory. Both are templates, not submission commands. Before use, replace every `REPLACE_*` value, verify the source snapshot and worktree state, create a dedicated result root, and bind the exact current artifacts, full-time VSV directory, source manifest, `Ne`, Wilson `r`, and resource contract. The templates use one DCU, eight CPUs, one node, and array concurrency `1`; they must still pass the site preflight and monitor requirements in `.cursor/skills/dcu-slurm-submit/SKILL.md`.

No full-time production job is authorized merely because this template exists. First run the read-only audit, then perform an approved resource pilot, then submit the eight configuration jobs with a durable ledger and monitor. A completed matrix is an artifact result, not an ensemble physics conclusion.

## Existing VSV smoke

The existing smoke CLI supports a single rank-6 VSV or a complete rank-5 source-timeslice family. Its time axis must be explicitly declared as `source-relative` or `source-sink`. See `contract_existing_vsv.py` and `contract_existing_vsv_pair.py` for the exact execution-record and source-manifest checks.
