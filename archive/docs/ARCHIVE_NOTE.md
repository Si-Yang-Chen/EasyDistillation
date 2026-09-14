# Archived documentation

**Archived:** 2026-09-14
**Reason:** the codebase changed substantially (quark-diagram monolithic refactor, current
insertion integrated, localized-blending production hooks ported, test layout reorganized),
so most of this documentation no longer describes the code. Rather than patch documents whose
underlying design has moved, the whole set was archived read-only and will be replaced by a
fresh documentation pass.

## Status: read-only

Do not edit these files. They are retained only so the original intent and derivations remain
traceable. Anything here may contradict the current implementation.

## Known drift in this set

- `PROJECT_ARCHITECTURE.md` still describes a `lattice/quark_diagram/` sub-package and names
  `hadron_irrep.py`, `quark_contract.py`, and `filedata/sliceloader.py`. All four were removed.
  It also describes `openspec/`, which was removed.
- `FILEDATA_DETAILED.md` documents `filedata/sliceloader.py`, which was removed.
- `DOCUMENTATION_INDEX.md` links to `openspec/` and to `doc/` paths; both are gone.
- `propagator_theory_and_usage.md` and several `preset.py` docstrings claim that `load()`
  returns an assembled full-shape array. It returns a lazy `FileData` handle; materialize it
  with `[:]` (or `[(t,)]` for timeslice loaders, whose suffix must contain `.t???`).
- `WORKFLOW_ANALYSIS.md` and `DISTILLATION_WORKFLOW.md` reference `tests/`, which was renamed
  to `test/`.
- `example-README.md` documents example scripts by name; the example directory contents have
  since changed.
- `localized_blending/*` predates the current-aware contraction integration.

## Current, non-archived documentation

Only these remain live and are not part of the archive:

- `README.md` (required by `pyproject.toml`'s `readme` field)
- `CHANGELOG.md`

## Provenance

Every file here was moved with `git mv`, so `git log --follow <path>` still reaches its full
history.
