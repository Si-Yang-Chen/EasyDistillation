---
name: easy-distillation-skill-testing
description: Validate an EasyDistillation project skill with a real smoke run, configuration-list handling, comparison against complete existing data, and controlled artifact cleanup. Use when testing or reviewing any easy-distillation eigensystem, elemental, perambulator, or correlator skill.
---

# EasyDistillation skill-testing workflow

Use this skill to test another project skill as an agent would use it. The test
must validate behavior and outputs, not only whether the prose sounds correct.

## 1. Read and isolate

Read the target skill first. Do not inspect unrelated project source until the
target skill has determined what it requires. Use a separate temporary output
directory. Do not modify ensemble data, existing products, or the target skill
during the test.

## 2. Resolve inputs before asking

Inspect the supplied data directory before asking for values that can be
discovered. Resolve and report:

- actual cfg filenames, prefixes, suffixes, and lattice shape;
- ensemble-specific physics metadata such as smearing parameters;
- `conf_list` or `conf_text` when supplied; it contains one cfg label per
  non-empty, non-comment line. For an authorized one-cfg smoke test, derive the
  only unambiguous cfg from the data when no list exists;
- the exact cfg sequence to be processed.

If no configuration-list file exists, create a temporary one-line list only when
the user authorized a one-cfg smoke test and the driver requires it. Do not
invent a production list. If several cfgs remain ambiguous, ask the user; if a
directory or metadata source cannot be accessed, report the concrete failure and
stop rather than using placeholders or guesses.

## 3. Confirm parameters with meaning

Separate parameters into three groups:

1. discovered facts, which the agent reports;
2. defaults or test proposals, which the user may review;
3. required choices, which the user must provide or approve.

For every proposed or required parameter, state its physical or computational
meaning and its effect on output, accuracy, memory, or runtime. Do not ask the
user to reconfirm values that were unambiguously discovered. Do not silently
apply a generic default when ensemble metadata is unresolved.

## 4. Run a real smoke test

Use the target skill's own driver or documented command. Prefer a small but real
run: reduced mode counts, a short explicit time list, and one cfg from the
resolved list. Record the exact command, backend, branch, input cfg, parameters,
wall time, warnings, and exit status.

Before starting, verify that the selected backend is feasible for the actual
diagram and time extent. If the host provides the project's GPU/CuPy path, do
not silently fall back to full-Lt NumPy. A process that is interrupted or leaves
no output is a failed run, regardless of successful preparation or binding.

The test must exercise the target skill's main output path. A dry-run, parser
check, or statement that a command “would run” is not a completed test.

When a delegated agent is expected to create or modify a remote driver, verify
the agent can actually write the selected copy and report its content hash or
diff. A main-agent-only fallback is a workflow failure even if the fallback
calculation succeeds. For Dispatch-capable drivers, use a unique one-line
smoke cfg list, inspect the generated `.tmp` queue, confirm the cfg is consumed
exactly once, and remove only that test queue after the process exits.

## 5. Compare with complete existing data

Before cleanup, if a reference script or driver was supplied, read that exact
file first (the main agent may read it even when a delegated implementation
agent is confined to an isolated code copy). Resolve its actual `save` path,
`PROJECT_ROOT`, output directory, filename template, and parameter mapping.
Search the resolved reference directory, including project-level directories
outside the gauge-data root. Do not use the temporary smoke output directory
as the reference location merely because it has a similar filename.

Then search the data and product directories for a complete reference product
for the same cfg or ensemble. Compare as applicable:

- shape and axis order;
- dtype and precision;
- populated cfg/time ranges;
- metadata and parameter provenance;
- finite values, norms, overlaps, or numerical residuals;
- a matching subset when the smoke run uses fewer modes or time slices.

First compare the output contracts: channel, operator set, axis order, shape,
dtype, cfg, source-time range, sink-time range, and mode truncation. A successful
run with a different contract is only `generated/incomparable`; it is not a
validation. Explain any difference caused by deliberate truncation or branch
selection. If no reference product exists, say so explicitly and report the
resolved reference path and every candidate path checked.

For any group-element axis, compare the recorded ordered element names before
comparing values. If the names are a permutation, reorder by labels and report
the permutation; do not silently compare positional slices.

## 6. Keep artifacts until comparison is finished

Never delete test outputs immediately after generation. Retain them through
reference discovery, comparison, and reporting. Delete temporary outputs and
temporary configuration lists only after comparison succeeds and the user has
not asked to retain them. If the user says to retain them, leave them in place
and report their exact paths.

## 7. Final report

Report in this order:

1. discovered input files and cfg sequence;
2. confirmed and proposed parameters with their meanings;
3. exact test command and runtime;
4. output shape, axis order, dtype, and populated range;
5. reference product and comparison metrics;
6. warnings, limitations, and cleanup status.

Do not claim success if execution, comparison, or cleanup was skipped. Distinguish
“generated”, “validated against reference”, and “cleaned up” as separate states.
