---
name: easy-distillation-eigensystem
description: Generate and dispatch covariant-Laplacian low-mode eigensystems for one or many gauge configurations, with explicit parameter confirmation and output validation.
---

# Eigensystem workflow

Use `scripts/gen_eigenvector.py` and `lattice.generator.EigenvectorGenerator`.
Input order is `[Lx,Ly,Lz,Lt]`; outputs are eigenvectors `[Lt,Ne,Lz,Ly,Lx,Nc]`
and eigenvalues `[Lt,Ne]`, little-endian complex128.

## Interaction gate

Inspect the supplied directory and any `conf_list`/`conf_text` before asking for
choices. Resolve actual cfg labels, filename prefix/suffix, lattice shape, file
format, and ensemble smearing metadata. Lists contain one cfg label per
non-empty, non-comment line. Do not invent a cfg, geometry, prefix, or stout
parameter. If several cfgs remain, show them and ask the user.

Separate discovered facts, user choices, and proposals. Required choices usually
include cfg scope, `Ne`, time slices, tolerance, stout parameters, backend,
projection/phase conventions, output root, and production Dispatch/resources.
A temporary output request does not authorize reducing cfg or time-slice scope.

Before any solver, `sbatch`, MPI launch, or Dispatch creation, present one final
summary containing exact cfg list and lattice; `Ne`, slices, tolerance, stout,
projection, phase, backend and Chebyshev; output and overwrite policy; host,
partition, ranks, CPU/GPU, memory, time limit, concurrency; Dispatch strategy,
task mapping, suffix, retry policy, and job count. Ask whether to submit/run with the summary
and wait for an affirmative answer (for example, `yes`, `submit`, or equivalent). A parameter value or output-path approval is
not submission approval. If a result/resource field changes, reconfirm.

## Single cfg and production

For one cfg invoke the driver with explicit `--cfg`; this bypasses Dispatch.
Save both arrays for a smoke run, validate them, then clean only the confirmed
temporary scope. `--tslice-list` leaves other rows zero; report populated slices
and never call such output complete for all `Lt`.

For multiple cfgs choose exactly one mode:

1. **Slurm array:** each element receives a disjoint manifest/shard and calls the
   driver with `--cfg`; preferred for independent single-process jobs.
2. **MPI Dispatch:** use `lattice.dispatch.Dispatch(cfglist, unique_suffix)`.
   Rank 0 atomically removes one cfg line and broadcasts it to all ranks. The
   cfg list, `.tmp` queue and outputs must be shared storage.

Do not combine array indexing and a shared Dispatch queue accidentally. Inspect
matching `.tmp` files and active jobs before reuse; keep a queue for a live run,
remove a stale one only after proving no corresponding job exists, and call
`clear()` only after all workers stop and the manifest is complete. Use the
`csns-submit` skill for code mirroring, submission, monitoring and resources.

## Manifest, recovery and validation

Production requires a persistent JSONL/TSV manifest beside outputs with parameter
hash, cfg, job/array ID, Dispatch suffix, paths, status
(`planned/submitted/running/failed/needs-validation/validated`), attempts, and
validation results. Never let active tasks share an output. Resume only after
checking scheduler state and existing files. Do not silently change `Ne`,
tolerance, backend, or resources after failure; propose and reconfirm.

Validate readability, shape, dtype, finite values, populated slices, eigenvalue
ordering, norms, orthogonality, and covariant-Laplacian residual. Compare
references using eigenvalues and subspace overlap, accounting for arbitrary
phases and degenerate-subspace rotations. Distinguish scheduler success,
generated, validated, and cleaned states; retain logs and manifest.

## Driver contract

The driver accepts `--cfg`, `--gauge-dir`, `--gauge-prefix`, `--latt-size`,
`--Ne`, `--tol`, `--nstep`, `--rho`, `--tslices`/`--tslice-list`,
`--backend {numpy,cupy}`, `--chebyshev`, `--project-su3`,
`--no-renorm-phase`, `--out`, and `--out-eval`. Require
`1 <= Ne < Lx*Ly*Lz*Nc`; CUDA/QUDA and Chebyshev require the GPU environment.
Smearing is spatial and must be applied exactly once per cfg according to the
confirmed convention.

## Production prerequisite routing

Before production, inspect the required cfg list, gauge files, metadata, and branch-specific products. If a prerequisite is missing, stop before submission and identify the producer skill: use easy-distillation-eigensystem for Laplace eigenvectors, easy-distillation-sparsened-point for point tables, easy-distillation-elemental for ordinary/current elementals, and easy-distillation-perambulator for VSV/PSV/PSP or density/generalized blocks. Propose the missing generation step and its parameters; do not silently create random or default prerequisite data. After generation, validate and record its path, parameter provenance, and content hash in the production manifest.

The CLI smearing defaults nstep=10 and rho=0.12 are reviewable proposals only. Other ensemble-dependent values must come from metadata or user confirmation. Filename-derived values are inferences and must be labeled and confirmed when physics-relevant.
