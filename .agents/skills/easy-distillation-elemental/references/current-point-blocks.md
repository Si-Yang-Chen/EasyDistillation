# Current and point-source elemental branch

Read this reference only when the user explicitly requests current elementals,
point-source blocks, or `v2v`, `v2p`, `p2v`, `p2p`. If a point table must be
created, use the `easy-distillation-sparsened-point` skill first and then treat
the validated `.npy` table as immutable input.

## Required choices and inputs

Confirm the requested blocks (`v2v`, `v2p`, `p2v`, `p2p`, or `all`), point-source
file/table, `Np`, `usedNp`, `usedNe`, displacement order (`num_nabla`), time
slices, and output paths. Point coordinates are zero-based `[x,y,z]` with input
shape `(Np,Lt,3)`. Do not infer a point source from a directory name. `v2v`
uses the ordered momentum list; `v2p`, `p2v`, and `p2p` are momentum-independent.

Only request dilution/blending parameters if the selected implementation and
user request enable blending. Do not show ordinary-elemental `num_derivative`
settings for a current-only request.

## Block contracts

For `num_disp` displacements and `Nc=3`, the driver writes:

| Block | Shape | Meaning |
|---|---|---|
| `v2v` | `(Lt,num_disp,num_mom,Ne,Ne)` | eigenvector-to-eigenvector current elemental |
| `v2p` | `(Lt,num_disp,Ne,Np,Nc)` | eigenvector sink to point source |
| `p2v` | `(Lt,num_disp,Np,Nc,Ne)` | point source to eigenvector source |
| `p2p` | one file per time slice | point-to-point sparse or identity blocks |

The preset loaders expose the v2v axis order as displacement, momentum, time,
sink eigenvector, source eigenvector; preserve the on-disk driver order in
metadata. `p2p` entries are displacement-major and contain either
`{"type":"identity"}` or a sparse record with `indices (K,2)` and
`values (K,Nc,Nc)`; `K=0` is valid.

## Confirmation and production

Before creating point/current outputs or submitting a job, include the selected
blocks, point-source path, `Np/usedNp`, ordered momentum list when applicable,
output files, and the same cfg Dispatch/resource summary required by the parent
skill. Wait for explicit submission approval. For multiple cfgs, each cfg task
produces the complete confirmed set of blocks; use either explicit Slurm array
selection or one uniquely suffixed MPI `Dispatch` queue, never both accidentally.

## Validation

Check each requested file or per-time p2p archive for readability, expected
shape, complex128 dtype (and integer sparse indices), finite values, populated
time slices, and correct block count. Verify `0 <= usedNe <= Ne` and
`0 <= usedNp <= Np`. For p2p, verify identity blocks and sparse entries against
the displacement order; an empty valid-pair set is not an error. Keep a
provenance sidecar recording point-source path, coordinates convention, block
list, displacement order, momentum list, cfg, and parameter hash.
