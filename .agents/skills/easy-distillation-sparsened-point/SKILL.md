---
name: easy-distillation-sparsened-point
description: Generate reproducible sparse point-source coordinate tables for EasyDistillation point-source and current workflows.
---

# Sparsened point generation

Use `scripts/gen_sparsened_point.py`, which wraps
`lattice.generator.sparsened_point.generate_sparsened_points`.

## Inputs and confirmation

The user must supply or approve `[Lx,Ly,Lz,Lt]` and `Np`, the number of spatial
points per time slice. `Np` must satisfy `1 <= Np <= Lx*Ly*Lz`. A seed is
optional but should be proposed for reproducibility. If the lattice shape can
be discovered from the selected ensemble or eigenvector header, report it as a
discovered fact; do not infer `Np` from a downstream solver setting.

Before writing a file, summarize lattice, `Np`, temporal point policy, seed, output path, overwrite
policy, and whether the same table will be used as source, sink, or both. Wait
for explicit approval when any of these values was proposed. A temporary output
request does not authorize changing lattice or point count.

## Output contract

The saved `.npy` array has shape `(Np,Lt,3)`, dtype `int32`, and zero-based
coordinates ordered `[x,y,z]`. For every time slice, coordinates are unique and
within `0 <= x < Lx`, `0 <= y < Ly`, `0 <= z < Lz`. Validate these invariants
after writing and record lattice, `Np`, seed, generator version/commit, and a
content hash in a JSON sidecar.

The table is compatible with `PointSourceNpy` and the point branches of
`CurrentElementalGenerator` and `PerambulatorGenerator`. It is not a gauge
configuration, an eigenvector, or a momentum list. Do not regenerate it during
each cfg job: generate once, validate it, and reference the immutable file.

## Production and cleanup

For a production ensemble, keep one approved point table per point-set
definition and include its absolute path and hash in every cfg manifest. Do not
overwrite a table that active jobs may be reading. Temporary smoke tables may be
removed only after dependent validation and within the confirmed cleanup scope.

## Command

```text
python scripts/gen_sparsened_point.py --latt-size Lx Ly Lz Lt --num-points Np \
  --seed SEED --temporal-policy independent|same --out points.npy
```

Use `--seed` when results must be reproducible; omitting it produces a new
random table. The generator creates independent random spatial samples for
each time slice, so point index `p` does not identify a fixed spatial location
across time.
