---
name: easy-distillation-perambulator
description: Generate perambulators (VSV/PSV/PSP and density/generalized VSSV blocks) with the EasyDistillation lattice QCD framework on GPU via PyQUDA/QUDA. Use when running distillation-framework perambulator generation in a PyQUDA 0.10.x + CuPy environment, configuring PerambulatorGenerator / DensityPerambulatorGenerator / GeneralizedPerambulatorGenerator, or staging small smoke runs before production job arrays.
---

# EasyDistillation perambulator generation

## Branch selection and submission gate

The default request is plain VSV. Point-source branches (PSV/PSP), density, and
generalized VSSV are opt-in; inspect point-source inputs only after the user
selects one. Before any GPU launch, MPI start, scheduler submission, or output
creation, summarize the resolved cfg list, generator/mode, all input files,
mass, solver tolerances, source times, backend, smearing, output paths,
resources, and distribution strategy. Wait for explicit approval to submit/run.
A supplied parameter or output directory is not submission approval; reconfirm
after any physics or resource change.

## 1. Required interaction

Use a `conf_list` or `conf_text` file when one is supplied; each non-empty,
non-`#` line is one cfg label. For an authorized one-cfg smoke test, use the
only unambiguous cfg discovered in the data when no list exists. If several
cfgs remain, show them and ask which one to test. Use one cfg per invocation and
keep output paths cfg-specific.

For unspecified plain-perambulator work, use `generator=perambulator` and
`mode=vsv` as the baseline. Point sources are out of scope by default: do not
ask for, scan, or configure point-source/sink files unless the user explicitly
requests a point-source product or names `PSV`/`PSP`. Density and generalized
calculations remain separate opt-in branches and must be named explicitly.

The user must provide the gauge-data directory. Unless another branch is
explicitly requested, use `generator=perambulator`, `mode=vsv`, and no point
sources. The generator/mode pairs are:

| Generator | Compatible mode(s) | Required data / choices |
|---|---|---|
| `perambulator` | `vsv`, `psv`, `psp` | source eigenvectors for VSV/PSV; point source **and** point sink for PSP; a point sink is also required for PSV |
| `density` | `density` | source eigenvectors, `gamma_list`, momentum list, `ti`, `tf`, `tau` |
| `generalized` | `generalized` | source eigenvectors, `gamma_list`, momentum list, `ti`, `tf` |

Inspect the input directory before proposing a command. If only a complete file
prefix is provided, use its containing directory and preserve the exact
prefix/suffix.

After receiving the gauge directory and, when needed, the source-eigenvector
directory, discover and validate:

- the configuration label(s), gauge prefix/suffix, and lattice
  `[Lx,Ly,Lz,Lt]` from filenames or metadata;
- the exact source/sink eigenvector files and their `.npy` header shapes and
  dtypes; infer `Ne` from axis 1 rather than asking the user to repeat it;
- the exact point-source/sink files and their `[Np,Lt,3]` shapes only when a
  point branch is explicitly selected; and
- whether the selected generator/mode has every required input.

If a file, lattice extent, action parameter, or smearing convention cannot be
resolved, say exactly what is missing and ask for it. Do not add point sources
or a point sink unless the user opts into PSV/PSP. For VSV, use
`products=("VSV",)` explicitly.

After inspection, show a short resolved configuration. Mark values as
discovered, user-supplied, proposed, or derived. Include the extra lines only
for density/generalized or explicitly requested point branches:

```text
Perambulator configuration
- generator/mode: perambulator / vsv — default plain branch [proposal]
- gauge-dir and cfg: <values> — input directory and selected configuration [user/discovered]
- lattice: [Lx,Ly,Lz,Lt] — discovered lattice size [discovered]
- gauge/eigenvector files: <prefix/suffix>, shape=<shape>, Ne=<Ne> — inputs [discovered]
- usedNe_src: <value> — retained source modes; lower values reduce test cost [user/proposal]
- mass / tol / maxiter: <values> — solver settings [user/metadata/unresolved]
- stout and boundary: <values> — ensemble smearing and temporal boundary [metadata/proposal]
- source times: <list> — slices used in this run [proposal/user]
- output: <temporary path> — deleted after smoke validation [derived]
```

`mass`, `tol`, and `maxiter` are site/ensemble specific: recover them from
trusted metadata or ask the user; never invent them. For density/generalized
generators, stout is not implemented, so use `stout nstep=0`.

## 2. Test and production

For a smoke test, use temporary output and a small branch:

- plain VSV (default): `generator=perambulator`, `mode=vsv`, `usedNe_src=4`
  (capped at the discovered `Ne`), `t_src=[0,1]` when available,
  `products=[vsv]`, `same_eigenvector=True`, vectorized calculation, and no
  point-source inspection;
- plain PSV/PSP: only when explicitly requested; then inspect only the required
  point/eigenvector inputs;
- density: `gamma_list=[0]`, momentum `[(0,0,0)]`, `ti=0`, `tf=1`, and a
  user-approved `tau` within the discovered time extent;
- generalized: `gamma_list=[0]`, momentum `[(0,0,0)]`, `ti=0`, `tf=1`.

The driver still allocates the full output shape in a smoke run. Check shape,
dtype, and (for VSV) the optional `--readback-check` before deleting temporary
outputs. For production, use the requested `Ne`, all requested source times,
one cfg per invocation, and unique paths. Use a scheduler array for many cfgs.

Code of record: `lattice/generator/perambulator.py`, `lattice/generator/density_perambulator.py`,
`lattice/generator/generalized_perambulator.py` in the EasyDistillation repo (verified at git `e19a399`).
`docs/modules/generator.md` is a useful map, but the code wins on any conflict.

## Environment

- Python (3.11 used for the reference timings below) with numpy, scipy, opt_einsum, mpi4py,
 `pyquda` + `pyquda_utils` 0.10.x, and `cupy` (13.x used for the reference timings).
- A CUDA GPU with a working PyQUDA/QUDA install is required (`set_backend("cupy")`,
 `check_QUDA` must pass).
- **Always set `CUDA_VISIBLE_DEVICES=<dev>`** — with it unset, CuPy init fails.
- On some systems the CUDA driver init (`cuInit`) fails in a bare non-login remote shell even with
 `CUDA_VISIBLE_DEVICES` set; run GPU jobs through a login shell (`bash -l -c '...'`) if so.
- MPI: `mpirun` may not be on PATH at every site (e.g. an MPI module with a missing modulefile).
 pyquda initializes fine as an MPI singleton (`rank 0 / size 1`), so single-GPU jobs run
 **without** mpirun. If you must multi-rank, locate a matching OpenMPI first.
- Batch: submit jobs with your site's scheduler (e.g. `sbatch -p <partition> -n 1 --mem 60G`, where
 `<partition>` is your site's partition name); pick an idle GPU partition/queue for GPU work.
- Stay inside your own work directory; never write into another analysis' data directories.

## Typical input files

| Object | Preset class | File | Shape / dtype |
|---|---|---|---|
| Gauge | `GaugeFieldIldg` | QIO/LIME `{prefix}{cfg}{suffix}` | `[Lt, Lz, Ly, Lx, Nd, Nc, Nc]`, `>c16` (`lattice/preset.py`) |
| Eigenvector | `EigenvectorNpy` | `.npy` | `[Lt, Ne, Lz, Ly, Lx, Nc]`, complex (`test/test_perambulator.py` layout) |
| Optional point source/sink | `PointSourceNpy` | `.npy` | `[Np, Lt, 3]` int, zero-based `[x, y, z]` (`lattice/preset.py`) |
| Generated perambulator | `PerambulatorNpy` | `.npy` | `[Lt, Lt, Ns, Ns, Ne, Ne]`, `<c8` (`lattice/preset.py`) |

All generator buffers/outputs use `contract_prec` (default little-endian `complex128`, `"<c16"`).

## 3. Execution workflow

1. **Staging** — put an EasyDistillation checkout (or unpacked source archive) into a work dir;
 `import lattice` works from that root (or pass `--ed-path <repo_root>` to `gen_peram.py`).
2. **Smoke** — after resolving the actual cfg, mass, tolerance, and iteration limit, use a
 small lattice/Ne and one or two `t_src`, e.g.
 `bash -l -c 'cd <workdir> && CUDA_VISIBLE_DEVICES=0 python scripts/gen_peram.py --cfg <cfg>
 --latt-size 4 4 4 8 --gauge-prefix test/weak_field --gauge-suffix .lime
 --eig-src-prefix test/weak_field --eig-src-suffix .eigenvector.input.npy --mass <mass>
 --tol <tol> --maxiter <maxiter> --used-ne-src 4 --t-src 0 1 --mode vsv
 --products vsv --out out/smoke.npy'`.
3. **Production** — full `Ne`, all `t_src`, one config per invocation. For the plain
 `perambulator` generator, stout-smear once before the `t_src` loop when the ensemble
 convention requires it; smearing is in-place and shared across all source times.
 Density/generalized runs keep stout disabled because their current generators do not
 implement smearing.
4. **Job array** — one task per config key (`--cfg`); keep `--out` unique per config. A single
 invocation loops over all `t_src` internally; do not split `t_src` across MPI ranks (each rank
 only fills sinks it owns — see Pitfalls).

## Call sequence (PerambulatorGenerator)

```python
from lattice import set_backend, check_QUDA
set_backend("cupy"); assert check_QUDA
from lattice import PerambulatorGenerator, GaugeFieldIldg, EigenvectorNpy
# Import PointSourceNpy only for an explicitly requested PSV/PSP branch.

gen = PerambulatorGenerator(latt_size=[Lx,Ly,Lz,Lt], gauge_field=..., mass=..., tol=..., maxiter=...,
                    eigenvector_src=..., point_src=..., point_snk=...,...)
gen.load(cfg) # gauge -> QUDA handle; eigenvectors -> cb2 V-dagger on device
gen.stout_smear(nstep, rho, dir_ignore=3) # optional, GPU only, after load
vsv, _ = gen.calc_new(t_src, products=("VSV",))
# Explicit point branches use products=("PSV",) or calc_point_sources(...).
gen.dirac.destroy
```

On-disk convention: `peramb[t_src] = np.roll(calc(t_src).get, -t_src, axis=0)`
(`test/test_perambulator.py`) so axis 1 is the relative sink time `(t_snk - t_src) mod Lt`.

## Parameter branch matrix - PerambulatorGenerator

Signature: `lattice/generator/perambulator.py`. All values below verified against code.

| Parameter | Default | Branches / notes (code ref) |
|---|---|---|
| `latt_size` | — | `[Lx, Ly, Lz, Lt]` order |
| `gauge_field` | — | `GaugeField` preset handle; read in `load` into `gauge_field_smear` |
| `mass` | — | → `core.getDirac`; internally kappa = 1/(2(mass + 1 + (Nd-1)/anisotropy)) in pyquda `dirac/clover_wilson.py`. **No `kappa` parameter exists.** |
| `tol` | — | solver tolerance |
| `maxiter` | — | solver max iterations |
| `xi_0` | `1.0` | anisotropy numerator; `LatticeInfo.anisotropy = xi_0/nu` |
| `nu` | `1.0` | anisotropy denominator |
| `clover_coeff_t` | `0.0` | temporal clover coefficient |
| `clover_coeff_r` | `1.0` | spatial clover coefficient; in pyquda `getDirac`, if anisotropy != 1: `csw = xi_0*ct^2/cr`, `clover_xi = sqrt(xi_0*ct/cr)`; else `csw = clover_coeff_t` (`pyquda_utils/core.py`). `csw == 0` → plain Wilson (no clover) |
| `t_boundary` | `1` | Literal `1` (periodic in t) or `-1` (anti-periodic); goes into `LatticeInfo` |
| `multigrid` | `None` | `None`/falsy → disabled; a list like `[[4,4,4,4],[4,4,4,4]]` enables QUDA MG; a non-list truthy value falls back to `[[2,2,2,2],[4,4,4,4]]` (`pyquda_utils/core.py`) |
| `contract_prec` | `"<c16"` | dtype of every buffer/output |
| `eigenvector_src` | `None` | `None` ⇒ `Ne_src = 0` ⇒ no eigenvector inversions (point-source-only run) |
| `usedNe_src` | `None` | defaults to `eigenvector_src.Ne`; explicit value ≠ file Ne only **prints a warning** |
| `eigenvector_snk` | `None` | ignored when `same_eigenvector=True` (forced to `eigenvector_src`); needed only for unequal source/sink bases |
| `usedNe_snk` | `None` | defaults to `eigenvector_snk.Ne` (or `Ne_src` if same); **no** mismatch warning here |
| `point_src` | `None` | `PointSource` handle; only used by `calc_point_sources` (PSP) |
| `point_snk` | `None` | sink sample table; drives PSV extraction indices |
| `usedNp_src` / `usedNp_snk` | `None` | default to `point_src.Np` / `point_snk.Np`; 0 if handle is None |
| `MRHS` | `False` | `True` → QUDA multi-RHS: one `MultiLatticeFermion` with Ns RHS per eigenvector, and 12 spin-color RHS per point source; needs more device memory |
| `use_vectorized` | `True` | `calc` dispatch: `True` → `calc_new` (vectorized PSV extraction), `False` → `calc_old` (sequential reference) |
| `same_eigenvector` | `True` | see `eigenvector_snk` above |

Method-level branches:

- `stout_smear(nstep, rho, dir_ignore=3)`: numpy backend → `NotImplementedError`;
 cupy → QUDA `smearSTOUT` on the gauge loaded by `load`. Calling before `load` → `ValueError`.
- `calc_new(t_src, products=("VSV","PSV"))`: `products` whitelist `{"VSV","PSV"}`
 (case-insensitive); requesting `VSV` without a sink eigenvector or `PSV` without `point_snk`
 raises `ValueError`. Returns `(VSV, PSV)` — buffers are **reused**, copy if you keep them.
- `calc(t_src)` = `calc_new(t_src)` with the default `products` — so it **raises** when
 `point_snk` is absent (PSV requested unconditionally). Use `calc_new(t, products=("VSV"))`.
- `calc_point_sources(t_src)`: requires both `point_src` and `point_snk` and
 `Np_src > 0` and `Np_snk > 0` (else buffers are `None`); point-source RHS batches of
 12 (MRHS) or 1; fills only sinks owned by this rank.
- `calc_old`'s PSV extraction is broken; its VSV part is fine. For PSV always use `calc_new`.

## Density vs Generalized vs plain

| | `PerambulatorGenerator` | `DensityPerambulatorGenerator` | `GeneralizedPerambulatorGenerator` |
|---|---|---|---|
| Init | `perambulator.py` | `density_perambulator.py` | `generalized_perambulator.py` |
| Extra params | src/snk eigenvectors, points, MRHS, use_vectorized, same_eigenvector | `gamma_list` (default `0..15` bitmask), `momentum_list` (default `[(0,0,0)]`) | identical to density |
| QUDA check at init | commented out | hard `raise ImportError` | hard `raise ImportError` |
| `load` | gauge + cb2 V† on device + point tables | gauge straight into Dirac; no cb2 | same as density |
| Smearing | `stout_smear` (QUDA) | **not implemented** (TODO) | not implemented |
| Calc | `calc(t_src)` / `calc_new` / `calc_point_sources` | `calc(ti, tf, tau)` | `calc(ti, tf)` |
| Output shape | VSV `[Lt,Ns,Ns,Ne_snk,Ne_src]`; PSV `[Lt,Ns,Ns,Np_snk,Nc,Ne_src]`; PSP `[Lt,Ns,Ns,Np_snk,Nc,Np_src,Nc]` | `[N_gamma, N_mom, Lz, Ly, Lx, Ns, Ns, Ne_f, Ne_i]` | `[N_gamma, N_mom, Lt, Ns, Ns, Ne_f, Ne_i]` |
| Physics | perambulator tau blocks | gamma5-conjugated sink × Gamma × source, fixed sink time `tau`, per-space-point | same contraction, all sink times (pinned host buffers + 2 CUDA streams) |
| gamma5 conjugation | n/a | applied inside the eigen loop — suspected double application, see Pitfalls | applied once per inversion |

## Output contract

- VSV (perambulator): single `.npy`, `[Lt, Lt, Ns, Ns, Ne_snk, Ne_src]`, `<c16`;
 `peramb[t_src, t_rel, s_snk, s_src, e_snk, e_src]` → loadable with `PerambulatorNpy(prefix, suffix, shape, Ne)`.
- PSV: `[Lt(t_src), Lt(t_rel), Ns, Ns, Np_snk, Nc, Ne_src]` → `PropagatorPSVNpy` full shape
 (`lattice/preset.py`). VSP is not generated; reconstruct from PSV (`lattice/propagators.py`).
- PSP: `[Lt(t_src), Lt(t_rel), Ns, Ns, Np_snk, Nc, Np_src, Nc]` → `PropagatorPSPNpy` full shape.
- Density VSSV: `[N_gamma, N_mom, Lz, Ly, Lx, Ns, Ns, Ne_f, Ne_i]`;
 Generalized VSSV: `[N_gamma, N_mom, Lt, Ns, Ns, Ne_f, Ne_i]` (no preset loader; save/load raw npy).
- `gen.last_metrics` (after `calc_new`/`calc_point_sources`) reports `inversion_seconds`,
 `vsv_seconds`, `psv_seconds`, `elapsed_seconds`, `peak_device_used_bytes`.

## Runtime (reference order-of-magnitude: 1x 32GB-class GPU, single CPU core, 4^3x8 lattice)

| Step | Measured |
|---|---|
| pyquda/QUDA init (`check_QUDA`, singleton MPI) | ~1.7 s (`initQuda Total time`) |
| `stout_smear(nstep=20, rho=0.1)` on 4^3x8 | 0.41 s (2 steps: 0.38 s) |
| `calc_new(t_src)`, Ne_src=4, first t_src (incl. `loadGauge` + kernel tuning) | 1.46 s total, inversion 1.05 s |
| `calc_new(t_src)`, Ne_src=4, subsequent t_src | 0.11 s total, inversion 0.10–0.11 s each |

Scale from these: per `t_src` the cost is ~`Ne_src` Dirac inversions (Ns spin RHS each; MRHS
batches them) plus one einsum per product. PSP adds `Np_src * Ns*Nc` point-source inversions per
`t_src`. Log `gen.last_metrics` per t_src and extrapolate to production sizes. GPU init can be
intermittent on some systems (see Pitfalls #1) — when a run aborts at import time, retrying after
a short wait generally succeeds.

## Pitfalls

1. **`CUDA_VISIBLE_DEVICES`** must be `0` (or a concrete device) — CuPy aborts otherwise; on some
 systems the CUDA driver init additionally needs a login shell (`bash -l -c`). `cuInit` can
 intermittently return error 3 on idle GPUs (worked, then stopped); retry or move to a
 GPU-enabled allocation. Do not fake timings when this happens.
2. **`gen.calc(t_src)` fails without a point sink** (PSV in default products → `ValueError`,
 `perambulator.py`). Pass `products=("VSV")`.
3. `calc_old` (`use_vectorized=False`) has a broken PSV path; VSV only.
4. **Multi-rank MPI**: a rank writes only eigen/time slices it owns (`gt*Lt <= t_src < (gt+1)*Lt`) and only sinks it owns; per-rank arrays must be combined with
 `gatherLattice(peramb, axes=[1,-1,-1,-1], reduce_op="sum", root=0)`
 (`test/test_perambulator_mpi.py`). With MPI size > 1 a naive rank-0 save is *partial*.
5. `mpirun` may be missing from PATH at some sites; pyquda 0.10.2 works as an MPI singleton for -np 1.
6. Eigenvector files store V, but the generator keeps **V† on device** (`load`); source
 insertion re-conjugates. Don't pre-conjugate inputs.
7. `stout_smear` is QUDA-only here (no ndarray fallback), is in-place, and must run
 after `load`; it is shared by all subsequent `t_src` calls (smear once per config).
8. Density generator traps: `gamma_list` entries are gamma **bitmasks** `0..15` (`gamma(15)` =
 gamma5); the `(ti, tf)` cache ignores `tau` — with the same `(ti, tf)` and a
 new `tau` the inversions are **not** redone, giving stale results; the gamma5 sink conjugation is
 applied inside the eigen loop and is suspected to compound (density outputs should be
 treated with suspicion until cross-checked; Generalized does not have this issue).
9. `usedNe_src` mismatch with the file's Ne only warns; `usedNe_snk` mismatch is silent.
10. `PerambulatorNpy` default dtype is `<c8` while generation buffers are `<c16`; give the loader an
 explicit shape and cast consistently.
11. returned blocks are reused buffers — `np.asarray(x.get()).copy()` before the next `calc` call.
12. **`same_eigenvector=True` shares one basis, so the sink cannot exceed the source.**
 Truncating the source (`usedNe_src < file Ne`) makes the sink follow that truncation, and
 asking for a larger `usedNe_snk` now raises a `ValueError` saying why. Do not set them
 inconsistently; `gen_peram.py` keeps them equal.

## Script

`scripts/gen_peram.py` — argparse CLI covering all branches above
(`--generator {perambulator,density,generalized}`, `--mode {vsv,psv,psp,density,generalized}`,
`--used-ne-src/snk`, `--point-*-prefix/np`, `--mRHS`, `--no-vectorized`, `--no-same-eigenvector`,
`--stout-*`, `--t-boundary`, `--multigrid "4 4 4 4 4 4 4 4"`, `--gamma-list`, `--momentum-list`,
`--ti/--tf/--tau`, `--products`, `--readback-check`). It probes eigenvector `.npy` shapes from the
real cfg file header, rolls outputs into the on-disk `[t_src, t_rel]` convention, and can reload its
own VSV output through `PerambulatorNpy` for a round-trip check.

## Production dispatch and recovery

For production, write a persistent manifest beside outputs with parameter hash, cfg, job or array ID, output paths, status (planned/submitted/running/failed/needs-validation/validated), attempts, and validation metrics. Choose either explicit Slurm array cfg selection or a uniquely suffixed MPI Dispatch queue; do not combine them accidentally. Inspect matching .tmp files and active jobs before reuse, retain a live queue, and call clear only after all workers stop. Resume only after checking scheduler state and existing files; do not silently change mass, tolerances, mode, backend, or resources after failure.


## Point-table dependency

When PSV or PSP is explicitly requested and no point table exists, use the easy-distillation-sparsened-point skill first. Treat its validated .npy output as immutable shared input; never generate a new random table inside each cfg job.


## Production prerequisite routing

Before production, inspect the required cfg list, gauge files, metadata, and branch-specific products. If a prerequisite is missing, stop before submission and identify the producer skill: use easy-distillation-eigensystem for Laplace eigenvectors, easy-distillation-sparsened-point for point tables, easy-distillation-elemental for ordinary/current elementals, and easy-distillation-perambulator for VSV/PSV/PSP or density/generalized blocks. Propose the missing generation step and its parameters; do not silently create random or default prerequisite data. After generation, validate and record its path, parameter provenance, and content hash in the production manifest.

The CLI stout defaults are reviewable proposals only. Resolve mass, clover/action, boundary, anisotropy, solver, and other ensemble-dependent values from metadata; filename suggestions are inferences and require confirmation before execution.
